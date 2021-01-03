import ray
import multiprocessing as mp
import threading
from queue import Queue as ThreadSafeQueue
from torch import from_numpy
from scipy.signal import lfilter
import numpy as np


class RolloutWorker:
    def __init__(self, worker_id, model_fn, worker_env_fn, ps, kwargs):
        self.id = worker_id
        self.verbose = kwargs['verbose']

        self.env = worker_env_fn(worker_id, kwargs)
        self.model = model_fn(kwargs)

        self.ps = ps
        self.weight_hash = None
        self.get_new_weights()

        self._data_g = self._data_generator()

    def get_new_weights(self):
        # if model parameters in parameter server is different from local model, download it
        weights_job = self.ps.get_weights.remote()
        weights, weights_hash = ray.get(weights_job)
        self.model.load_state_dict(weights)
        old_hash = self.weight_hash
        self.weight_hash = weights_hash
        if self.verbose:
            print("RolloutWorker {} load state dict, "
                  "hashing changed from "
                  "{} to {}".format(self.id, old_hash, self.weight_hash))
        del weights_job, weights, weights_hash

    def _data_generator(self):
        self.pull_job = self.ps.pull.remote()
        model_inputs = self.env.get_model_inputs()
        while True:
            model_outputs = self.model.select_action(*model_inputs)
            datas, infos, model_inputs = self.env.step(*model_outputs)
            if len(datas) == 0 or len(infos) == 0:
                continue
            yield datas, infos
            # get new weights only when at least one of vectorized envs is done
            if ray.get(self.pull_job) != self.weight_hash:
                self.get_new_weights()
            self.pull_job = self.ps.pull.remote()

    def get(self):
        return next(self._data_g)


class PostProcessor:
    def __init__(self, processor_id, rollout_keys, collect_keys, model_fn, worker_env_fn, ps, kwargs):
        assert kwargs['num_workers'] % kwargs['num_postprocessors'] == 0
        self.num_workers = kwargs['num_workers'] // kwargs['num_postprocessors']
        self.workers = [
            ray.remote(num_cpus=kwargs['cpu_per_worker'])(RolloutWorker).remote(worker_id=i + kwargs['num_postprocessors'] * processor_id,
                                                                                model_fn=model_fn,
                                                                                worker_env_fn=worker_env_fn,
                                                                                ps=ps,
                                                                                kwargs=kwargs)
            for i in range(self.num_workers)
        ]
        self.working_jobs = []
        # job_hashing maps job id to worker index
        self.job_hashing = {}
        self.ready_queue = ThreadSafeQueue(maxsize=self.num_workers)

        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.chunk_len = kwargs['chunk_len']

        self.stored_chunk_num = 0
        self.stored_infos = []
        self.stored_concat_datas = []

        self.min_return_chunk_num = kwargs['min_return_chunk_num'] * self.num_workers
        self._generator = self._data_generator()

        self.rollout_keys = rollout_keys
        self.collect_keys = collect_keys

    def collect_traj(self):
        # iteratively make worker active
        for i in range(self.num_workers):
            job = self.workers[i].get.remote()
            self.working_jobs.append(job)
            self.job_hashing[job] = i
        while True:
            [ready_job], self.working_jobs = ray.wait(self.working_jobs, num_returns=1)

            worker_id = self.job_hashing[ready_job]
            self.job_hashing.pop(ready_job)

            new_job = self.workers[worker_id].get.remote()
            self.working_jobs.append(new_job)
            self.job_hashing[new_job] = worker_id

            self.ready_queue.put(ready_job)

    def _data_generator(self):
        collect_job = threading.Thread(target=self.collect_traj, daemon=True)
        collect_job.start()
        while True:
            ready_job = self.ready_queue.get()
            datas, infos = ray.get(ready_job)
            self.stored_infos += infos
            for seg, bootstrap_value in datas:
                eplen = len(seg[0])
                value_target, adv = self.compute_gae(getattr(seg, 'reward'), getattr(seg, 'value'), bootstrap_value)
                blks = []
                for k in self.collect_keys[:-1]:
                    if k in self.rollout_keys:
                        blks.append(getattr(seg, k).reshape(eplen, -1))
                    elif k == 'value_target':
                        blks.append(value_target)
                    elif k == 'adv':
                        blks.append(adv)
                    else:
                        raise NotImplementedError
                storage_block = np.concatenate(blks, axis=-1)
                self.stored_concat_datas.append(self.to_chunk(storage_block, getattr(seg, 'rnn_hidden', None)))

            if self.stored_chunk_num >= self.min_return_chunk_num:
                storage_blocks, rnn_hiddens = zip(*self.stored_concat_datas)
                storage_blocks = np.concatenate(storage_blocks, axis=1)
                if rnn_hiddens[0] is not None:
                    rnn_hiddens = np.concatenate(rnn_hiddens, axis=1)
                else:
                    rnn_hiddens = None
                yield (storage_blocks, rnn_hiddens), self.stored_infos

                self.stored_concat_datas = []
                self.stored_infos = []
                self.stored_chunk_num = 0
    
    def to_chunk(self, storage_block, redundant_rnn_hidden=None):
        chunk_num = int(np.ceil(len(storage_block) / self.chunk_len))
        target_len = chunk_num * self.chunk_len
        if len(storage_block) < target_len:
            pad = tuple([(0, target_len - len(storage_block))] + [(0, 0)] * (storage_block.ndim - 1))
            storage_block = np.pad(storage_block, pad, 'constant', constant_values=0)
        storage_block = storage_block.reshape(self.chunk_len, chunk_num, *storage_block.shape[1:])
        self.stored_chunk_num += chunk_num
        if redundant_rnn_hidden is not None:
            indices = np.arange(chunk_num) * self.chunk_len
            rnn_hidden = np.transpose(redundant_rnn_hidden[indices], (1, 0, 2))
        else:
            rnn_hidden = None
        return storage_block, rnn_hidden

    def compute_gae(self, reward, value, bootstrap_value):
        eplen = len(reward)
        discounted_r = lfilter([1], [1, -self.gamma], reward.squeeze(-1)[::-1])[::-1]
        discount_factor = self.gamma**np.arange(eplen, 0, -1)
        n_step_v = discounted_r + bootstrap_value * discount_factor
        td_err = n_step_v - value.squeeze(-1)
        adv = lfilter([1], [1, -self.lmbda], td_err[::-1])[::-1]
        return np.expand_dims(n_step_v, 1), np.expand_dims(adv, 1)
    
    def get(self):
        return next(self._generator)


class RolloutCollector:
    def __init__(self, rollout_keys, collect_keys, model_fn, worker_env_fn, ps, kwargs):
        self.num_postprocessors = kwargs['num_postprocessors']
        self.postprocessors = [
            ray.remote(num_cpus=1)(PostProcessor).remote(processor_id=i,
                                                        rollout_keys=rollout_keys,
                                                        collect_keys=collect_keys,
                                                        model_fn=model_fn,
                                                        worker_env_fn=worker_env_fn,
                                                        ps=ps,
                                                        kwargs=kwargs)
            for i in range(self.num_postprocessors)
        ]
        self.working_jobs = []
        # job_hashing maps job id to worker index
        self.job_hashing = {}

        self._data_id_g = self._data_id_generator()
        print("---------------------------------------------------")
        print("              Workers starting ......              ")
        print("---------------------------------------------------")

    def _start(self):
        for i in range(self.num_postprocessors):
            job = self.postprocessors[i].get.remote()
            self.working_jobs.append(job)
            self.job_hashing[job] = i

    def _data_id_generator(self):
        # iteratively make worker active
        self._start()
        while True:
            [ready_job], self.working_jobs = ray.wait(self.working_jobs, num_returns=1)

            worker_id = self.job_hashing[ready_job]
            self.job_hashing.pop(ready_job)

            new_job = self.postprocessors[worker_id].get.remote()
            self.working_jobs.append(new_job)
            self.job_hashing[new_job] = worker_id
            yield ready_job

    def get_sample_ids(self):
        return next(self._data_id_g)


# class BufferWriter(mp.Process):
#     def __init__(self, ready_queue, buffer):
#         super().__init__()
#         self.daemon = True
#         self.ready_queue = ready_queue
#         self.buffer = buffer

#     def run(self):
#         while True:
#             blk = self.ready_queue.get()
#             self.buffer.put(blk)


class SimulationSupervisor:
    def __init__(self, rollout_keys, collect_keys, model_fn, worker_env_fn, ps, recorder, global_buffer, kwargs):
        self.rollout_collector = RolloutCollector(rollout_keys=rollout_keys, collect_keys=collect_keys, model_fn=model_fn, worker_env_fn=worker_env_fn, ps=ps, kwargs=kwargs)
        self.global_buffer = global_buffer
        self.recorder = recorder
        self.ready_id_queue = ThreadSafeQueue(maxsize=128)

        self.sample_job = threading.Thread(target=self.sample_from_rollout_collector, daemon=True)
        self.put_jobs = [
            threading.Thread(target=self.put_sample_into_buffer, daemon=True) for _ in range(kwargs['num_writers'])
        ]

    def start(self):
        self.sample_job.start()
        for job in self.put_jobs:
            job.start()

    def sample_from_rollout_collector(self):
        self.record_job = []
        while True:
            ready_sample_id = self.rollout_collector.get_sample_ids()
            self.ready_id_queue.put(ready_sample_id)

    # TODO: i/o bound, may use asyncio?
    def put_sample_into_buffer(self):
        self.record_job = []
        while True:
            ready_sample_id = self.ready_id_queue.get()
            storage_block, infos = ray.get(ready_sample_id)
            self.global_buffer.put(*storage_block)
            ray.get(self.record_job)
            self.record_job = self.recorder.push.remote(infos)

    def get_ready_queue_util(self):
        return self.ready_id_queue.qsize() / 128


class BufferCollector(mp.Process):
    def __init__(self, buffer, shm_tensor_dict, available_flag, ready):
        super().__init__()
        self.daemon = True
        self.ready = ready
        self.buffer = buffer
        self.shm_tensor_dict = shm_tensor_dict
        self.available_flag = available_flag

    def run(self):
        while True:
            self.ready.acquire()
            self.ready.wait_for(lambda: self.available_flag == 0)
            numpy_data_batch = self.buffer.get()
            for k, v in numpy_data_batch.items():
                self.shm_tensor_dict[k].copy_(from_numpy(v))
            self.available_flag += 1
            self.ready.release()
