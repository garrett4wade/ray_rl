import threading
import numpy as np
import ray
from scipy.signal import lfilter
from queue import Queue as ThreadSafeQueue
from system_utils.worker import RolloutWorker


class PostProcessor:
    def __init__(self, processor_id, rollout_keys, collect_keys, model_fn, worker_env_fn, ps, kwargs):
        """(remote) postprocessor of rollouts (e.g. compute GAE), however slow down performance, tentatively archived
        """
        assert kwargs['num_workers'] % kwargs['num_postprocessors'] == 0
        self.num_workers = kwargs['num_workers'] // kwargs['num_postprocessors']
        self.workers = [
            ray.remote(num_cpus=kwargs['cpu_per_worker'])(RolloutWorker).remote(
                worker_id=i + kwargs['num_postprocessors'] * processor_id,
                model_fn=model_fn,
                worker_env_fn=worker_env_fn,
                ps=ps,
                kwargs=kwargs) for i in range(self.num_workers)
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
