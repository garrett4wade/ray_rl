import ray
import multiprocessing as mp
import threading
from queue import Queue as ThreadSafeQueue
from torch import from_numpy


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
            data_batches, infos, model_inputs = self.env.step(*model_outputs)
            if len(data_batches) == 0 or len(infos) == 0:
                continue
            yield data_batches, infos
            # get new weights only when at least one of vectorized envs is done
            if ray.get(self.pull_job) != self.weight_hash:
                self.get_new_weights()
            self.pull_job = self.ps.pull.remote()

    def get(self):
        return next(self._data_g)


class RolloutCollector:
    def __init__(self, model_fn, worker_env_fn, ps, kwargs):
        self.num_workers = int(kwargs['num_workers'] / kwargs['num_supervisors'])
        self.ps = ps
        self.workers = [
            ray.remote(num_cpus=kwargs['cpu_per_worker'])(RolloutWorker).remote(worker_id=i,
                                                                                model_fn=model_fn,
                                                                                worker_env_fn=worker_env_fn,
                                                                                ps=ps,
                                                                                kwargs=kwargs)
            for i in range(self.num_workers)
        ]
        self.working_jobs = []
        # job_hashing maps job id to worker index
        self.job_hashing = {}

        self._data_id_g = self._data_id_generator()
        print("---------------------------------------------------")
        print("              Workers starting ......              ")
        print("---------------------------------------------------")

    def _start(self):
        for i in range(self.num_workers):
            job = self.workers[i].get.remote()
            self.working_jobs.append(job)
            self.job_hashing[job] = i

    def _data_id_generator(self):
        # iteratively make worker active
        self._start()
        while True:
            [ready_job], self.working_jobs = ray.wait(self.working_jobs, num_returns=1)

            worker_id = self.job_hashing[ready_job]
            self.job_hashing.pop(ready_job)

            new_job = self.workers[worker_id].get.remote()
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
    def __init__(self, model_fn, worker_env_fn, ps, recorder, global_buffer, kwargs):
        self.rollout_collector = RolloutCollector(model_fn=model_fn, worker_env_fn=worker_env_fn, ps=ps, kwargs=kwargs)
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
            self.global_buffer.put(storage_block)
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
