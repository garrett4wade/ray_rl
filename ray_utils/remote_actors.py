import ray
import itertools
from ray.util.queue import Queue
from threading import Thread, Lock
from ray_utils.remote_functions import isNone
from rl_utils.buffer import CircularBuffer


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
            if len(data_batches) == 0:
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

        self._data_id_g = self._data_id_generator(num_returns=kwargs['num_returns'])
        print("---------------------------------------------------")
        print("              Workers starting ......              ")
        print("---------------------------------------------------")

    def _start(self):
        for i in range(self.num_workers):
            job = self.workers[i].get.remote()
            self.working_jobs.append(job)
            self.job_hashing[job] = i

    def _data_id_generator(self, num_returns, timeout=None):
        # iteratively make worker active
        self._start()
        while True:
            ready_jobs, self.working_jobs = ray.wait(self.working_jobs, num_returns=num_returns, timeout=timeout)

            for ready_job in ready_jobs:
                worker_id = self.job_hashing[ready_job]
                self.job_hashing.pop(ready_job)

                new_job = self.workers[worker_id].get.remote()
                self.working_jobs.append(new_job)
                self.job_hashing[new_job] = worker_id

            yield ready_jobs

    def get_sample_ids(self):
        return next(self._data_id_g)


@ray.remote(num_cpus=1)
class ReadWorker:
    def __init__(self, ready_id_queue, recorder, buffer):
        self.recorder = recorder
        self.buffer = buffer
        self.ready_id_queue = ready_id_queue

    def send(self):
        ready_sample_ids = self.ready_id_queue.get()
        all_batch_returns = ray.get(ready_sample_ids)
        nested_data_batch, nested_info = zip(*all_batch_returns)
        job = [
            self.recorder.push.remote(list(itertools.chain.from_iterable(nested_info))),
            self.buffer.put_batch.remote(list(itertools.chain.from_iterable(nested_data_batch)))
        ]
        while job:
            _, job = ray.wait(job, num_returns=1)


@ray.remote(num_cpus=1)
class QueueReader:
    def __init__(self, ready_id_queue, recorder, buffer, num_readers):
        self.num_readers = num_readers
        self.read_workers = [ReadWorker.remote(ready_id_queue, recorder, buffer) for _ in range(self.num_readers)]
        self.working_jobs = []
        # job_hashing maps job id to worker index
        self.job_hashing = {}

    def start(self):
        for i in range(self.num_readers):
            job = self.read_workers[i].send.remote()
            self.working_jobs.append(job)
            self.job_hashing[job] = i
        while True:
            ready_jobs, self.working_jobs = ray.wait(self.working_jobs, num_returns=1)
            ready_job = ready_jobs[0]
            worker_id = self.job_hashing[ready_job]
            self.job_hashing.pop(ready_job)

            new_job = self.read_workers[worker_id].send.remote()
            self.working_jobs.append(new_job)
            self.job_hashing[new_job] = worker_id
            ray.get(ready_job)


@ray.remote(num_cpus=2)
class SimulationSupervisor:
    def __init__(self, model_fn, worker_env_fn, ps, recorder, remote_buffer, kwargs):
        self.rollout_collector = RolloutCollector(model_fn=model_fn, worker_env_fn=worker_env_fn, ps=ps, kwargs=kwargs)
        self.ready_id_queue = Queue(maxsize=128)
        self.buffer = remote_buffer
        self.recorder = recorder
        # self.queue_reader = QueueReader.remote(self.ready_id_queue, recorder, remote_buffer, kwargs['num_readers'])

        self.sample_job = Thread(target=self.sample_from_rollout_collector)
        self.send_job = Thread(target=self.send_sample_to_buffer)

    def start(self):
        self.sample_job.start()
        self.send_job.start()

    def sample_from_rollout_collector(self):
        while True:
            ready_sample_ids = self.rollout_collector.get_sample_ids()
            self.ready_id_queue.put(ready_sample_ids)

    def send_sample_to_buffer(self):
        while True:
            ready_sample_ids = self.ready_id_queue.get()
            all_batch_returns = ray.get(ready_sample_ids)
            nested_data_batch, nested_info = zip(*all_batch_returns)
            job = [
                self.recorder.push.remote(list(itertools.chain.from_iterable(nested_info))),
                self.buffer.put_batch.remote(list(itertools.chain.from_iterable(nested_data_batch)))
            ]
            while job:
                _, job = ray.wait(job, num_returns=1)

    def get_ready_queue_util(self):
        return self.ready_id_queue.qsize() / 128


@ray.remote(num_cpus=1)
class CollectWorker:
    def __init__(self, buffer, batch_size):
        self.batch_size = batch_size
        self.buffer = buffer

    def get(self):
        data_batch = self.buffer.get.remote(self.batch_size)
        while ray.get(isNone.remote(data_batch)):
            data_batch = self.buffer.get.remote(self.batch_size)
        return data_batch


class BufferCollector:
    def __init__(self, buffer, batch_size, num_collectors):
        self.num_collectors = num_collectors
        self.collect_workers = [CollectWorker.remote(buffer, batch_size) for _ in range(self.num_collectors)]
        self.working_jobs = []
        # job_hashing maps job id to worker index
        self.job_hashing = {}

        self._data_id_g = self._data_id_generator()

    def _start(self):
        for i in range(self.num_collectors):
            job = self.collect_workers[i].get.remote()
            self.working_jobs.append(job)
            self.job_hashing[job] = i

    def _data_id_generator(self):
        # iteratively make worker active
        self._start()
        while True:
            ready_jobs, self.working_jobs = ray.wait(self.working_jobs, num_returns=1)
            ready_job = ready_jobs[0]
            worker_id = self.job_hashing[ready_job]
            self.job_hashing.pop(ready_job)

            new_job = self.collect_workers[worker_id].get.remote()
            self.working_jobs.append(new_job)
            self.job_hashing[new_job] = worker_id
            yield ray.get(ready_job)

    def get_batch_ref(self):
        return next(self._data_id_g)


@ray.remote(num_cpus=2)
class RemoteBuffer(CircularBuffer):
    def __init__(self, maxsize, reuse_times, keys, ready_queue, batch_size):
        self.ready_queue = ready_queue
        self.batch_size = batch_size
        self.lock = Lock()
        super().__init__(maxsize, reuse_times, keys)

        get_job = Thread(target=self.get_batch_into_queue)
        get_job.start()

    def get_batch_into_queue(self):
        while True:
            if len(self._storage) > self.batch_size and not self.ready_queue.full():
                self.lock.acquire()
                self.ready_queue.put(self.get(self.batch_size))
                self.lock.release()


# @ray.remote(num_cpus=1, num_gpus=len(nvgpu.available_gpus()))
# class GPULoader:
#     def __init__(self, tpdv):
#         self.tpdv = tpdv

#     def load(self, data_batch):
#         for k, v in data_batch.items():
#             data_batch[k] = torch.from_numpy(v.copy()).to(**self.tpdv)
#         return data_batch
