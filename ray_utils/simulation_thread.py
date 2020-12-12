import ray
import itertools
from copy import deepcopy
from queue import Queue
from threading import Thread


class Worker:
    '''
    A Worker wraps 1 env and 1 model together.
    Model continuously interacts with env to get data,
    and waits RolloutCollector to collect generated data.
    '''
    def __init__(self, worker_id, model_fn, worker_env_fn, ps, kwargs):
        self.id = worker_id

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
        # old_hash = self.weight_hash
        self.weight_hash = weights_hash
        # print("Worker {} load state dict, "
        #       "hashing changed from "
        #       "{} to {}".format(self.id, old_hash, self.weight_hash))
        del weights_job, weights, weights_hash

    def _data_generator(self):
        self.pull_job = self.ps.pull.remote()
        model_inputs = self.env.get_model_inputs()
        while True:
            actions, action_logits, values = self.model.select_action(*model_inputs)
            data_batches, ep_returns, wons, model_inputs = self.env.step(actions, action_logits, values)
            if len(data_batches) == 0:
                continue
            yield (data_batches, ep_returns, wons)
            # get new weights only when at least one of vectorized envs is done
            if ray.get(self.pull_job) != self.weight_hash:
                self.get_new_weights()
            self.pull_job = self.ps.pull.remote()

    def get(self):
        return next(self._data_g)

    def close_env(self):
        self.env.close()


class RolloutCollector:
    def __init__(self, model_fn, worker_env_fn, ps, kwargs):
        self.num_workers = int(kwargs['num_workers'])
        self.ps = ps
        self.workers = [
            ray.remote(num_cpus=kwargs['cpu_per_worker'])(Worker).remote(worker_id=i,
                                                                         model_fn=model_fn,
                                                                         worker_env_fn=worker_env_fn,
                                                                         ps=ps,
                                                                         kwargs=kwargs) for i in range(self.num_workers)
        ]
        self.working_jobs = []
        # job_hashing maps job id to worker index
        self.job_hashing = {}

        self._data_id_g = self._data_id_generator(num_returns=kwargs['num_returns'])
        print("###################################################")
        print("############# Workers starting ...... #############")
        print("###################################################")

    def _start(self):
        for i in range(self.num_workers):
            job = self.workers[i].get.remote()
            self.working_jobs.append(job)
            self.job_hashing[job] = i

    def _data_id_generator(self, num_returns, timeout=None):
        import time
        # iteratively make worker active
        self._start()
        while True:
            start = time.time()
            ready_jobs, self.working_jobs = ray.wait(self.working_jobs, num_returns=num_returns, timeout=timeout)
            wait_time = time.time() - start

            for ready_job in ready_jobs:
                worker_id = self.job_hashing[ready_job]
                self.job_hashing.pop(ready_job)

                new_job = self.workers[worker_id].get.remote()
                self.working_jobs.append(new_job)
                self.job_hashing[new_job] = worker_id

            yield ready_jobs, wait_time

    def get_sample_ids(self):
        return next(self._data_id_g)

    def close_env(self):
        close_jobs = []
        for worker in self.workers:
            close_jobs.append(worker.close_env.remote())
        while close_jobs:
            _, close_jobs = ray.wait(close_jobs, num_returns=1)


class SimulationThread(Thread):
    def __init__(self, model_fn, worker_env_fn, ps, recorder, global_buffer, kwargs):
        super().__init__()
        self.rollout_collector = RolloutCollector(model_fn=model_fn, worker_env_fn=worker_env_fn, ps=ps, kwargs=kwargs)
        self.global_buffer = global_buffer
        self.recorder = recorder
        self.ready_id_queue = Queue(maxsize=128)
        self.daemon = True
        self.wait_times = []

    def sample_from_rollout_collector(self):
        while True:
            ready_sample_ids = self.rollout_collector.get_sample_ids()
            self.ready_id_queue.put(ready_sample_ids)

    def run(self):
        # asynchronous simlation thread main loop
        sample_job = Thread(target=self.sample_from_rollout_collector)
        sample_job.start()

        while True:
            ready_sample_ids, wait_time = self.ready_id_queue.get()
            self.wait_times.append(wait_time)
            all_batch_return = ray.get(ready_sample_ids)

            nested_data_batches, nested_ep_returns, nested_wons = zip(*deepcopy(all_batch_return))
            ep_returns = list(itertools.chain.from_iterable(nested_ep_returns))
            wons = list(itertools.chain.from_iterable(nested_wons))
            push_job = self.recorder.push.remote(ep_returns, wons)
            self.global_buffer.put_batch(list(itertools.chain.from_iterable(nested_data_batches)))

            ray.get(push_job)
            # del object references & ray.get returns
            # hopefully this can delete object in ray object store
            del push_job, ready_sample_ids, all_batch_return

    def get_wait_time(self):
        import numpy as np
        t = np.mean(self.wait_times)
        self.wait_times = []
        return t


class BufferCollector(Thread):
    def __init__(self, batch_queue, buffer, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.buffer = buffer
        self.batch_queue = batch_queue
        self.daemon = True

    def run(self):
        while True:
            batch = self.buffer.get(self.batch_size)
            while batch is None:
                batch = self.buffer.get(self.batch_size)
            self.batch_queue.put(batch)
