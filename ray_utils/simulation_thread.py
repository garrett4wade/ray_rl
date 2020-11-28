from threading import Thread
import time

import ray
import sys
from queue import Queue

_LAST_FREE_TIME = 0.0
_TO_FREE = []


def ray_get_and_free(object_ids):
    # borrowed from rllib
    """
    Call ray.get and then queue the object ids for deletion.

    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.

    Args:
        object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.

    Returns:
        The result of ray.get(object_ids).
    """

    free_delay_s = 10.0
    max_free_queue_size = 100

    global _LAST_FREE_TIME
    global _TO_FREE

    result = ray.get(object_ids)
    if type(object_ids) is not list:
        object_ids = [object_ids]
    _TO_FREE.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_TO_FREE) > max_free_queue_size or now - _LAST_FREE_TIME > free_delay_s):
        ray.internal.free(_TO_FREE)
        _TO_FREE = []
        _LAST_FREE_TIME = now

    return result


class Worker():
    '''
    A Worker wraps 1 env and 1 model together.
    Model continuously interacts with env to get data,
    and waits RolloutCollector to collect generated data.
    '''
    def __init__(self, worker_id, model_fn, env_fn, ps, kwargs):
        self.id = worker_id

        self.env = env_fn(worker_id, kwargs)
        self.model = model_fn(kwargs)

        self.ps = ps
        self.load_period = kwargs['load_period']
        self.weight_hash = None
        self.get_new_weights()

        self._data_g = self._data_generator()

    def get_new_weights(self):
        # if model parameters in parameter server is different from local model, download it
        result = ray.get(self.ps.get_weights.remote(self.weight_hash))
        if result is not None:
            self.model.load_state_dict(result[0])
            old_hash = self.weight_hash
            self.weight_hash = result[1]
            print("Worker {} load state dict, "
                  "hashing changed from "
                  "{} to {}".format(self.id, old_hash, self.weight_hash))

    def _data_generator(self):
        # invoke env.step to generate data
        global_step = 0
        while True:
            if global_step % self.load_period == 0:
                self.get_new_weights()
            global_step += 1

            data_batches, ep_returns = self.env.step(self.model)
            if len(data_batches) == 0:
                continue
            yield (data_batches, ep_returns)

    def get(self):
        return next(self._data_g)


class RolloutCollector():
    def __init__(self, model_fn, env_fn, ps, kwargs):
        self.num_workers = int(kwargs['num_workers'])
        self.ps = ps
        self.workers = [
            ray.remote(num_cpus=kwargs['cpu_per_worker'])(Worker).remote(worker_id=i,
                                                                         model_fn=model_fn,
                                                                         env_fn=env_fn,
                                                                         ps=ps,
                                                                         kwargs=kwargs) for i in range(self.num_workers)
        ]
        self._data_id_g = self._data_id_generator(num_returns=kwargs['num_returns'])
        print("###################################################")
        print("############# Workers starting ...... #############")
        print("###################################################")

    def _data_id_generator(self, num_returns, timeout=None):
        # iteratively make worker active
        self.worker_done = [True for _ in range(self.num_workers)]
        self.working_job_ids = []
        self.id2job_idx = dict()
        while True:
            for i in range(self.num_workers):
                if self.worker_done[i]:
                    job_id = self.workers[i].get.remote()
                    self.working_job_ids.append(job_id)
                    self.id2job_idx[job_id] = i
                    self.worker_done[i] = False

            ready_ids, self.working_job_ids = ray.wait(self.working_job_ids, num_returns=num_returns, timeout=timeout)

            for ready_id in ready_ids:
                self.worker_done[self.id2job_idx[ready_id]] = True
                self.id2job_idx.pop(ready_id)

            yield ready_ids

    def get_sample_ids(self):
        ready_ids = next(self._data_id_g)
        return ready_ids


class SimulationThread(Thread):
    def __init__(self, model_fn, env_fn, ps, recorder, global_buffer, kwargs):
        super().__init__()
        self.rollout_collector = RolloutCollector(model_fn=model_fn, env_fn=env_fn, ps=ps, kwargs=kwargs)
        self.global_buffer = global_buffer
        self.recorder = recorder
        self.ready_id_queue = Queue(maxsize=128)
        self.daemon = True

    def sample_from_rollout_collector(self):
        while True:
            ready_sample_ids = self.rollout_collector.get_sample_ids()
            self.ready_id_queue.put(ready_sample_ids)

    def run(self):
        # asynchronous simlation thread main loop
        sample_job = Thread(target=self.sample_from_rollout_collector)
        sample_job.start()

        while True:
            ready_sample_ids = self.ready_id_queue.get()

            # get samples
            try:
                all_batch_return = ray_get_and_free(ready_sample_ids)
            except ray.exceptions.UnreconstructableError as e:
                all_batch_return = []
                print(str(e))
            except ray.exceptions.RayError as e:
                all_batch_return = []
                print(str(e))

            push_jobs = []
            for data_batches, ep_returns in all_batch_return:
                push_jobs.append(self.recorder.push.remote(ep_returns))
                for data_batch in data_batches:
                    self.global_buffer.put(data_batch)
            ray.get(push_jobs)
