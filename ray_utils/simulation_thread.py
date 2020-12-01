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
        get_weights_job = self.ps.get_weights.remote(self.weight_hash)
        result = ray.get(get_weights_job)
        if result is not None:
            self.model.load_state_dict(result[0])
            old_hash = self.weight_hash
            self.weight_hash = result[1]
            print("Worker {} load state dict, "
                  "hashing changed from "
                  "{} to {}".format(self.id, old_hash, self.weight_hash))
        del get_weights_job
        del result

    def _data_generator(self):
        model_inputs = self.env.get_model_inputs()
        while True:
            actions, action_logits, values = self.model.select_action(*model_inputs)
            data_batches, ep_returns, model_inputs = self.env.step(actions, action_logits, values)
            if len(data_batches) == 0:
                continue
            yield (data_batches, ep_returns)
            # get new weights only when at least one of vectorized envs is done
            self.get_new_weights()

    def get(self):
        return next(self._data_g)


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
    def __init__(self, model_fn, worker_env_fn, ps, recorder, global_buffer, kwargs):
        super().__init__()
        self.rollout_collector = RolloutCollector(model_fn=model_fn, worker_env_fn=worker_env_fn, ps=ps, kwargs=kwargs)
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
            all_batch_return = ray.get(ready_sample_ids)

            nested_data_batches, nested_ep_returns = zip(*deepcopy(all_batch_return))
            push_job = self.recorder.push.remote(list(itertools.chain.from_iterable(nested_ep_returns)))
            for data_batch in itertools.chain.from_iterable(nested_data_batches):
                self.global_buffer.put(data_batch)

            ray.get(push_job)
            # del object references & ray.get returns
            # hopefully this can delete object in ray object store
            del push_job
            del ready_sample_ids
            del all_batch_return


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
