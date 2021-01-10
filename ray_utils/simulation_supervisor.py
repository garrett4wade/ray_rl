import multiprocessing as mp
from ray_utils.remote_actors import RolloutCollector
from ray_utils.init_ray import initialize_single_machine_ray_on_supervisor
from ray_utils.remote_server import ParameterServer
import ray
import torch
import threading
import queue
from copy import deepcopy


class SimulationSupervisor(mp.Process):
    def __init__(self, supervisor_id, rollout_keys, collect_keys, model_fn, worker_env_fn, global_buffer, weights,
                 weights_available, info_queue, queue_util, kwargs):
        super().__init__()
        assert kwargs['num_workers'] % kwargs['num_supervisors'] == 0
        assert kwargs['num_postprocessors'] % kwargs['num_supervisors'] == 0
        self.id = supervisor_id
        self.rollout_keys = rollout_keys
        self.collect_keys = collect_keys
        self.model_fn = model_fn
        self.worker_env_fn = worker_env_fn
        self.global_buffer = global_buffer
        self.weights = weights
        self.weights_available = weights_available
        self.info_queue = info_queue
        self.queue_util = queue_util
        self.kwargs = kwargs

        self.ready_id_queue = queue.Queue(maxsize=128)

    def run(self):
        initialize_single_machine_ray_on_supervisor(self.id, self.kwargs)
        self.ps = ParameterServer.remote(self.weights)
        self.rollout_collector = RolloutCollector(
            supervisor_id=self.id,
            #   rollout_keys=self.rollout_keys,
            #   collect_keys=self.collect_keys,
            model_fn=self.model_fn,
            worker_env_fn=self.worker_env_fn,
            ps=self.ps,
            kwargs=self.kwargs)
        sample_job = threading.Thread(target=self.sample_from_rollout_collector, daemon=True)
        sample_job.start()
        put_jobs = [
            threading.Thread(target=self.put_sample_into_buffer, daemon=True) for _ in range(self.kwargs['num_writers'])
        ]
        for put_job in put_jobs:
            put_job.start()
        upload_job = []
        while True:
            if self.weights_available[self.id]:
                ray.get(upload_job)
                upload_job = self.ps.set_weights.remote(deepcopy(self.weights))
                self.weights_available[self.id].copy_(torch.tensor(0))
                self.queue_util.copy_(torch.tensor(self.ready_id_queue.qsize() / self.ready_id_queue.maxsize))

    def sample_from_rollout_collector(self):
        while True:
            ready_sample_id = self.rollout_collector.get_sample_ids()
            self.ready_id_queue.put(ready_sample_id)

    def put_sample_into_buffer(self):
        while True:
            ready_sample_id = self.ready_id_queue.get()
            segs, infos = ray.get(ready_sample_id)
            for seg in segs:
                self.global_buffer.put(seg)
            self.info_queue.put(infos)
