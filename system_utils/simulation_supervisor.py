import ray
import torch
from utils.drain_queue import drain_ray_queue
from ray.util.queue import Queue as RayQueue
import zmq
import itertools
import os
import numpy as np
from copy import deepcopy
from collections import OrderedDict, namedtuple
import multiprocessing as mp
from system_utils.worker import RolloutCollector
from system_utils.init_ray import initialize_ray_on_supervisor
from system_utils.parameter_server import ParameterServer


@ray.remote
def send_seg(socket, seg, flags=0, copy=True, track=False):
    data = np.concatenate([x.reshape(-1) for x in seg]).astype(np.float32)
    socket.send(data, flags, copy=copy, track=track)
    socket.recv()


@ray.remote(num_cpus=0, resources={'head': 1})
class Ray2ProcessSender:
    def __init__(self, sender_id, rollout_collector, info_queue, wait_time_queue):
        self.id = sender_id
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.bind('ipc:///dev/shm/ray2proc_' + str(sender_id))
        self.rollout_collector = rollout_collector
        self.info_queue = info_queue
        self.wait_time_queue = wait_time_queue

    def send_seg(self, seg, flags=0, copy=True, track=False):
        data = np.concatenate([x.reshape(-1) for x in seg]).astype(np.float32)
        return self.socket.send(data, flags, copy=copy, track=track)

    def run(self):
        job = self.rollout_collector.get_sample_ids.remote()
        while True:
            seg, infos, wait_time = ray.get(job)
            job = self.rollout_collector.get_sample_ids.remote()
            self.info_queue.put(infos)
            self.wait_time_queue.put(wait_time)
            self.send_seg(seg)
            self.socket.recv()


def receive_and_put(receiver_id, buffer, flags=0, copy=True, track=False):
    os.setpriority(os.PRIO_PROCESS, os.getpid(), 1)
    Seg = namedtuple('Seg', list(buffer.shapes.keys()))
    datalen_per_batch = 0
    for k, shp in buffer.shapes.items():
        if 'rnn_hidden' in k:
            datalen_per_batch += np.prod(shp)
        else:
            datalen_per_batch += buffer.chunk_len * np.prod(shp)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect('ipc:///dev/shm/ray2proc_' + str(receiver_id))
    while True:
        msg = socket.recv(flags=flags, copy=copy, track=track)
        buf = memoryview(msg)
        data = np.frombuffer(buf, dtype=np.float32)
        assert len(data) % datalen_per_batch == 0
        batch_size = int(len(data) // datalen_per_batch)

        # get split
        simplex_shapes = []
        for k, shp in buffer.shapes.items():
            if 'rnn_hidden' in k:
                simplex_shapes.append(np.prod(shp) * batch_size)
            else:
                simplex_shapes.append(buffer.chunk_len * np.prod(shp) * batch_size)
        split = [int(sum(simplex_shapes[:i])) for i in range(1, len(simplex_shapes))]
        seg_data = np.split(data, split, -1)

        # get shapes with batch
        shapes_with_batch = OrderedDict({})
        for k, shp in buffer.shapes.items():
            if 'rnn_hidden' in k:
                shapes_with_batch[k] = (shp[0], batch_size, *shp[1:])
            else:
                shapes_with_batch[k] = (buffer.chunk_len, batch_size, *shp)
        seg_dict = {}
        for x, (k, shp) in zip(seg_data, shapes_with_batch.items()):
            seg_dict[k] = x.reshape(shp)

        # put into buffer
        buffer.put(Seg(**seg_dict))
        socket.send(b'1')


class SimulationSupervisor(mp.Process):
    def __init__(self, rollout_keys, collect_keys, model_fn, worker_env_fn, global_buffer, weights, weights_available,
                 ep_info_dict, queue_util, wait_time, config):
        super().__init__()
        self.rollout_keys = rollout_keys
        self.collect_keys = collect_keys
        self.model_fn = model_fn
        self.worker_env_fn = worker_env_fn
        self.global_buffer = global_buffer
        self.weights = weights
        self.weights_available = weights_available
        self.ep_info_dict = ep_info_dict
        self.queue_util = queue_util
        self.wait_time = wait_time
        self.config = config

    def run(self):
        initialize_ray_on_supervisor(self.config)
        self.info_queue = RayQueue(maxsize=16 * self.config.num_workers)
        self.wait_time_queue = RayQueue(maxsize=16 * self.config.num_workers)
        self.ps = ParameterServer.remote(deepcopy(self.weights))
        self.rollout_collectors = [
            RolloutCollector.remote(collector_id=i,
                                    model_fn=self.model_fn,
                                    worker_env_fn=self.worker_env_fn,
                                    ps=self.ps,
                                    config=self.config) for i in range(self.config.num_collectors)
        ]

        assert self.config.num_writers % self.config.num_collectors == 0
        senders = [[
            Ray2ProcessSender.remote(i + j * self.config.num_collectors, self.rollout_collectors[j], self.info_queue,
                                     self.wait_time_queue)
            for i in range(self.config.num_writers // self.config.num_collectors)
        ] for j in range(self.config.num_collectors)]
        for sender_grp in senders:
            for sender in sender_grp:
                sender.run.remote()

        receivers = [
            mp.Process(target=receive_and_put, args=(i, self.global_buffer)) for i in range(self.config.num_writers)
        ]
        for receiver in receivers:
            receiver.start()

        upload_job = []
        while True:
            if self.weights_available:
                ray.get(upload_job)
                upload_job = self.ps.set_weights.remote(deepcopy(self.weights))
                self.weights_available.copy_(torch.tensor(0))
                # self.queue_util.copy_(torch.tensor(self.ready_id_queue.qsize() / (4 * self.config.num_workers)))
                wts = drain_ray_queue(self.wait_time_queue)
                if len(wts) > 0:
                    self.wait_time.copy_(torch.tensor(np.mean(wts)))
                history_infos = list(itertools.chain(*drain_ray_queue(self.info_queue)))
                if len(history_infos) > 0:
                    for k in history_infos[0]._fields:
                        info_data = [getattr(info, k) for info in history_infos]
                        self.ep_info_dict[k + '/avg'].copy_(torch.tensor(np.mean(info_data)))
                        self.ep_info_dict[k + '/min'].copy_(torch.tensor(np.min(info_data)))
                        self.ep_info_dict[k + '/max'].copy_(torch.tensor(np.max(info_data)))

    # def sample_from_rollout_collector(self):
    #     while True:
    #         ready_seg_id, ready_info_id, wait_time = self.rollout_collector.get_sample_ids()
    #         self.ready_id_queue.put(ready_seg_id)
    #         self.history_info += ray.get(ready_info_id)
    #         self.wait_times.append(wait_time)

    def terminate(self):
        ray.shutdown()
        super().terminate()
