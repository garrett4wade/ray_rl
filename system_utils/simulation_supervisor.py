import ray
import torch
import threading
from ray.util.queue import Queue as RayQueue
import zmq
import numpy as np
from copy import deepcopy
from collections import OrderedDict, namedtuple
import multiprocessing as mp
from system_utils.worker import RolloutCollector
from system_utils.init_ray import initialize_ray_on_supervisor
from system_utils.parameter_server import ParameterServer


@ray.remote(num_cpus=0.5)
class Ray2ProcessSender:
    def __init__(self, sender_id, ready_id_queue, port=257894):
        self.id = sender_id
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.bind('tcp://*:' + str(port + self.id))
        self.ready_id_queue = ready_id_queue

    def send_seg(self, seg, flags=0, copy=True, track=False):
        data = np.concatenate([x.reshape(-1) for x in seg])
        return self.socket.send(data, flags, copy=copy, track=track)

    def run(self):
        while True:
            segs = self.ready_id_queue.get()
            for seg in segs:
                self.send_seg(seg)
                self.socket.recv()


def receive_and_put(receiver_id, buffer, port=257894, flags=0, copy=True, track=False):
    Seg = namedtuple('Seg', list(buffer.shapes.keys()))
    datalen_per_batch = 0
    for k, shp in buffer.shapes.items():
        if 'rnn_hidden' in k:
            datalen_per_batch += np.prod(shp)
        else:
            datalen_per_batch += buffer.chunk_len * np.prod(shp)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect('tcp://localhost:' + str(port + receiver_id))
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
    def __init__(self, rollout_keys, collect_keys, model_fn, worker_env_fn, global_buffer, weights,
                 weights_available, ep_info_dict, queue_util, kwargs):
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
        self.kwargs = kwargs

        self.history_info = []

    def run(self):
        initialize_ray_on_supervisor(self.kwargs)
        self.ready_id_queue = RayQueue(maxsize=128)
        self.ps = ParameterServer.remote(self.weights)
        self.rollout_collector = RolloutCollector(model_fn=self.model_fn,
                                                  worker_env_fn=self.worker_env_fn,
                                                  ps=self.ps,
                                                  kwargs=self.kwargs)
        sample_job = threading.Thread(target=self.sample_from_rollout_collector, daemon=True)
        sample_job.start()

        senders = [Ray2ProcessSender.remote(i, self.ready_id_queue) for i in range(self.kwargs['num_writers'])]
        for sender in senders:
            sender.run.remote()
        receivers = [
            mp.Process(target=receive_and_put, args=(i, self.global_buffer)) for i in range(self.kwargs['num_writers'])
        ]
        for receiver in receivers:
            receiver.start()

        upload_job = []
        while True:
            if self.weights_available:
                ray.get(upload_job)
                upload_job = self.ps.set_weights.remote(deepcopy(self.weights))
                self.weights_available.copy_(torch.tensor(0))
                self.queue_util.copy_(torch.tensor(self.ready_id_queue.qsize() / 128))
                if len(self.history_info) > 0:
                    for k in self.history_info[0]._fields:
                        self.ep_info_dict[k + '/avg'].copy_(
                            torch.tensor(np.mean([getattr(info, k) for info in self.history_info])))
                        self.ep_info_dict[k + '/min'].copy_(
                            torch.tensor(np.min([getattr(info, k) for info in self.history_info])))
                        self.ep_info_dict[k + '/max'].copy_(
                            torch.tensor(np.max([getattr(info, k) for info in self.history_info])))
                    self.history_info = []

    def sample_from_rollout_collector(self):
        while True:
            ready_seg_id, ready_info_id = self.rollout_collector.get_sample_ids()
            self.ready_id_queue.put(ready_seg_id)
            self.history_info += ray.get(ready_info_id)
