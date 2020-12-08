import ray
import time
import numpy as np


@ray.remote(num_cpus=0.5)
class ParameterServer:
    def __init__(self, weights):
        self.weights = weights
        self.weights_hash = int(time.time())

    def pull(self):
        return self.weights_hash

    def get_weights(self):
        return self.weights, self.weights_hash

    def set_weights(self, weights):
        self.weights = weights
        self.weights_hash = int(time.time())


@ray.remote(num_cpus=0.5)
class ReturnRecorder:
    def __init__(self):
        self.storage = []
        self.statistics = dict(max=0, min=0, avg=0)

    def push(self, ep_returns):
        self.storage += ep_returns

    def pull(self):
        if len(self.storage) > 0:
            self.statistics['max'] = np.max(self.storage)
            self.statistics['min'] = np.min(self.storage)
            self.statistics['avg'] = np.mean(self.storage)
            self.storage = []
        return self.statistics
