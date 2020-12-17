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
        self.statistics = dict(max=0, min=0, avg=0, winning_rate=0)
        self.wons = []

    def push(self, ep_returns, wons):
        self.storage += ep_returns
        self.wons += wons

    def pull(self):
        if len(self.storage) > 0:
            self.statistics['max'] = np.max(self.storage)
            self.statistics['min'] = np.min(self.storage)
            self.statistics['avg'] = np.mean(self.storage)
            self.statistics['winning_rate'] = np.mean(self.wons)
            self.storage = []
            self.wons = []
        return self.statistics


@ray.remote(num_cpus=1)
class PopArtServer:
    def __init__(self, beta):
        self.beta = beta
        self.running_mean = 0.0
        self.running_mean_sq = 0.0

    def pull(self):
        clipped_std = np.clip(np.sqrt(self.running_mean_sq - self.running_mean**2), 1e-4, 1e6)
        return self.running_mean, clipped_std

    def push(self, mean, mean_sq):
        self.running_mean = self.running_mean * self.beta + (1 - self.beta) * mean
        self.running_mean_sq = self.running_mean_sq * self.beta + (1 - self.beta) * mean_sq
