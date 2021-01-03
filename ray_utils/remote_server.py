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
class EpisodeRecorder:
    def __init__(self):
        self.storage = dict()
        self.statistics = dict()

    def push(self, stat_list):
        if len(stat_list) == 0:
            return
        stat_dict = {k: [getattr(x, k) for x in stat_list] for k in stat_list[0]._fields}
        for k, v in stat_dict.items():
            if k not in self.storage.keys():
                self.storage[k] = v
            else:
                self.storage[k] += v

    def pull(self):
        if len(self.storage) > 0:
            for k, v in self.storage.items():
                self.statistics[k + '/max'] = np.max(v)
                self.statistics[k + '/min'] = np.min(v)
                self.statistics[k + '/avg'] = np.mean(v)
            self.storage = dict()
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
