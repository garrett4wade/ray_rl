import ray
import time


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
