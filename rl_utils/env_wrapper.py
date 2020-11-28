"""
Modified from OpenAI Baselines code to work with EnvWithMemory
"""

from abc import ABC, abstractmethod


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnvWithMemory(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_envs):
        self.num_envs = num_envs

    @abstractmethod
    def step_async(self, actions, *args):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def step(self, actions, *args):
        self.step_async(actions, *args)
        return self.step_wait()
