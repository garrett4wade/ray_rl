import numpy as np
from itertools import chain
from multiprocessing import Process, Pipe
from rl_utils.env_wrapper import VecEnvWithMemory, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, action, logits, value = remote.recv()
        if cmd == 'step':
            data_batches, ep_return, model_input = env.step(action, logits, value)
            remote.send((data_batches, ep_return, model_input))
        elif cmd == 'get_model_inputs':
            remote.send((env.obs, ))
        else:
            raise NotImplementedError


class SubprocVecEnvWithMemory(VecEnvWithMemory):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        VecEnvWithMemory.__init__(self, len(env_fns))

    def step_async(self, actions, action_logits, values):
        for remote, action, logits, value in zip(self.remotes, actions, action_logits, values):
            remote.send(('step', action, logits, value))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        nested_data_batches, ep_returns, model_inputs = zip(*results)
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return list(chain.from_iterable(nested_data_batches)), list(filter(None, ep_returns)), stacked_model_inputs

    def get_model_inputs(self):
        for remote in self.remotes:
            remote.send(('get_model_inputs', None, None, None))
        model_inputs = [remote.recv() for remote in self.remotes]
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return stacked_model_inputs
