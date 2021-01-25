import numpy as np
from collections import namedtuple
from env.atari.registry import get_shapes, ROLLOUT_KEYS, DTYPES, Info


class EnvWithMemory:
    def __init__(self, env_fn, config):
        self.env = env_fn(config)
        self.shapes = get_shapes(config)
        self.RolloutSegCls = namedtuple('RolloutSeg', ROLLOUT_KEYS)

        self.stored_chunk_num = 0
        self.history_ep_datas = []
        self.history_ep_infos = []

        self.chunk_len = config.chunk_len
        self.step_size = self.chunk_len
        self.min_return_chunk_num = config.min_return_chunk_num

        self.gamma = config.gamma
        self.lmbda = config.lmbda
        self.max_timesteps = config.max_timesteps

        self.verbose = config.verbose
        self.reset()

    def _preprocess(self, obs):
        return np.transpose(obs, [2, 0, 1]).astype(np.uint8)

    def _get_model_input(self):
        return self.obs, self.rnn_hidden

    def _init_rnn_hidden(self):
        return np.zeros(self.shapes['rnn_hidden'], dtype=np.float32)

    def reset(self):
        if self.verbose and self.done:
            print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.obs = self._preprocess(self.env.reset())
        self.rnn_hidden = self._init_rnn_hidden()
        self.done = False
        self.ep_step = self.ep_return = 0
        self.reset_history()

    def step(self, action, action_logits, value, rnn_hidden):
        n_obs, r, d, _ = self.env.step(action)
        n_obs = self._preprocess(n_obs)

        self.ep_return += r
        self.ep_step += 1

        self.history['obs'].append(self.obs)
        self.history['action'].append(action)
        self.history['action_logits'].append(action_logits)
        self.history['reward'].append(r)
        self.history['value'].append(value)
        self.history['rnn_hidden'].append(self.rnn_hidden)

        self.obs = n_obs
        self.rnn_hidden = rnn_hidden
        self.done = d or self.ep_step >= self.max_timesteps
        if self.done:
            seg = self.RolloutSegCls(**{k: np.stack(v).astype(DTYPES[k]) for k, v in self.history.items()})
            self.stored_chunk_num += int(np.ceil(self.ep_step / self.chunk_len))
            self.history_ep_datas.append(seg)
            self.history_ep_infos.append(Info(ep_return=self.ep_return))
            self.reset()
            if self.stored_chunk_num >= self.min_return_chunk_num:
                datas = self.history_ep_datas.copy()
                infos = self.history_ep_infos.copy()
                self.history_ep_datas = []
                self.history_ep_infos = []
                self.stored_chunk_num = 0
                return datas, infos, self._get_model_input()
            else:
                return [], [], self._get_model_input()
        return [], [], self._get_model_input()

    def reset_history(self):
        self.history = {}
        for k in ROLLOUT_KEYS:
            self.history[k] = []


class VecEnvWithMemory:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]

    def step(self, *args):
        datas, infos, model_inputs = [], [], []
        for i, env in enumerate(self.envs):
            cur_datas, cur_infos, model_input = env.step(*[arg[i] for arg in args])
            datas += cur_datas
            infos += cur_infos
            model_inputs.append(model_input)
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return datas, infos, stacked_model_inputs

    def get_model_inputs(self):
        model_inputs = [env._get_model_input() for env in self.envs]
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return stacked_model_inputs

    def close(self):
        for env in self.envs:
            env.close()
