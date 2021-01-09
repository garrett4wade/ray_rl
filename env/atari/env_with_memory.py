import numpy as np
from collections import OrderedDict, namedtuple

# NOTE: rnn_hidden must be the last one in keys
ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward', 'pad_mask', 'rnn_hidden']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target', 'pad_mask', 'rnn_hidden']
assert 'rnn_hidden' not in ROLLOUT_KEYS or 'rnn_hidden' == ROLLOUT_KEYS[-1]
assert 'rnn_hidden' not in COLLECT_KEYS or 'rnn_hidden' == COLLECT_KEYS[-1]
Seg = namedtuple('Seg', ROLLOUT_KEYS)
Info = namedtuple('Info', ['ep_return'])


class EnvWithMemory:
    def get_shapes(kwargs):
        return {
            'obs': kwargs['obs_dim'],
            'action': (1, ),
            'action_logits': (kwargs['action_dim'], ),
            'value': (1, ),
            'adv': (1, ),
            'value_target': (1, ),
            'pad_mask': (1, ),
        }

    def get_rnn_hidden_shape(kwargs):
        return (1, kwargs['hidden_dim'] * 2)

    def __init__(self, env_fn, kwargs):
        self.env = env_fn(kwargs)
        self.rnn_hidden_shape = EnvWithMemory.get_rnn_hidden_shape(kwargs)

        self.stored_chunk_num = 0
        self.history_ep_datas = []
        self.history_ep_infos = []

        self.chunk_len = kwargs['chunk_len']
        self.step_size = self.chunk_len
        self.min_return_chunk_num = kwargs['min_return_chunk_num']

        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.max_timesteps = kwargs['max_timesteps']

        self.verbose = kwargs['verbose']
        self.reset()

    def _preprocess(self, obs):
        obs = np.transpose(obs, [2, 0, 1]).astype(np.float32)
        return obs / 255.0

    def _get_model_input(self):
        return self.obs, self.rnn_hidden

    def _init_rnn_hidden(self):
        return np.zeros(self.rnn_hidden_shape, dtype=np.float32)

    def reset(self):
        if self.verbose and self.done:
            print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.obs = self._preprocess(self.env.reset())
        self.rnn_hidden = self._init_rnn_hidden()
        self.done = False
        self.ep_step = self.ep_return = 0
        self.reset_history()

    def step(self, action, action_logits, value, rnn_hidden):
        if self.done:
            # if env is done in the previous step, use bootstrap value
            # to compute gae and collect history data
            self.history_ep_datas.append((Seg(**{k: np.stack(v) for k, v in self.history.items()}), value))
            self.history_ep_infos.append(Info(ep_return=self.ep_return))
            self.stored_chunk_num += np.ceil(self.ep_step / self.chunk_len)
            self.reset()
            if self.stored_chunk_num >= self.min_return_chunk_num:
                datas = self.history_ep_datas
                infos = self.history_ep_infos
                self.history_ep_datas = []
                self.history_ep_infos = []
                self.stored_chunk_num = 0
                return datas, infos, self._get_model_input()
            else:
                return [], [], self._get_model_input()

        n_obs, r, d, _ = self.env.step(action)
        n_obs = self._preprocess(n_obs)

        self.ep_return += r
        self.ep_step += 1

        self.history['obs'].append(self.obs)
        self.history['action'].append(action)
        self.history['action_logits'].append(action_logits)
        self.history['reward'].append(np.array([r], dtype=np.float32))
        self.history['value'].append(value)
        self.history['rnn_hidden'].append(self.rnn_hidden)
        self.history['pad_mask'].append(np.array([1 - self.done], dtype=np.float32))

        self.obs = n_obs
        self.rnn_hidden = rnn_hidden
        self.done = d or self.ep_step >= self.max_timesteps
        return [], [], self._get_model_input()

    def reset_history(self):
        self.history = OrderedDict({})
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
