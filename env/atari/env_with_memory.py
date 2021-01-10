import numpy as np
from scipy.signal import lfilter
from collections import OrderedDict, namedtuple

# NOTE: rnn_hidden must be the last one in keys
ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward', 'pad_mask', 'rnn_hidden']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target', 'pad_mask', 'rnn_hidden']
assert 'rnn_hidden' not in ROLLOUT_KEYS or 'rnn_hidden' == ROLLOUT_KEYS[-1]
assert 'rnn_hidden' not in COLLECT_KEYS or 'rnn_hidden' == COLLECT_KEYS[-1]
Seg = namedtuple('Seg', COLLECT_KEYS)
Info = namedtuple('Info', ['ep_return'])


class EnvWithMemory:
    def get_shapes(kwargs):
        return OrderedDict({
            'obs': kwargs['obs_dim'],
            'action': (),
            'action_logits': (kwargs['action_dim'], ),
            'value': (),
            'adv': (),
            'value_target': (),
            'pad_mask': (),
            'rnn_hidden': (1, kwargs['hidden_dim'] * 2),
        })

    def __init__(self, env_fn, kwargs):
        self.env = env_fn(kwargs)
        self.shapes = EnvWithMemory.get_shapes(kwargs)

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
        if self.done:
            # if env is done in the previous step, use bootstrap value
            # to compute gae and collect history data
            # import time
            # start = time.time()
            self.history_ep_datas.append(self.collect(value))
            # print("collect time: {}".format(1e3 * (time.time() - start)))
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
        self.history['pad_mask'].append(1 - self.done)

        self.obs = n_obs
        self.rnn_hidden = rnn_hidden
        self.done = d or self.ep_step >= self.max_timesteps
        return [], [], self._get_model_input()

    def reset_history(self):
        self.history = OrderedDict({})
        for k in ROLLOUT_KEYS:
            self.history[k] = []

    def collect(self, bootstrap_value):
        v_target, adv = self.compute_gae(bootstrap_value)
        data_batch = OrderedDict({})
        for k in COLLECT_KEYS:
            if k in ROLLOUT_KEYS:
                data_batch[k] = np.stack(self.history[k], axis=0)
            elif k == 'value_target':
                data_batch[k] = v_target
            elif k == 'adv':
                data_batch[k] = adv
        return self.to_chunk(data_batch)

    def to_chunk(self, data_batch):
        chunk_num = int(np.ceil(self.ep_step / self.chunk_len))
        target_len = chunk_num * self.chunk_len
        chunks = {}
        for k, v in data_batch.items():
            if 'rnn_hidden' in k:
                indices = np.arange(chunk_num) * self.chunk_len
                chunks[k] = np.transpose(v[indices], (1, 0, 2))
            else:
                if len(v) < target_len:
                    pad = tuple([(0, target_len - len(v))] + [(0, 0)] * (v.ndim - 1))
                    pad_v = np.pad(v, pad, 'constant', constant_values=0)
                else:
                    pad_v = v
                chunks[k] = pad_v.reshape(self.chunk_len, chunk_num, *v.shape[1:])
        self.stored_chunk_num += chunk_num
        return Seg(**chunks)

    def compute_gae(self, bootstrap_value):
        discounted_r = lfilter([1], [1, -self.gamma], self.history['reward'][::-1])[::-1]
        discount_factor = self.gamma**np.arange(self.ep_step, 0, -1)
        n_step_v = discounted_r + bootstrap_value * discount_factor
        td_err = n_step_v - np.array(self.history['value'], dtype=np.float32)
        adv = lfilter([1], [1, -self.lmbda], td_err[::-1])[::-1]
        return n_step_v, adv


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
