import numpy as np
from scipy.signal import lfilter


class EnvWithMemory:
    def __init__(self, env_fn, rollout_keys, collect_keys, burn_in_input_keys, shapes, kwargs):
        self.env = env_fn(kwargs)
        self.rollout_keys = rollout_keys
        self.collect_keys = collect_keys
        self.burn_in_input_keys = burn_in_input_keys
        self.shapes = shapes

        self.history_data_batches = []
        self.history_ep_infos = []

        self.chunk_len = kwargs['chunk_len']
        self.burn_in_len = kwargs['burn_in_len']
        if np.all(['rnn_hidden' not in k for k in self.collect_keys]):
            assert self.burn_in_len == 0 and len(self.burn_in_input_keys) == 0
        self.replay = kwargs['replay']
        self.min_return_chunk_num = kwargs['min_return_chunk_num']
        assert self.chunk_len % self.replay == 0
        self.step_size = self.chunk_len // self.replay

        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.max_timesteps = kwargs['max_timesteps']

        self.verbose = kwargs['verbose']
        self.reset()

    def _preprocess(self, obs):
        return obs.astype(np.float32)

    def _get_model_input(self):
        return (self.obs, )

    def _init_rnn_hidden(self):
        return np.zeros(self.shapes['rnn_hidden'], dtype=np.float32)

    def reset(self):
        if self.verbose and self.done:
            print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.obs = self._preprocess(self.env.reset())
        self.done = False
        self.ep_step = self.ep_return = 0
        self.reset_history()

    def step(self, action, action_logits, value):
        if self.done:
            # if env is done in the previous step, use bootstrap value
            # to compute gae and collect history data
            self.history_data_batches += self.collect(value)
            self.history_ep_infos.append(dict(ep_return=self.ep_return))
            self.reset()
            if len(self.history_data_batches) >= self.min_return_chunk_num:
                data_batches = self.history_data_batches
                infos = self.history_ep_infos
                self.history_data_batches = []
                self.history_ep_infos = []
                return data_batches, infos, self._get_model_input()
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

        self.obs = n_obs
        self.done = d or self.ep_step >= self.max_timesteps
        return [], [], self._get_model_input()

    def reset_history(self):
        self.history = {}
        for k in self.rollout_keys:
            if k in self.burn_in_input_keys:
                self.history[k] = [np.zeros(self.shapes[k], dtype=np.float32) for _ in range(self.burn_in_len)]
            else:
                self.history[k] = []

    def collect(self, bootstrap_value):
        v_target, adv = self.compute_gae(bootstrap_value)
        data_batch = {}
        for k in self.collect_keys:
            if k in self.rollout_keys:
                data_batch[k] = np.stack(self.history[k], axis=0)
            elif k == 'value_target':
                data_batch[k] = v_target
            elif k == 'adv':
                data_batch[k] = adv
        return self.split_and_padding(data_batch)

    def split_and_padding(self, data_batch):
        data_length = self.ep_step + self.burn_in_len
        chunk_cnt = 0
        chunks = []
        while data_length > self.burn_in_len:
            chunk = {}
            for k, v in data_batch.items():
                target_len = (self.chunk_len + self.burn_in_len) if k in self.burn_in_input_keys else self.chunk_len
                chunk[k] = v[chunk_cnt * self.step_size:chunk_cnt * self.step_size + target_len]
            for k, v in chunk.items():
                target_len = (self.chunk_len + self.burn_in_len) if k in self.burn_in_input_keys else self.chunk_len
                if 'rnn_hidden' in k:
                    chunk[k] = v[0]
                elif len(v) < target_len:
                    pad = tuple([(0, target_len - len(v))] + [(0, 0)] * (v.ndim - 1))
                    chunk[k] = np.pad(v, pad, 'constant', constant_values=0)
            chunks.append(chunk)
            data_length -= self.step_size
            chunk_cnt += 1
        return chunks

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
        data_batches, infos, model_inputs = [], [], []
        for i, env in enumerate(self.envs):
            cur_data_batches, cur_infos, model_input = env.step(*[arg[i] for arg in args])
            data_batches += cur_data_batches
            infos += cur_infos
            model_inputs.append(model_input)
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return data_batches, infos, stacked_model_inputs

    def get_model_inputs(self):
        model_inputs = [env._get_model_input() for env in self.envs]
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return stacked_model_inputs

    def close(self):
        for env in self.envs:
            env.close()
