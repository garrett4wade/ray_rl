import numpy as np
from scipy.signal import lfilter

ROLLOUT_KEYS = ['obs', 'state', 'action', 'action_logits', 'avail_action', 'value', 'reward']
COLLECT_KEYS = ['obs', 'state', 'action', 'action_logits', 'avail_action', 'value', 'adv', 'value_target']


class EnvWithMemory:
    def __init__(self, env_fn, kwargs):
        self.env = env_fn(kwargs)

        self.is_continuous = False

        self.history_data_batches = []
        self.history_ep_returns = []
        self.history_wons = []

        self.chunk_len = kwargs['chunk_len']
        self.min_return_chunk_num = kwargs['min_return_chunk_num']

        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.max_timesteps = kwargs['max_timesteps']

        self.verbose = kwargs['verbose']
        self.reset()

    def reset(self):
        if self.done and self.verbose:
            print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.obs, self.state, self.avail_action = self.preprocess(*self.env.reset())
        self.done = self.won = False
        self.ep_step = self.ep_return = 0
        self.reset_history()

    def reset_history(self):
        self.history = {}
        for k in ROLLOUT_KEYS:
            self.history[k] = []

    def step(self, action, action_logits, value):
        if self.done:
            # if env is done in the previous step, use bootstrap value
            # to compute gae and collect history data
            self.history_data_batches += self.collect(value)
            self.history_ep_returns.append(self.ep_return)
            self.history_wons.append(self.won)
            self.reset()
            if len(self.history_data_batches) >= self.min_return_chunk_num:
                data_batches = self.history_data_batches
                ep_returns = self.history_ep_returns
                wons = self.history_wons
                self.history_data_batches = []
                self.history_ep_returns = []
                self.history_wons = []
                return data_batches, ep_returns, wons, (self.obs, self.state, self.avail_action)
            else:
                return [], [], [], (self.obs, self.state, self.avail_action)

        n_obs, n_state, n_avail_action, r, d, info = self.env.step(action)
        self.won = info['battle_won']
        n_obs, n_state, n_avail_action = self.preprocess(n_obs, n_state, n_avail_action)

        self.ep_return += r
        self.ep_step += 1

        self.history['obs'].append(self.obs)
        self.history['state'].append(self.state)
        self.history['action'].append(action)
        self.history['action_logits'].append(action_logits)
        self.history['avail_action'].append(self.avail_action)
        self.history['reward'].append(r)
        self.history['value'].append(value)

        self.obs = n_obs
        self.state = n_state
        self.avail_action = n_avail_action
        self.done = d or self.ep_step >= self.max_timesteps
        return [], [], [], (self.obs, self.state, self.avail_action)

    def collect(self, bootstrap_value):
        v_target, adv = self.compute_gae(bootstrap_value)
        data_batch = {}
        for k in COLLECT_KEYS:
            if k in ROLLOUT_KEYS:
                data_batch[k] = np.stack(self.history[k], axis=0)
            elif k == 'value_target':
                data_batch[k] = v_target
            elif k == 'adv':
                data_batch[k] = adv
        return self.split_and_padding(data_batch)

    def split_and_padding(self, data_batch):
        episode_length = len(data_batch['adv'])
        chunk_num = int(np.ceil(episode_length / self.chunk_len))
        split_indices = [self.chunk_len * i for i in range(1, chunk_num)]
        chunks = [{} for _ in range(chunk_num)]
        for k, v in data_batch.items():
            splitted_v = np.split(v, split_indices, axis=0)
            if len(splitted_v[-1]) < self.chunk_len:
                pad = tuple([(0, self.chunk_len - len(splitted_v[-1]))] + [(0, 0)] * (v.ndim - 1))
                splitted_v[-1] = np.pad(splitted_v[-1], pad, 'constant', constant_values=0)
            for i, chunk in enumerate(chunks):
                chunk[k] = splitted_v[i]
        return chunks

    def preprocess(self, obs, state, avail_action):
        return obs.astype(np.float32), state.astype(np.float32), avail_action.astype(np.float32)

    def compute_gae(self, bootstrap_value):
        discounted_r = lfilter([1], [1, -self.gamma], self.history['reward'][::-1])[::-1]
        episode_length = len(self.history['reward'])
        n_step_v = discounted_r + bootstrap_value * self.gamma**np.arange(episode_length, 0, -1)
        td_err = n_step_v - np.array(self.history['value'], dtype=np.float32)
        adv = lfilter([1], [1, -self.lmbda], td_err[::-1])[::-1]
        return n_step_v, adv


class VecEnvWithMemory:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]

    def step(self, actions, action_logits, values):
        data_batches, ep_returns, wons, model_inputs = [], [], [], []
        for i, env in enumerate(self.envs):
            cur_data_batches, cur_ep_returns, cur_wons, model_input = env.step(actions[i], action_logits[i], values[i])
            data_batches += cur_data_batches
            ep_returns += cur_ep_returns
            wons += cur_wons
            model_inputs.append(model_input)
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return data_batches, ep_returns, wons, stacked_model_inputs

    def get_model_inputs(self):
        return np.stack([env.obs for env in self.envs],
                        axis=0), np.stack([env.state for env in self.envs],
                                          axis=0), np.stack([env.avail_action for env in self.envs], axis=0)

    def close(self):
        for env in self.envs:
            env.close()
