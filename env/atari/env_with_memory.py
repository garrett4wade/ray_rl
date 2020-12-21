import numpy as np
from scipy.signal import lfilter
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target']


class EnvWithMemory:
    def __init__(self, env_fn, env_id, kwargs):
        self.env = env_fn(kwargs)
        self.env.seed(env_id * 12345 + kwargs['seed'])

        if isinstance(self.env.action_space, Box):
            self.is_continuous = True
        elif isinstance(self.env.action_space, Discrete):
            self.is_continuous = False
        assert not self.is_continuous, "this branch is for Atari game, environment action space must be discrete"

        self.history_data_batches = []
        self.history_ep_returns = []

        self.chunk_len = kwargs['chunk_len']
        self.min_return_chunk_num = kwargs['min_return_chunk_num']

        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.max_timesteps = kwargs['max_timesteps']
        self.done = False
        self.ep_step = self.ep_return = 0
        self.reset()

    def reset(self):
        # if self.done:
        #     print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.obs = self.preprocess_obs(self.env.reset())
        self.done = False
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
            self.reset()
            if len(self.history_data_batches) >= self.min_return_chunk_num:
                data_batches = self.history_data_batches
                ep_returns = self.history_ep_returns
                self.history_data_batches = []
                self.history_ep_returns = []
                return data_batches, ep_returns, (self.obs, )
            else:
                return [], [], (self.obs, )

        n_obs, r, d, _ = self.env.step(action)
        n_obs = self.preprocess_obs(n_obs)

        self.ep_return += r
        self.ep_step += 1

        self.history['obs'].append(self.obs)
        self.history['action'].append(action)
        self.history['action_logits'].append(action_logits)
        self.history['reward'].append(r)
        self.history['value'].append(value)

        self.obs = n_obs
        self.done = d or self.ep_step >= self.max_timesteps
        return [], [], (self.obs, )

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

    def preprocess_obs(self, obs):
        obs = np.transpose(obs, [2, 0, 1]).astype(np.float32)
        return (obs - 128.0) / 128.0

    def compute_gae(self, bootstrap_value):
        discounted_r = lfilter([1], [1, -self.gamma], self.history['reward'][::-1])[::-1]
        episode_length = len(self.history['reward'])
        n_step_v = discounted_r + bootstrap_value * self.gamma**np.arange(episode_length, 0, -1)
        td_err = n_step_v - np.array(self.history['value'], dtype=np.float32)
        adv = lfilter([1], [1, -self.lmbda], td_err[::-1])[::-1]
        # td_lmbda = lfilter([1], [1, -self.lmbda], n_step_v[::-1])[::-1]
        return n_step_v, adv


class DummyVecEnvWithMemory:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]

    def step(self, actions, action_logits, values):
        data_batches, ep_returns, model_inputs = [], [], []
        for i, env in enumerate(self.envs):
            cur_data_batches, cur_ep_returns, model_input = env.step(actions[i], action_logits[i], values[i])
            if len(cur_data_batches) > 0:
                data_batches += cur_data_batches
            if len(cur_ep_returns) > 0:
                ep_returns += cur_ep_returns
            model_inputs.append(model_input)
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return data_batches, ep_returns, stacked_model_inputs

    def get_model_inputs(self):
        return [np.stack([env.obs for env in self.envs], axis=0)]
