import numpy as np
from scipy.signal import lfilter
from collections import namedtuple
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target']


class Env:
    def __init__(self, env_fn, env_id, kwargs):
        self.env = env_fn(kwargs)
        self.env.seed(env_id * 12345 + kwargs['seed'])

        if isinstance(self.env.action_space, Box):
            self.is_continuous = True
        elif isinstance(self.env.action_space, Discrete):
            self.is_continuous = False
        assert not self.is_continuous, "this branch is for Atari game, environment action space must be discrete"

        self.chunk_len = kwargs['chunk_len']

        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.max_timesteps = kwargs['max_timesteps']
        self.done = False
        self.ep_step = self.ep_return = 0
        self.reset()

    def reset(self):
        if self.done:
            print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.obs = self.preprocess_obs(self.env.reset())
        self.done = False
        self.ep_step = self.ep_return = 0
        self.reset_history()

    def reset_history(self):
        self.history = {}
        for k in ROLLOUT_KEYS:
            self.history[k] = []

    def step(self, action, action_logits, value, model):
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

        if self.done:
            bootstrap_value = model.value(n_obs)
            data_batch = self.collect(bootstrap_value)
            ep_return = self.ep_return
            self.reset()
            return data_batch, ep_return
        else:
            return None, None

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
        chunk_num = int(np.ceil(episode_length // self.chunk_len))
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


class Envs:
    def __init__(self, env_fn, worker_id, kwargs):
        self.envs = [Env(env_fn, worker_id + i * kwargs['num_workers'], kwargs) for i in range(kwargs['env_num'])]

    def step(self, model):
        actions, action_logits, values = model.select_action(self.get_model_inputs())

        data_batches, ep_returns = [], []
        for i, env in enumerate(self.envs):
            cur_data_batches, cur_ep_return = env.step(actions[i], action_logits[i], values[i], model)
            if len(cur_data_batches) > 0:
                data_batches += cur_data_batches
            if cur_ep_return is not None:
                ep_returns.append(cur_ep_return)
        return data_batches, ep_returns

    def get_model_inputs(self):
        return np.stack([env.obs for env in self.envs], axis=0)
