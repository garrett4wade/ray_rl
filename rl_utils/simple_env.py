import numpy as np
from scipy.signal import lfilter
from collections import namedtuple
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target']


class Env:
    def __init__(self, env_fn, kwargs):
        self.env = env_fn(kwargs)

        if isinstance(self.env.action_space, Box):
            self.is_continuous = True
        elif isinstance(self.env.action_space, Discrete):
            self.is_continuous = False
        assert not self.is_continuous, "this branch is for Atari game, environment action space must be discrete"

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
        return data_batch

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
    def __init__(self, env_fn, kwargs):
        self.envs = [Env(env_fn, kwargs) for _ in range(kwargs['env_num'])]

    def step(self, model):
        actions, action_logits, values = model.select_action(self.get_model_inputs())

        data_batches, ep_returns = [], []
        for i, env in enumerate(self.envs):
            cur_data_batch, cur_ep_return = env.step(actions[i], action_logits[i], values[i], model)
            if cur_data_batch is not None:
                data_batches.append(cur_data_batch)
            if cur_ep_return is not None:
                ep_returns.append(cur_ep_return)
        return data_batches, ep_returns

    def get_model_inputs(self):
        return np.stack([env.obs for env in self.envs], axis=0)
