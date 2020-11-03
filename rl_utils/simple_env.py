import numpy as np
from collections import namedtuple
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

# Seg is a tuple that gathers all kinds of data we need
Seg = namedtuple("Seg", ['state', 'action', 'action_logits', 'value', 'reward', 'done_mask'])


class Env:
    '''
    Gym env wrapper for recurrent network training.

    When stepping along one episode,
    store required data into (self.**_history).

    When a episode is done or there's plenty of history data,
    cut history data into (burn_in_len + chunk_len + 1) sequences
    for training.
    NOTE:
    1. 'burn_in_len' data is for GRU hidden state computation for
       'chunk_len' data.
    2. 'chunk_len' data is for loss computation.
    3. Last time dimension is for value bootstraping.
    '''
    def __init__(self, env_fn, kwargs):
        self.env = env_fn(kwargs)
        if isinstance(self.env.action_space, Box):
            self.is_continuous = True
        elif isinstance(self.env.action_space, Discrete):
            self.is_continuous = False
        assert not self.is_continuous, "this branch is for Atari game, environment action space must be discrete"

        # basic network parameters
        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']

        # for recurrent network training, hyperparameters of sequece data
        self.burn_in_len = 0
        self.chunk_len = kwargs['chunk_len']
        self.replay = 1
        self.max_timesteps = kwargs['max_timesteps']

        # parameters that decide how much data we need before collecting them
        self.min_return_chunk_num = kwargs['min_return_chunk_num']
        assert self.chunk_len % self.replay == 0
        self.step_size = self.chunk_len // self.replay
        self.target_len = self.burn_in_len + self.chunk_len + 1

        self.done = False
        self.ep_return = 0
        self.ep_step = 0
        self.reset()

    def reset(self):
        if self.done:
            print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.state = self.preprocess_state(self.env.reset())
        self.done = False
        self.ep_return = 0
        self.ep_step = 0

        self.reset_history()
        '''
            state, done & hidden are aligned
            others are aligned with 1 less element than them
        '''
        self.state_history.append(self.state)
        self.done_mask.append(1)

    def reset_history(self):
        self.state_history = [np.zeros(self.state_dim, dtype=np.float32)] * self.burn_in_len
        # for discrete env, action is an integer number
        self.action_history = [0] * self.burn_in_len
        self.action_logits_history = [np.zeros(self.action_dim, dtype=np.float32)] * self.burn_in_len
        self.reward_history = [0] * self.burn_in_len
        self.done_mask = [0] * self.burn_in_len
        self.value_history = [0] * self.burn_in_len

    def step(self, action, action_logits, value):
        nex_state, r, d, _ = self.env.step(action)
        nex_state = self.preprocess_state(nex_state)

        self.ep_return += r
        self.ep_step += 1

        self.action_history.append(action)
        self.action_logits_history.append(action_logits)
        self.reward_history.append(r)
        self.value_history.append(value)

        self.state = nex_state
        self.done = d or self.ep_step >= self.max_timesteps

        self.state_history.append(nex_state)
        self.done_mask.append(0 if self.done else 1)

        segs = self.collect()
        ep_return = None
        if self.done:
            ep_return = self.ep_return
            self.reset()
        return segs, ep_return

    def collect(self):
        seg = None
        # while not done, if history data can compose k chunks,
        # then return them early before done
        k = self.min_return_chunk_num
        burn_in = self.burn_in_len
        chunk = self.chunk_len
        step_size = self.step_size
        cut_len = (k - 1) * step_size + burn_in + chunk + 1

        if self.done:
            seg = Seg(np.stack(self.state_history, axis=0), np.stack(self.action_history, axis=0),
                      np.stack(self.action_logits_history, axis=0), np.stack(self.value_history, axis=0),
                      np.stack(self.reward_history, axis=0), np.stack(self.done_mask, axis=0))
        elif len(self.value_history) >= cut_len:
            seg = Seg(np.stack(self.state_history[:cut_len], axis=0), np.stack(self.action_history[:cut_len], axis=0),
                      np.stack(self.action_logits_history[:cut_len], axis=0),
                      np.stack(self.value_history[:cut_len], axis=0), np.stack(self.reward_history[:cut_len], axis=0),
                      np.stack(self.done_mask[:cut_len], axis=0))

            cut_off = k * step_size
            self.state_history = self.state_history[cut_off:]
            self.action_history = self.action_history[cut_off:]
            self.action_logits_history = self.action_logits_history[cut_off:]
            self.value_history = self.value_history[cut_off:]
            self.reward_history = self.reward_history[cut_off:]
            self.done_mask = self.done_mask[cut_off:]

        if seg is None:
            return []
        else:
            return self.split_and_padding(seg)

    def split_and_padding(self, seg):
        results = []
        step_size = self.chunk_len // self.replay
        target_len = self.burn_in_len + self.chunk_len + 1
        while len(seg[0]) > 0:
            '''
                NOTE need to ensure first dim is time dimension
            '''
            sequence = [s[:target_len] for s in seg]
            keys = ['state', 'action', 'action_logits', 'value', 'reward', 'done_mask']
            seg_dict = dict()
            for i, s in enumerate(sequence):
                key = keys[i]
                pad = tuple([(0, target_len - s.shape[0])] + [(0, 0)] * (s.ndim - 1))
                seg_dict[key] = np.pad(s, pad, 'constant', constant_values=0)
            results.append(seg_dict)
            seg = Seg(*[s[step_size:] for s in seg])
        return results

    def get_state(self):
        return self.state
    
    def preprocess_state(self, state):
        state = np.transpose(state, [2, 0, 1]).astype(np.float32)
        return (state - 128.0) / 128.0


class Envs:
    def __init__(self, env_fn, kwargs):
        self.envs = [Env(env_fn, kwargs) for _ in range(kwargs['env_num'])]

    def step(self, model):
        actions, action_logits, values = model.step(self.get_model_inputs())

        segs, ep_returns = [], []
        for i, env in enumerate(self.envs):
            cur_segs, cur_ep_return = env.step(actions[i], action_logits[i], values[i])
            if len(cur_segs) > 0:
                segs += cur_segs
            if cur_ep_return is not None:
                ep_returns.append(cur_ep_return)
        return segs, ep_returns

    def get_model_inputs(self):
        return np.stack([env.get_state() for env in self.envs], axis=0)
