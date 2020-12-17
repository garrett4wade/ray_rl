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
        self.history_ep_returns = []
        self.history_wons = []

        self.chunk_len = kwargs['chunk_len']
        self.burn_in_len = kwargs['burn_in_len']
        self.replay = kwargs['replay']
        self.min_return_chunk_num = kwargs['min_return_chunk_num']
        assert self.chunk_len % self.replay == 0
        self.step_size = self.chunk_len // self.replay

        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.max_timesteps = kwargs['max_timesteps']

        self.verbose = kwargs['verbose']
        self.reset()

    def preprocess(self, obs, state, avail_action, agent_death):
        return obs.astype(np.float32), state.astype(np.float32), avail_action.astype(np.float32), agent_death.astype(
            np.float32)

    def reset(self):
        if self.verbose and self.done:
            print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.obs, self.state, self.avail_action, self.agent_death = self.preprocess(*self.env.reset())
        self.actor_rnn_hidden, self.critic_rnn_hidden = self._init_rnn_hidden()
        self.done = self.won = False
        self.ep_step = self.ep_return = 0
        self.reset_history()

    def _init_rnn_hidden(self):
        return (np.zeros(self.shapes['actor_rnn_hidden'],
                         dtype=np.float32), np.zeros(self.shapes['critic_rnn_hidden'], dtype=np.float32))

    def reset_history(self):
        self.history = {}
        for k in self.rollout_keys:
            if k in self.burn_in_input_keys:
                self.history[k] = [np.zeros(self.shapes[k], dtype=np.float32) for _ in range(self.burn_in_len)]
            else:
                self.history[k] = []

    def _get_model_input(self):
        return self.obs, self.state, self.avail_action, self.actor_rnn_hidden, self.critic_rnn_hidden

    def step(self, **kwargs):
        action, action_logits, value = kwargs.get('action'), kwargs.get('action_logits'), kwargs.get('value')
        actor_rnn_hidden, critic_rnn_hidden = kwargs.get('actor_rnn_hidden'), kwargs.get('critic_rnn_hidden')
        if self.done:
            # if env is done in the previous step, use bootstrap value
            # to compute gae and collect history data
            self.history_data_batches += self.collect(value)
            self.history_ep_returns.append(self.ep_return)
            self.history_wons.append(self.won)
            self.reset()
            if len(self.history_data_batches) > self.min_return_chunk_num:
                data_batches = self.history_data_batches
                ep_returns = self.history_ep_returns
                wons = self.history_wons
                self.history_data_batches = []
                self.history_ep_returns = []
                self.history_wons = []
                return data_batches, ep_returns, wons, self._get_model_input()
            else:
                return [], [], [], self._get_model_input()

        n_obs, n_state, n_avail_action, r, n_agent_death, d, info = self.env.step(action)
        n_obs, n_state, n_avail_action, n_agent_death = self.preprocess(n_obs, n_state, n_avail_action, n_agent_death)

        self.ep_return += r
        self.ep_step += 1

        self.history['obs'].append(self.obs)
        self.history['state'].append(self.state)
        self.history['action'].append(action)
        self.history['action_logits'].append(action_logits)
        self.history['avail_action'].append(self.avail_action)
        self.history['reward'].append(r)
        self.history['value'].append(value)
        self.history['actor_rnn_hidden'].append(self.actor_rnn_hidden)
        self.history['critic_rnn_hidden'].append(self.critic_rnn_hidden)
        self.history['live_mask'].append(1 - self.agent_death)
        self.history['pad_mask'].append(1 - self.done)

        self.obs = n_obs
        self.state = n_state
        self.avail_action = n_avail_action
        self.actor_rnn_hidden = actor_rnn_hidden
        self.critic_rnn_hidden = critic_rnn_hidden
        self.agent_death = n_agent_death
        self.done = d or self.ep_step >= self.max_timesteps
        self.won = info.get('battle_won', False)
        return [], [], [], self._get_model_input()

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
                target_len = self.chunk_len + self.burn_in_len if k in self.burn_in_input_keys else self.chunk_len
                chunk[k] = v[chunk_cnt * self.step_size:chunk_cnt * self.step_size + target_len]
            for k, v in chunk.items():
                target_len = self.chunk_len + self.burn_in_len if k in self.burn_in_input_keys else self.chunk_len
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
        episode_length = len(self.history['reward'])
        n_step_v = discounted_r + bootstrap_value * self.gamma**np.arange(episode_length, 0, -1)
        td_err = n_step_v - np.array(self.history['value'], dtype=np.float32)
        adv = lfilter([1], [1, -self.lmbda], td_err[::-1])[::-1]
        return n_step_v, adv


class VecEnvWithMemory:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]

    def step(self, actions, action_logits, values, actor_rnn_hiddens, critic_rnn_hiddens):
        data_batches, ep_returns, wons, model_inputs = [], [], [], []
        for i, env in enumerate(self.envs):
            cur_data_batches, cur_ep_returns, cur_wons, model_input = env.step(
                **{
                    'action': actions[i],
                    'action_logits': action_logits[i],
                    'value': values[i],
                    'actor_rnn_hidden': actor_rnn_hiddens[i],
                    'critic_rnn_hidden': critic_rnn_hiddens[i],
                })
            data_batches += cur_data_batches
            ep_returns += cur_ep_returns
            wons += cur_wons
            model_inputs.append(model_input)
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return data_batches, ep_returns, wons, stacked_model_inputs

    def get_model_inputs(self):
        model_inputs = [env._get_model_input() for env in self.envs]
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return stacked_model_inputs

    def close(self):
        for env in self.envs:
            env.close()
