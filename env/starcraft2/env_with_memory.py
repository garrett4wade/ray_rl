import numpy as np
from scipy.signal import lfilter
from env.starcraft2.registry import get_shapes, ROLLOUT_KEYS, COLLECT_KEYS, DTYPES, Seg, Info


class EnvWithMemory:
    def __init__(self, env_fn, config):
        self.env = env_fn(config)
        self.shapes = get_shapes(config)

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

    def _preprocess(self, obs, state, avail_action, agent_death):
        return obs.astype(np.float32), state.astype(np.float32), avail_action.astype(np.uint8), agent_death.astype(
            np.uint8)

    def _get_model_input(self):
        return self.obs, self.state, self.avail_action, self.actor_rnn_hidden, self.critic_rnn_hidden

    def _init_rnn_hidden(self):
        return (np.zeros(self.shapes['actor_rnn_hidden'],
                         dtype=np.float32), np.zeros(self.shapes['critic_rnn_hidden'], dtype=np.float32))

    def reset(self):
        if self.verbose and self.done:
            print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.obs, self.state, self.avail_action, self.agent_death = self._preprocess(*self.env.reset())
        self.actor_rnn_hidden, self.critic_rnn_hidden = self._init_rnn_hidden()
        self.done = self.won = False
        self.ep_step = self.ep_return = 0
        self.reset_history()

    def step(self, action, action_logits, value, actor_rnn_hidden, critic_rnn_hidden):
        n_obs, n_state, n_avail_action, r, n_agent_death, d, info = self.env.step(action)
        n_obs, n_state, n_avail_action, n_agent_death = self._preprocess(n_obs, n_state, n_avail_action, n_agent_death)

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

        if self.done:
            self.history_ep_datas.append(self.collect())
            self.history_ep_infos.append(Info(ep_return=self.ep_return, winning_rate=float(self.won)))
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

    def collect(self):
        v_target, adv = self.compute_gae()
        data_batch = {}
        for k in COLLECT_KEYS:
            if k in ROLLOUT_KEYS:
                data_batch[k] = np.stack(self.history[k], axis=0).astype(DTYPES[k])
            elif k == 'value_target':
                data_batch[k] = v_target.astype(DTYPES[k])
            elif k == 'adv':
                data_batch[k] = adv.astype(DTYPES[k])
        return self.to_chunk(data_batch)

    def to_chunk(self, data_batch):
        chunk_num = int(np.ceil(self.ep_step / self.chunk_len))
        target_len = chunk_num * self.chunk_len
        chunks = {}
        for k, v in data_batch.items():
            if 'rnn_hidden' in k:
                indices = np.arange(chunk_num) * self.chunk_len
                chunks[k] = np.swapaxes(v[indices], 1, 0)
            else:
                if len(v) < target_len:
                    pad = tuple([(0, target_len - len(v))] + [(0, 0)] * (v.ndim - 1))
                    pad_v = np.pad(v, pad, 'constant', constant_values=0)
                else:
                    pad_v = v
                chunks[k] = np.swapaxes(pad_v.reshape(chunk_num, self.chunk_len, *v.shape[1:]), 1, 0)
        self.stored_chunk_num += chunk_num
        return Seg(**chunks)

    def compute_gae(self, bootstrap_step=np.inf):
        reward = np.array(self.history['reward'], dtype=np.float32)
        value = np.array(self.history['value'], dtype=np.float32)
        assert reward.ndim == 1 and value.ndim == 1
        bootstrap_v = np.array(self.history['value'][1:] + [0], dtype=np.float32)
        one_step_td = reward + self.gamma * bootstrap_v - value
        adv = lfilter([1], [1, -self.lmbda * self.gamma], one_step_td[::-1])[::-1]

        if bootstrap_step >= self.ep_step:
            # monte carlo return
            v_target = lfilter([1], [1, -self.gamma], reward[::-1])[::-1]
        else:
            # n-step TD return
            discounted_r = lfilter(self.gamma**np.arange(self.chunk_len), [1], reward[::-1])[::-1]
            bootstrapped_v = np.concatenate([value[bootstrap_step:], np.zeros(bootstrap_step, dtype=np.float32)])
            v_target = discounted_r + self.gamma**bootstrap_step * bootstrapped_v
        return v_target, adv


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
        if len(datas) > 0:
            seg = Seg(*[np.concatenate([getattr(x, k) for x in datas], axis=1) for k in Seg._fields])
        else:
            seg = ()
        return seg, infos, stacked_model_inputs

    def get_model_inputs(self):
        model_inputs = [env._get_model_input() for env in self.envs]
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return stacked_model_inputs

    def close(self):
        for env in self.envs:
            env.close()
