import numpy as np
from collections import namedtuple
from scipy.signal import lfilter
from collections import OrderedDict
from rl_utils.utils import StorageProperty, get_simplex_shapes

StorageBlock = namedtuple('StorageBlock', ['main'])


class EnvWithMemory:
    ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward']
    COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target']

    def get_shapes(kwargs):
        return OrderedDict({
            'obs': (kwargs['obs_dim'], ),
            'action': (kwargs['action_dim'], ),
            'action_logits': (kwargs['action_dim'] * 2, ),
            'value': (1, ),
            'adv': (1, ),
            'value_target': (1, ),
        })


    def __init__(self, env_fn, kwargs):
        self.env = env_fn(kwargs)

        self.stored_chunk_num = 0
        self.history_storage_blocks = []
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
        return obs.astype(np.float32)

    def _get_model_input(self):
        return (self.obs, )

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
            self.history_storage_blocks.append(self.collect(value))
            self.history_ep_infos.append(dict(ep_return=self.ep_return))
            self.reset()
            if self.stored_chunk_num >= self.min_return_chunk_num:
                storage_blocks = self.history_storage_blocks
                infos = self.history_ep_infos
                self.history_storage_blocks = []
                self.history_ep_infos = []
                self.stored_chunk_num = 0
                return storage_blocks, infos, self._get_model_input()
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

        self.obs = n_obs
        self.done = d or self.ep_step >= self.max_timesteps
        return [], [], self._get_model_input()

    def reset_history(self):
        self.history = OrderedDict({})
        for k in EnvWithMemory.ROLLOUT_KEYS:
            self.history[k] = []

    def collect(self, bootstrap_value):
        v_target, adv = self.compute_gae(bootstrap_value)
        data_batch = OrderedDict({})
        for k in EnvWithMemory.COLLECT_KEYS:
            if k in EnvWithMemory.ROLLOUT_KEYS:
                data_batch[k] = np.stack(self.history[k], axis=0)
            elif k == 'value_target':
                data_batch[k] = v_target
            elif k == 'adv':
                data_batch[k] = adv
        concat_data_batch = np.concatenate([v.reshape(-1) for v in data_batch.values()], axis=-1)
        return self.to_chunk(concat_data_batch)

    def to_chunk(self, concat_data_batch):
        chunk_num = int(np.ceil(self.ep_step / self.chunk_len))
        target_len = chunk_num * self.chunk_len
        if len(concat_data_batch) < target_len:
            pad = tuple([(0, target_len - len(concat_data_batch))] + [(0, 0)] * (concat_data_batch.ndim - 1))
            concat_data_batch = np.pad(concat_data_batch, pad, 'constant', constant_values=0)
        concat_data_batch = concat_data_batch.reshape(self.chunk_len, chunk_num, *concat_data_batch.shape[1:])
        self.stored_chunk_num += chunk_num
        return concat_data_batch

    def compute_gae(self, bootstrap_value):
        reward = np.array(self.history['reward'], dtype=np.float32).squeeze(-1)
        value = np.array(self.history['value'], dtype=np.float32).squeeze(-1)
        discounted_r = lfilter([1], [1, -self.gamma], reward[::-1])[::-1]
        discount_factor = self.gamma**np.arange(self.ep_step, 0, -1)
        n_step_v = discounted_r + bootstrap_value * discount_factor
        td_err = n_step_v - value
        adv = lfilter([1], [1, -self.lmbda], td_err[::-1])[::-1]
        return np.expand_dims(n_step_v, 1), np.expand_dims(adv, 1)


class VecEnvWithMemory:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]

    def step(self, *args):
        storage_blocks, infos, model_inputs = [], [], []
        for i, env in enumerate(self.envs):
            cur_storage_blocks, cur_infos, model_input = env.step(*[arg[i] for arg in args])
            storage_blocks += cur_storage_blocks
            infos += cur_infos
            model_inputs.append(model_input)
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        concat_storage_block = np.concatenate(storage_blocks, axis=1) if len(storage_blocks) > 0 else None
        return concat_storage_block, infos, stacked_model_inputs

    def get_model_inputs(self):
        model_inputs = [env._get_model_input() for env in self.envs]
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return stacked_model_inputs

    def close(self):
        for env in self.envs:
            env.close()
