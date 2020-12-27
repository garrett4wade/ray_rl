import numpy as np
from collections import namedtuple
from scipy.signal import lfilter
from collections import OrderedDict
from rl_utils.utils import StorageProperty, get_simplex_shapes

StorageBlock = namedtuple('StorageBlock', ['main'])


class EnvWithMemory:
    ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward']
    COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target']

    def get_collect_shapes(kwargs):
        return OrderedDict({
            'obs': (kwargs['obs_dim'], ),
            'action': (kwargs['action_dim'], ),
            'action_logits': (kwargs['action_dim'] * 2, ),
            'value': (1, ),
            'adv': (1, ),
            'value_target': (1, ),
        })

    def get_storage_properties(kwargs):
        return OrderedDict({
            "main":
            StorageProperty(length=kwargs['chunk_len'],
                            agent_num=1,
                            keys=EnvWithMemory.COLLECT_KEYS,
                            simplex_shapes=get_simplex_shapes(EnvWithMemory.get_collect_shapes(kwargs)))
        })

    def __init__(self, env_fn, kwargs):
        self.env = env_fn(kwargs)
        self.storage_properties = EnvWithMemory.get_storage_properties(kwargs)

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
        self.history = {}
        for k in EnvWithMemory.ROLLOUT_KEYS:
            self.history[k] = []

    def collect(self, bootstrap_value):
        v_target, adv = self.compute_gae(bootstrap_value)
        data_batch = {}
        for k in EnvWithMemory.COLLECT_KEYS:
            if k in EnvWithMemory.ROLLOUT_KEYS:
                data_batch[k] = np.stack(self.history[k], axis=0)
            elif k == 'value_target':
                data_batch[k] = v_target
            elif k == 'adv':
                data_batch[k] = adv
        concat_data_batch = self.concat_storage_type(data_batch)
        return self.pad_then_reshape(concat_data_batch)

    def pad_then_reshape(self, concat_data_batch):
        chunk_num = int(np.ceil(self.ep_step / self.chunk_len))
        target_len = chunk_num * self.chunk_len
        for st, concat_v in concat_data_batch.items():
            if len(concat_v) < target_len:
                pad = tuple([(0, target_len - len(concat_v))] + [(0, 0)] * (concat_v.ndim - 1))
                pad_concat_v = np.pad(concat_v, pad, 'constant', constant_values=0)
            else:
                pad_concat_v = concat_v
            concat_data_batch[st] = pad_concat_v.reshape(self.chunk_len, chunk_num, *concat_v.shape[1:])
        self.stored_chunk_num += chunk_num
        return StorageBlock(**concat_data_batch)

    def concat_storage_type(self, data_batch):
        results = {}
        for st, ppty in self.storage_properties.items():
            results[st] = []
            # NOTE: ppty.keys is ordered, thus concatenated result is also ordered
            for k in ppty.keys:
                results[st].append(data_batch[k])
        return {st: np.concatenate(all_v, axis=-1) for st, all_v in results.items()}

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
        if len(storage_blocks) > 0:
            concat_storage_block = StorageBlock(*[np.concatenate(blks, axis=1) for blks in zip(*storage_blocks)])
        else:
            concat_storage_block = ()
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
