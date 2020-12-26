# MuJoCo
from rl_utils.utils import StorageProperty, get_simplex_shapes
ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target']
BURN_IN_INPUT_KEYS = []


def get_shapes(kwargs):
    return {
        'obs': (kwargs['obs_dim'], ),
        'action': (kwargs['action_dim'], ),
        'action_logits': (kwargs['action_dim'] * 2, ),
        'value': (1, ),
        'reward': (1, ),
        'adv': (1, ),
        'value_target': (1, ),
    }


def get_storage_property(kwargs):
    return {
        "main":
        StorageProperty(length=kwargs['chunk_len'],
                        agent_num=1,
                        keys=COLLECT_KEYS,
                        simplex_shapes=get_simplex_shapes(get_shapes(kwargs)))
    }
