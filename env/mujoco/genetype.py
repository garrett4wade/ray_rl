# MuJoCo
ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target']
BURN_IN_INPUT_KEYS = []


def get_shapes(kwargs):
    return {
        'obs': (kwargs['obs_dim'], ),
        'action': (kwargs['action_dim'], ),
        'action_logits': (kwargs['action_dim'] * 2, ),
        'value': (),
        'reward': (),
    }
