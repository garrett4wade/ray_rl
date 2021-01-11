from collections import namedtuple, OrderedDict

ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward', 'pad_mask']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target', 'pad_mask']

# Seg is a combination of rollout data in 1 episode
Seg = namedtuple('Seg', COLLECT_KEYS)

# Info is statistic infomation in 1 episode, e.g. return, winning rate
Info = namedtuple('Info', ['ep_return'])


def get_shapes(kwargs):
    return OrderedDict({
        'obs': (kwargs['obs_dim'], ),
        'action': (kwargs['action_dim'], ),
        'action_logits': (kwargs['action_dim'] * 2, ),
        'value': (),
        'adv': (),
        'value_target': (),
        'pad_mask': (),
    })
