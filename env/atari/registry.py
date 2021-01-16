from collections import namedtuple, OrderedDict
from numpy import uint8, float32

ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward', 'pad_mask', 'rnn_hidden']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target', 'pad_mask', 'rnn_hidden']

# Seg is a combination of rollout data in 1 episode
Seg = namedtuple('Seg', COLLECT_KEYS)

# Info is statistic infomation in 1 episode, e.g. return, winning rate
Info = namedtuple('Info', ['ep_return'])

DTYPES = OrderedDict({
    'obs': uint8,
    'action': uint8,
    'action_logits': float32,
    'value': float32,
    'adv': float32,
    'value_target': float32,
    'pad_mask': uint8,
    'rnn_hidden': float32,
})


def get_shapes(kwargs):
    return OrderedDict({
        'obs': kwargs['obs_dim'],
        'action': (),
        'action_logits': (kwargs['action_dim'], ),
        'value': (),
        'adv': (),
        'value_target': (),
        'pad_mask': (),
        'rnn_hidden': (1, kwargs['hidden_dim'] * 2),
    })
