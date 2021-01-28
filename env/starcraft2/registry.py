from collections import namedtuple, OrderedDict
from numpy import float32, uint8

ROLLOUT_KEYS = [
    'obs', 'state', 'action', 'action_logits', 'avail_action', 'value', 'reward', 'actor_rnn_hidden',
    'critic_rnn_hidden', 'pad_mask', 'live_mask'
]
COLLECT_KEYS = [
    'obs', 'state', 'action', 'action_logits', 'avail_action', 'value', 'adv', 'value_target', 'actor_rnn_hidden',
    'critic_rnn_hidden', 'pad_mask', 'live_mask'
]

# Seg is a combination of rollout data in 1 episode
Seg = namedtuple('Seg', COLLECT_KEYS)

# Info is statistic infomation in 1 episode, e.g. return, winning rate
Info = namedtuple('Info', ['ep_return', 'winning_rate'])

DTYPES = OrderedDict({
    'obs': float32,
    'state': float32,
    'action': uint8,
    'action_logits': float32,
    'avail_action': uint8,
    'value': float32,
    'adv': float32,
    'value_target': float32,
    'actor_rnn_hidden': float32,
    'critic_rnn_hidden': float32,
    'pad_mask': uint8,
    'live_mask': uint8,
})


def get_shapes(config):
    return {
        'obs': (config.agent_num, config.obs_dim),
        'state': (config.state_dim, ),
        'action': (config.agent_num, ),
        'action_logits': (config.agent_num, config.action_dim),
        'avail_action': (config.agent_num, config.action_dim),
        'value': (),
        'adv': (),
        'value_target': (),
        'actor_rnn_hidden': (config.actor_rnn_layers, config.agent_num, config.hidden_dim),
        'critic_rnn_hidden': (config.critic_rnn_layers, config.hidden_dim),
        'pad_mask': (),
        'live_mask': (config.agent_num, ),
    }
