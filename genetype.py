# StarCraft2
ROLLOUT_KEYS = [
    'obs', 'state', 'action', 'action_logits', 'avail_action', 'value', 'reward', 'actor_rnn_hidden',
    'critic_rnn_hidden', 'pad_mask', 'live_mask'
]
COLLECT_KEYS = [
    'obs', 'state', 'action', 'action_logits', 'avail_action', 'value', 'adv', 'value_target', 'actor_rnn_hidden',
    'critic_rnn_hidden', 'pad_mask', 'live_mask'
]
BURN_IN_INPUT_KEYS = ['obs', 'state', 'actor_rnn_hidden', 'critic_rnn_hidden']


def get_shapes(kwargs):
    return {
        'obs': (
            kwargs['agent_num'],
            kwargs['obs_dim'],
        ),
        'state': (kwargs['state_dim'], ),
        'action': (kwargs['agent_num'], ),
        'action_logits': (
            kwargs['agent_num'],
            kwargs['action_dim'],
        ),
        'avail_action': (
            kwargs['agent_num'],
            kwargs['action_dim'],
        ),
        'value': (),
        'reward': (),
        'actor_rnn_hidden': (
            kwargs['actor_rnn_layers'],
            kwargs['agent_num'],
            kwargs['hidden_dim'],
        ),
        'critic_rnn_hidden': (
            kwargs['critic_rnn_layers'],
            kwargs['hidden_dim'],
        ),
        'pad_mask': (),
        'live_mask': (kwargs['agent_num'], ),
    }
