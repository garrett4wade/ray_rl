ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward', 'rnn_hidden', 'pad_mask']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target', 'rnn_hidden', 'pad_mask']
BURN_IN_INPUT_KEYS = ['obs', 'rnn_hidden']


def get_shapes(kwargs):
    return {
        'obs': kwargs['obs_dim'],
        'action': (),
        'action_logits': (kwargs['action_dim'], ),
        'value': (),
        'reward': (),
        'rnn_hidden': (
            1,
            kwargs['hidden_dim'] * 2,
        ),
        'pad_mask': (),
    }
