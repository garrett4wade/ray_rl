from collections import namedtuple

import torch as T

GernelizedAdvantageEsitimationReturns = namedtuple(
    'GAE', ['advantages', 'deltas', 'discounted_returns'])


@T.no_grad()
def from_rewards(rewards,
                 values,
                 target_values,
                 bootstrap_value,
                 gamma=0.99,
                 lamb=0.97):

    # Make sure tensor ranks are consistent.
    assert isinstance(gamma, float)
    assert isinstance(lamb, float)
    r_rank = rewards.dim()  # Usually 2.
    assert values.dim() == r_rank
    assert bootstrap_value.dim() == r_rank - 1
    assert rewards.dim() == r_rank

    ###########################################
    # before transpose, batch first
    rewards = rewards.transpose(0, 1)
    values = values.transpose(0, 1)
    target_values = target_values.transpose(0, 1)
    # after transpose, time first
    ###########################################

    bootstrap_r = T.cat([rewards, bootstrap_value.unsqueeze(0)], dim=0)
    bootstrap_v = T.cat([target_values, bootstrap_value.unsqueeze(0)], dim=0)

    discounted_returns = T.zeros_like(bootstrap_r, dtype=T.float)
    for i in range(bootstrap_r.shape[0] - 1, -1, -1):
        if i == bootstrap_r.shape[0] - 1:
            discounted_returns[i] = bootstrap_r[i]
        else:
            discounted_returns[
                i] = discounted_returns[i + 1] * gamma + bootstrap_r[i]
    discounted_returns = discounted_returns[:-1]

    deltas = bootstrap_r[:-1] + gamma * bootstrap_v[1:] - bootstrap_v[:-1]

    advantages = T.zeros_like(deltas, dtype=T.float)
    for i in range(deltas.shape[0] - 1, -1, -1):
        if i == deltas.shape[0] - 1:
            advantages[i] = deltas[i]
        else:
            advantages[i] = advantages[i + 1] * gamma * lamb + deltas[i]

    advantages = (advantages - advantages.mean()) / advantages.std()

    ###########################################
    # before transpose, time first
    advantages = advantages.transpose(0, 1)
    deltas = deltas.transpose(0, 1)
    discounted_returns = discounted_returns.transpose(0, 1)
    # after transpose, batch first
    ###########################################

    return GernelizedAdvantageEsitimationReturns(
        advantages=advantages.detach(),
        deltas=deltas.detach(),
        discounted_returns=discounted_returns.detach())
