# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.
For details and theory see:
"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.
See https://arxiv.org/abs/1802.01561 for the full paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import torch as T

VTraceReturns = namedtuple('VTraceReturns',
                           ['vs', 'advantages', 'pg_advantages'])


@T.no_grad()
def from_importance_weights(log_rhos,
                            discounts,
                            rewards,
                            values,
                            bootstrap_value,
                            clip_rho_threshold=1.0,
                            clip_c_threshold=1.0):

    # Make sure tensor ranks are consistent.
    rho_rank = log_rhos.dim()  # Usually 2.
    assert values.dim() == rho_rank
    assert bootstrap_value.dim() == rho_rank - 1
    assert discounts.dim() == rho_rank
    assert rewards.dim() == rho_rank

    ###########################################
    # before transpose, batch first
    log_rhos = log_rhos.transpose(0, 1)
    discounts = discounts.transpose(0, 1)
    rewards = rewards.transpose(0, 1)
    values = values.transpose(0, 1)
    # after transpose, time first
    ###########################################

    if clip_rho_threshold is not None:
        assert isinstance(clip_rho_threshold, float)
    if clip_c_threshold is not None:
        assert isinstance(clip_c_threshold, float)

    rhos = log_rhos.exp()
    if clip_rho_threshold is not None:
        clipped_rhos = T.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = T.clamp(rhos, max=clip_c_threshold)

    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = T.cat(
        [values[1:], bootstrap_value.unsqueeze(0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    # V-trace vs are calculated through a scan from the back to the beginning
    # of the given trajectory.
    vs_minus_v_xs = T.zeros_like(values)
    for i in range(discounts.shape[0] - 1, -1, -1):
        discount_t, c_t, delta_t = discounts[i], cs[i], deltas[i]
        if i == discounts.shape[0] - 1:
            vs_minus_v_xs[i] = delta_t
        else:
            vs_minus_v_xs[i] = delta_t + discount_t * c_t * vs_minus_v_xs[i +
                                                                          1]

    # Add V(x_s) to get v_s.
    vs = vs_minus_v_xs + values

    # Advantage for policy gradient.
    vs_t_plus_1 = T.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
    advantages = rewards + discounts * vs_t_plus_1 - values

    pg_advantages = clipped_rhos * advantages

    ###########################################
    # before transpose, time first
    vs = vs.transpose(0, 1)
    advantages = advantages.transpose(0, 1)
    pg_advantages = pg_advantages.transpose(0, 1)
    # after transpose, batch first
    ###########################################

    # Make sure no gradients backpropagated through the returned values.
    return VTraceReturns(vs=vs.detach(),
                         advantages=advantages.detach(),
                         pg_advantages=pg_advantages.detach())
