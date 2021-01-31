import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.parallel import DistributedDataParallel as DDP

# borrowed from soft-actor-critic
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class ActorCritic(nn.Module):
    def __init__(self, is_training, config):
        super().__init__()
        self.is_training = is_training
        action_scale = torch.tensor(config.action_scale.copy())
        action_loc = torch.tensor(config.action_loc.copy())
        self.action_scale = nn.Parameter(action_scale)
        self.action_scale.requires_grad = False
        self.action_loc = nn.Parameter(action_loc)
        self.action_loc.requires_grad = False

        obs_dim = config.obs_dim
        action_dim = config.action_dim
        hidden_dim = config.hidden_dim
        self.feature_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(normalized_shape=[hidden_dim]),
                                         nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                         nn.LayerNorm(normalized_shape=[hidden_dim]), nn.ReLU())
        # actor
        self.action_layer = nn.Linear(hidden_dim, 2 * action_dim)
        # critic
        self.value_layer = nn.Linear(hidden_dim, 1)

    def core(self, state):
        return self.feature_net(state)

    def forward(self, state):
        x = self.core(state)
        action_logits = self.action_layer(x)
        value = self.value_layer(x).squeeze(-1)

        mu, log_std = torch.split(action_logits, action_logits.shape[-1] // 2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        mu = torch.tanh(mu) * self.action_scale + self.action_loc
        dist = Normal(loc=mu, scale=log_std.exp())

        action = dist.sample()

        return action, action_logits, value

    @torch.no_grad()
    def select_action(self, obs):
        assert not self.is_training
        obs = torch.from_numpy(obs)
        results = self(obs)
        return [result.numpy() for result in results]


def compute_loss(learner,
                 obs,
                 action,
                 action_logits,
                 adv,
                 value,
                 value_target,
                 pad_mask,
                 clip_ratio,
                 world_size,
                 value_clip=True):
    if isinstance(learner, DDP):
        assert world_size > 1
        action_scale = learner.module.action_scale
        action_loc = learner.module.action_loc
        adv_mean = adv.mean()
        adv_meansq = adv.pow(2).mean()

        # all reduce to advantage mean & std
        dist.all_reduce_multigpu([adv_mean])
        dist.all_reduce_multigpu([adv_meansq])
        adv_mean = adv_mean / world_size
        adv_meansq = adv_meansq / world_size

        adv_std = (adv_meansq - adv_mean.pow(2)).sqrt()
    else:
        action_scale = learner.action_scale
        action_loc = learner.action_loc
        adv_mean = adv.mean()
        adv_std = adv.std(unbiased=False)
    adv = (adv - adv_mean) / (adv_std + 1e-8)
    _, target_action_logits, cur_state_value = learner(obs)

    action_dim = action_logits.shape[-1] // 2
    target_mu, target_log_std = torch.split(target_action_logits, action_dim, dim=-1)
    target_mu = torch.tanh(target_mu) * action_scale + action_loc
    target_log_std = target_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    target_dist = Normal(target_mu, target_log_std.exp())
    target_action_logprobs = target_dist.log_prob(action).sum(-1)

    mu, log_std = torch.split(action_logits, action_dim, dim=-1)
    mu = torch.tanh(mu) * action_scale + action_loc
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    behavior_dist = Normal(mu, log_std.exp())
    behavior_action_logprobs = behavior_dist.log_prob(action).sum(-1)

    log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
    rhos = log_rhos.exp()

    p_surr = adv * torch.clamp(rhos, 1 - clip_ratio, 1 + clip_ratio)
    p_loss = (-torch.min(p_surr, rhos * adv) * pad_mask).mean()

    if value_clip:
        cur_v_clipped = value + torch.clamp(cur_state_value - value, -clip_ratio, clip_ratio)
        v_loss = torch.max((cur_state_value - value_target)**2, (cur_v_clipped - value_target)**2)
    else:
        v_loss = (cur_state_value - value_target)**2
    v_loss = (v_loss * pad_mask).mean()

    entropy_loss = (-target_dist.entropy().sum(-1) * pad_mask).mean()
    return v_loss, p_loss, entropy_loss
