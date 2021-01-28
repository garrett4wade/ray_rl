import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class ActorCritic(nn.Module):
    def __init__(self, is_training, config):
        super().__init__()
        self.is_training = is_training
        self.obs_dim = obs_dim = config.obs_dim
        self.state_dim = state_dim = config.state_dim
        self.action_dim = action_dim = config.action_dim
        self.hidden_dim = hidden_dim = config.hidden_dim

        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, obs, state):
        return self.actor(obs), self.critic(state).squeeze(-1)

    @torch.no_grad()
    def select_action(self, obs, state, avail_action):
        assert not self.is_training
        obs = torch.from_numpy(obs)
        state = torch.from_numpy(state)
        avail_action = torch.from_numpy(avail_action)
        logits, value = self(obs, state)
        logits[avail_action == 0.0] = -1e10
        action = Categorical(logits=logits).sample()
        return action.numpy(), logits.numpy(), value.numpy()


def compute_loss(learner, obs, state, action, action_logits, avail_action, adv, value, value_target, clip_ratio,
                 world_size):
    if isinstance(learner, DDP):
        assert world_size > 1
        adv_mean = adv.mean()
        adv_meansq = adv.pow(2).mean()
        # all reduce to advantage mean & std
        dist.all_reduce_multigpu([adv_mean])
        dist.all_reduce_multigpu([adv_meansq])
        adv_mean = adv_mean / world_size
        adv_meansq = adv_meansq / world_size
        adv_std = (adv_meansq - adv_mean.pow(2)).sqrt()
    else:
        adv_mean = adv.mean()
        adv_std = adv.std(unbiased=False)
    adv = (adv - adv_mean) / (adv_std + 1e-8)
    action = action.to(torch.long)
    # to remove padding values
    valid_mask = (value != 0.0).to(torch.float32)

    target_action_logits, cur_state_value = learner(obs, state)
    target_action_logits[avail_action == 0.0] = -1e10

    target_dist = Categorical(logits=target_action_logits)
    target_action_logprobs = target_dist.log_prob(action)
    behavior_action_logprobs = Categorical(logits=action_logits).log_prob(action)

    log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
    # taking average along agent dim
    rhos = log_rhos.exp().mean(-1)

    p_surr = adv * torch.clamp(rhos, 1 - clip_ratio, 1 + clip_ratio)
    p_loss = (-torch.min(p_surr, rhos * adv) * valid_mask).mean()

    cur_v_clipped = value + torch.clamp(cur_state_value - value, -clip_ratio, clip_ratio)
    v_loss = torch.max(F.mse_loss(cur_state_value, value_target), F.mse_loss(cur_v_clipped, value_target))
    v_loss = (v_loss * valid_mask).mean()

    entropy_loss = (-target_dist.entropy().mean(-1) * valid_mask).mean()
    return v_loss, p_loss, entropy_loss
