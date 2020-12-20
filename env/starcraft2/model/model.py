import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, is_training, kwargs):
        super().__init__()
        self.is_training = is_training
        if not is_training:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(kwargs['gpu_id'] if torch.cuda.is_available() else 'cpu')

        self.obs_dim = obs_dim = kwargs['obs_dim']
        self.state_dim = state_dim = kwargs['state_dim']
        self.action_dim = action_dim = kwargs['action_dim']
        self.hidden_dim = hidden_dim = kwargs['hidden_dim']

        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.clip_ratio = kwargs['clip_ratio']
        self.tpdv = dict(device=self.device, dtype=torch.float32)
        self.to(self.device)

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

    def compute_loss(self, obs, state, action, action_logits, avail_action, adv, value, value_target):
        action = action.to(torch.long)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # to remove padding values
        valid_mask = (value != 0.0).to(torch.float32)

        target_action_logits, cur_state_value = self(obs, state)
        target_action_logits[avail_action == 0.0] = -1e10

        target_dist = Categorical(logits=target_action_logits)
        target_action_logprobs = target_dist.log_prob(action)
        behavior_action_logprobs = Categorical(logits=action_logits).log_prob(action)

        log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
        # taking average along agent dim
        rhos = log_rhos.exp().mean(-1)

        p_surr = adv * torch.clamp(rhos, 1 - self.clip_ratio, 1 + self.clip_ratio)
        p_loss = (-torch.min(p_surr, rhos * adv) * valid_mask).mean()

        cur_v_clipped = value + torch.clamp(cur_state_value - value, -self.clip_ratio, self.clip_ratio)
        v_loss = torch.max(F.mse_loss(cur_state_value, value_target), F.mse_loss(cur_v_clipped, value_target))
        v_loss = (v_loss * valid_mask).mean()

        entropy_loss = (-target_dist.entropy().mean(-1) * valid_mask).mean()
        return v_loss, p_loss, entropy_loss

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}
