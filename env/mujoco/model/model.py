import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

# borrowed from soft-actor-critic
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class ActorCritic(nn.Module):
    def __init__(self, is_training, kwargs):
        super().__init__()
        self.is_training = is_training
        if not is_training:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(kwargs['gpu_id'] if torch.cuda.is_available() else 'cpu')

        action_scale = torch.from_numpy(kwargs['action_scale'].copy())
        action_loc = torch.from_numpy(kwargs['action_loc'].copy())
        self.action_scale = nn.Parameter(action_scale).detach().to(self.device)
        self.action_loc = nn.Parameter(action_loc).detach().to(self.device)

        obs_dim = kwargs['obs_dim']
        action_dim = kwargs['action_dim']
        hidden_dim = kwargs['hidden_dim']

        self.action_dim = kwargs['action_dim']
        self.feature_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(normalized_shape=[hidden_dim]),
                                         nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                         nn.LayerNorm(normalized_shape=[hidden_dim]), nn.ReLU())
        # actor
        self.action_layer = nn.Linear(hidden_dim, 2 * action_dim)
        # critic
        self.value_layer = nn.Linear(hidden_dim, 1)

        self.clip_ratio = kwargs['clip_ratio']
        self.tpdv = dict(device=self.device, dtype=torch.float32)
        self.to(self.device)

    def core(self, state):
        return self.feature_net(state)

    def forward(self, state):
        x = self.core(state)
        action_logits = self.action_layer(x)
        value = self.value_layer(x)

        mu, log_std = torch.split(action_logits, self.action_dim, dim=-1)
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

    def compute_loss(self, obs, action, action_logits, adv, value, value_target):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # to remove padding values
        valid_mask = (value != 0.0).to(torch.float32)
        _, target_action_logits, cur_state_value = self(obs)

        target_mu, target_log_std = torch.split(target_action_logits, self.action_dim, dim=-1)
        target_mu = torch.tanh(target_mu) * self.action_scale + self.action_loc
        target_log_std = target_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        target_dist = Normal(target_mu, target_log_std.exp())
        target_action_logprobs = target_dist.log_prob(action).sum(-1, keepdim=True)

        mu, log_std = torch.split(action_logits, self.action_dim, dim=-1)
        mu = torch.tanh(mu) * self.action_scale + self.action_loc
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        behavior_dist = Normal(mu, log_std.exp())
        behavior_action_logprobs = behavior_dist.log_prob(action).sum(-1, keepdim=True)

        log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
        rhos = log_rhos.exp()

        p_surr = adv * torch.clamp(rhos, 1 - self.clip_ratio, 1 + self.clip_ratio)
        p_loss = (-torch.min(p_surr, rhos * adv) * valid_mask).mean()

        cur_v_clipped = value + torch.clamp(cur_state_value - value, -self.clip_ratio, self.clip_ratio)
        v_loss = torch.max(F.mse_loss(cur_state_value, value_target), F.mse_loss(cur_v_clipped, value_target))
        v_loss = (v_loss * valid_mask).mean()

        entropy_loss = (-target_dist.entropy().sum(-1, keepdim=True) * valid_mask).mean()
        return v_loss, p_loss, entropy_loss

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}
