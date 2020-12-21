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

        self.action_dim = action_dim = kwargs['action_dim']
        self.hidden_dim = hidden_dim = kwargs['hidden_dim']

        # default convolutional model in rllib
        self.feature_net = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
                                         nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
                                         nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0), nn.ReLU(),
                                         nn.Flatten(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LSTM(512, 512))

        # actor
        self.action_layer = nn.Linear(hidden_dim, action_dim)
        # critic
        self.value_layer = nn.Linear(hidden_dim, 1)

        self.clip_ratio = kwargs['clip_ratio']
        self.tpdv = dict(device=self.device, dtype=torch.float32)
        self.to(self.device)

    def core(self, obs):
        if obs.ndim == 4:
            return self.feature_net(obs)
        else:
            b, t, c, h, w = obs.shape
            return self.feature_net(obs.view(b * t, c, h, w)).reshape(b, t, -1)

    def forward(self, obs):
        x = self.core(obs)
        action_logits = self.action_layer(x)
        value = self.value_layer(x).squeeze(-1)

        action = Categorical(logits=action_logits).sample()
        return action, action_logits, value

    @torch.no_grad()
    def select_action(self, obs):
        assert not self.is_training
        obs = torch.from_numpy(obs)
        results = self(obs)
        return [result.numpy() for result in results]

    def compute_loss(self, obs, action, action_logits, adv, value, value_target):
        action = action.to(torch.long)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # to remove padding values
        valid_mask = (value != 0.0).to(torch.float32)

        _, target_action_logits, cur_state_value = self(obs)

        target_dist = Categorical(logits=target_action_logits)
        target_action_logprobs = target_dist.log_prob(action)
        behavior_action_logprobs = Categorical(logits=action_logits).log_prob(action)

        log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
        rhos = log_rhos.exp()

        p_surr = adv * torch.clamp(rhos, 1 - self.clip_ratio, 1 + self.clip_ratio)
        p_loss = (-torch.min(p_surr, rhos * adv) * valid_mask).mean()

        cur_v_clipped = value + torch.clamp(cur_state_value - value, -self.clip_ratio, self.clip_ratio)
        v_loss = torch.max(F.mse_loss(cur_state_value, value_target), F.mse_loss(cur_v_clipped, value_target))
        v_loss = (v_loss * valid_mask).mean()

        entropy_loss = (-target_dist.entropy() * valid_mask).mean()
        return v_loss, p_loss, entropy_loss

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}