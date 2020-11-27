import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from module.vtrace import from_importance_weights as vtrace_from_importance_weights
from module.gae import from_rewards as gae_from_rewards
from module.conv import ConvMaxpoolResModule


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
        self.feature_net = nn.Sequential(nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0), nn.ReLU(),
                                         ConvMaxpoolResModule(8), ConvMaxpoolResModule(16), ConvMaxpoolResModule(32),
                                         nn.ReLU(), nn.Flatten())

        # actor
        self.action_layer = nn.Linear(hidden_dim, action_dim)

        # critic
        self.value_layer = nn.Linear(hidden_dim, 1)

        self.clip_ratio = kwargs['clip_ratio']
        self.tpdv = dict(device=self.device, dtype=torch.float32)
        self.to(self.device)

    def core(self, state):
        assert torch.all((state >= -1).logical_and(state <= 1))
        return self.feature_net(state)

    def forward(self, state):
        x = self.core(state)
        action_logits = self.action_layer(x)
        value = self.value_layer(x).squeeze(-1)

        dist = Categorical(logits=action_logits)
        action = dist.sample()

        return action, action_logits, value

    @torch.no_grad()
    def value(self, obs):
        assert not self.is_training
        obs = torch.from_numpy(obs).to(**self.tpdv).unsqueeze(0)
        return self.value_layer(self.core(obs)).cpu().item()

    @torch.no_grad()
    def select_action(self, obs):
        assert not self.is_training
        obs = torch.from_numpy(obs).to(**self.tpdv)
        results = self(obs)
        return [result.detach().cpu().numpy() for result in results]

    def compute_loss(self, obs, action, action_logits, adv, value, value_target):
        action = action.to(torch.long)

        # all following code is for loss computation
        _, target_action_logits, cur_state_value = self(obs)

        target_dist = Categorical(logits=target_action_logits)
        target_action_logprobs = target_dist.log_prob(action)
        behavior_action_logprobs = Categorical(logits=action_logits).log_prob(action)

        log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
        rhos = log_rhos.exp()

        p_surr = adv * torch.clamp(rhos, 1 - self.clip_ratio, 1 + self.clip_ratio)
        p_loss = -torch.min(p_surr, rhos * adv).mean()

        cur_v_clipped = value + torch.clamp(cur_state_value - value, -self.clip_ratio, self.clip_ratio)
        v_loss = torch.max(F.mse_loss(cur_state_value, value_target), F.mse_loss(cur_v_clipped, value_target))
        v_loss = v_loss.mean()

        entropy_loss = -target_dist.entropy().mean()
        return v_loss, p_loss, entropy_loss

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}
