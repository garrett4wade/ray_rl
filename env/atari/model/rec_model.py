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
        self.feature_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 512)
        # actor
        self.action_layer = nn.Linear(hidden_dim, action_dim)
        # critic
        self.value_layer = nn.Linear(hidden_dim, 1)

        self.clip_ratio = kwargs['clip_ratio']
        self.tpdv = dict(device=self.device, dtype=torch.float32)
        self.to(self.device)

    def forward(self, obs, rnn_hidden):
        t, b, c, h, w = obs.shape
        h0, c0 = torch.split(rnn_hidden, self.hidden_dim, -1)
        x = self.feature_net(obs.view(t * b, c, h, w)).reshape(t, b, self.hidden_dim)
        x, (h1, c1) = self.rnn(x, (h0.contiguous(), c0.contiguous()))
        h_out = torch.cat((h1, c1), dim=-1)
        action_logits = self.action_layer(x)
        value = self.value_layer(x)
        return action_logits, value, h_out

    @torch.no_grad()
    def select_action(self, obs, rnn_hidden):
        assert not self.is_training
        obs = torch.from_numpy(obs)
        rnn_hidden = torch.from_numpy(rnn_hidden).transpose(0, 1)
        x = self.feature_net(obs).unsqueeze(0)
        h0, c0 = torch.split(rnn_hidden, self.hidden_dim, -1)
        x, (h1, c1) = self.rnn(x, (h0.contiguous(), c0.contiguous()))
        x = x.squeeze_(0)
        h_out = torch.cat((h1, c1), dim=-1).transpose(0, 1)
        action_logits = self.action_layer(x)
        value = self.value_layer(x)

        action = Categorical(logits=action_logits).sample()
        return action.numpy(), action_logits.numpy(), value.numpy(), h_out.numpy()

    def compute_loss(self, obs, action, action_logits, adv, value, value_target, pad_mask, rnn_hidden):
        action = action.to(torch.long).squeeze(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        target_action_logits, cur_state_value, _ = self(obs, rnn_hidden)

        target_dist = Categorical(logits=target_action_logits)
        target_action_logprobs = target_dist.log_prob(action)
        behavior_action_logprobs = Categorical(logits=action_logits).log_prob(action)

        log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
        rhos = log_rhos.exp().unsqueeze(-1)

        p_surr = adv * torch.clamp(rhos, 1 - self.clip_ratio, 1 + self.clip_ratio)
        p_loss = (-torch.min(p_surr, rhos * adv) * pad_mask).mean()

        cur_v_clipped = value + torch.clamp(cur_state_value - value, -self.clip_ratio, self.clip_ratio)
        v_loss = torch.max(F.mse_loss(cur_state_value, value_target), F.mse_loss(cur_v_clipped, value_target))
        v_loss = (v_loss * pad_mask).mean()

        entropy_loss = (-target_dist.entropy() * pad_mask.squeeze(-1)).mean()
        return v_loss, p_loss, entropy_loss

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}
