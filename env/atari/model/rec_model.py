import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP


class ActorCritic(nn.Module):
    def __init__(self, is_training, config):
        super().__init__()
        self.is_training = is_training
        action_dim = config.action_dim
        hidden_dim = config.hidden_dim

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

    def forward(self, obs, rnn_hidden):
        t, b, c, h, w = obs.shape
        hidden_dim = rnn_hidden.shape[-1] // 2
        h0, c0 = torch.split(rnn_hidden, hidden_dim, -1)
        x = self.feature_net(obs.view(t * b, c, h, w)).reshape(t, b, hidden_dim)
        x, (h1, c1) = self.rnn(x, (h0.contiguous(), c0.contiguous()))
        h_out = torch.cat((h1, c1), dim=-1)
        action_logits = self.action_layer(x)
        value = self.value_layer(x).squeeze(-1)
        return action_logits, value, h_out

    @torch.no_grad()
    def select_action(self, obs, rnn_hidden):
        assert not self.is_training
        obs = torch.from_numpy(obs).to(torch.float32) / 255
        rnn_hidden = torch.from_numpy(rnn_hidden).transpose(0, 1)
        x = self.feature_net(obs).unsqueeze(0)
        h0, c0 = torch.split(rnn_hidden, rnn_hidden.shape[-1] // 2, -1)
        x, (h1, c1) = self.rnn(x, (h0.contiguous(), c0.contiguous()))
        x = x.squeeze_(0)
        h_out = torch.cat((h1, c1), dim=-1).transpose(0, 1)
        action_logits = self.action_layer(x)
        value = self.value_layer(x).squeeze(-1)

        action = Categorical(logits=action_logits).sample()
        return action.numpy(), action_logits.numpy(), value.numpy(), h_out.numpy()


def compute_loss(learner, obs, action, action_logits, adv, value, value_target, pad_mask, rnn_hidden, clip_ratio,
                 world_size, value_clip=True):
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
    obs = obs.to(torch.float32) / 255
    action = action.to(torch.long)

    target_action_logits, cur_state_value, _ = learner(obs, rnn_hidden)

    target_dist = Categorical(logits=target_action_logits)
    target_action_logprobs = target_dist.log_prob(action)
    behavior_action_logprobs = Categorical(logits=action_logits).log_prob(action)

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

    entropy_loss = (-target_dist.entropy() * pad_mask).mean()
    return v_loss, p_loss, entropy_loss
