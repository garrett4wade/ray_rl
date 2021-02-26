import torch
import torch.nn as nn
from torch.distributions import Categorical
# from rl_utils.initialization import init
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from copy import deepcopy


class ActorCritic(nn.Module):
    def __init__(self, is_training, config):
        super().__init__()
        self.is_training = is_training
        self.obs_dim = obs_dim = config.obs_dim
        self.state_dim = state_dim = config.state_dim
        self.action_dim = action_dim = config.action_dim
        self.hidden_dim = hidden_dim = config.hidden_dim
        self.actor_rnn_layers = actor_rnn_layers = config.actor_rnn_layers
        self.critic_rnn_layers = critic_rnn_layers = config.critic_rnn_layers

        # actor model
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm([hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm([hidden_dim]),
        )
        self.actor_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=actor_rnn_layers)
        self.actor_gate = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # critic model
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm([hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm([hidden_dim]),
        )
        self.critic_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=critic_rnn_layers)
        self.critic_gate = nn.Linear(hidden_dim, hidden_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # # target critic model
        # # TODO: remain target critic model in actors only, since it will accelerate parameter loading
        # self.tgt_critic_net = deepcopy(self.critic_net)
        # self.tgt_critic_rnn = deepcopy(self.critic_rnn)
        # self.tgt_critic_gate = deepcopy(self.critic_gate)
        # self.tgt_critic_head = deepcopy(self.critic_head)
        # for p in self.tgt_critic_net.parameters():
        #     p.requires_grad = False
        # for p in self.tgt_critic_rnn.parameters():
        #     p.requires_grad = False
        # for p in self.tgt_critic_gate.parameters():
        #     p.requires_grad = False
        # for p in self.tgt_critic_head.parameters():
        #     p.requires_grad = False

        # for c in self.children():
        #     init(c)

    def forward(self, obs, state, actor_rnn_hidden, critic_rnn_hidden):
        """core function invoked during both rollout and backpropagation

        Args:
            obs (PyTorch tensor [chunk_len, batch_size, agent_num, obs_dim]): decentralized agent observation
            state (PyTorch tensor [chunk_len, batch_size, state_dim]): centralzied all agents' state
            actor_rnn_hidden (PyTorch tensor [actor_rnn_layers, batch_size, agent_num, hidden_dim]):
                actor GRU hidden state
            critic_rnn_hidden (PyTorch tensor [critic_rnn_layers, batch_size, hidden_dim]):
                critic GRU hidden state

        Returns:
            actor_feature: input of actor head
            critic_feature: input of critic head
            ahout: actor GRU output hidden state
            chout: critic GRU output hidden state
        """
        chunk_len, bs, agn = obs.shape[:3]
        actor_mlp_out = self.actor_net(obs)
        actor_rnn_out, ahout = self.actor_rnn(
            actor_mlp_out.view(chunk_len, bs * agn, self.hidden_dim),
            actor_rnn_hidden.reshape(self.actor_rnn_layers, bs * agn, self.hidden_dim))
        actor_rnn_out = actor_rnn_out.view(chunk_len, bs, agn, self.hidden_dim)
        ahout = ahout.view(self.actor_rnn_layers, bs, agn, self.hidden_dim)
        actor_feature = actor_mlp_out + torch.sigmoid(self.actor_gate(actor_mlp_out)) * actor_rnn_out
        logits = self.actor_head(actor_feature)

        # if use_target:
        #     with torch.no_grad():
        #         critic_mlp_out = self.tgt_critic_net(state)
        #         critic_rnn_out, chout = self.tgt_critic_rnn(critic_mlp_out, critic_rnn_hidden)
        #         critic_feature = critic_mlp_out + torch.sigmoid(self.tgt_critic_gate(critic_mlp_out)) * critic_rnn_out
        #         value = self.tgt_critic_head(critic_feature).squeeze(-1)
        # else:
        critic_mlp_out = self.critic_net(state)
        critic_rnn_out, chout = self.critic_rnn(critic_mlp_out, critic_rnn_hidden)
        critic_feature = critic_mlp_out + torch.sigmoid(self.critic_gate(critic_mlp_out)) * critic_rnn_out
        value = self.critic_head(critic_feature).squeeze(-1)

        return logits, value, ahout, chout

    @torch.no_grad()
    def select_action(self, obs, state, avail_action, actor_rnn_hidden, critic_rnn_hidden):
        """method invoked during rollout only

        Args:
            obs (numpy array [env_num_per_worker, agent_num, obs_dim]):
                decentralized agent observation
            state (numpy array [env_num_per_worker, state_dim]):
                centralized all agents' state
            avail_action (numpy array [env_num_per_worker, agent_dim, action_dim]):
                current available action
            actor_rnn_hidden (numpy array [env_num_per_worker, actor_rnn_layers, agent_dim, hidden_dim]):
                actor rnn hidden state
            critic_rnn_hidden (numpy array [env_num_per_worker, critic_rnn_layers, hidden_dim]):
                critic rnn hidden state

        Returns:
            action: sampled action
            action_logits: action head output
            value: current state value
            ahout: actor GRU output hidden state
            chout: critic GRU output hidden state
        """
        assert not self.is_training
        obs = torch.from_numpy(obs).unsqueeze(0)
        state = torch.from_numpy(state).unsqueeze(0)
        avail_action = torch.from_numpy(avail_action)
        actor_rnn_hidden = torch.from_numpy(actor_rnn_hidden).transpose(0, 1)
        critic_rnn_hidden = torch.from_numpy(critic_rnn_hidden).transpose(0, 1)

        logits, value, ahout, chout = self(obs, state, actor_rnn_hidden, critic_rnn_hidden)
        logits, value = logits.squeeze_(0), value.squeeze_(0)
        logits[avail_action == 0.0] = -1e10
        action = Categorical(logits=logits).sample()
        ahout, chout = ahout.transpose_(0, 1), chout.transpose_(0, 1)
        return action.numpy(), logits.numpy(), value.numpy(), ahout.numpy(), chout.numpy()


def compute_loss(learner,
                 obs,
                 state,
                 action,
                 action_logits,
                 avail_action,
                 adv,
                 value,
                 value_target,
                 actor_rnn_hidden,
                 critic_rnn_hidden,
                 pad_mask,
                 live_mask,
                 clip_ratio,
                 world_size,
                 value_clip=False,
                 adv_norm=False):
    if adv_norm:
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
        # TODO: performance better with no advantage normalization
        adv = (adv - adv_mean) / (adv_std + 1e-8)
    action = action.to(torch.long)

    target_action_logits, cur_state_value, _, _ = learner(obs, state, actor_rnn_hidden, critic_rnn_hidden)
    target_action_logits[avail_action == 0.0] = -1e10
    action_logits[avail_action == 0.0] = -1e10

    target_dist = Categorical(logits=target_action_logits)
    target_action_logprobs = target_dist.log_prob(action)
    behavior_action_logprobs = Categorical(logits=action_logits).log_prob(action)

    log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
    rhos = log_rhos.exp() * live_mask

    p_surr = adv * (torch.clamp(rhos, 1 - clip_ratio, 1 + clip_ratio) * live_mask).sum(-1)
    p_loss = (-torch.min(p_surr, rhos.sum(-1) * adv)).sum() / live_mask.sum()

    if value_clip:
        cur_v_clipped = value + torch.clamp(cur_state_value - value, -clip_ratio, clip_ratio)
        v_loss = torch.max((cur_state_value - value_target)**2, (cur_v_clipped - value_target)**2)
    else:
        v_loss = (cur_state_value - value_target)**2
    v_loss = (v_loss * pad_mask).sum() / pad_mask.sum()

    entropy_loss = (-target_dist.entropy() * live_mask).sum() / live_mask.sum()
    return v_loss, p_loss, entropy_loss
