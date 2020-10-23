import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from module.vtrace import from_importance_weights as vtrace_from_importance_weights
from module.gae import from_rewards as gae_from_rewards

# borrowed from soft-actor-critic
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class ContinuousActorCritic(nn.Module):
    def __init__(self, is_training, kwargs):
        super().__init__()
        self.is_training = is_training
        if not is_training:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        action_scale = torch.from_numpy(kwargs['action_scale'].copy())
        action_loc = torch.from_numpy(kwargs['action_loc'].copy())
        self.action_scale = nn.Parameter(action_scale).detach().to(self.device)
        self.action_scale.requires_grad_(False)
        self.action_loc = nn.Parameter(action_loc).detach().to(self.device)
        self.action_loc.requires_grad_(False)

        state_dim = kwargs['state_dim']
        action_dim = kwargs['action_dim']
        hidden_dim = kwargs['hidden_dim']

        self.action_dim = kwargs['action_dim']
        self.feature_net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(normalized_shape=[hidden_dim]),
                                         nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                         nn.LayerNorm(normalized_shape=[hidden_dim]), nn.ReLU())
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # actor
        self.action_layer = nn.Linear(hidden_dim, 2 * action_dim)

        # critic
        self.value_layer = nn.Linear(hidden_dim, 1)

        self.gamma = kwargs['gamma']
        self.lamb = kwargs['lamb']
        self.clip_ratio = kwargs['clip_ratio']

        self.to(self.device)

    def core(self, state, hidden_state):
        x = self.feature_net(state)
        feature, hidden_state_out = self.rnn(x, hidden_state)
        return feature, hidden_state_out

    def forward(self, state, hidden_state):
        x, hidden_state_out = self.core(state, hidden_state)
        action_logits = self.action_layer(x)
        value = self.value_layer(x).squeeze(-1)

        mu, log_std = torch.split(action_logits, self.action_dim, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        mu = torch.tanh(mu) * self.action_scale + self.action_loc
        dist = Normal(loc=mu, scale=log_std.exp())

        action = dist.sample()

        return (action, action_logits, value, hidden_state_out)

    def compute_policy_phase_loss(self,
                                  state,
                                  hidden_state,
                                  action,
                                  reward,
                                  done_mask,
                                  action_logits,
                                  value,
                                  use_vtrace=False):
        # convert remaining input to PyTorch tensors
        inputs = [action, reward, action_logits, value, done_mask]
        for i, item in enumerate(inputs):
            if not torch.is_tensor(item):
                item = torch.from_numpy(item)
            inputs[i] = item.to(device=self.device, dtype=torch.float)
        (action, reward, action_logits, value, done_mask) = inputs

        # all following code is for loss computation
        x, _ = self.core(state, hidden_state)
        target_action_logits = self.action_layer(x)
        cur_state_value = self.value_layer(x.detach()).squeeze(-1)
        '''
            NOTE done_mask can remove values calculated from padding 0
        '''
        cur_state_value = cur_state_value * done_mask
        value = value * done_mask

        target_mu, target_log_std = torch.split(target_action_logits, self.action_dim, dim=-1)
        target_mu = torch.tanh(target_mu) * self.action_scale + self.action_loc
        target_log_std = target_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        target_dist = Normal(target_mu, target_log_std.exp())
        target_action_logprobs = target_dist.log_prob(action).sum(-1)

        mu, log_std = torch.split(action_logits, self.action_dim, dim=-1)
        mu = torch.tanh(mu) * self.action_scale + self.action_loc
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        behavior_dist = Normal(mu, log_std.exp())
        behavior_action_logprobs = behavior_dist.log_prob(action).sum(-1)

        log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
        rhos = log_rhos.exp()
        '''
            NOTE input padding 0 from sequence end doesn't affect vtrace
        '''
        if use_vtrace:
            ''' vtrace '''
            vtrace_returns = vtrace_from_importance_weights(log_rhos=log_rhos.detach()[:, :-1],
                                                            discounts=self.gamma *
                                                            torch.ones_like(reward[:, :-1], dtype=torch.float),
                                                            rewards=reward[:, :-1],
                                                            values=cur_state_value[:, :-1],
                                                            bootstrap_value=cur_state_value[:, -1])
            self.vtrace_returns = vtrace_returns

            adv = vtrace_returns.advantages
            v_target = vtrace_returns.vs
        else:
            ''' gae '''
            gae_return = gae_from_rewards(reward[:, :-1], cur_state_value[:, :-1], cur_state_value[:, :-1],
                                          cur_state_value[:, -1], self.gamma, self.lamb)
            adv = gae_return.advantages
            v_target = gae_return.discounted_returns

        p_surr1 = adv * torch.clamp(rhos[:, :-1], 1 - self.clip_ratio, 1 + self.clip_ratio)
        p_surr2 = rhos[:, :-1] * adv
        p_loss = -torch.min(p_surr1, p_surr2)
        p_loss = (p_loss * done_mask[:, :-1]).mean()

        cur_v_clipped = value[:, :-1] + torch.clamp(cur_state_value[:, :-1] - value[:, :-1], -self.clip_ratio,
                                                    self.clip_ratio)
        v_surr1 = ((v_target - cur_state_value[:, :-1])**2)
        v_surr2 = (v_target - cur_v_clipped)**2
        v_loss = .5 * torch.max(v_surr1, v_surr2)
        v_loss = (v_loss * done_mask[:, :-1]).mean()

        entropy_loss = -target_dist.entropy()
        entropy_loss = entropy_loss.sum(-1)
        entropy_loss = (entropy_loss[:, :-1] * done_mask[:, :-1]).mean()
        return v_loss, p_loss, entropy_loss

    def compute_aux_phase_loss(self, state, hidden_state, action_logits, value):
        # convert inputs to PyTorch tensors
        inputs = [action_logits, value]
        for i, item in enumerate(inputs):
            if not torch.is_tensor(item):
                item = torch.from_numpy(item)
            inputs[i] = item.to(device=self.device, dtype=torch.float)
        action_logits, value = inputs

        (_, new_action_logits, cur_state_value, _) = self(state, hidden_state)

        new_mu, new_log_std = torch.split(new_action_logits, self.action_dim, dim=-1)
        new_log_std = new_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        new_mu = new_mu.tanh() * self.action_scale + self.action_loc

        old_mu, old_log_std = torch.split(action_logits, self.action_dim, dim=-1)
        old_log_std = old_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        old_mu = old_mu.tanh() * self.action_scale + self.action_loc

        # KL divergence for gaussian distributions
        p_loss = 2 * (new_log_std - old_log_std) + (2 * (old_log_std - new_log_std)).exp() + (
            (old_mu - new_mu) / new_log_std.exp())**2 - 1
        p_loss = p_loss.sum(-1).mean()
        assert not torch.any(torch.isnan(p_loss)), 'p_loss has nan!'

        v_loss = F.mse_loss(cur_state_value[:, :-1], value)
        return p_loss, v_loss

    def compute_value_target(self,
                             state,
                             hidden_state,
                             action,
                             reward,
                             done_mask,
                             action_logits,
                             value,
                             use_vtrace=False):
        # convert remaining input to PyTorch tensors
        inputs = [action, reward, action_logits, value, done_mask]
        for i, item in enumerate(inputs):
            if not torch.is_tensor(item):
                item = torch.from_numpy(item)
            inputs[i] = item.to(device=self.device, dtype=torch.float)
        (action, reward, action_logits, value, done_mask) = inputs

        # all following code is for loss computation
        x, _ = self.core(state, hidden_state)
        cur_state_value = self.value_layer(x).squeeze(-1)
        '''
            NOTE done_mask can remove values calculated from padding 0
        '''
        cur_state_value = cur_state_value * done_mask
        value = value * done_mask
        if use_vtrace:
            target_action_logits = self.action_layer(x)
            target_mu, target_log_std = torch.split(target_action_logits, self.action_dim, dim=-1)
            target_mu = torch.tanh(target_mu) * self.action_scale + self.action_loc
            target_log_std = target_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
            target_dist = Normal(target_mu, target_log_std.exp())
            target_action_logprobs = target_dist.log_prob(action).sum(-1)

            mu, log_std = torch.split(action_logits, self.action_dim, dim=-1)
            mu = torch.tanh(mu) * self.action_scale + self.action_loc
            log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
            behavior_dist = Normal(mu, log_std.exp())
            behavior_action_logprobs = behavior_dist.log_prob(action).sum(-1)

            log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
            '''
                NOTE input padding 0 from sequence end doesn't affect vtrace
            '''
            ''' vtrace '''
            vtrace_returns = vtrace_from_importance_weights(log_rhos=log_rhos.detach()[:, :-1],
                                                            discounts=self.gamma *
                                                            torch.ones_like(reward[:, :-1], dtype=torch.float),
                                                            rewards=reward[:, :-1],
                                                            values=cur_state_value[:, :-1],
                                                            bootstrap_value=cur_state_value[:, -1])
            v_target = vtrace_returns.vs
        else:
            ''' gae '''
            gae_return = gae_from_rewards(reward[:, :-1], cur_state_value[:, :-1], cur_state_value[:, :-1],
                                          cur_state_value[:, -1], self.gamma, self.lamb)
            v_target = gae_return.discounted_returns
        return v_target.detach().cpu().numpy()

    def step(self,
             state,
             hidden_state,
             phase='act',
             action=None,
             reward=None,
             done_mask=None,
             action_logits=None,
             value=None):
        '''
            the only function needs to be invoked for all workers and learners

            There may be 3 scenarios:
            1. For workers, invoke model.step(s, h) to get action,
               and store (logprobs, state_values, hidden_states)
            2. For leanrer burn-in, feed hidden_state to get hidden_state_out
               (similar as R2D2)
            3. For learner loss computation, need to in addition feed
               (action, reward, done_mask, action_logits, value)

            Shapes of inputs in each scenario (except hidden_state):
            1. env simulation with shape (env_num, 1, *)
            2. burn-in with shape (batch_size, burn_in_len, *)
            3. loss computation with shape (batch_size, chunk_len + 1, *)

            NOTE: hidden_state with shape (1, batch_size, hidden_size)
            NOTE: loss computation has 1 more time length for value bootstrap
        '''
        assert phase in ['act', 'burn_in', 'policy', 'value', 'action_logits', 'aux']
        # convert input to PyTorch tensors
        inputs = [state, hidden_state]
        for i, item in enumerate(inputs):
            if not torch.is_tensor(item):
                item = torch.from_numpy(item)
            inputs[i] = item.to(device=self.device, dtype=torch.float)
        state, hidden_state = inputs

        # when not training, just rollout action
        if phase == 'act':
            with torch.no_grad():
                results = self(state, hidden_state)
                return [result.detach().cpu().numpy() for result in results]
        # when training, first burn-in to get hidden state
        elif phase == 'burn_in':
            with torch.no_grad():
                return self(state, hidden_state)[-1].detach()
        elif phase == 'policy':
            return self.compute_policy_phase_loss(state, hidden_state, action, reward, done_mask, action_logits, value)
        elif phase == 'value':
            with torch.no_grad():
                return self.compute_value_target(state, hidden_state, action, reward, done_mask, action_logits, value)
        elif phase == 'action_logits':
            with torch.no_grad():
                feature, _ = self.core(state, hidden_state)
                return self.action_layer(feature).detach()
        elif phase == 'aux':
            return self.compute_aux_phase_loss(state, hidden_state, action_logits, value)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}
