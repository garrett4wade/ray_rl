import torch.nn as nn
import torch

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
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.action_dim = action_dim = kwargs['action_dim']
        self.hidden_dim = hidden_dim = kwargs['hidden_dim']
        
        # default convolutional model in rllib
        self.feature_net = nn.Sequential(nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0), nn.ReLU(),
                                         ConvMaxpoolResModule(8), ConvMaxpoolResModule(16), 
                                         ConvMaxpoolResModule(32), nn.ReLU(), nn.Flatten())

        # actor
        self.action_layer = nn.Linear(hidden_dim, action_dim)

        # critic
        self.value_layer = nn.Linear(hidden_dim, 1)

        self.gamma = kwargs['gamma']
        self.lamb = kwargs['lamb']
        self.clip_ratio = kwargs['clip_ratio']

        self.use_vtrace = kwargs['use_vtrace']

        self.to(self.device)

    def core(self, state):
        assert torch.all((state>=-1).logical_and(state<=1))
        if not self.is_training:
            assert state.ndim == 4
            return self.feature_net(state)
        else:
            assert state.ndim == 5
            batch_size, t_dim = state.shape[:2]
            return self.feature_net(state.view(batch_size * t_dim, *state.shape[2:])).reshape(batch_size, t_dim, self.hidden_dim)

    def forward(self, state):
        x = self.core(state)
        action_logits = self.action_layer(x)
        value = self.value_layer(x).squeeze(-1)

        dist = Categorical(logits=action_logits)

        action = dist.sample()

        return action, action_logits, value

    def step(self,
             state,
             action=None,
             reward=None,
             done_mask=None,
             action_logits=None,
             value=None):
        # convert input to PyTorch tensors
        if not torch.is_tensor(state):
            state = torch.from_numpy(state)
        state = state.to(device=self.device, dtype=torch.float)

        if not self.is_training:
            with torch.no_grad():
                results = self(state)
                return [result.detach().cpu().numpy() for result in results]

        # convert remaining input to PyTorch tensors
        inputs = [action, reward, action_logits, value, done_mask]
        for i, item in enumerate(inputs):
            if not torch.is_tensor(item):
                item = torch.from_numpy(item)
            inputs[i] = item.to(device=self.device, dtype=torch.float)
        (action, reward, action_logits, value, done_mask) = inputs
        action = action.to(torch.long)

        # all following code is for loss computation
        _, target_action_logits, cur_state_value = self(state)
        '''
            NOTE done_mask can remove values calculated from padding 0
        '''
        cur_state_value = cur_state_value * done_mask
        value = value * done_mask

        target_dist = Categorical(logits=target_action_logits)
        target_action_logprobs = target_dist.log_prob(action)
        behavior_dist = Categorical(logits=action_logits)
        behavior_action_logprobs = behavior_dist.log_prob(action)

        log_rhos = target_action_logprobs - behavior_action_logprobs.detach()
        rhos = log_rhos.exp()
        '''
            NOTE input padding 0 from sequence end doesn't affect vtrace
        '''
        if self.use_vtrace:
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
        v_loss = torch.max(v_surr1, v_surr2)
        v_loss = (v_loss * done_mask[:, :-1]).mean()

        entropy_loss = -target_dist.entropy()
        entropy_loss = (entropy_loss[:, :-1] * done_mask[:, :-1]).mean()
        return v_loss, p_loss, entropy_loss

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}
