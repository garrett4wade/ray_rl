import os
import gym
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.optim as optim

from env import Envs
from buffer import ReplayQueue
from continuous_model import ContinuousActorCritic
from discrete_model import DiscreteActorCritic
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from torch.utils.tensorboard import SummaryWriter

# global configuration
parser = argparse.ArgumentParser(description='run asynchronous PPO')
parser.add_argument("--exp_name", type=str, default='ray_test0', help='experiment name')

# environment
parser.add_argument('--env_name', type=str, default='Humanoid-v2', help='name of env')
parser.add_argument('--env_num', type=int, default=4, help='number of evironments per worker')
parser.add_argument('--total_frames', type=int, default=int(1.2e6), help='optimization batch size')

# important parameters of model and algorithm
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden layer size of mlp & gru')
parser.add_argument('--batch_size', type=int, default=512, help='optimization batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--entropy_coef', type=float, default=.01, help='entropy loss coefficient')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lamb', type=float, default=0.97, help='gae discount factor')
parser.add_argument('--clip_ratio', type=float, default=0.2, help='ppo clip ratio')
parser.add_argument('--update_every', type=int, default=6, help='update times in each training step')
parser.add_argument('--max_grad_norm', type=float, default=40.0, help='maximum gradient norm')

# recurrent model parameters
parser.add_argument('--burn_in_len', type=int, default=20, help='rnn hidden state burn-in length')
parser.add_argument('--chunk_len', type=int, default=20, help='rnn BPTT chunk length')
parser.add_argument('--replay', type=int, default=2, help='sequence cross-replay times')
parser.add_argument('--max_timesteps', type=int, default=1000, help='episode maximum timesteps')
parser.add_argument('--min_return_chunk_num', type=int, default=5, help='minimal chunk number before env.collect')

# Ray distributed training parameters
parser.add_argument('--push_period', type=int, default=1, help='learner parameter upload period')
parser.add_argument('--load_period', type=int, default=50, help='load period from parameter server')
parser.add_argument('--num_workers', type=int, default=6, help='remote worker numbers')
parser.add_argument('--num_returns', type=int, default=2, help='number of returns in ray.wait')
parser.add_argument('--cpu_per_worker', type=int, default=1, help='cpu used per worker')
parser.add_argument('--q_size', type=int, default=4, help='number of batches in replay buffer')

# random seed
parser.add_argument('--seed', type=int, default=0, help='random seed')

kwargs = vars(parser.parse_args())


def build_env(kwargs):
    return gym.make(kwargs['env_name'])


# get state/action information from env
init_env = build_env(kwargs)
kwargs['state_dim'] = init_env.observation_space.shape[0]
if isinstance(init_env.action_space, Discrete):
    kwargs['action_dim'] = init_env.action_space.n
    kwargs['continuous_env'] = False
elif isinstance(init_env.action_space, Box):
    kwargs['action_dim'] = init_env.action_space.shape[0]
    a = init_env.action_space.high
    b = init_env.action_space.low
    kwargs['action_loc'] = (a + b) / 2
    kwargs['action_scale'] = (b - a) / 2
    kwargs['continuous_env'] = True
del init_env

if __name__ == "__main__":
    exp_start_time = time.time()

    # set random seed
    torch.manual_seed(kwargs['seed'])
    np.random.seed(kwargs['seed'] + 13563)

    # initialized TensorBoard summary writer
    writer = SummaryWriter(log_dir=os.path.join('./runs', kwargs['exp_name']), comment='Humanoid-v2')

    # initialize wrapped env & model
    envs = Envs(build_env, kwargs)
    if kwargs['continuous_env']:
        model = ContinuousActorCritic(True, kwargs)
    else:
        model = DiscreteActorCritic(True, kwargs)
    optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])

    # initialize buffer
    keys = ['state', 'action', 'action_logits', 'value', 'reward', 'hidden_state', 'done_mask']
    buffer = ReplayQueue(kwargs['batch_size'] * kwargs['q_size'], keys)

    # main loop
    global_step = 0
    num_frames = 0
    while num_frames < kwargs['total_frames']:
        global_step += 1
        model.is_training = False
        # interact with env and collect data into buffer
        iter_start = time.time()
        ep_return_records = []
        while buffer.size() <= kwargs['batch_size']:
            segs, ep_returns = envs.step(model)
            if len(segs) > 0:
                for seg in segs:
                    buffer.put(seg)
                ep_return_records += ep_returns

        # sample from buffer
        data_batch = buffer.get(kwargs['batch_size'])
        sample_time = time.time() - iter_start

        # split data for burn-in and loss computation respectively
        model.is_training = True
        pre, post = dict(), dict()
        for k, v in data_batch.items():
            if k == 'hidden_state':
                pre[k] = v
            else:
                pre[k], post[k] = np.split(v, [kwargs['burn_in_len']], axis=1)

        num_frames += np.sum(post['done_mask'])
        start = time.time()
        # burn-in stage: compute RNN hidden state for loss computation
        state, hidden_state = pre['state'], pre['hidden_state']
        post['hidden_state'] = model.step(state, hidden_state, burn_in=True)

        # training loop: compute loss and backpropogate gradient update
        loss_record = dict(p_loss=[], v_loss=[], entropy_loss=[])
        for _ in range(kwargs['update_every']):
            optimizer.zero_grad()
            v_loss, p_loss, entropy_loss = model.step(**post)
            loss = p_loss + v_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), kwargs['max_grad_norm'])
            optimizer.step()

            # record loss
            loss_record['p_loss'].append(p_loss.item())
            loss_record['v_loss'].append(v_loss.item())
            loss_record['entropy_loss'].append(entropy_loss.item())
        optimize_time = time.time() - start

        # write summary into TensorBoard
        if len(ep_return_records) > 0:
            return_stat = dict(max=np.max(ep_return_records),
                               min=np.min(ep_return_records),
                               avg=np.mean(ep_return_records))
            for k, v in return_stat.items():
                writer.add_scalar('ep_return/' + k, v, global_step=num_frames, walltime=time.time() - exp_start_time)
        writer.add_scalar('loss/p_loss', np.mean(loss_record['p_loss']), global_step=global_step)
        writer.add_scalar('loss/v_loss', np.mean(loss_record['v_loss']), global_step=global_step)
        writer.add_scalar('loss/entropy_loss', np.mean(loss_record['entropy_loss']), global_step=global_step)

        dur = time.time() - iter_start
        print(("Global Step: {}, Frames: {}, Sample Time: {:.2f}s, Optimization Time: {:.2f}s, " +
               "Iteration Step Time: {:.2f}s").format(global_step, num_frames, sample_time, optimize_time, dur))
    writer.close()
    print("############ experiment finished ############")
