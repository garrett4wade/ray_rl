import os
import gym
import ray
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
from ps import ParameterServer, ReturnRecorder
from simulation_thread import SimulationThread

# global configuration
parser = argparse.ArgumentParser(description='run asynchronous PPO')
parser.add_argument("--exp_name", type=str, default='ray_test0', help='experiment name')

# environment
parser.add_argument('--env_name', type=str, default='Humanoid-v2', help='name of env')
parser.add_argument('--env_num', type=int, default=2, help='number of evironments per worker')
parser.add_argument('--total_frames', type=int, default=int(2e6), help='optimization batch size')

# important parameters of model and algorithm
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden layer size of mlp & gru')
parser.add_argument('--batch_size', type=int, default=512, help='optimization batch size')
parser.add_argument('--n_epoch', type=int, default=6, help='update epoches in each training step')
parser.add_argument('--n_mini_batch', type=int, default=4, help='number of minibatches in 1 training epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--entropy_coef', type=float, default=.01, help='entropy loss coefficient')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lamb', type=float, default=0.97, help='gae discount factor')
parser.add_argument('--clip_ratio', type=float, default=0.2, help='ppo clip ratio')
parser.add_argument('--max_grad_norm', type=float, default=50.0, help='maximum gradient norm')

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


def build_simple_env(kwargs):
    return gym.make(kwargs['env_name'])


# get state/action information from env
init_env = build_simple_env(kwargs)
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


def build_worker_env(kwargs):
    return Envs(build_simple_env, kwargs)


def build_worker_model(kwargs):
    # model for interacting with env, does not need gradient update
    # instead, download parameters from parameter server (ps)
    if kwargs['continuous_env']:
        return ContinuousActorCritic(False, kwargs)
    else:
        return DiscreteActorCritic(False, kwargs)


def build_learner_model(kwargs):
    # model for gradient update
    # after update, upload parameters to parameter server
    if kwargs['continuous_env']:
        return ContinuousActorCritic(True, kwargs)
    else:
        return DiscreteActorCritic(True, kwargs)


if __name__ == "__main__":
    exp_start_time = time.time()

    # set random seed
    torch.manual_seed(kwargs['seed'])
    np.random.seed(kwargs['seed'] + 13563)

    # initialize ray
    ray.init()

    # initialized TensorBoard summary writer
    writer = SummaryWriter(log_dir=os.path.join('./runs', kwargs['exp_name']), comment='Humanoid-v2')

    # initialize learner, who is responsible for gradient update
    learner = build_learner_model(kwargs)
    optimizer = optim.Adam(learner.parameters(), lr=kwargs['lr'])
    init_weights = learner.get_weights()

    # initialize buffer
    keys = ['state', 'action', 'action_logits', 'value', 'reward', 'hidden_state', 'done_mask']
    buffer = ReplayQueue(kwargs['batch_size'] * kwargs['q_size'], keys)

    # initialize workers, who are responsible for interacting with env (simulation)
    ps = ParameterServer.remote(weights=init_weights)
    recorder = ReturnRecorder.remote()
    simulation_thread = SimulationThread(model_fn=build_worker_model,
                                         env_fn=build_worker_env,
                                         ps=ps,
                                         recorder=recorder,
                                         global_queue=buffer,
                                         kwargs=kwargs)
    # after starting simulation thread, workers asynchronously interact with
    # environments and send data into buffer via Ray backbone
    simulation_thread.start()

    # main loop
    global_step = 0
    num_frames = 0
    while num_frames < kwargs['total_frames']:
        '''
            sample from buffer
        '''
        iter_start = time.time()
        # wait until there's enough data in buffer
        data_batch = buffer.get(kwargs['batch_size'])
        while data_batch is None:
            data_batch = buffer.get(kwargs['batch_size'])
        sample_time = time.time() - iter_start
        '''
            split data for burn-in and backpropogation
        '''
        pre, post = dict(), dict()
        for k, v in data_batch.items():
            if k == 'hidden_state':
                pre[k] = v
            else:
                pre[k], post[k] = np.split(v, [kwargs['burn_in_len']], axis=1)

        # pull from remote return recorder
        return_stat_job = recorder.pull.remote()
        num_frames += np.sum(post['done_mask'])
        global_step += 1
        '''
            update !
        '''
        start = time.time()
        # burn-in stage: compute RNN hidden state for loss computation
        state, hidden_state = pre['state'], pre['hidden_state']
        post['hidden_state'] = learner.step(state, hidden_state, burn_in=True)
        # training loop: compute loss and backpropogate gradient update
        loss_record = dict(p_loss=[], v_loss=[], entropy_loss=[], grad_norm=[])
        for _ in range(kwargs['n_epoch']):
            minibatch_idxes = torch.split(torch.randperm(kwargs['batch_size']),
                                          kwargs['batch_size'] // kwargs['n_mini_batch'])
            for idx in minibatch_idxes:
                optimizer.zero_grad()
                v_loss, p_loss, entropy_loss = learner.step(
                    **{k: v[idx] if k != 'hidden_state' else v[:, idx]
                       for k, v in post.items()})
                loss = p_loss + v_loss
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(learner.parameters(), kwargs['max_grad_norm'])
                optimizer.step()

                # record loss
                loss_record['p_loss'].append(p_loss.item())
                loss_record['v_loss'].append(v_loss.item())
                loss_record['entropy_loss'].append(entropy_loss.item())
                loss_record['grad_norm'].append(grad_norm.item())
        optimize_time = time.time() - start

        # upload updated learner parameter to parameter server
        if global_step % kwargs['push_period'] == 0:
            # TODO: this line will stall to wait upload job finishing, can be optimize?
            ray.get(ps.set_weights.remote(learner.get_weights()))

        # write summary into TensorBoard
        return_stat = ray.get(return_stat_job)
        for k, v in return_stat.items():
            writer.add_scalar('ep_return/' + k, v, global_step=num_frames, walltime=time.time() - exp_start_time)
        writer.add_scalar('loss/p_loss', np.mean(loss_record['p_loss']), global_step=global_step)
        writer.add_scalar('loss/v_loss', np.mean(loss_record['v_loss']), global_step=global_step)
        writer.add_scalar('loss/entropy_loss', np.mean(loss_record['entropy_loss']), global_step=global_step)
        writer.add_scalar('grad_norm', np.mean(loss_record['grad_norm']), global_step=global_step)

        dur = time.time() - iter_start
        print(("Global Step: {}, Frames: {}, " + "Sample Time: {:.2f}s, Optimization Time: {:.2f}s, " +
               "Iteration Step Time: {:.2f}s").format(global_step, num_frames, sample_time, optimize_time, dur))
    print("############ prepare to shut down ray ############")
    ray.shutdown()
    writer.close()
    print("Experiment Time Consume: {}".format(time.time() - exp_start_time))
    print("############ experiment finished ############")
