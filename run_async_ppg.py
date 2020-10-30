import os
import gym
import ray
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.optim as optim

from rl_utils.env import Envs
from rl_utils.buffer import ReplayQueue, AuxReplayQueue
from model.ppg_continuous_model import ContinuousActorCritic
from model.ppg_discrete_model import DiscreteActorCritic
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from torch.utils.tensorboard import SummaryWriter
from ray_utils.ps import ParameterServer, ReturnRecorder
from ray_utils.simulation_thread import SimulationThread

# global configuration
parser = argparse.ArgumentParser(description='run asynchronous PPO')
parser.add_argument("--exp_name", type=str, default='ray_test0', help='experiment name')

# environment
parser.add_argument('--env_name', type=str, default='Humanoid-v2', help='name of env')
parser.add_argument('--env_num', type=int, default=2, help='number of evironments per worker')
parser.add_argument('--total_frames', type=int, default=int(10e6), help='total environment frames')
parser.add_argument('--total_steps', type=int, default=int(2e3), help='optimization batch size')

# important parameters of model and algorithm
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden layer size of mlp & gru')
parser.add_argument('--batch_size', type=int, default=512, help='optimization batch size')
parser.add_argument('--aux_interval', type=int, default=16, help='auxilary phase interval')
parser.add_argument('--n_epoch_policy', type=int, default=1, help='update epoches in policy phase')
parser.add_argument('--n_epoch_aux', type=int, default=9, help='update epoches in auxilary phase')
parser.add_argument('--n_mini_batch', type=int, default=4, help='number of minibatches in 1 training epoch')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--entropy_coef', type=float, default=.01, help='entropy loss coefficient')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lamb', type=float, default=0.97, help='gae discount factor')
parser.add_argument('--clip_ratio', type=float, default=0.2, help='ppo clip ratio')
parser.add_argument('--max_grad_norm', type=float, default=40.0, help='maximum gradient norm')
parser.add_argument('--use_vtrace', type=bool, default=False, help='maximum gradient norm')

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

    aux_keys = ['state', 'hidden_state', 'value']
    aux_buffer = AuxReplayQueue(kwargs['aux_interval'], aux_keys)

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
    while num_frames < kwargs['total_frames'] or global_step < kwargs['total_steps']:
        '''
            sample from buffer
        '''
        iter_start = time.time()
        # wait until there's enough data in buffer
        data_batch = buffer.get(kwargs['batch_size'])
        while data_batch is None:
            data_batch = buffer.get(kwargs['batch_size'])
        sample_time = time.time() - iter_start

        # policy phase
        # split data for burn-in and backpropogation
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
        post['hidden_state'] = learner.step(state, hidden_state, phase='burn_in')
        # training loop: compute loss and backpropogate gradient update
        policy_loss_record = dict(p_loss=[], v_loss=[], entropy_loss=[], grad_norm=[])
        for _ in range(kwargs['n_epoch_policy']):
            minibatch_idxes = torch.split(torch.randperm(kwargs['batch_size']),
                                          kwargs['batch_size'] // kwargs['n_mini_batch'])
            for idx in minibatch_idxes:
                optimizer.zero_grad()
                v_loss, p_loss, entropy_loss = learner.step(
                    phase='policy', **{k: v[idx] if k != 'hidden_state' else v[:, idx]
                                       for k, v in post.items()})
                loss = p_loss + v_loss
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(learner.parameters(), kwargs['max_grad_norm'])
                optimizer.step()

                # record loss
                policy_loss_record['p_loss'].append(p_loss.item())
                policy_loss_record['v_loss'].append(v_loss.item())
                policy_loss_record['entropy_loss'].append(entropy_loss.item())
                policy_loss_record['grad_norm'].append(grad_norm.item())

        # store into auxilary buffer
        v_target = learner.step(**post, phase='value')
        aux_seg = dict(state=data_batch['state'], hidden_state=data_batch['hidden_state'], value=v_target)
        aux_buffer.put(aux_seg)
        policy_phase_time = time.time() - start

        # auxilary phase
        start = time.time()
        if global_step % kwargs['aux_interval'] == 0:
            aux_data_batch = aux_buffer.get(kwargs['aux_interval'])
            # auxilary phase burn-in
            pre, post = dict(), dict()
            for k, v in aux_data_batch.items():
                if k == 'hidden_state':
                    pre[k] = v
                elif k != 'value':
                    # target value has length of 'chunk_len'
                    pre[k], post[k] = np.split(v, [kwargs['burn_in_len']], axis=1)
            state, hidden_state = pre['state'], pre['hidden_state']
            post['hidden_state'] = learner.step(state, hidden_state, phase='burn_in')
            # prepare data needed in auxilary phase update
            post['action_logits'] = learner.step(phase='action_logits', **post)
            post['value'] = aux_data_batch['value']
            # auxilary phase update
            aux_loss_record = dict(p_loss=[], v_loss=[], grad_norm=[])
            for _ in range(kwargs['n_epoch_aux']):
                minibatch_idxes = torch.split(torch.randperm(kwargs['batch_size']),
                                              kwargs['batch_size'] // kwargs['n_mini_batch'])
                for idx in minibatch_idxes:
                    optimizer.zero_grad()
                    aux_p_loss, aux_v_loss = learner.step(
                        phase='aux', **{k: v[idx] if k != 'hidden_state' else v[:, idx]
                                        for k, v in post.items()})
                    (aux_p_loss + aux_v_loss).backward()
                    aux_grad_norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), kwargs['max_grad_norm'])
                    optimizer.step()

                    # record auxilary phase losses
                    aux_loss_record['p_loss'].append(aux_p_loss.item())
                    aux_loss_record['v_loss'].append(aux_v_loss.item())
                    aux_loss_record['grad_norm'].append(aux_grad_norm.item())
        auxilary_phase_time = time.time() - start

        # upload updated learner parameter to parameter server
        if global_step % kwargs['push_period'] == 0:
            # TODO: this line will stall to wait upload job finishing, can be optimize?
            ray.get(ps.set_weights.remote(learner.get_weights()))

        # write summary into TensorBoard
        return_stat = ray.get(return_stat_job)
        for k, v in return_stat.items():
            writer.add_scalar('ep_return/frames/' + k, v, global_step=num_frames, walltime=time.time() - exp_start_time)
            writer.add_scalar('ep_return/steps/' + k, v, global_step=global_step, walltime=time.time() - exp_start_time)
        for k, v in policy_loss_record.items():
            writer.add_scalar('policy_phase/' + k, np.mean(v), global_step=global_step)
        if global_step % kwargs['aux_interval'] == 0:
            for k, v in aux_loss_record.items():
                writer.add_scalar('auxilary_phase/' + k, np.mean(v), global_step=global_step)

        dur = time.time() - iter_start
        print(("Global Step: {}, Frames: {}, Sample Time: {:.2f}s, " +
               "Policy Phase Time: {:.2f}s, Auxilary Phase Time: {:.2f}s, Iteration Step Time: {:.2f}s").format(
                   global_step, num_frames, sample_time, policy_phase_time, auxilary_phase_time, dur))
    print("############ prepare to shut down ray ############")
    ray.shutdown()
    writer.close()
    print("Experiment Time Consume: {}".format(time.time() - exp_start_time))
    print("############ experiment finished ############")
