import os
import gym
import ray
import time
import torch
import wandb
import torch.nn as nn
import argparse
import numpy as np
import torch.optim as optim

from rl_utils.simple_env import Envs
from rl_utils.buffer import ReplayQueue
from model.model import ActorCritic
from ray_utils.ps import ParameterServer, ReturnRecorder
from ray_utils.simulation_thread import SimulationThread
from ray.rllib.env.atari_wrappers import wrap_deepmind

# global configuration
parser = argparse.ArgumentParser(description='run asynchronous PPO')
parser.add_argument("--exp_name", type=str, default='ray_test0', help='experiment name')
parser.add_argument("--write_summary", type=bool, default=True, help='whether to write summary')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

# environment
parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='name of env')
parser.add_argument('--env_num', type=int, default=8, help='number of evironments per worker')
parser.add_argument('--total_frames', type=int, default=int(20e6), help='optimization batch size')

# important parameters of model and algorithm
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden layer size of mlp & gru')
parser.add_argument('--batch_size', type=int, default=512, help='optimization batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--entropy_coef', type=float, default=.0, help='entropy loss coefficient')
parser.add_argument('--value_coef', type=float, default=1.0, help='entropy loss coefficient')
parser.add_argument('--gamma', type=float, default=0.997, help='discount factor')
parser.add_argument('--lmbda', type=float, default=0.97, help='gae discount factor')
parser.add_argument('--clip_ratio', type=float, default=0.2, help='ppo clip ratio')
parser.add_argument('--n_epoch', type=int, default=2, help='update times in each training step')
parser.add_argument('--n_minibatch', type=int, default=4, help='update times in each training step')
parser.add_argument('--max_grad_norm', type=float, default=40.0, help='maximum gradient norm')
parser.add_argument('--use_vtrace', type=bool, default=False, help='whether to use vtrace')

# recurrent model parameters
parser.add_argument('--burn_in_len', type=int, default=0, help='rnn hidden state burn-in length')
parser.add_argument('--chunk_len', type=int, default=64, help='rnn BPTT chunk length')
parser.add_argument('--replay', type=int, default=1, help='sequence cross-replay times')
parser.add_argument('--max_timesteps', type=int, default=int(1e6), help='episode maximum timesteps')
parser.add_argument('--min_return_chunk_num', type=int, default=5, help='minimal chunk number before env.collect')

# Ray distributed training parameters
parser.add_argument('--push_period', type=int, default=1, help='learner parameter upload period')
parser.add_argument('--load_period', type=int, default=25, help='load period from parameter server')
parser.add_argument('--num_workers', type=int, default=32, help='remote worker numbers')
parser.add_argument('--num_returns', type=int, default=4, help='number of returns in ray.wait')
parser.add_argument('--cpu_per_worker', type=int, default=1, help='cpu used per worker')
parser.add_argument('--q_size', type=int, default=16, help='number of batches in replay buffer')

# random seed
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()
kwargs = vars(args)


def build_simple_env(kwargs):
    return wrap_deepmind(gym.make(kwargs['env_name']), dim=42)


# get state/action information from env
init_env = build_simple_env(kwargs)
kwargs['state_dim'] = (42, 42, 4)
kwargs['action_dim'] = init_env.action_space.n
kwargs['continuous_env'] = False
del init_env


def build_worker_env(worker_id, kwargs):
    return Envs(build_simple_env, worker_id, kwargs)


def build_worker_model(kwargs):
    # model for interacting with env, does not need gradient update
    # instead, download parameters from parameter server (ps)
    return ActorCritic(False, kwargs)


def build_learner_model(kwargs):
    # model for gradient update
    # after update, upload parameters to parameter server
    return ActorCritic(True, kwargs)


def train_learner_on_minibatch(learner, optimizer, data_batch, config):
    batch_size = config.batch_size
    n_minibatch = config.n_minibatch
    stat = dict(v_loss=[], p_loss=[], entropy_loss=[])
    minibatch_idxes = torch.split(torch.randperm(batch_size), batch_size // n_minibatch)
    for idx in minibatch_idxes:
        optimizer.zero_grad()
        v_loss, p_loss, entropy_loss = learner.compute_loss(**{k: v[idx] for k, v in data_batch.items()})
        loss = p_loss + config.value_coef * v_loss + config.entropy_coef * entropy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(learner.parameters(), config.max_grad_norm)
        optimizer.step()

        stat['v_loss'].append(v_loss.item())
        stat['p_loss'].append(p_loss.item())
        stat['entropy_loss'].append(entropy_loss.item())
    return {k: np.mean(v) for k, v in stat.items()}


if __name__ == "__main__":
    exp_start_time = time.time()

    # set random seed
    torch.manual_seed(kwargs['seed'])
    np.random.seed(kwargs['seed'] + 13563)

    # initialize ray
    ray.init()

    if args.write_summary:
        # initialized weights&biases summary
        run = wandb.init(project='distributed rl',
                         group='atari pong',
                         job_type='performace_evaluation',
                         name=kwargs['exp_name'],
                         entity='garrett4wade',
                         config=kwargs)
        config = wandb.config
    else:
        config = args

    # initialize learner, who is responsible for gradient update
    learner = build_learner_model(kwargs)
    optimizer = optim.Adam(learner.parameters(), lr=config.lr)
    init_weights = learner.get_weights()

    # initialize buffer
    keys = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target']
    buffer = ReplayQueue(config.batch_size * config.q_size, keys)

    # initialize workers, who are responsible for interacting with env (simulation)
    ps = ParameterServer.remote(weights=init_weights)
    recorder = ReturnRecorder.remote()
    simulation_thread = SimulationThread(model_fn=build_worker_model,
                                         env_fn=build_worker_env,
                                         ps=ps,
                                         recorder=recorder,
                                         global_buffer=buffer,
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
        data_batch = buffer.get(config.batch_size)
        while data_batch is None:
            data_batch = buffer.get(config.batch_size)
        sample_time = time.time() - iter_start
        '''
            split data for burn-in and backpropogation
        '''
        # pull from remote return recorder
        return_stat_job = recorder.pull.remote()
        num_frames += config.batch_size
        global_step += 1
        '''
            update !
        '''
        start = time.time()
        # training loop: compute loss and backpropogate gradient update
        loss_record = dict(p_loss=[], v_loss=[], entropy_loss=[])
        # convert numpy array to tensor and send them to desired device
        for k, v in data_batch.items():
            data_batch[k] = torch.from_numpy(v).to(**learner.tpdv)
        for _ in range(config.n_epoch):
            stat = train_learner_on_minibatch(learner, optimizer, data_batch, config)
            for k, v in stat.items():
                loss_record[k].append(v)
        optimize_time = time.time() - start

        # upload updated learner parameter to parameter server
        if global_step % config.push_period == 0:
            ray.get(ps.set_weights.remote(learner.get_weights()))

        return_stat = ray.get(return_stat_job)

        dur = time.time() - iter_start
        print("----------------------------------------------")
        print(("Global Step: {}, Frames: {}, " + "Average Return: {:.2f}, " + "Sample Time: {:.2f}s, " +
               "Optimization Time: {:.2f}s, " + "Iteration Step Time: {:.2f}s").format(
                   global_step, num_frames, return_stat['avg'], sample_time, optimize_time, dur))
        print("----------------------------------------------")

        # collect statistics to record
        return_stat = {'ep_return/' + k: v for k, v in return_stat.items()}
        loss_stat = {'loss/' + k: np.mean(v) for k, v in loss_record.items()}
        time_stat = {'time/sample': sample_time, 'time/optimization': optimize_time, 'time/iteration': dur}
        if args.write_summary:
            # write summary into weights&biases
            wandb.log({**return_stat, **loss_stat, **time_stat}, step=num_frames)

    print("############ prepare to shut down ray ############")
    ray.shutdown()
    run.finish()
    print("Experiment Time Consume: {}".format(time.time() - exp_start_time))
    print("############ experiment finished ############")
