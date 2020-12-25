import gym
import ray
import time
import wandb
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# import os
# import psutil
# from pgrep import pgrep

from env.atari.env_with_memory import EnvWithMemory, VecEnvWithMemory
from env.atari.genetype import get_shapes, COLLECT_KEYS, ROLLOUT_KEYS, BURN_IN_INPUT_KEYS
from env.atari.model.rec_model import ActorCritic
from env.atari.wrappers import WarpFrame, FrameStack
from rl_utils.buffer import CircularBuffer
from ray_utils.remote_server import ParameterServer, EpisodeRecorder
from ray_utils.remote_actors import SimulationSupervisor  # , BufferCollector  # , GPULoader
from ray_utils.init_ray import initialize_single_machine_ray

# global configuration
parser = argparse.ArgumentParser(description='run asynchronous PPO')
parser.add_argument("--exp_name", type=str, default='sample_only_thread', help='experiment name')
parser.add_argument("--wandb_group", type=str, default='sample_speed', help='weights & biases group name')
parser.add_argument("--wandb_job", type=str, default='sample_speed', help='weights & biases job name')
parser.add_argument("--no_summary", action='store_true', help='whether to write summary')
parser.add_argument("--record_mem", action='store_true', help='whether to store mem info, which is slow')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--verbose', action='store_true')

# environment
parser.add_argument('--env_name', type=str, default='Breakout-v0', help='name of env')
parser.add_argument('--env_num', type=int, default=16, help='# evironments per worker')
parser.add_argument('--total_frames', type=int, default=int(10e6), help='optimization batch size')

# important parameters of model and algorithm
parser.add_argument('--hidden_dim', type=int, default=512, help='hidden layer size of mlp & gru')
parser.add_argument('--batch_size', type=int, default=512, help='optimization batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--entropy_coef', type=float, default=.01, help='entropy loss coefficient')
parser.add_argument('--value_coef', type=float, default=1.0, help='entropy loss coefficient')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lmbda', type=float, default=0.95, help='gae discount factor')
parser.add_argument('--clip_ratio', type=float, default=0.2, help='ppo clip ratio')
parser.add_argument('--reuse_times', type=int, default=2, help='sample reuse times')
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='maximum gradient norm')
parser.add_argument('--use_vtrace', type=bool, default=False, help='whether to use vtrace')

# recurrent model parameters
parser.add_argument('--burn_in_len', type=int, default=0, help='rnn hidden state burn-in length')
parser.add_argument('--chunk_len', type=int, default=16, help='rnn BPTT chunk length')
parser.add_argument('--replay', type=int, default=1, help='sequence cross-replay times')
parser.add_argument('--max_timesteps', type=int, default=int(1e6), help='episode maximum timesteps')
parser.add_argument('--min_return_chunk_num', type=int, default=16, help='minimal chunk number before env.collect')

# Ray distributed training parameters
parser.add_argument('--num_supervisors', type=int, default=1, help='# of simulation supervisors')
parser.add_argument('--num_collectors', type=int, default=4, help='# of buffer collectors')
parser.add_argument('--num_readers', type=int, default=1, help='# of data sendors')
parser.add_argument('--push_period', type=int, default=1, help='learner parameter upload period')
parser.add_argument('--num_workers', type=int, default=32, help='remote worker numbers')
parser.add_argument('--num_returns', type=int, default=1, help='number of returns in ray.wait')
parser.add_argument('--cpu_per_worker', type=float, default=1, help='cpu used per worker')
parser.add_argument('--q_size', type=int, default=16, help='number of batches in replay buffer')

# random seed
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()
kwargs = vars(args)


def build_simple_env(kwargs, seed=0):
    env = gym.make(kwargs['env_name'], frameskip=4)
    env = FrameStack(WarpFrame(env, 84), 4)
    env.seed(seed)
    return env


def build_worker_env(worker_id, kwargs):
    env_fns = []
    for i in range(kwargs['env_num']):
        env_id = worker_id + i * kwargs['num_workers']
        seed = 12345 * env_id + kwargs['seed']
        env_fns.append(lambda: EnvWithMemory(lambda kwargs: build_simple_env(kwargs, seed=seed), ROLLOUT_KEYS,
                                             COLLECT_KEYS, BURN_IN_INPUT_KEYS, SHAPES, kwargs))
    return VecEnvWithMemory(env_fns)


def build_worker_model(kwargs):
    # model for interacting with env, does not need gradient update
    # instead, download parameters from parameter server (ps)
    return ActorCritic(False, kwargs)


def build_learner_model(kwargs):
    # model for gradient update
    # after update, upload parameters to parameter server
    return ActorCritic(True, kwargs)


def train_learner_on_batch(learner, optimizer, data_batch, config):
    optimizer.zero_grad()
    v_loss, p_loss, entropy_loss = learner.compute_loss(**data_batch)
    loss = p_loss + config.value_coef * v_loss + config.entropy_coef * entropy_loss
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(learner.parameters(), config.max_grad_norm)
    optimizer.step()
    return dict(v_loss=v_loss.item(),
                p_loss=p_loss.item(),
                entropy_loss=entropy_loss.item(),
                grad_norm=grad_norm.item())


# get state/action information from env
init_env = build_simple_env(kwargs)
kwargs['obs_dim'] = (84, 84, 4)
kwargs['action_dim'] = init_env.action_space.n
del init_env

SHAPES = get_shapes(kwargs)

if __name__ == "__main__":
    exp_start_time = time.time()

    # set random seed
    torch.manual_seed(kwargs['seed'])
    np.random.seed(kwargs['seed'] + 13563)

    if args.no_summary:
        config = args
    else:
        # initialized weights&biases summary
        run = wandb.init(project='distributed rl',
                         group=kwargs['wandb_group'],
                         job_type=kwargs['wandb_job'],
                         name=kwargs['exp_name'],
                         entity='garrett4wade',
                         config=kwargs)
        config = wandb.config

    initialize_single_machine_ray(config)

    # initialize learner, who is responsible for gradient update
    learner = build_learner_model(kwargs)
    optimizer = optim.Adam(learner.parameters(), lr=config.lr)
    init_weights = learner.get_weights()

    # initialize buffer
    buffer_maxsize = config.batch_size * config.q_size
    buffer = ray.remote(num_cpus=2)(CircularBuffer).remote(buffer_maxsize, config.reuse_times, COLLECT_KEYS)

    # initialize workers, who are responsible for interacting with env (simulation)
    ps = ParameterServer.remote(weights=init_weights)
    recorder = EpisodeRecorder.remote()
    supervisors = [
        SimulationSupervisor.remote(model_fn=build_worker_model,
                                    worker_env_fn=build_worker_env,
                                    ps=ps,
                                    recorder=recorder,
                                    remote_buffer=buffer,
                                    kwargs=kwargs) for _ in range(config.num_supervisors)
    ]
    # after starting simulation thread, workers asynchronously interact with
    # environments and send data into buffer via Ray backbone
    for supervisor in supervisors:
        supervisor.start.remote()

    # buffer_collector = BufferCollector(buffer, config.batch_size, config.num_collectors)

    # main loop
    global_step = 0
    num_frames = 0
    while num_frames < kwargs['total_frames']:
        num_frames += kwargs['batch_size'] * kwargs['chunk_len']
        global_step += 1
        time.sleep(0.5)
        if not args.no_summary:
            # write summary into weights&biases
            wandb.log(
                {
                    'buffer/received_sample':
                    ray.get(buffer.get_received_sample.remote()),
                    'buffer/ready_id_queue_util':
                    np.mean(ray.get([supervisor.get_ready_queue_util.remote() for supervisor in supervisors])),
                },
                step=num_frames)
    if not args.no_summary:
        run.finish()
    print("Experiment Time Consume: {}".format(time.time() - exp_start_time))
    ray.shutdown()
