import os
import gym
import time
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp
from env.mujoco.env_with_memory import EnvWithMemory, VecEnvWithMemory
from env.mujoco.model.model import ActorCritic, compute_loss
from env.mujoco.registry import get_shapes, ROLLOUT_KEYS, COLLECT_KEYS, DTYPES, Info
from rollout_runner import RolloutRunner
from trainer.trainer import Trainer
from utils.find_free_gpu import find_free_gpu

# global configuration
parser = argparse.ArgumentParser(description='run asynchronous PPO')
parser.add_argument("--exp_name", type=str, default='ray_test0', help='experiment name')
parser.add_argument("--wandb_group", type=str, default='mujoco', help='weights & biases group name')
parser.add_argument("--wandb_job", type=str, default='run 100M', help='weights & biases job name')
parser.add_argument("--no_summary", action='store_true', help='whether to write summary')
parser.add_argument('--num_gpus', type=int, default=4, help='utilized gpu num')
parser.add_argument('--verbose', action='store_true', help='whether to output debug imformation')
parser.add_argument('--cluster', action='store_true', help='whether running on cluster')

# environment
parser.add_argument('--env_name', type=str, default='Humanoid-v2', help='name of env')
parser.add_argument('--env_num', type=int, default=16, help='# evironments per worker')
parser.add_argument('--total_frames', type=int, default=int(100e6), help='optimization batch size')

# important parameters of model and algorithm
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden layer size of mlp & gru')
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
parser.add_argument('--min_return_chunk_num', type=int, default=32, help='minimal chunk number before env.collect')

# Ray distributed training parameters
parser.add_argument('--ray_dashboard', action='store_true', help='use ray dashboard')
parser.add_argument('--num_writers', type=int, default=4, help='# of buffer writers')
parser.add_argument('--push_period', type=int, default=1, help='learner parameter upload period')
parser.add_argument('--num_workers', type=int, default=32, help='remote worker numbers')
parser.add_argument('--cpu_per_worker', type=int, default=1, help='cpu used per worker')
parser.add_argument('--q_size', type=int, default=8, help='number of batches in replay buffer')

# random seed
parser.add_argument('--seed', type=int, default=0, help='random seed')

config = parser.parse_args()


def build_simple_env(config, seed=0):
    env = gym.make(config.env_name)
    env.seed(seed)
    return env


def build_worker_env(worker_id, config):
    env_fns = []
    for i in range(config.env_num):
        env_id = worker_id + i * config.num_workers
        seed = 12345 * env_id + config.seed
        env_fns.append(lambda: EnvWithMemory(lambda config: build_simple_env(config, seed=seed), config))
    return VecEnvWithMemory(env_fns)


def build_worker_model(config):
    # model for interacting with env, does not need gradient update
    # instead, download parameters from parameter server (ps)
    return ActorCritic(False, config)


def build_learner_model(config):
    # model for gradient update
    # after update, upload parameters to parameter server
    return ActorCritic(True, config)


# get state/action information from env
init_env = build_simple_env(config)
config.obs_dim = init_env.observation_space.shape[0]
config.action_dim = init_env.action_space.shape[0]
high = init_env.action_space.high
low = init_env.action_space.low
config.action_loc = (high + low) / 2
config.action_scale = (high - low) / 2
config.continuous_env = True
del init_env

SHAPES = get_shapes(config)

if __name__ == "__main__":
    exp_start_time = time.time()
    os.setpriority(os.PRIO_PROCESS, os.getpid(), 0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed + 13563)

    # learner prototype is used for initializing worker models and learner DDP model
    # it will never be trained
    learner_prototype = build_learner_model(config)

    # initialize Ray based remote rollout workers
    init_weights = learner_prototype.state_dict()
    rollout_runner = RolloutRunner(init_weights=init_weights,
                                   worker_model_fn=build_worker_model,
                                   worker_env_fn=build_worker_env,
                                   rollout_keys=ROLLOUT_KEYS,
                                   collect_keys=COLLECT_KEYS,
                                   ep_info_keys=Info._fields,
                                   shapes=SHAPES,
                                   dtypes=DTYPES,
                                   config=config)
    rollout_runner.run()

    # initialize trainer for each GPU and start DDP training
    ranks = find_free_gpu(config.num_gpus)
    trainers = [
        Trainer(rank=rank,
                world_size=config.num_gpus,
                model=learner_prototype,
                loss_fn=compute_loss,
                buffer=rollout_runner.buffer,
                global_weights=rollout_runner.global_weights,
                weights_available=rollout_runner.weights_available,
                ep_info_dict=rollout_runner.ep_info_dict,
                queue_util=rollout_runner.queue_util,
                wait_time=rollout_runner.wait_time,
                config=config) for rank in ranks
    ]
    jobs = [mp.Process(target=trainer.run) for trainer in trainers]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()

    # terminate rollout workers
    rollout_runner.shutdown()
    print("Experiment Time Consume: {}".format(time.time() - exp_start_time))
