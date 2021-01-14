import gym
import ray
import time
import wandb
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from env.mujoco.env_with_memory import EnvWithMemory, VecEnvWithMemory
from env.mujoco.model.model import ActorCritic
from env.mujoco.registry import get_shapes, ROLLOUT_KEYS, COLLECT_KEYS, Seg, Info
from rl_utils.buffer import SharedCircularBuffer
from system_utils.simulation_supervisor import SimulationSupervisor

# global configuration
parser = argparse.ArgumentParser(description='run asynchronous PPO')
parser.add_argument("--exp_name", type=str, default='ray_test0', help='experiment name')
parser.add_argument("--wandb_group", type=str, default='mujoco', help='weights & biases group name')
parser.add_argument("--wandb_job", type=str, default='run 100M', help='weights & biases job name')
parser.add_argument("--no_summary", action='store_true', help='whether to write summary')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
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
parser.add_argument('--num_collectors', type=int, default=4, help='# of buffer collectors')
parser.add_argument('--num_writers', type=int, default=4, help='# of buffer writers')
parser.add_argument('--push_period', type=int, default=1, help='learner parameter upload period')
parser.add_argument('--num_workers', type=int, default=32, help='remote worker numbers')
parser.add_argument('--cpu_per_worker', type=int, default=1, help='cpu used per worker')
parser.add_argument('--q_size', type=int, default=8, help='number of batches in replay buffer')

# random seed
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()
kwargs = vars(args)


def build_simple_env(kwargs, seed=0):
    env = gym.make(kwargs['env_name'])
    env.seed(seed)
    return env


def build_worker_env(worker_id, kwargs):
    env_fns = []
    for i in range(kwargs['env_num']):
        env_id = worker_id + i * kwargs['num_workers']
        seed = 12345 * env_id + kwargs['seed']
        env_fns.append(lambda: EnvWithMemory(lambda kwargs: build_simple_env(kwargs, seed=seed), kwargs))
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
kwargs['obs_dim'] = init_env.observation_space.shape[0]
kwargs['action_dim'] = init_env.action_space.shape[0]
high = init_env.action_space.high
low = init_env.action_space.low
kwargs['action_loc'] = (high + low) / 2
kwargs['action_scale'] = (high - low) / 2
kwargs['continuous_env'] = True
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

    # initialize learner, who is responsible for gradient update
    learner = build_learner_model(kwargs)
    optimizer = optim.Adam(learner.parameters(), lr=config.lr)
    # to efficiently broadcast updated weights into workers,
    # use shared memory tensor to communicate with each subprocess
    global_weights = learner.get_weights()
    for v in global_weights.values():
        v.share_memory_()
    weights_available = torch.tensor(1).share_memory_()

    # initialize buffer
    buffer_maxsize = config.batch_size * config.q_size
    buffer = SharedCircularBuffer(buffer_maxsize, config.chunk_len, config.reuse_times, SHAPES, config.batch_size,
                                  config.num_collectors, Seg)

    # initialize workers, who are responsible for interacting with env (simulation)
    queue_util = torch.tensor(0.0).share_memory_()
    ep_info_dict = {
        k + '/' + stat_k: torch.tensor(0.0).share_memory_()
        for k in Info._fields for stat_k in ['avg', 'min', 'max']
    }
    supervisor = SimulationSupervisor(
        rollout_keys=ROLLOUT_KEYS,
        collect_keys=COLLECT_KEYS,
        model_fn=build_worker_model,
        worker_env_fn=build_worker_env,
        global_buffer=buffer,
        weights=global_weights,
        weights_available=weights_available,
        ep_info_dict=ep_info_dict,
        queue_util=queue_util,
        kwargs=kwargs,
    )
    # after starting simulation thread, workers asynchronously interact with
    # environments and send data into buffer via Ray backbone
    supervisor.start()

    shm_tensor_dict = {}
    for k, shp in SHAPES.items():
        if 'rnn_hidden' in k:
            shm_tensor_dict[k] = torch.zeros(shp[0], config.batch_size, *shp[1:]).to(**learner.tpdv)
        else:
            shm_tensor_dict[k] = torch.zeros(config.chunk_len, config.batch_size, *shp).to(**learner.tpdv)

    # main loop
    global_step = 0
    num_frames = 0
    coll_cnt = 0
    while num_frames < kwargs['total_frames']:
        # sample and load into gpu
        iter_start = time.time()
        buffer.get(shm_tensor_dict)
        sample_time = time.time() - iter_start

        num_frames += int(config.chunk_len * config.batch_size)
        global_step += 1
        '''
            update !
        '''
        start = time.time()
        loss_stat = train_learner_on_batch(learner, optimizer, shm_tensor_dict, config)
        optimize_time = time.time() - start

        # broadcast updated learner parameters to each subprocess
        # then subprocess uploads them into remote parameter server
        if global_step % config.push_period == 0:
            new_weights = learner.get_weights()
            for k, v in new_weights.items():
                global_weights[k].copy_(v)
            weights_available.copy_(torch.tensor(1))

        dur = time.time() - iter_start

        return_record = {k: v.item() for k, v in ep_info_dict.items()}
        print("----------------------------------------------")
        print(("Global Step: {}, Frames: {}, " + "Average Return: {:.2f}, " + "Sample Time: {:.2f}s, " +
               "Optimization Time: {:.2f}s, " + "Iteration Step Time: {:.2f}s").format(
                   global_step, num_frames, return_record['ep_return/avg'], sample_time, optimize_time, dur))
        print("----------------------------------------------")

        if not args.no_summary:
            # collect statistics to record
            return_stat = {'ep_return/' + k: v for k, v in return_record.items()}
            loss_stat = {'loss/' + k: v for k, v in loss_stat.items()}
            time_stat = {
                'time/sample': sample_time,
                'time/optimization': optimize_time,
                'time/iteration': dur,
            }
            buffer_stat = {
                'buffer/utilization': buffer.get_util(),
                'buffer/received_sample': buffer.get_received_sample(),
                'buffer/consumed_sample': num_frames / kwargs['chunk_len'],
                'buffer/ready_id_queue_util': queue_util.item(),
            }

            # write summary into weights&biases
            wandb.log({
                **return_stat,
                **loss_stat,
                **time_stat,
                **buffer_stat,
            }, step=num_frames)

    if not args.no_summary:
        run.finish()
    print("Experiment Time Consume: {}".format(time.time() - exp_start_time))
    ray.shutdown()
