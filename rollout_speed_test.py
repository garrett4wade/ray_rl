import gym
import argparse
from env.atari.model.rec_model import ActorCritic
from env.atari.wrappers import WarpFrame, FrameStack
import time
import ray
import numpy as np
import matplotlib.pyplot as plt

# global configuration
parser = argparse.ArgumentParser(description='run asynchronous PPO')
parser.add_argument("--exp_name", type=str, default='ray_test0', help='experiment name')
parser.add_argument("--wandb_group", type=str, default='atari', help='weights & biases group name')
parser.add_argument("--wandb_job", type=str, default='run 100M', help='weights & biases job name')
parser.add_argument("--no_summary", action='store_true', help='whether to write summary')
parser.add_argument("--record_mem", action='store_true', help='whether to store mem info, which is slow')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--verbose', action='store_true')

# environment
parser.add_argument('--env_name', type=str, default='Breakout-v0', help='name of env')
parser.add_argument('--env_num', type=int, default=2, help='# evironments per worker')
parser.add_argument('--env_split', type=int, default=2, help='# splitted vectorized env copies per worker')
parser.add_argument('--total_frames', type=int, default=int(100e6), help='optimization batch size')

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
parser.add_argument('--min_return_chunk_num', type=int, default=64, help='minimal chunk number before env.collect')

# Ray distributed training parameters
parser.add_argument('--num_supervisors', type=int, default=1, help='# of simulation supervisors')
parser.add_argument('--num_collectors', type=int, default=4, help='# of buffer collectors')
parser.add_argument('--num_readers', type=int, default=4, help='# of data sendors')
parser.add_argument('--push_period', type=int, default=1, help='learner parameter upload period')
parser.add_argument('--num_workers', type=int, default=32, help='remote worker numbers')
parser.add_argument('--num_returns', type=int, default=1, help='number of returns in ray.wait')
parser.add_argument('--cpu_per_worker', type=float, default=1, help='cpu used per worker')
parser.add_argument('--q_size', type=int, default=16, help='number of batches in replay buffer')

# random seed
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()
kwargs = vars(args)


def build_simple_env(seed=0):
    env = gym.make("Breakout-v0", frameskip=4)
    env = FrameStack(WarpFrame(env, 84), 4)
    env.seed(seed)
    return env


def _preprocess(obs, h):
    obs = np.transpose(obs, [2, 0, 1]).astype(np.float32)
    return np.stack([obs / 255.0]), np.stack([h])


@ray.remote(num_cpus=.5)
class Worker:
    def __init__(self):
        self.model = ActorCritic(False, kwargs)
        self.env = build_simple_env()

    def rollout(self):
        obs = self.env.reset()
        h = np.zeros((1, 2 * kwargs['hidden_dim']), dtype=np.float32)
        d = False
        step = 0
        while not d:
            action, _, _, h = self.model.select_action(*_preprocess(obs, h))
            h = h[0]
            obs, _, d, _ = self.env.step(action[0])
            step += 1
        return step


init_env = build_simple_env()
kwargs['obs_dim'] = (84, 84, 4)
kwargs['action_dim'] = init_env.action_space.n
del init_env

if __name__ == "__main__":
    ray.init()
    worker = Worker.remote()
    global_step = 0
    start = time.time()
    for _ in range(10):
        global_step += ray.get(worker.rollout.remote())
    print(global_step / (time.time() - start))
