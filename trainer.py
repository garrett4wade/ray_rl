import os
import time
import wandb
import torch
import torch.distributed as dist
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self,
                 rank,
                 world_size,
                 model,
                 loss_fn,
                 buffer,
                 global_weights,
                 weights_available,
                 ep_info_dict,
                 queue_util,
                 wait_time,
                 config,
                 optimizer='adam'):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.learner_prototype = model
        self.loss_fn = loss_fn
        self.buffer = buffer
        self.optimizer_nickname = optimizer

        # for parameters broadcasting to workers
        self.global_weights = global_weights
        self.weights_available = weights_available

        # for summary information
        self.ep_info_dict = ep_info_dict
        self.queue_util = queue_util
        self.wait_time = wait_time

    def setup(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.rank)
        if self.rank == 0 and not self.config.no_summary:
            # initialized weights&biases summary
            self.wandb_exp = wandb.init(project='distributed rl',
                                        group=self.config.wandb_group,
                                        job_type=self.config.wandb_job,
                                        name=self.config.exp_name,
                                        entity='garrett4wade',
                                        config=vars(self.config))
        # initialize the process group
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        print(f"Initializing DDP training on rank {self.rank}.")
        # construct DDP model and optimizer
        self.learner = DDP(deepcopy(self.learner_prototype).cuda())
        if self.optimizer_nickname == 'adam':
            self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=self.config.lr)
        else:
            raise NotImplementedError
        # pre-allocate cuda tensor to copy data from buffer
        self.tensor_dict = {}
        for k, shp in self.buffer.shapes.items():
            if 'rnn_hidden' in k:
                self.tensor_dict[k] = torch.zeros((shp[0], self.config.batch_size, *shp[1:]),
                                                  dtype=getattr(torch, self.buffer.dtypes[k].__name__)).cuda()
            else:
                self.tensor_dict[k] = torch.zeros((self.config.chunk_len, self.config.batch_size, *shp),
                                                  dtype=getattr(torch, self.buffer.dtypes[k].__name__)).cuda()
        self.num_frames = 0
        self.global_step = 0

    def teardown(self):
        dist.destroy_process_group()

    def train_learner_on_batch(self):
        self.optimizer.zero_grad()
        v_loss, p_loss, entropy_loss = self.loss_fn(self.learner, clip_ratio=self.config.clip_ratio, **self.tensor_dict)
        loss = p_loss + self.config.value_coef * v_loss + self.config.entropy_coef * entropy_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.learner.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return v_loss, p_loss, entropy_loss, grad_norm

    def run(self):
        self.setup()
        while self.num_frames < self.config.total_frames:
            # sample and load into gpu
            iter_start = time.time()
            self.buffer.get(self.tensor_dict)
            sample_time = time.time() - iter_start

            self.num_frames += int(self.config.chunk_len * self.config.batch_size * self.config.num_gpus)
            self.global_step += 1
            '''
                update !
            '''
            start = time.time()
            v_loss, p_loss, entropy_loss, grad_norm = self.train_learner_on_batch()
            optimize_time = time.time() - start

            # broadcast updated learner parameters to each subprocess
            # then subprocess uploads them into remote parameter server
            if self.rank == 0 and self.global_step % self.config.push_period == 0:
                new_weights = {k: v.cpu() for k, v in self.learner.state_dict().items()}
                for k, v in new_weights.items():
                    self.global_weights[k.replace('module.', '')].copy_(v)
                self.weights_available.copy_(torch.tensor(1))

            dist.all_reduce_multigpu([v_loss])
            dist.all_reduce_multigpu([p_loss])
            dist.all_reduce_multigpu([entropy_loss])
            dist.all_reduce_multigpu([grad_norm])
            loss_stat = {
                'v_loss': v_loss.item(),
                'p_loss': p_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'grad_norm': grad_norm.item()
            }
            iter_dur = time.time() - iter_start

            return_stat = {k: v.item() for k, v in self.ep_info_dict.items()}
            print("-" * 50)
            print(("Process: {}, Global Step: {}, Frames: {}, " + "Average Return: {:.2f}, " +
                   "Sample Time: {:.2f}s, " + "Optimization Time: {:.2f}s, " + "Iteration Step Time: {:.2f}s").format(
                       self.rank, self.global_step, self.num_frames, return_stat['ep_return/avg'], sample_time,
                       optimize_time, iter_dur))
            print("-" * 50)

            if self.rank == 0 and not self.config.no_summary:
                self.summary(loss_stat, return_stat, sample_time, optimize_time, iter_dur)
        self.teardown()
        if self.rank == 0 and not self.config.no_summary:
            self.wandb_exp.finish()

    def summary(self, loss_stat, return_stat, sample_time, optimize_time, iter_dur):
        # collect statistics to record
        time_stat = {
            'time/sample': sample_time,
            'time/optimization': optimize_time,
            'time/iteration': iter_dur,
        }
        buffer_stat = {
            'buffer/utilization': self.buffer.get_util(),
            'buffer/received_sample': self.buffer.get_received_sample(),
            'buffer/consumed_sample': self.num_frames / self.config.chunk_len,
            'buffer/ready_id_queue_util': self.queue_util.item(),
            'buffer/ray_wait_time': self.wait_time.item(),
        }

        # write summary into weights&biases
        wandb.log(
            {
                'global_step': self.global_step,
                **{'ep_return/' + k: v
                   for k, v in return_stat.items()},
                **{'loss/' + k: v
                   for k, v in loss_stat.items()},
                **time_stat,
                **buffer_stat,
            },
            step=self.num_frames)
