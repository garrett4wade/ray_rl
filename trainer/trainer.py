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
                 ctrl,
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
        self.ctrl = ctrl
        self.optimizer_nickname = optimizer

        # for parameters broadcasting to workers
        self.global_weights = global_weights
        self.weights_available = weights_available

        # for summary information
        self.ep_info_dict = ep_info_dict
        self.queue_util = queue_util
        self.wait_time = wait_time

    def setup(self):
        """
        if world_size == 1, set up simple trainer on single GPU,
        else set up PyTorch DDP trainer on corresponding GPU
        """
        os.setpriority(os.PRIO_PROCESS, os.getpid(), 0)
        # initialized weights & biases summary
        if (self.rank == 0 or self.world_size == 1) and not self.config.no_summary:
            self.wandb_exp = wandb.init(project='distributed rl',
                                        group=self.config.wandb_group,
                                        job_type=self.config.wandb_job,
                                        name=self.config.exp_name,
                                        entity='garrett4wade',
                                        config=vars(self.config))

        # set up learner
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.rank)
        if self.world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            # initialize the process group
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
            print(f"Trainer Set Up: initializing DDP training on GPU rank {self.rank}.")
            # construct DDP model
            self.learner = DDP(deepcopy(self.learner_prototype).cuda())
        else:
            self.learner = self.learner_prototype.cuda()
            print(f"Trainer Set Up: initializing training on single GPU rank {self.rank}.")

        # load ckpt if needed
        if self.config.load_ckpt:
            if self.world_size > 1:
                dist.barrier()
            self.learner.load_state_dict(torch.load(self.config.load_ckpt_file, map_location={'cpu': 'cuda'}))

        # set up optimizer
        if self.optimizer_nickname == 'adam':
            self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=self.config.lr)
        else:
            raise NotImplementedError

        self.num_frames = 0
        self.global_step = 0

    def teardown(self):
        if self.world_size > 1:
            dist.destroy_process_group()

    def train_learner_on_batch(self, data_batch, n_minibatch):
        if n_minibatch == 1:
            self.optimizer.zero_grad()
            v_loss, p_loss, entropy_loss = self.loss_fn(self.learner,
                                                        clip_ratio=self.config.clip_ratio,
                                                        world_size=self.world_size,
                                                        **data_batch)
            loss = p_loss + self.config.value_coef * v_loss + self.config.entropy_coef * entropy_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.learner.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            return v_loss, p_loss, entropy_loss, grad_norm
        else:
            batch_size = data_batch[list(data_batch.keys())[0]].shape[1]
            minibs = batch_size // n_minibatch
            assert batch_size % n_minibatch == 0
            indices = torch.split(torch.randperm(batch_size), minibs)
            v_losses, p_losses, entropy_losses, grad_norms = [], [], [], []
            for idx in indices:
                self.optimizer.zero_grad()
                mini_databatch = {k: v[:, idx] for k, v in data_batch.items()}
                v_loss, p_loss, entropy_loss = self.loss_fn(self.learner,
                                                            clip_ratio=self.config.clip_ratio,
                                                            world_size=self.world_size,
                                                            **mini_databatch)
                loss = p_loss + self.config.value_coef * v_loss + self.config.entropy_coef * entropy_loss
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.learner.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                v_losses.append(v_loss)
                p_losses.append(p_loss)
                entropy_losses.append(entropy_loss)
                grad_norms.append(grad_norm)
            return (sum(v_losses) / n_minibatch, sum(p_losses) / n_minibatch, sum(entropy_losses) / n_minibatch,
                    sum(grad_norms) / n_minibatch)

    def run(self):
        self.setup()
        while self.num_frames < self.config.total_frames:
            if self.world_size > 1:
                dist.barrier()
            # sample and load into gpu
            iter_start = time.time()
            data_batch = self.buffer.get(self.ctrl.barrier)
            sample_time = time.time() - iter_start

            self.num_frames += int(self.config.chunk_len * self.config.batch_size * self.config.num_gpus)
            self.global_step += 1
            '''
                update !
            '''
            start = time.time()
            v_loss, p_loss, entropy_loss, grad_norm = self.train_learner_on_batch(data_batch, self.config.n_minibatch)
            optimize_time = time.time() - start

            # broadcast updated learner parameters to each subprocess
            # then subprocess uploads them into remote parameter server
            if (self.rank == 0 or self.world_size == 1) and self.global_step % self.config.push_period == 0:
                new_weights = {k: v.cpu() for k, v in self.learner.state_dict().items()}
                for k, v in new_weights.items():
                    self.global_weights[k.replace('module.', '')].copy_(v)
                self.weights_available.copy_(torch.tensor(1))

            if self.world_size > 1:
                dist.all_reduce_multigpu([v_loss])
                dist.all_reduce_multigpu([p_loss])
                dist.all_reduce_multigpu([entropy_loss])
                dist.all_reduce_multigpu([grad_norm])
            loss_stat = {
                'v_loss': v_loss.item() / self.world_size,
                'p_loss': p_loss.item() / self.world_size,
                'entropy_loss': entropy_loss.item() / self.world_size,
                'grad_norm': grad_norm.item() / self.world_size
            }
            iter_dur = time.time() - iter_start

            return_stat = {k: v.item() for k, v in self.ep_info_dict.items()}
            self.ctrl.lock.acquire()
            print("Process: {:1d} | ".format(self.rank) + "Global Step: {: >5d} | ".format(self.global_step) +
                  "Frames: {: >10d} | ".format(self.num_frames) +
                  "Average Return: {: >8.2f} | ".format(return_stat['ep_return/avg']) +
                  "Sample Time: {: >6.2f}s | ".format(sample_time) +
                  "Optimization Time: {: >6.2f}s | ".format(optimize_time) +
                  "Iteration Step Time: {: >6.2f}s".format(iter_dur))
            self.ctrl.lock.release()

            if (self.rank == 0 or self.world_size == 1) and not self.config.no_summary:
                self.summary(loss_stat, return_stat, sample_time, optimize_time, iter_dur)

            if (self.rank == 0 or self.world_size == 1) and self.config.save_ckpt and (self.global_step %
                                                                                       self.config.save_interval == 0):
                ckpt_file = os.path.join(self.config.save_ckpt_dir, str(self.global_step) + '.pt')
                torch.save({k: v.cpu() for k, v in self.learner.state_dict().items()}, ckpt_file)

        self.teardown()
        if (self.rank == 0 or self.world_size == 1) and not self.config.no_summary:
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
