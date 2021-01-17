import time
import wandb
import torch
import torch.nn as nn
from rl_utils.buffer import SharedCircularBuffer
from system_utils.simulation_supervisor import SimulationSupervisor


class Runner:
    def __init__(
        self,
        learner_model_fn,
        worker_model_fn,
        worker_env_fn,
        rollout_keys,
        collect_keys,
        info_fn,
        seg_fn,
        shapes,
        dtypes,
        config,
        optimizer='adam',
    ):
        # initialize learner, who is responsible for gradient update
        self.learner = learner_model_fn(config)
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=config.lr)
        # to efficiently broadcast updated weights into workers,
        # use shared memory tensor to communicate with each subprocess
        self.global_weights = self.learner.get_weights()
        for v in self.global_weights.values():
            v.share_memory_()
        self.weights_available = torch.tensor(1).share_memory_()

        # initialize buffer
        buffer_maxsize = config.batch_size * config.q_size
        self.buffer = SharedCircularBuffer(buffer_maxsize, config.chunk_len, config.reuse_times, shapes, dtypes,
                                           config.batch_size, seg_fn)
        self.shm_tensor_dict = {}
        for k, shp in shapes.items():
            if 'rnn_hidden' in k:
                self.shm_tensor_dict[k] = torch.zeros((shp[0], config.batch_size, *shp[1:]),
                                                      dtype=getattr(torch, dtypes[k].__name__),
                                                      device=self.learner.device)
            else:
                self.shm_tensor_dict[k] = torch.zeros((config.chunk_len, config.batch_size, *shp),
                                                      dtype=getattr(torch, dtypes[k].__name__),
                                                      device=self.learner.device)

        # initialize workers, who are responsible for interacting with env (simulation)
        self.queue_util = torch.tensor(0.0).share_memory_()
        self.wait_time = torch.tensor(0.0).share_memory_()
        self.ep_info_dict = {
            k + '/' + stat_k: torch.tensor(0.0).share_memory_()
            for k in info_fn._fields for stat_k in ['avg', 'min', 'max']
        }
        self.supervisor = SimulationSupervisor(rollout_keys=rollout_keys,
                                               collect_keys=collect_keys,
                                               model_fn=worker_model_fn,
                                               worker_env_fn=worker_env_fn,
                                               global_buffer=self.buffer,
                                               weights=self.global_weights,
                                               weights_available=self.weights_available,
                                               ep_info_dict=self.ep_info_dict,
                                               queue_util=self.queue_util,
                                               wait_time=self.wait_time,
                                               config=config)
        self.config = config

    def train_learner_on_batch(self):
        self.optimizer.zero_grad()
        v_loss, p_loss, entropy_loss = self.learner.compute_loss(**self.shm_tensor_dict)
        loss = p_loss + self.config.value_coef * v_loss + self.config.entropy_coef * entropy_loss
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.learner.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return dict(v_loss=v_loss.item(),
                    p_loss=p_loss.item(),
                    entropy_loss=entropy_loss.item(),
                    grad_norm=grad_norm.item())

    def run(self):
        # after starting supervisor, workers asynchronously interact with
        # environments and send data into buffer via Ray backbone
        self.supervisor.start()

        # main loop
        self.global_step = 0
        self.num_frames = 0
        while self.num_frames < self.config.total_frames:
            # sample and load into gpu
            iter_start = time.time()
            self.buffer.get(self.shm_tensor_dict)
            sample_time = time.time() - iter_start

            self.num_frames += int(self.config.chunk_len * self.config.batch_size)
            self.global_step += 1
            '''
                update !
            '''
            start = time.time()
            loss_stat = self.train_learner_on_batch()
            optimize_time = time.time() - start

            # broadcast updated learner parameters to each subprocess
            # then subprocess uploads them into remote parameter server
            if self.global_step % self.config.push_period == 0:
                new_weights = self.learner.get_weights()
                for k, v in new_weights.items():
                    self.global_weights[k].copy_(v)
                self.weights_available.copy_(torch.tensor(1))

            iter_dur = time.time() - iter_start

            return_stat = {k: v.item() for k, v in self.ep_info_dict.items()}
            print("----------------------------------------------")
            print(("Global Step: {}, Frames: {}, " + "Average Return: {:.2f}, " + "Sample Time: {:.2f}s, " +
                   "Optimization Time: {:.2f}s, " + "Iteration Step Time: {:.2f}s").format(
                       self.global_step, self.num_frames, return_stat['ep_return/avg'], sample_time, optimize_time,
                       iter_dur))
            print("----------------------------------------------")

            if not self.config.no_summary:
                self.summary(loss_stat, return_stat, sample_time, optimize_time, iter_dur)
        self.supervisor.terminate()

    def summary(self, loss_stat, return_stat, sample_time, optimize_time, iter_dur):
        # collect statistics to record
        return_stat = {'ep_return/' + k: v for k, v in return_stat.items()}
        loss_stat = {'loss/' + k: v for k, v in loss_stat.items()}
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
        wandb.log({
            **return_stat,
            **loss_stat,
            **time_stat,
            **buffer_stat,
        }, step=self.num_frames)
