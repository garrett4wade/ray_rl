import torch
from system_utils.simulation_supervisor import SimulationSupervisor
from rl_utils.buffer import SharedCircularBuffer


class RolloutRunner:
    def __init__(
        self,
        init_weights,
        worker_model_fn,
        worker_env_fn,
        rollout_keys,
        collect_keys,
        ep_info_keys,
        shapes,
        dtypes,
        config,
    ):
        # initialize buffer
        buffer_maxsize = config.batch_size * config.q_size
        self.buffer = SharedCircularBuffer(buffer_maxsize, config.chunk_len, config.reuse_times, shapes, dtypes,
                                           config.num_gpus, config.batch_size)
        self.shapes = shapes
        self.dtypes = dtypes
        self.rollout_keys = rollout_keys
        self.collect_keys = collect_keys
        self.worker_env_fn = worker_env_fn
        self.worker_model_fn = worker_model_fn
        self.config = config
        # shared memory tensor for summary
        self.queue_util = torch.tensor(0.0).share_memory_()
        self.wait_time = torch.tensor(0.0).share_memory_()
        self.ep_info_dict = {
            k + '/' + stat_k: torch.tensor(0.0).share_memory_()
            for k in ep_info_keys for stat_k in ['avg', 'min', 'max']
        }

        self.global_weights = init_weights
        for v in self.global_weights.values():
            v.share_memory_()
        self.weights_available = torch.tensor(1).share_memory_()
        self.supervisor = SimulationSupervisor(rollout_keys=self.rollout_keys,
                                               collect_keys=self.collect_keys,
                                               model_fn=self.worker_model_fn,
                                               worker_env_fn=self.worker_env_fn,
                                               global_buffer=self.buffer,
                                               weights=self.global_weights,
                                               weights_available=self.weights_available,
                                               ep_info_dict=self.ep_info_dict,
                                               queue_util=self.queue_util,
                                               wait_time=self.wait_time,
                                               config=self.config)

    def run(self):
        self.supervisor.start()

    def shutdown(self):
        self.supervisor.terminate()
