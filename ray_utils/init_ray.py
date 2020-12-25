import ray
import nvgpu
from numpy import ceil


def initialize_single_machine_ray(config):
    # initialize ray
    # additional 2 cpus are for parameter server & main script respectively
    worker_cpus = config.cpu_per_worker * config.num_workers
    supervisor_cpus = config.num_supervisors * 2
    collector_cpus = config.num_collectors
    ps_rtrecorder_cpus = 1
    buffer_cpus = 2
    main_process_cpus = 1
    num_cpus = int(
        ceil(worker_cpus + supervisor_cpus + collector_cpus + ps_rtrecorder_cpus + buffer_cpus + main_process_cpus))
    num_gpus = len(nvgpu.available_gpus())
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    print("Ray utilized cpu number: {}, gpu number: {}".format(num_cpus, num_gpus))
