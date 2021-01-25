import ray
from numpy import ceil


def initialize_ray_on_supervisor(config):
    """initialize ray on supervisor, which is a subprocess, to decouple sampling and optimization
    """
    worker_cpus = config.cpu_per_worker * config.num_workers
    # postprocessor_cpus = config.num_postprocessors // config.num_supervisors
    ps_recorder_cpus = 1
    ctrl_process_cpus = 1
    ray2proc_sender_cpus = int(ceil(config.num_writers * 0.2))
    num_cpus = worker_cpus + ps_recorder_cpus + ctrl_process_cpus + ray2proc_sender_cpus  # + postprocessor_cpus
    # 1.5GB object store memory per worker, empirically can't use that much
    if config.cluster:
        ray.init(address='auto', include_dashboard=config.ray_dashboard)
    else:
        ray.init(num_cpus=num_cpus, include_dashboard=config.ray_dashboard, resources={'head': 1000})
        print("Ray utilized cpu number: {}".format(num_cpus))
