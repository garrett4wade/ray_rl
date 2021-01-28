import ray
from numpy import ceil


def initialize_ray_on_supervisor(config):
    """initialize ray on supervisor, which is a subprocess, to decouple sampling and optimization
    """
    worker_cpus = config.cpu_per_worker * config.num_workers
    # postprocessor_cpus = config.num_postprocessors // config.num_supervisors
    ctrl_process_cpus = 1
    ps_sender_cpus = int(ceil(config.num_writers * 0.2 + 0.5))
    num_cpus = worker_cpus + ctrl_process_cpus + ps_sender_cpus  # + postprocessor_cpus
    # 1.5GB object store memory per worker, empirically can't use that much
    if config.cluster:
        ray.init(address='auto', include_dashboard=config.ray_dashboard)
    else:
        ray.init(num_cpus=num_cpus, include_dashboard=config.ray_dashboard, resources={'head': 1000})
        print("Ray utilized cpu number: {}".format(num_cpus))
