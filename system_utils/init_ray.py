import ray
from numpy import ceil


def initialize_single_machine_ray(config):
    worker_cpus = config.cpu_per_worker * config.num_workers
    postprocessor_cpus = config.num_postprocessors
    ps_recorder_cpus = 1
    main_process_cpus = 1
    num_cpus = worker_cpus + ps_recorder_cpus + main_process_cpus + postprocessor_cpus
    ray.init(num_cpus=num_cpus)
    print("Ray utilized cpu number: {}".format(num_cpus))


def initialize_ray_on_supervisor(config):
    """initialize ray on supervisor, which is a subprocess to decouple sampling and optimization

    Args:
        supervisor_id (int): to determine dashboard port
        config (wandb.config): configs
    """
    worker_cpus = config.cpu_per_worker * config.num_workers
    # postprocessor_cpus = config.num_postprocessors // config.num_supervisors
    ps_recorder_cpus = 1
    main_process_cpus = 1
    ray2proc_sender_cpus = int(ceil(config.num_writers * 0.5))
    num_cpus = worker_cpus + ps_recorder_cpus + main_process_cpus + ray2proc_sender_cpus  # + postprocessor_cpus
    # 1.5GB object store memory per worker, empirically can't use that much
    if config.cluster:
        ray.init(address='auto', include_dashboard=config.ray_dashboard)
    else:
        ray.init(num_cpus=num_cpus, include_dashboard=config.ray_dashboard)
        print("Ray utilized cpu number: {}".format(num_cpus))
