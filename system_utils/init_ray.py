import ray


def initialize_ray_on_supervisor(config):
    """initialize ray on supervisor, which is a subprocess, to decouple sampling and optimization
    """
    num_cpus = config.cpu_per_worker * config.num_workers + 1
    # 1.5GB object store memory per worker, empirically can't use that much
    if config.cluster:
        ray.init(address='auto', include_dashboard=config.ray_dashboard)
    else:
        ray.init(num_cpus=num_cpus, include_dashboard=config.ray_dashboard, resources={'head': 1000})
        print("Ray utilized cpu number: {}".format(num_cpus))
