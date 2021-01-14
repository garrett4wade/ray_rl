import ray


def initialize_single_machine_ray(config):
    worker_cpus = config.cpu_per_worker * config.num_workers
    postprocessor_cpus = config.num_postprocessors
    ps_recorder_cpus = 1
    main_process_cpus = 1
    num_cpus = worker_cpus + ps_recorder_cpus + main_process_cpus + postprocessor_cpus
    ray.init(num_cpus=num_cpus)
    print("Ray utilized cpu number: {}".format(num_cpus))


def initialize_ray_on_supervisor(supervisor_id, kwargs):
    """initialize ray on supervisor, which is a subprocess to decouple sampling and optimization

    Args:
        supervisor_id (int): to determine dashboard port
        kwargs (dict): configs
    """
    worker_cpus = kwargs['cpu_per_worker'] * kwargs['num_workers'] // kwargs['num_supervisors']
    # postprocessor_cpus = kwargs['num_postprocessors'] // kwargs['num_supervisors']
    ps_recorder_cpus = 1
    main_process_cpus = 1
    num_cpus = worker_cpus + ps_recorder_cpus + main_process_cpus  # + postprocessor_cpus
    # 1.5GB object store memory per worker, empirically can't use that much
    dashboard_kwargs = {}
    if kwargs['ray_dashboard']:
        dashboard_kwargs['dashboard_port'] = int(8265 + supervisor_id)
    else:
        dashboard_kwargs['include_dashboard'] = False
    if kwargs['cluster']:
        ray.init(address='auto', **dashboard_kwargs)
    else:
        ray.init(num_cpus=num_cpus,
                 object_store_memory=int(1.5 * 1024**3 * kwargs['num_workers'] // kwargs['num_supervisors']),
                 **dashboard_kwargs)
        print("Ray utilized cpu number: {}".format(num_cpus))
