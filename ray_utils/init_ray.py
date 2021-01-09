import ray
# import nvgpu


def initialize_single_machine_ray(config):
    worker_cpus = config.cpu_per_worker * config.num_workers
    postprocessor_cpus = config.num_postprocessors
    ps_recorder_cpus = 1
    main_process_cpus = 1
    num_cpus = worker_cpus + ps_recorder_cpus + main_process_cpus + postprocessor_cpus
    # num_gpus = len(nvgpu.available_gpus())
    ray.init(num_cpus=num_cpus)
    print("Ray utilized cpu number: {}".format(num_cpus))


def initialize_single_machine_ray_on_supervisor(supervisor_id, kwargs):
    with open('/proc/meminfo') as fd:
        for line in fd:
            if line.startswith('MemAvailable'):
                mem_avail = int(line.split()[1])
                break
    worker_cpus = kwargs['cpu_per_worker'] * kwargs['num_workers'] // kwargs['num_supervisors']
    postprocessor_cpus = kwargs['num_postprocessors'] // kwargs['num_supervisors']
    ps_recorder_cpus = 1
    main_process_cpus = 1
    num_cpus = worker_cpus + ps_recorder_cpus + main_process_cpus + postprocessor_cpus
    # num_gpus = len(nvgpu.available_gpus())
    ray.init(num_cpus=num_cpus,
             dashboard_port=8265 + supervisor_id,
             object_store_memory=mem_avail * 1024 // kwargs['num_supervisors'])
    print("Ray utilized cpu number: {}".format(num_cpus))
