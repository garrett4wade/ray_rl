import ray
import nvgpu


def initialize_single_machine_ray(config):
    worker_cpus = config.cpu_per_worker * config.num_workers
    postprocessor_cpus = config.num_postprocessors
    ps_recorder_cpus = 1
    main_process_cpus = 1
    num_cpus = worker_cpus + ps_recorder_cpus + main_process_cpus + postprocessor_cpus
    num_gpus = len(nvgpu.available_gpus())
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    print("Ray utilized cpu number: {}, gpu number: {}".format(num_cpus, num_gpus))
