import ray
import time


@ray.remote(num_cpus=1)
class RolloutWorker:
    def __init__(self, worker_id, model_fn, worker_env_fn, ps, config):
        self.id = worker_id
        self.verbose = config.verbose

        self.env = worker_env_fn(worker_id, config)
        self.model = model_fn(config)

        self.ps = ps
        self.weight_hash = None
        self.get_new_weights()

        self._data_g = self._data_generator()

    def get_new_weights(self):
        # if model parameters in parameter server is different from local model, download it
        weights_job = self.ps.get_weights.remote()
        weights, weights_hash = ray.get(weights_job)
        self.model.load_state_dict(weights)
        old_hash = self.weight_hash
        self.weight_hash = weights_hash
        if self.verbose:
            print("RolloutWorker {} load state dict, "
                  "hashing changed from "
                  "{} to {}".format(self.id, old_hash, self.weight_hash))
        del weights_job, weights, weights_hash

    def _data_generator(self):
        self.pull_job = self.ps.pull.remote()
        model_inputs = self.env.get_model_inputs()
        while True:
            model_outputs = self.model.select_action(*model_inputs)
            datas, infos, model_inputs = self.env.step(*model_outputs)
            if len(datas) == 0 or len(infos) == 0:
                continue
            yield datas, infos
            # get new weights only when at least one of vectorized envs is done
            if ray.get(self.pull_job) != self.weight_hash:
                self.get_new_weights()
            self.pull_job = self.ps.pull.remote()

    @ray.method(num_returns=2)
    def get(self):
        return next(self._data_g)


@ray.remote(num_cpus=0, resources={'head': 1})
class RolloutCollector:
    def __init__(self, collector_id, model_fn, worker_env_fn, ps, config):
        self.id = collector_id
        self.num_returns = 1
        self.num_workers = config.num_workers // config.num_collectors
        assert config.num_workers % config.num_collectors == 0
        self.workers = [
            RolloutWorker.remote(worker_id=i + collector_id * config.num_collectors,
                                 model_fn=model_fn,
                                 worker_env_fn=worker_env_fn,
                                 ps=ps,
                                 config=config) for i in range(self.num_workers)
        ]
        self.working_jobs = []
        # job_hashing maps job id to worker index
        self.job_hashing = {}

        self._data_id_g = self._data_id_generator()
        print("---------------------------------------------------")
        print("   Starting workers in rollout collector {} ......  ".format(self.id))
        print("---------------------------------------------------")

    def _start(self):
        for i in range(self.num_workers):
            sample_job, info_job = self.workers[i].get.remote()
            self.working_jobs.append(sample_job)
            self.job_hashing[sample_job] = (i, info_job)

    def _data_id_generator(self):
        # iteratively make worker active
        self._start()
        while True:
            start = time.time()
            [ready_sample_job], self.working_jobs = ray.wait(self.working_jobs, num_returns=self.num_returns)
            wait_time = time.time() - start

            worker_id, ready_info_job = self.job_hashing[ready_sample_job]
            self.job_hashing.pop(ready_sample_job)

            new_sample_job, new_info_job = self.workers[worker_id].get.remote()
            self.working_jobs.append(new_sample_job)
            self.job_hashing[new_sample_job] = (worker_id, new_info_job)
            yield ray.get(ready_sample_job), ray.get(ready_info_job), wait_time

    def get_sample_ids(self):
        return next(self._data_id_g)
