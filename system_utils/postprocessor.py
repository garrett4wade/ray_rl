import time
import numpy as np
import ray
from collections import namedtuple
from scipy.signal import lfilter
from system_utils.worker import RawdataCollector


@ray.remote(num_cpus=1, resources={'head': 1})
class PostProcessor:
    def __init__(self, processor_id, num_workers, rollout_keys, collect_keys, model_fn, worker_env_fn, ps, config):
        """(remote) postprocessor of rollouts (e.g. compute GAE), however slow down performance, tentatively archived
        """
        self.rawdata_collector = RawdataCollector.remote(num_workers, model_fn, worker_env_fn, ps, config)

        self.rollout_keys = rollout_keys
        self.collect_keys = collect_keys
        self.gamma = config.gamma
        self.lmbda = config.lmbda
        self.chunk_len = config.chunk_len
        self.CollectSegCls = namedtuple('CollectSeg', self.collect_keys)

        self._generator = self._data_generator()

    def _data_generator(self):
        job = self.rawdata_collector.get_sample_ids.remote()
        while True:
            segs, infos, wait_time = ray.get(job)
            job = self.rawdata_collector.get_sample_ids.remote()
            collect_segs = [self.postprocess(seg) for seg in segs]
            merged_seg = self.CollectSegCls(*[
                np.concatenate([getattr(collect_seg, k) for collect_seg in collect_segs], axis=1)
                for k in self.CollectSegCls._fields
            ])
            yield merged_seg, infos, wait_time

    def postprocess(self, seg):
        adv = self.compute_gae(seg.reward, seg.value)
        value_target = self.n_step_return(seg.reward, seg.value)
        data_batch = {}
        for k in self.collect_keys:
            if k in self.rollout_keys:
                data_batch[k] = getattr(seg, k)
            elif k == 'adv':
                data_batch[k] = adv
            elif k == 'value_target':
                data_batch[k] = value_target
            elif k == 'pad_mask':
                data_batch[k] = np.ones(len(adv), dtype=np.uint8)
        return self.to_chunk(data_batch)

    def to_chunk(self, data_batch):
        chunk_num = int(np.ceil(len(data_batch['adv']) / self.chunk_len))
        target_len = chunk_num * self.chunk_len
        chunks = {}
        for k, v in data_batch.items():
            if 'rnn_hidden' in k:
                indices = np.arange(chunk_num) * self.chunk_len
                chunks[k] = np.swapaxes(v[indices], 1, 0)
            else:
                if len(v) < target_len:
                    pad = tuple([(0, target_len - len(v))] + [(0, 0)] * (v.ndim - 1))
                    pad_v = np.pad(v, pad, 'constant', constant_values=0)
                else:
                    pad_v = v
                chunks[k] = np.swapaxes(pad_v.reshape(chunk_num, self.chunk_len, *v.shape[1:]), 1, 0)
        return self.CollectSegCls(**chunks)

    def compute_gae(self, reward, value):
        assert reward.ndim == 1 and value.ndim == 1
        bootstrapped_v = np.concatenate([value[1:], np.zeros(1, dtype=np.float32)])
        one_step_td = reward + self.gamma * bootstrapped_v - value
        adv = lfilter([1], [1, -self.lmbda * self.gamma], one_step_td[::-1])[::-1]
        return adv

    def n_step_return(self, reward, value, bootstrap_step=None):
        # compute n-step return given full episode data
        assert reward.ndim == 1 and value.ndim == 1
        bootstrap_step = min(self.chunk_len, len(reward)) if bootstrap_step is None else bootstrap_step
        if bootstrap_step == np.inf:
            # monte carlo return
            return lfilter([1], [1, -self.gamma], reward[::-1])[::-1]
        else:
            # n-step TD return
            bootstrap_step = min(self.chunk_len, len(reward)) if bootstrap_step is None else min(
                bootstrap_step, len(reward))
            discounted_r = lfilter(self.gamma**np.arange(self.chunk_len), [1], reward[::-1])[::-1]
            bootstrapped_v = np.concatenate([value[bootstrap_step:], np.zeros(bootstrap_step, dtype=np.float32)])
            return discounted_r + self.gamma**bootstrap_step * bootstrapped_v

    @ray.method(num_returns=3)
    def get(self):
        return next(self._generator)


class RolloutCollector:
    def __init__(self, rollout_keys, collect_keys, model_fn, worker_env_fn, ps, config):
        self.total_workers = config.num_workers
        self.total_postprocessors = config.num_postprocessors
        self.worker_num_allocation = self.allocate_worker()
        self.postprocessors = [
            PostProcessor.remote(processor_id=i,
                                 num_workers=self.worker_num_allocation[i],
                                 rollout_keys=rollout_keys,
                                 collect_keys=collect_keys,
                                 model_fn=model_fn,
                                 worker_env_fn=worker_env_fn,
                                 ps=ps,
                                 config=config) for i in range(self.total_postprocessors)
        ]
        self.working_jobs = []
        # job_hashing maps job id to worker index
        self.job_hashing = {}

        self._data_id_g = self._data_id_generator()
        print("---------------------------------------------------")
        print("              Workers starting ......              ")
        print("---------------------------------------------------")

    def _start(self):
        for i in range(self.total_postprocessors):
            sample_job, info_job, worker_wait_time_job = self.postprocessors[i].get.remote()
            self.working_jobs.append(sample_job)
            self.job_hashing[sample_job] = (i, info_job, worker_wait_time_job)

    def _data_id_generator(self):
        # iteratively make worker active
        self._start()
        while True:
            start = time.time()
            [ready_sample_job], self.working_jobs = ray.wait(self.working_jobs, num_returns=1)
            wait_time = time.time() - start

            worker_id, ready_info_job, worker_wait_time_job = self.job_hashing[ready_sample_job]
            self.job_hashing.pop(ready_sample_job)

            new_sample_job, new_info_job, new_worker_wait_time_job = self.postprocessors[worker_id].get.remote()
            self.working_jobs.append(new_sample_job)
            self.job_hashing[new_sample_job] = (worker_id, new_info_job, new_worker_wait_time_job)
            yield ready_sample_job, ready_info_job, worker_wait_time_job, wait_time

    def get_sample_ids(self):
        return next(self._data_id_g)

    def allocate_worker(self):
        allocation = [self.total_workers // self.total_postprocessors] * self.total_postprocessors
        for i in range(self.total_workers % self.total_postprocessors):
            allocation[i] += 1
        assert sum(allocation) == self.total_workers
        return allocation
