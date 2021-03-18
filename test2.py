import numpy as np
import gym
import time
from scipy.signal import lfilter
from env.mujoco.registry import get_shapes, ROLLOUT_KEYS, COLLECT_KEYS, DTYPES, Seg, Info
ROLLOUT_KEYS = ['obs', 'action', 'action_logits', 'value', 'reward', 'pad_mask']
COLLECT_KEYS = ['obs', 'action', 'action_logits', 'value', 'adv', 'value_target', 'pad_mask']


class EnvWithMemory:
    def __init__(self, env_id, env_fn, rollout_buffer, config):
        self.id = env_id
        self.env = env_fn(config)
        self.rollout_buffer = rollout_buffer
        self.config = config

        self.chunk_len = config.chunk_len
        self.step_size = self.chunk_len
        self.min_return_chunk_num = config.min_return_chunk_num

        self.gamma = config.gamma
        self.lmbda = config.lmbda
        self.max_timesteps = config.max_timesteps

        self.verbose = config.verbose
        self.reset()

    def _preprocess(self, obs):
        return obs.astype(np.float32)

    def _get_model_input(self):
        return (self.obs, )

    def reset(self):
        if self.verbose and self.done:
            print("Episode End: Step {}, Return {}".format(self.ep_step, self.ep_return))
        self.ep_step = self.ep_return = 0
        self.rollout_buffer.obs[self.id, 0] = self._preprocess(self.env.reset())
        self.rollout_buffer.pad_mask[self.id, 0] = 1

    def step(self):
        act = self.rollout_buffer.action[self.id, self.ep_step]
        n_obs, r, d, _ = self.env.step(act)

        self.rollout_buffer.reward[self.id, self.ep_step] = r
        self.ep_return += r
        self.ep_step += 1

        done = d or self.ep_step >= self.max_timesteps

        if not done:
            self.rollout_buffer.obs[self.id, self.ep_step] = self._preprocess(n_obs)
            self.rollout_buffer.pad_mask[self.id, self.ep_step] = 1 - d
            return [], [], self._get_model_input()
        else:
            seg = Seg(*[getattr(self.rollout_buffer, k)[self.id, :self.ep_step] for k in ROLLOUT_KEYS])
            self.history_ep_datas.append(seg)
            self.history_ep_infos.append(Info(ep_return=self.ep_return))
            self.stored_chunk_num += int(np.ceil(self.ep_step / self.chunk_len))
            self.reset()
            if self.stored_chunk_num >= self.min_return_chunk_num:
                datas = self.history_ep_datas.copy()
                infos = self.history_ep_infos.copy()
                self.history_ep_datas = []
                self.history_ep_infos = []
                self.stored_chunk_num = 0
                return datas, infos, self._get_model_input()
            else:
                return [], [], self._get_model_input()

    # def collect(self):
    #     v_target, adv = self.compute_gae()
    #     data_batch = {}
    #     for k in COLLECT_KEYS:
    #         if k in ROLLOUT_KEYS:
    #             data_batch[k] = np.stack(self.history[k], axis=0).astype(DTYPES[k])
    #         elif k == 'value_target':
    #             data_batch[k] = v_target.astype(DTYPES[k])
    #         elif k == 'adv':
    #             data_batch[k] = adv.astype(DTYPES[k])
    #     return self.to_chunk(data_batch)

    # def to_chunk(self, data_batch):
    #     chunk_num = int(np.ceil(self.ep_step / self.chunk_len))
    #     target_len = chunk_num * self.chunk_len
    #     chunks = {}
    #     for k, v in data_batch.items():
    #         if 'rnn_hidden' in k:
    #             indices = np.arange(chunk_num) * self.chunk_len
    #             chunks[k] = np.swapaxes(v[indices], 1, 0)
    #         else:
    #             if len(v) < target_len:
    #                 pad = tuple([(0, target_len - len(v))] + [(0, 0)] * (v.ndim - 1))
    #                 pad_v = np.pad(v, pad, 'constant', constant_values=0)
    #             else:
    #                 pad_v = v
    #             chunks[k] = np.swapaxes(pad_v.reshape(chunk_num, self.chunk_len, *v.shape[1:]), 1, 0)
    #     self.stored_chunk_num += chunk_num
    #     return Seg(**chunks)

    # def compute_gae(self, bootstrap_step=np.inf):
    #     reward = np.array(self.history['reward'], dtype=np.float32)
    #     value = np.array(self.history['value'], dtype=np.float32)
    #     assert reward.ndim == 1 and value.ndim == 1
    #     bootstrap_v = np.array(self.history['value'][1:] + [0], dtype=np.float32)
    #     one_step_td = reward + self.gamma * bootstrap_v - value
    #     adv = lfilter([1], [1, -self.lmbda * self.gamma], one_step_td[::-1])[::-1]

    #     if bootstrap_step >= self.ep_step:
    #         # monte carlo return
    #         v_target = lfilter([1], [1, -self.gamma], reward[::-1])[::-1]
    #     else:
    #         # n-step TD return
    #         discounted_r = lfilter(self.gamma**np.arange(self.chunk_len), [1], reward[::-1])[::-1]
    #         bootstrapped_v = np.concatenate([value[bootstrap_step:], np.zeros(bootstrap_step, dtype=np.float32)])
    #         v_target = discounted_r + self.gamma**bootstrap_step * bootstrapped_v
    #     return v_target, adv


class VecEnvWithMemory:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]

    def step(self):
        datas, infos, model_inputs = [], [], []
        for i, env in enumerate(self.envs):
            cur_datas, cur_infos, model_input = env.step()
            datas += cur_datas
            infos += cur_infos
            model_inputs.append(model_input)
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        # if len(datas) > 0:
        #     seg = Seg(*[np.concatenate([getattr(x, k) for x in datas], axis=1) for k in Seg._fields])
        # else:
        #     seg = ()
        return datas, infos, stacked_model_inputs

    def get_model_inputs(self):
        model_inputs = [env._get_model_input() for env in self.envs]
        stacked_model_inputs = []
        for inp in zip(*model_inputs):
            stacked_model_inputs.append(np.stack(inp))
        return stacked_model_inputs

    def close(self):
        for env in self.envs:
            env.close()


def build_simple_env(config, seed=0):
    env = gym.make('Ant-v2')
    env.seed(seed)
    return env


def build_worker_env(worker_id, rollout_buffer, slc, config):
    env_fns = []
    for i in range(config.env_num):
        env_id = worker_id + i * config.num_workers
        seed = 12345 * env_id + config.seed
        env_fns.append(lambda: EnvWithMemory(env_id + config.num_workers * worker_id, lambda config: build_simple_env(
            config, seed=seed), rollout_buffer, config))
    return VecEnvWithMemory(env_fns)


class RolloutWorker:
    def __init__(self, worker_id, rollout_buffer, worker_env_fn, act_ready, model_input_queue, config):
        self.id = worker_id
        self.rollout_buffer = rollout_buffer
        self.slc = slice(self.id * config.env_num, (self.id + 1) * config.env_num)
        self.verbose = config.verbose
        self.env = worker_env_fn(worker_id, rollout_buffer, self.slc, config)
        self.act_ready = act_ready
        self.model_input_queue = model_input_queue

        self._data_g = self._data_generator()

    def _data_generator(self):
        self.model_input_queue.put((0, self.slc, self.env._get_model_inputs()))
        while True:
            self.act_ready.acquire()
            datas, infos, model_inputs = self.env.step()
            if len(datas) > 0 and len(infos) > 0:
                yield datas, infos
            self.model_input_queue.put((self.env.ep_step, self.slc, model_inputs))

    def get(self):
        return next(self._data_g)

class ActionServer:
    def __init__(self, model_fn, rollout_buffer, act_readies, model_input_queue, config):
        self.model = model_fn(config)
        self.rollout_buffer = rollout_buffer
        self.act_readies = act_readies
        self.model_input_queue = model_input_queue
        self.config = config
    
    def serve_actions(self):
        while True:
            ep_t, worker_slc, model_input = model_input_queue.get()
            action, action_logits, value = model.select_action(*model_input)
            self.rollout_buffer.action[worker_slc, ep_t] = action
            self.rollout_buffer.action_logits[worker_slc, ep_t] = action_logits
            self.rollout_buffer.value[worker_slc, ep_t] = value
@ray.remote(num_cpus=0, resources={'head': 1})
class RolloutCollector:
    def __init__(self, collector_id, model_fn, worker_env_fn, ps, config):
        self.id = collector_id
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
            [ready_sample_job], self.working_jobs = ray.wait(self.working_jobs, num_returns=1)
            wait_time = time.time() - start

            worker_id, ready_info_job = self.job_hashing[ready_sample_job]
            self.job_hashing.pop(ready_sample_job)

            new_sample_job, new_info_job = self.workers[worker_id].get.remote()
            self.working_jobs.append(new_sample_job)
            self.job_hashing[new_sample_job] = (worker_id, new_info_job)
            yield ray.get(ready_sample_job), ray.get(ready_info_job), wait_time

    def get_sample_ids(self):
        return next(self._data_id_g)