import multiprocessing as mp


def build_samples_buffer(agent,
                         env,
                         batch_spec,
                         bootstrap_value=False,
                         agent_shared=True,
                         env_shared=True,
                         subprocess=True,
                         examples=None):
    """Recommended to step/reset agent and env in subprocess, so it doesn't
    affect settings in master before forking workers (e.g. torch num_threads
    (MKL) may be set at first forward computation.)"""
    if examples is None:
        if subprocess:
            mgr = mp.Manager()
            examples = mgr.dict()  # Examples pickled back to master.
            w = mp.Process(target=get_example_outputs, args=(agent, env, examples, subprocess))
            w.start()
            w.join()
        else:
            examples = dict()
            get_example_outputs(agent, env, examples)

    T, B = batch_spec
    all_action = buffer_from_example(examples["action"], (T + 1, B), agent_shared)
    action = all_action[1:]
    prev_action = all_action[:-1]  # Writing to action will populate prev_action.
    agent_info = buffer_from_example(examples["agent_info"], (T, B), agent_shared)
    agent_buffer = AgentSamples(
        action=action,
        prev_action=prev_action,
        agent_info=agent_info,
    )
    if bootstrap_value:
        bv = buffer_from_example(examples["agent_info"].value, (1, B), agent_shared)
        agent_buffer = AgentSamplesBsv(*agent_buffer, bootstrap_value=bv)

    observation = buffer_from_example(examples["observation"], (T, B), env_shared)
    all_reward = buffer_from_example(examples["reward"], (T + 1, B), env_shared)
    reward = all_reward[1:]
    prev_reward = all_reward[:-1]  # Writing to reward will populate prev_reward.
    done = buffer_from_example(examples["done"], (T, B), env_shared)
    env_info = buffer_from_example(examples["env_info"], (T, B), env_shared)
    env_buffer = EnvSamples(
        observation=observation,
        reward=reward,
        prev_reward=prev_reward,
        done=done,
        env_info=env_info,
    )
    samples_np = Samples(agent=agent_buffer, env=env_buffer)
    samples_pyt = torchify_buffer(samples_np)
    return samples_pyt, samples_np, examples


def get_example_outputs(agent, env, examples, subprocess=False):
    """Do this in a sub-process to avoid setup conflict in master/workers (e.g.
    MKL)."""
    if subprocess:  # i.e. in subprocess.
        import torch
        torch.set_num_threads(1)  # Some fix to prevent MKL hang.
    o = env.reset()
    a = env.action_space.sample()
    o, r, d, env_info = env.step(a)
    r = np.asarray(r, dtype="float32")  # Must match torch float dtype here.
    agent.reset()
    agent_inputs = torchify_buffer(AgentInputs(o, a, r))
    a, agent_info = agent.step(*agent_inputs)
    if "prev_rnn_state" in agent_info:
        # Agent leaves B dimension in, strip it: [B,N,H] --> [N,H]
        agent_info = agent_info._replace(prev_rnn_state=agent_info.prev_rnn_state[0])
    examples["observation"] = o
    examples["reward"] = r
    examples["done"] = d
    examples["env_info"] = env_info
    examples["action"] = a  # OK to put torch tensor here, could numpify.
    examples["agent_info"] = agent_info


class BaseCollector:
    """Class that steps environments, possibly in worker process."""
    def __init__(
        self,
        rank,
        envs,
        samples_np,
        batch_T,
        TrajInfoCls,
        agent=None,  # Present or not, depending on collector class.
        sync=None,
        step_buffer_np=None,
        global_B=1,
        env_ranks=None,
    ):
        save__init__args(locals())

    def start_envs(self):
        """e.g. calls reset() on every env."""
        raise NotImplementedError

    def start_agent(self):
        """In CPU-collectors, call ``agent.collector_initialize()`` e.g. to set up
        vector epsilon-greedy, and reset the agent.
        """
        if getattr(self, "agent", None) is not None:  # Not in GPU collectors.
            self.agent.collector_initialize(
                global_B=self.global_B,  # Args used e.g. for vector epsilon greedy.
                env_ranks=self.env_ranks,
            )
            self.agent.reset()
            self.agent.sample_mode(itr=0)

    def collect_batch(self, agent_inputs, traj_infos):
        """Main data collection loop."""
        raise NotImplementedError

    def reset_if_needed(self, agent_inputs):
        """Reset agent and or env as needed, if doing between batches."""
        pass


class DecorrelatingStartCollector(BaseCollector):
    """Collector which can step all environments through a random number of random
    actions during startup, to decorrelate the states in training batches.
    """
    def start_envs(self, max_decorrelation_steps=0):
        """Calls ``reset()`` on every environment instance, then steps each
        one through a random number of random actions, and returns the
        resulting agent_inputs buffer (`observation`, `prev_action`,
        `prev_reward`)."""
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, obs in enumerate(observations):
            observation[b] = obs  # numpy array or namedarraytuple
        prev_action = np.stack([env.action_space.null_value() for env in self.envs])
        prev_reward = np.zeros(len(self.envs), dtype="float32")
        if self.rank == 0:
            logger.log("Sampler decorrelating envs, max steps: " f"{max_decorrelation_steps}")
        if max_decorrelation_steps != 0:
            for b, env in enumerate(self.envs):
                n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)
                for _ in range(n_steps):
                    a = env.action_space.sample()
                    o, r, d, info = env.step(a)
                    traj_infos[b].step(o, a, r, d, None, info)
                    if getattr(info, "traj_done", d):
                        o = env.reset()
                        traj_infos[b] = self.TrajInfoCls()
                    if d:
                        a = env.action_space.null_value()
                        r = 0
                observation[b] = o
                prev_action[b] = a
                prev_reward[b] = r
        # For action-server samplers.
        if hasattr(self, "step_buffer_np") and self.step_buffer_np is not None:
            self.step_buffer_np.observation[:] = observation
            self.step_buffer_np.action[:] = prev_action
            self.step_buffer_np.reward[:] = prev_reward
        return AgentInputs(observation, prev_action, prev_reward), traj_infos


def initialize_worker(rank, seed=None, cpu=None, torch_threads=None):
    """Assign CPU affinity, set random seed, set torch_threads if needed to
    prevent MKL deadlock.
    """
    log_str = f"Sampler rank {rank} initialized"
    cpu = [cpu] if isinstance(cpu, int) else cpu
    p = psutil.Process()
    try:
        if cpu is not None:
            p.cpu_affinity(cpu)
        cpu_affin = p.cpu_affinity()
    except AttributeError:
        cpu_affin = "UNAVAILABLE MacOS"
    log_str += f", CPU affinity {cpu_affin}"
    torch_threads = (1 if torch_threads is None and cpu is not None else torch_threads
                     )  # Default to 1 to avoid possible MKL hang.
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    log_str += f", Torch threads {torch.get_num_threads()}"
    if seed is not None:
        set_seed(seed)
        time.sleep(0.3)  # (so the printing from set_seed is not intermixed)
        log_str += f", Seed {seed}"
    logger.log(log_str)


def sampling_process(common_kwargs, worker_kwargs):
    """Target function used for forking parallel worker processes in the
    samplers. After ``initialize_worker()``, it creates the specified number
    of environment instances and gives them to the collector when
    instantiating it.  It then calls collector startup methods for
    environments and agent.  If applicable, instantiates evaluation
    environment instances and evaluation collector.

    Then enters infinite loop, waiting for signals from master to collect
    training samples or else run evaluation, until signaled to exit.
    """
    c, w = AttrDict(**common_kwargs), AttrDict(**worker_kwargs)
    initialize_worker(w.rank, w.seed, w.cpus, c.torch_threads)
    envs = [c.EnvCls(**c.env_kwargs) for _ in range(w.n_envs)]
    set_envs_seeds(envs, w.seed)

    collector = c.CollectorCls(
        rank=w.rank,
        envs=envs,
        samples_np=w.samples_np,
        batch_T=c.batch_T,
        TrajInfoCls=c.TrajInfoCls,
        agent=c.get("agent", None),  # Optional depending on parallel setup.
        sync=w.get("sync", None),
        step_buffer_np=w.get("step_buffer_np", None),
        global_B=c.get("global_B", 1),
        env_ranks=w.get("env_ranks", None),
    )
    agent_inputs, traj_infos = collector.start_envs(c.max_decorrelation_steps)
    collector.start_agent()

    if c.get("eval_n_envs", 0) > 0:
        eval_envs = [c.EnvCls(**c.eval_env_kwargs) for _ in range(c.eval_n_envs)]
        set_envs_seeds(eval_envs, w.seed)
        eval_collector = c.eval_CollectorCls(
            rank=w.rank,
            envs=eval_envs,
            TrajInfoCls=c.TrajInfoCls,
            traj_infos_queue=c.eval_traj_infos_queue,
            max_T=c.eval_max_T,
            agent=c.get("agent", None),
            sync=w.get("sync", None),
            step_buffer_np=w.get("eval_step_buffer_np", None),
        )
    else:
        eval_envs = list()

    ctrl = c.ctrl
    ctrl.barrier_out.wait()
    while True:
        collector.reset_if_needed(agent_inputs)  # Outside barrier?
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        if ctrl.do_eval.value:
            eval_collector.collect_evaluation(ctrl.itr.value)  # Traj_infos to queue inside.
        else:
            agent_inputs, traj_infos, completed_infos = collector.collect_batch(agent_inputs, traj_infos,
                                                                                ctrl.itr.value)
            for info in completed_infos:
                c.traj_infos_queue.put(info)
        ctrl.barrier_out.wait()

    for env in envs + eval_envs:
        env.close()


class ActionServer:
    """Mixin class with methods for serving actions to worker processes which execute
    environment steps.
    """
    def serve_actions(self, itr):
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        step_np, agent_inputs = self.step_buffer_np, self.agent_inputs

        for t in range(self.batch_spec.T):
            for b in obs_ready:
                b.acquire()  # Workers written obs and rew, first prev_act.
                # assert not b.acquire(block=False)  # Debug check.
            if self.mid_batch_reset and np.any(step_np.done):
                for b_reset in np.where(step_np.done)[0]:
                    step_np.action[b_reset] = 0  # Null prev_action into agent.
                    step_np.reward[b_reset] = 0  # Null prev_reward into agent.
                    self.agent.reset_one(idx=b_reset)
            action, agent_info = self.agent.step(*agent_inputs)
            step_np.action[:] = action  # Worker applies to env.
            step_np.agent_info[:] = agent_info  # Worker sends to traj_info.
            for w in act_ready:
                # assert not w.acquire(block=False)  # Debug check.
                w.release()  # Signal to worker.

        for b in obs_ready:
            b.acquire()
            assert not b.acquire(block=False)  # Debug check.
        if "bootstrap_value" in self.samples_np.agent:
            self.samples_np.agent.bootstrap_value[:] = self.agent.value(*agent_inputs)
        if np.any(step_np.done):  # Reset at end of batch; ready for next.
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action into agent.
                step_np.reward[b_reset] = 0  # Null prev_reward into agent.
                self.agent.reset_one(idx=b_reset)
            # step_np.done[:] = False  # Worker resets at start of next.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.

    def serve_actions_evaluation(self, itr):
        """Similar to ``serve_actions()``.  If a maximum number of eval trajectories
        was specified, keeps track of the number completed and terminates evaluation
        if the max is reached.  Returns a list of completed trajectory-info objects.
        """
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        step_np, step_pyt = self.eval_step_buffer_np, self.eval_step_buffer_pyt
        traj_infos = list()
        self.agent.reset()
        agent_inputs = AgentInputs(step_pyt.observation, step_pyt.action, step_pyt.reward)  # Fixed buffer objects.

        for t in range(self.eval_max_T):
            if t % EVAL_TRAJ_CHECK == 0:  # (While workers stepping.)
                traj_infos.extend(drain_queue(self.eval_traj_infos_queue, guard_sentinel=True))
            for b in obs_ready:
                b.acquire()
                # assert not b.acquire(block=False)  # Debug check.
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action.
                step_np.reward[b_reset] = 0  # Null prev_reward.
                self.agent.reset_one(idx=b_reset)
            action, agent_info = self.agent.step(*agent_inputs)
            step_np.action[:] = action
            step_np.agent_info[:] = agent_info
            if self.eval_max_trajectories is not None and t % EVAL_TRAJ_CHECK == 0:
                self.sync.stop_eval.value = len(traj_infos) >= self.eval_max_trajectories
            for w in act_ready:
                # assert not w.acquire(block=False)  # Debug check.
                w.release()
            if self.sync.stop_eval.value:
                logger.log("Evaluation reach max num trajectories " f"({self.eval_max_trajectories}).")
                break
        if t == self.eval_max_T - 1 and self.eval_max_trajectories is not None:
            logger.log("Evaluation reached max num time steps " f"({self.eval_max_T}).")
        for b in obs_ready:
            b.acquire()  # Workers always do extra release; drain it.
            assert not b.acquire(block=False)  # Debug check.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.

        return traj_infos


class AlternatingSamplerBase(GpuSamplerBase):
    """Twice the standard number of worker processes are forked, and they may
    share CPU resources in pairs.  Environment instances are divided evenly
    among the two sets.  While one set of workers steps their environments,
    the action-server process computes the actions for the other set of
    workers, which are paused until their new actions are ready (this pause
    happens in the GpuSampler).  The two sets of workers alternate in this
    procedure, keeping the CPU maximally busy.  The intention is to hide the
    time to compute actions from the critical path of execution, which can
    provide up to a 2x speed boost in theory, if the environment-step time and
    agent-step time were othewise equal.

    If the latency in computing and returning the agent's action is longer
    than environment stepping, then this alternation might not be faster,
    because it calls agent action selection twice as many times.
    """

    alternating = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.batch_spec.B % 2 == 0, "Need even number for sampler batch_B."

    def initialize(self, agent, *args, **kwargs):
        """Like the super class's ``initialize()``, but creates additional set of
        synchronization and communication objects for the alternate workers."""
        if agent.recurrent and not agent.alternating:
            raise TypeError("If agent is recurrent, must be 'alternating' to use here.")
        elif not agent.recurrent:
            agent.alternating = True  # FF agent doesn't need special class, but tell it so.
        examples = super().initialize(agent, *args, **kwargs)
        self._make_alternating_pairs()
        return examples

    def _make_alternating_pairs(self):
        half_w = self.n_worker // 2  # Half of workers.
        self.half_B = half_B = self.batch_spec.B // 2  # Half of envs.
        self.obs_ready_pair = (self.sync.obs_ready[:half_w], self.sync.obs_ready[half_w:])
        self.act_ready_pair = (self.sync.act_ready[:half_w], self.sync.act_ready[half_w:])
        self.step_buffer_np_pair = (self.step_buffer_np[:half_B], self.step_buffer_np[half_B:])
        self.agent_inputs_pair = (self.agent_inputs[:half_B], self.agent_inputs[half_B:])
        if self.eval_n_envs > 0:
            assert self.eval_n_envs % 2 == 0
            eval_half_B = self.eval_n_envs // 2
            self.eval_step_buffer_np_pair = (self.eval_step_buffer_np[:eval_half_B],
                                             self.eval_step_buffer_np[eval_half_B:])
            self.eval_agent_inputs_pair = (self.eval_agent_inputs[:eval_half_B], self.eval_agent_inputs[eval_half_B:])
        if "bootstrap_value" in self.samples_np.agent:
            self.bootstrap_value_pair = (self.samples_np.agent.bootstrap_value[0, :half_B],
                                         self.samples_np.agent.bootstrap_value[0, half_B:])  # (leading dim T=1)

    def _get_n_envs_list(self, affinity=None, n_worker=None, B=None):
        if affinity is not None:
            assert affinity.get("alternating", False), "Need alternating affinity."
        n_worker = len(affinity["workers_cpus"]) if n_worker is None else n_worker
        assert n_worker % 2 == 0, "Need even number workers."
        B = self.batch_spec.B if B is None else B
        assert B % 2 == 0
        # To log warnings:
        n_envs_list = super()._get_n_envs_list(n_worker=n_worker, B=B)
        if B % n_worker > 0:
            # Redistribute extra envs.
            n_envs_list = [B // n_worker] * n_worker
            for w in range((B % n_worker) // 2):
                n_envs_list[w] += 1
                n_envs_list[w + n_worker // 2] += 1  # Paired worker.
        return n_envs_list


class GpuSamplerBase:
    """Base class for parallel samplers which use worker processes to execute
    environment steps on CPU resources but the master process to execute agent
    forward passes for action selection, presumably on GPU.  Use GPU-based
    collecter classes.

    In addition to the usual batch buffer for data samples, allocates a step
    buffer over shared memory, which is used for communication with workers.
    The step buffer includes `observations`, which the workers write and the
    master reads, and `actions`, which the master write and the workers read.
    (The step buffer has leading dimension [`batch_B`], for the number of 
    parallel environments, and each worker gets its own slice along that
    dimension.)  The step buffer object is held in both numpy array and torch
    tensor forms over the same memory; e.g. workers write to the numpy array
    form, and the agent is able to read the torch tensor form.

    (Possibly more information about how the stepping works, but write
    in action-server or smwr like that.)
    """

    gpu = True
    alternating = False

    def __init__(
            self,
            EnvCls,
            env_kwargs,
            batch_T,
            batch_B,
            CollectorCls,
            max_decorrelation_steps=100,
            TrajInfoCls=TrajInfo,
            eval_n_envs=0,  # 0 for no eval setup.
            eval_CollectorCls=None,  # Must supply if doing eval.
            eval_env_kwargs=None,
            eval_max_steps=None,  # int if using evaluation.
            eval_max_trajectories=None,  # Optional earlier cutoff.
    ):
        eval_max_steps = None if eval_max_steps is None else int(eval_max_steps)
        eval_max_trajectories = (None if eval_max_trajectories is None else int(eval_max_trajectories))
        save__init__args(locals())
        self.batch_spec = BatchSpec(batch_T, batch_B)
        self.mid_batch_reset = CollectorCls.mid_batch_reset

    @property
    def batch_size(self):
        return self.batch_spec.size  # For logging at least.

    def initialize(
        self,
        agent,
        affinity,
        seed,
        bootstrap_value=False,
        traj_info_kwargs=None,
        world_size=1,
        rank=0,
        worker_process=None,
    ):
        """
        Creates an example instance of the environment for agent initialization
        (which may differ by sub-class) and to pre-allocate batch buffers, then deletes
        the environment instance.  Batch buffers are allocated on shared memory, so
        that worker processes can read/write directly to them.

        Computes the number of parallel processes based on the ``affinity``
        argument.  Forks worker processes, which instantiate their own environment
        and collector objects.  Waits for the worker process to complete all initialization
        (such as decorrelating environment states) before returning.  Barriers and other
        parallel indicators are constructed to manage worker processes.
        
        .. warning::
            If doing offline agent evaluation, will use at least one evaluation environment
            instance per parallel worker, which might increase the total
            number of evaluation instances over what was requested.  This may
            result in bias towards shorter episodes if the episode length is
            variable, and if the max number of evalution steps divided over the
            number of eval environments (`eval_max_steps /
            actual_eval_n_envs`), is not large relative to the max episode
            length.
        """
        n_envs_list = self._get_n_envs_list(affinity=affinity)
        self.n_worker = n_worker = len(n_envs_list)
        B = self.batch_spec.B
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        self.world_size = world_size
        self.rank = rank

        if self.eval_n_envs > 0:
            self.eval_n_envs_per = max(1, self.eval_n_envs // n_worker)
            self.eval_n_envs = eval_n_envs = self.eval_n_envs_per * n_worker
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}.")
            self.eval_max_T = eval_max_T = int(self.eval_max_steps // eval_n_envs)

        env = self.EnvCls(**self.env_kwargs)
        self._agent_init(agent, env, global_B=global_B, env_ranks=env_ranks)
        examples = self._build_buffers(env, bootstrap_value)
        env.close()
        del env

        self._build_parallel_ctrl(n_worker)

        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing every init.

        common_kwargs = self._assemble_common_kwargs(affinity, global_B)
        workers_kwargs = self._assemble_workers_kwargs(affinity, seed, n_envs_list)

        target = sampling_process if worker_process is None else worker_process
        self.workers = [
            mp.Process(target=target, kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs
        ]
        for w in self.workers:
            w.start()

        self.ctrl.barrier_out.wait()  # Wait for workers ready (e.g. decorrelate).
        return examples  # e.g. In case useful to build replay buffer.

    def _get_n_envs_list(self, affinity=None, n_worker=None, B=None):
        B = self.batch_spec.B if B is None else B
        n_worker = len(affinity["workers_cpus"]) if n_worker is None else n_worker
        if B < n_worker:
            logger.log(f"WARNING: requested fewer envs ({B}) than available worker "
                       f"processes ({n_worker}). Using fewer workers (but maybe better to "
                       "increase sampler's `batch_B`.")
            n_worker = B
        n_envs_list = [B // n_worker] * n_worker
        if not B % n_worker == 0:
            logger.log("WARNING: unequal number of envs per process, from "
                       f"batch_B {self.batch_spec.B} and n_worker {n_worker} "
                       "(possible suboptimal speed).")
            for b in range(B % n_worker):
                n_envs_list[b] += 1
        return n_envs_list

    def obtain_samples(self, itr):
        """Signals worker to begin environment step execution loop, and drops
        into ``serve_actions()`` method to provide actions to workers based on
        the new observations at each step.
        """
        # self.samples_np[:] = 0  # Reset all batch sample values (optional).
        self.agent.sample_mode(itr)
        self.ctrl.barrier_in.wait()
        self.serve_actions(itr)  # Worker step environments here.
        self.ctrl.barrier_out.wait()
        traj_infos = drain_queue(self.traj_infos_queue)
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        """Signals workers to begin agent evaluation loop, and drops into
        ``serve_actions_evaluation()`` to provide actions to workers at each
        step.
        """
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.agent.eval_mode(itr)
        self.ctrl.barrier_in.wait()
        traj_infos = self.serve_actions_evaluation(itr)
        self.ctrl.barrier_out.wait()
        traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
                                      n_sentinel=self.n_worker))  # Block until all finish submitting.
        self.ctrl.do_eval.value = False
        return traj_infos

    def _agent_init(self, agent, env, global_B=1, env_ranks=None):
        """Initializes the agent, having it *not* share memory, because all
        agent functions (training and sampling) happen in the master process,
        presumably on GPU."""
        agent.initialize(
            env.spaces,
            share_memory=False,  # No share memory.
            global_B=global_B,
            env_ranks=env_ranks)
        self.agent = agent

    def _build_buffers(self, *args, **kwargs):
        _, _, examples = build_samples_buffer(self.agent,
                                              env,
                                              self.batch_spec,
                                              bootstrap_value,
                                              agent_shared=True,
                                              env_shared=True,
                                              subprocess=True)
        self.step_buffer_pyt, self.step_buffer_np = build_step_buffer(examples, self.batch_spec.B)
        self.agent_inputs = AgentInputs(self.step_buffer_pyt.observation, self.step_buffer_pyt.action,
                                        self.step_buffer_pyt.reward)
        if self.eval_n_envs > 0:
            self.eval_step_buffer_pyt, self.eval_step_buffer_np = \
                build_step_buffer(examples, self.eval_n_envs)
            self.eval_agent_inputs = AgentInputs(
                self.eval_step_buffer_pyt.observation,
                self.eval_step_buffer_pyt.action,
                self.eval_step_buffer_pyt.reward,
            )
        return examples

    def _build_parallel_ctrl(self, n_worker):
        self.ctrl = AttrDict(
            quit=mp.RawValue(ctypes.c_bool, False),
            barrier_in=mp.Barrier(n_worker + 1),
            barrier_out=mp.Barrier(n_worker + 1),
            do_eval=mp.RawValue(ctypes.c_bool, False),
            itr=mp.RawValue(ctypes.c_long, 0),
        )
        self.traj_infos_queue = mp.Queue()
        self.eval_traj_infos_queue = mp.Queue()
        self.sync = AttrDict(stop_eval=mp.RawValue(ctypes.c_bool, False))
        self.sync.obs_ready = [mp.Semaphore(0) for _ in range(n_worker)]
        self.sync.act_ready = [mp.Semaphore(0) for _ in range(n_worker)]

    def _assemble_common_kwargs(self, *args, **kwargs):
        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            agent=self.agent,
            batch_T=self.batch_spec.T,
            CollectorCls=self.CollectorCls,
            TrajInfoCls=self.TrajInfoCls,
            traj_infos_queue=self.traj_infos_queue,
            ctrl=self.ctrl,
            max_decorrelation_steps=self.max_decorrelation_steps,
            torch_threads=affinity.get("worker_torch_threads", 1),
            global_B=global_B,
        )
        if self.eval_n_envs > 0:
            common_kwargs.update(
                dict(
                    eval_n_envs=self.eval_n_envs_per,
                    eval_CollectorCls=self.eval_CollectorCls,
                    eval_env_kwargs=self.eval_env_kwargs,
                    eval_max_T=self.eval_max_T,
                    eval_traj_infos_queue=self.eval_traj_infos_queue,
                ))
        common_kwargs["agent"] = None  # Remove.
        return common_kwargs

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        workers_kwargs = list()
        i_env = 0
        g_env = sum(n_envs_list) * self.rank
        for rank in range(len(n_envs_list)):
            n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs)
            env_ranks = list(range(g_env, g_env + n_envs))
            worker_kwargs = dict(
                rank=rank,
                env_ranks=env_ranks,
                seed=seed + rank,
                cpus=(affinity["workers_cpus"][rank] if affinity.get("set_affinity", True) else None),
                n_envs=n_envs,
                samples_np=self.samples_np[:, slice_B],
                sync=self.sync,  # Only for eval, on CPU.
            )
            i_env += n_envs
            g_env += n_envs
            workers_kwargs.append(worker_kwargs)
        i_env = 0
        for rank, w_kwargs in enumerate(workers_kwargs):
            n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs)
            w_kwargs["sync"] = AttrDict(
                stop_eval=self.sync.stop_eval,
                obs_ready=self.sync.obs_ready[rank],
                act_ready=self.sync.act_ready[rank],
            )
            w_kwargs["step_buffer_np"] = self.step_buffer_np[slice_B]
            if self.eval_n_envs > 0:
                eval_slice_B = slice(self.eval_n_envs_per * rank, self.eval_n_envs_per * (rank + 1))
                w_kwargs["eval_step_buffer_np"] = \
                    self.eval_step_buffer_np[eval_slice_B]
            i_env += n_envs
        return workers_kwargs