# ray_rl
Reinforcement learning example code for analyzing performance of Ray.

# Requirements
    tensorflow==1.15
    torch==1.5.1
    ray
    gym
Even though we provide *requirement.txt*, it may have **great amount of redundancy**. We **strongly recommend** you just install above 4 packages and try to install other required packages by running the code and finding which required package is not installed yet.

Besides, gym environments can not be fully installed directly by using *pip install gym*. See https://github.com/openai/gym#installing-everything.

# Use
You can run *run_async_ppo.py* to run asynchronous proximal policy optimization (PPO, see https://arxiv.org/abs/1707.06347) algorighm locally. Asynchronous PPO uses local CPUs for interacting with gym evironment and GPU for updating model parameters, using dataflow very similar to IMPALA (see https://arxiv.org/abs/1802.01561).

By contrast, you can run *run_sync_ppo.py* to run classic synchronous PPO that does not use Ray.

Learning curve and losses will be recorded in tensorboard automatically. We provide example results on Humanoid-v2 environment in ./runs.

# Terminologies
**workers** refer to all models and environment copies that used for interacting with gym environment, and periodically download parameters from **parameter server**. They are implemented using Ray Actor (see https://docs.ray.io/en/latest/actors.html) and use CPU only. 

**learner** refers to the model that uses gathered data to update parameters, and periodically upload parameters to **parameter server**. It is implemented in the main thread and does not use Ray.

We hope comments in code be helpful. :)

# gym Environment Suggestion

Environments in OpenAI gym can be roughly divided into *discrete* and *continuous* ones. Actions in *discrete* environments are integer numbers, while actions in *continuous* environments are real value arrays. 

Personally, I often evaluate PPO algorithm in Mujoco environments, e.g. Humanoid-v2, which are continuous ones, but using Mujoco requires licsence. You can follow the instructions for installing gym (https://github.com/openai/gym#installing-everything) to get a licsence for free as a student. If you can't get one, using *box2d* environments e.g. BipedalWalker-v2 and LunarLander-v2 is also a good choice. See http://gym.openai.com/envs/#box2d.

# Flaw
Hyperparameters in the code are **NOT** well optimized, so the performance will be a little unstable. You may see losses do not decrease in tensorboard summary, but episode score indeed increases. Personally I think this is because of the bad selection of hyperparameters. We will provide stable version before long.
