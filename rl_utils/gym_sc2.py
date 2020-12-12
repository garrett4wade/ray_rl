import numpy as np
from smac.env import StarCraft2Env


class GymStarCraft2Env(StarCraft2Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        return np.stack(self.get_obs()), self.get_state(), np.stack(self.get_avail_actions())

    def step(self, actions):
        r, d, info = super().step(actions)
        return np.stack(self.get_obs()), self.get_state(), np.stack(self.get_avail_actions()), r, d, info
