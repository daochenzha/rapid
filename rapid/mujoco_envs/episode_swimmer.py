import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class EpisodeSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.t = 0
        self.r = 0
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        self.t += 1
        self.r += reward
        if self.t >= 1000:
            return_r = self.r
            self.r = 0
            self.t = 0
        else:
            return_r = 0
        return ob, return_r, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.t = 0
        self.r = 0
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
