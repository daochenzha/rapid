import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
import xml.etree.ElementTree as ET

class ViscosityEpisodeSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.t = 0
        self.r = 0
        self.seed()
        self._init()
        utils.EzPickle.__init__(self)

    def _init(self, model_path='swimmer.xml', frame_skip=4):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(mujoco_env.__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        # Read the XML
        viscosity = str(self.np_random.uniform(0.1, 0.5))
        with open(fullpath, 'r') as f:
            xml_text = f.read()
        xml_text = ET.fromstring(xml_text)
        xml_text.find('option').set('viscosity', viscosity)
        xml_text = ET.tostring(xml_text, encoding='unicode')
        
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_xml(xml_text)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

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

    def reset(self):
        self._init()
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def reset_model(self):
        self.t = 0
        self.r = 0
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
