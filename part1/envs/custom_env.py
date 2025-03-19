import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def step(self, action):
        observation = np.zeros(4, )
        reward = 0.
        terminated = True
        truncated = True
        info = {
            'step': 1,
            'extra_info': 'This is just an example',
            'done': True
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation = np.zeros(4, )
        info = {
            'step': 1,
            'extra_info': 'This is just an example',
            'done': True
        }

        return observation, info

    def render(self):
        pass

    def close(self):
        pass