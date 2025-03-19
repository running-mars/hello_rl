"""
Description:
    - Design a network (common part of Actor and Critic) for feature extraction

Authors:
    - Running-Mars
"""

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomACNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomACNetwork, self).__init__(observation_space, features_dim)

        self.common = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        features = self.common(obs)

        return features