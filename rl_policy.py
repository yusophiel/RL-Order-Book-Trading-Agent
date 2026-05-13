"""
PPO agent factory + custom feature extractor
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from config import AgentConfig


# Simple pass-through + linear projection. Used for all state types except 'encoded'.

class LinearFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(n_input, max(features_dim, n_input)),
            nn.LayerNorm(max(features_dim, n_input)),
            nn.ReLU(),
            nn.Linear(max(features_dim, n_input), features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# Build a PPO agent for the given environment.
def make_ppo_agent(
    env: gym.Env,
    agent_cfg: AgentConfig,
    state_type: str = "best_bidask",
    seed: int = 42,
) -> PPO:
    if state_type == "raw_lob":
        features_dim = 128
    elif state_type == "encoded":
        features_dim = 96
    else:
        features_dim = 64
    policy_kwargs = {
        "features_extractor_class": LinearFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": list(agent_cfg.net_arch),
        "activation_fn": nn.ReLU,
    }

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=agent_cfg.learning_rate,
        n_steps=agent_cfg.n_steps,
        batch_size=agent_cfg.batch_size,
        n_epochs=agent_cfg.n_epochs,
        gamma=agent_cfg.gamma,
        gae_lambda=agent_cfg.gae_lambda,
        clip_range=agent_cfg.clip_range,
        ent_coef=agent_cfg.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )
    return agent
