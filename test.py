import numpy as np

import matplotlib.pyplot as plt
import gym
import pybullet_envs

from itertools import product


"""### AT vs FD
"""


"""The cell below applies your ES implementation to the RL objective you've defined in the cell above."""
if __name__ == '__main__':
  # env_name = 'InvertedPendulumBulletEnv-v0'
  # env_name = 'IceHockey-ram-v0'
  env_name = 'BipedalWalker-v3'
  # env_name = 'HalfCheetah-v2'
  # env_name = 'Swimmer-v2'
  # env_name = 'CartPole-v0'
  env = gym.make(env_name)
  state = env.reset()
  print(state.size)
  print(env.action_space.shape[0])




