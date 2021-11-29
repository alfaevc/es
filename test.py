#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-

import numpy as np
import es
import policy

import matplotlib.pyplot as plt
import gym
import pybullet_envs
from networks import NN

import tensorflow as tf
from itertools import product


"""### AT vs FD
"""


"""The cell below applies your ES implementation to the RL objective you've defined in the cell above."""
if __name__ == '__main__':
  # env_name = 'InvertedPendulumBulletEnv-v0'
  env_name = 'IceHockey-ram-v0'
  # env_name = 'HalfCheetah-v2'
  # env_name = 'Swimmer-v2'
  # env_name = 'CartPole-v0'
  env = gym.make(env_name)
  state = env.reset()
  s = state.size
  a = env.action_space
  print(list(product([-1,0,1],repeat=3)))
  

  


# In[ ]:


# In[ ]:




