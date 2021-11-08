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
import tqdm



"""### AT vs FD
"""


"""The cell below applies your ES implementation to the RL objective you've defined in the cell above."""
if __name__ == '__main__':
  # env_name = 'InvertedPendulumBulletEnv-v0'
  # env_name = 'HalfCheetah-v2'
  env_name = 'Swimmer-v2'
  # env_name = 'CartPole-v0'
  env = gym.make(env_name)
  state = env.reset()
  a = env.action_space
  print(a)
  n, = a.shape
  print(n)
  print((state.size + 1) * 2 * n)
  a = np.ones(n)
  s, r, done, _ = env.step(a)
  print("The next state is {}, reward is {}, the termination status is {}".format(s, r, done))
  # theta_dim = state_dim + 1
  x = np.array([(1,np.array([2,3])),(4,np.array([5,6]))])
  print(np.mean(list(map(list, zip(*x)))[1], axis = 1).size)

# In[ ]:


# In[ ]:




