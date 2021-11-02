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
  env_name = 'HalfCheetah-v2'
  # env_name = 'CartPole-v0'
  env = gym.make(env_name)
  state = env.reset()
  print(state.size)
  n, = env.action_space.shape
  print(n)
  a = np.array([2,2,2,2,2,2])
  s, r, done, _ = env.step(a)
  print("The next state is {}, reward is {}, the termination status is {}".format(s, r, done))
  print()
  print(np.random.normal(np.array([0,10]),np.array([1,1])))
  # theta_dim = state_dim + 1


# In[ ]:





# In[ ]:




