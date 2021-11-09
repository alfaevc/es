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
  # x = np.array([(1,np.array([2,3])),(4,np.array([5,6]))])
  # print(np.mean(list(map(list, zip(*x)))[1], axis = 1).size)

  # for param, target_param in zip(self.critic.trainable_weights, self.critic_target.trainable_weights):
  #     target_param.assign(self.tau * param.numpy() + (1 - self.tau) * target_param.numpy())
  N = 10

  actor = NN(n)
  actor_lr = 1e-4
  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
  actor_loss = tf.keras.losses.MeanSquaredError()
  actor.compile(optimizer=actor_optimizer, loss=actor_loss)

  actor.fit(np.random.standard_normal((N,state.size+n)), np.random.standard_normal((N,n)), epochs=1, batch_size=N, verbose=0)

  print(type(actor.trainable_weights))
  # print(actor.trainable_weights)
  for param in actor.trainable_weights:
      print(param.shape)

  arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

  A = arr.reshape((2,4))
  print(A)

  print(A.reshape(-1))

  print(np.concatenate(tuple([arr, arr[:3], arr[4:8]])))

# In[ ]:


# In[ ]:




