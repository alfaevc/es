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
  # env_name = 'BipedalWalker-v3'
  env_name = "Acrobot-v1"
  # env_name = 'Walker2d-v2'
  # env_name = 'Swimmer-v2'
  # env_name = 'CartPole-v0'
  # env_name = 'Humanoid-v2'
  # env_name = 'HalfCheetah-v2'
  env = gym.make(env_name)

  state_dim, = env.observation_space.shape
  print(state_dim)
  nA = env.action_space.n
  print(nA)
  a_mean = np.zeros(nA)
  a_v = np.ones(nA)
  G = 0.0

  done = False
  state = env.reset()
  # obs = env.render(height=64, width=64, mode="rgb_array")
  # print(obs.shape)
  step = 0

  '''
  while not done:
      action = np.tanh(np.random.normal(a_mean, a_v))
  
      # action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
      state, reward, done, _ = env.step(action)

      #obs = env.render(height=64, width=64, mode="rgb_array")
      #imgplot = plt.imshow(obs)
      #plt.show()
      #plt.close()
      #plt.savefig("trial_render_imgs/{0} step {1}.png".format(env_name, step))
      
      step += 1

      G += reward
      
  print(step)
  '''
 