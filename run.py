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
  env_name = 'InvertedPendulumBulletEnv-v0'
  # env_name = 'CartPole-v0'
  env = gym.make(env_name)
  # theta_dim = 5
  state_dim = env.reset().size
  theta_dim = (state_dim + 1) * 2

  # lp = policy.Log(env)
  gp = policy.Gaus(env, state_dim)
  # fn_with_env = functools.partial(rl_fn, env=env)
  fn_with_env = gp.rl_fn
  num_seeds = 5
  # max_epoch = 151
  max_epoch = 251
  N = 5
  res = np.zeros((num_seeds, max_epoch))
  method = "AT"

  for k in tqdm.tqdm(range(num_seeds)):
    theta0 = np.random.standard_normal(size=theta_dim)

    # epsilons = np.random.multivariate_normal(np.zeros(5), np.identity(5), size=2)
    # print(epsilons)
    # fn = lambda x: fn_with_env(theta0 + x) * x
    # print(fn_with_env(theta0 + epsilons[0]) * epsilons[0])
    # print(np.array(list(map(fn, epsilons))))
  
    theta, accum_rewards = es.gradascent(theta0, fn_with_env, method=method, max_epoch=max_epoch, N=N)
    res[k] = np.array(accum_rewards)
  ns = range(1, len(accum_rewards)+1)

  avs = np.mean(res, axis=0)
  maxs = np.max(res, axis=0)
  mins = np.min(res, axis=0)
  
  plt.fill_between(ns, mins, maxs, alpha=0.1)
  plt.plot(ns, avs, '-o', markersize=1, label='return')

  plt.legend()
  plt.grid(True)
  plt.xlabel('Iterations', fontsize = 15)
  plt.ylabel('Return', fontsize = 15)

  if method == "FD":
    plt.title("FD ES {}".format(env_name), fontsize = 24)
    plt.savefig("plots/FD ES.png")
  elif method == "AT":
    plt.title("AT ES {}".format(env_name), fontsize = 24)
    plt.savefig("plots/AT ES.png")
  else:
    plt.title("Vanilla ES {}".format(env_name), fontsize = 24)
    plt.savefig("plots/Vanilla ES.png")


# In[ ]:





# In[ ]:




