#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# -*- coding: utf-8 -*-
import os 
import tqdm

import numpy as np
import es
import policy

import matplotlib.pyplot as plt
import gym
import pybullet_envs



"""### AT vs FD
"""

def test_video(policy, theta, env_name, method):
        		# Usage: 
		# 	you can pass the arguments within agent.train() as:
		# 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env_name, method)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(policy.env, save_path, force=True)
    G = 0.0
    state = env.reset()
    done = False
    a_dim = np.arange(policy.nA)
    while not done:
        # WRITE CODE HERE
        env.render()
        fn = lambda a: [theta[2*a*(policy.state_dim+1)] + state @ theta[2*a*(policy.state_dim+1)+1: (2*a+1)*(policy.state_dim+1)], 
                        theta[(2*a+1)*(policy.state_dim+1)] + state @ theta[(2*a+1)*(policy.state_dim+1)+1: (2*a+2)*(policy.state_dim+1)]]
        mvs = np.array(list(map(fn, a_dim))).flatten()
        a_mean, a_v  = policy.get_output(np.expand_dims(mvs, 0))
        action = np.tanh(np.random.normal(a_mean[0], a_v[0]))

        state, reward, done, _ = env.step(action)
        G += reward
    
    print("The return is {}".format(G))
  
"""The cell below applies your ES implementation to the RL objective you've defined in the cell above."""
if __name__ == '__main__':
  # env_name = 'InvertedPendulumBulletEnv-v0'
  # env_name = 'FetchPush-v1'
  # env_name = 'HalfCheetah-v2'
  env_name = 'Swimmer-v2'
  # env_name = 'InvertedPendulumBulletEnv-v0'
  env = gym.make(env_name)
  state_dim = env.reset().size
  # theta_dim = state_dim + 1
  nA, = env.action_space.shape
  theta_dim = (state_dim + 1) * 2 * nA

  # lp = policy.Log(env)
  pi = policy.Gaus(env, state_dim, nA=nA)
  # fn_with_env = functools.partial(rl_fn, env=env)
  num_seeds = 5
  # max_epoch = 151
  max_epoch = 301
  N = theta_dim
  res = np.zeros((num_seeds, max_epoch))
  method = "AT"
  print("The method is {}".format(method))

  for k in tqdm.tqdm(range(num_seeds)):
    theta0 = np.random.standard_normal(size=theta_dim)
    # test_video(pi, theta0, env_name, method)
    # epsilons = np.random.multivariate_normal(np.zeros(5), np.identity(5), size=2)
    # print(epsilons)
    # fn = lambda x: fn_with_env(theta0 + x) * x
    # print(fn_with_env(theta0 + epsilons[0]) * epsilons[0])
    # print(np.array(list(map(fn, epsilons))))
    # theta, accum_rewards, method = es.gradascent_autoSwitch(theta0, pi, method=method, sigma=0.1, eta=5e-3, max_epoch=max_epoch, N=N)
    theta, accum_rewards = es.gradascent(theta0, pi, method=method, sigma=0.1, eta=1e-2, max_epoch=max_epoch, N=N)
    test_video(pi, theta, env_name, method)
    # theta, accum_rewards = es.gradascent(theta, pi, method=method, sigma=0.1, eta=1e-2, max_epoch=max_epoch, N=N)
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
  
  plt.title("{0} ES {1}".format(method, env_name), fontsize = 24)
  plt.savefig("plots/{0} ES {1}".format(method, env_name))
  


# In[ ]:





# In[ ]:




