# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import gym
import functools
import tqdm


"""### AT vs FD
"""

def _sigmoid(x):
  return 1 / (1 + np.exp(-x))

def _get_action(s, theta):
  w = theta[:4]
  b = theta[4]
  p_left = _sigmoid(w @ s + b)
  a = np.random.choice(2, p=[p_left, 1 - p_left])
  return a

def rl_fn(theta, env):
  assert len(theta) == 5
  # START HIDE
  done = False
  s = env.reset()
  total_reward = 0
  while not done:
    a = _get_action(s, theta)
    s, r, done, _ = env.step(a)
    total_reward += r
  # END HIDE
  return total_reward

def vanilla_gradient(theta, sigma=1, N=100):
  assert len(theta) == 5
  epsilons = np.random.multivariate_normal(np.zeros(5), np.identity(5), size=N)
  fn = lambda x: rl_fn(theta + sigma * x) * x
  return np.mean(np.fromiter(map(fn, epsilons), dtype=np.float), axis=1)/sigma

def FD_gradient(theta, sigma=1, N=100):
  assert len(theta) == 5
  epsilons = np.random.multivariate_normal(np.zeros(5), np.identity(5), size=N)
  fn = lambda x: (rl_fn(theta + sigma * x) - rl_fn(theta)) * x
  return np.mean(np.fromiter(map(fn, epsilons), dtype=np.float), axis=1)/sigma

def AT_gradient(theta, sigma=1, N=100):
  assert len(theta) == 5
  epsilons = np.random.multivariate_normal(np.zeros(5), np.identity(5), size=N)
  fn = lambda x: (rl_fn(theta + sigma * x) - rl_fn(theta - sigma * x)) * x
  return np.mean(np.fromiter(map(fn, epsilons), dtype=np.float), axis=1)/sigma/2
    
def gradascent(theta0, method=None, eta=1e-2, max_epoch=1000):
  theta = np.copy(theta0)
  accum_rewards = np.zeros(max_epoch)
  for i in range(max_epoch):
    accum_rewards[i] = rl_fn(theta, env)
    if method == "FD":
      theta += eta * FD_gradient(theta, sigma=1, N=100)
    elif method == "AT":
      theta += eta * AT_gradient(theta, sigma=1, N=100)
    else: #vanilla
      theta += eta * vanilla_gradient(theta, sigma=1, N=100)
  return theta, accum_rewards

"""The cell below applies your CMA-ES implementation to the RL objective you've defined in the cell above."""
if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  fn_with_env = functools.partial(rl_fn, env=env)

  epsilons = np.random.multivariate_normal(np.zeros(5), np.identity(5), size=2)
  print(epsilons)
  fn = lambda x: np.dot(np.ones(5), x)
  print(np.fromiter(map(fn, epsilons), dtype=np.float))
  """
  plt.plot(range(1, len(mean_sample_vec)+1), mean_sample_vec, label='mean sample reward')
  plt.plot(range(1, len(best_sample_vec)+1), best_sample_vec, label='best sample reward')
  plt.legend()
  plt.grid(True)
  plt.xlabel('Iterations')
  plt.ylabel('Reward')

  r_vec = [fn_with_env(np.ones(5)) for _ in tqdm.trange(1000)]

  2 * np.array(r_vec).reshape((10, -1)).mean(axis=0).std()



  np.mean(r_vec), np.std(r_vec)

  np.mean(r_vec), np.std(r_vec)

  def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

  def _get_action(s, params):
    w = params[:4]
    b = params[4]
    p_left = _sigmoid(w @ s + b)
    a = np.random.choice(2, p=[p_left, 1 - p_left])
    return a

  def rl_fn(params, env):
    assert len(params) == 5
    reward = []
    s = env.reset()
    a = _get_action(s, params)
    obs, r, done, info = env.step(a)
    reward.append(r)
    while (done == False):
      obs, r, done, info = env.step(a)
      reward.append(r)
      a = _get_action(s, params)
    # get actions given params
    # step in environment until end of episode
    # record total undiscounted reward
    total_reward = sum(reward)
    return total_reward

  # env = gym.make('CartPole-v0')
  # fn_with_env = functools.partial(rl_fn, env=env)

  rewards = []
  for i in range(1000):
    r = fn_with_env(-np.ones(5))
    rewards.append(r)
  print(sum(rewards)/1000)

  rewards = []
  for i in range(1000):
    r = fn_with_env([1, 0, 1, 0, 1])
    rewards.append(r)
  print(sum(rewards)/1000)

  rewards = []
  for i in range(1000):
    r = fn_with_env([0, 1, 2, 3, 4])
    rewards.append(r)
  print(sum(rewards)/1000)

  N = 1000
  l = 20
  sample_means = np.zeros(N)
  for i in range(N):
    x = np.zeros(l)
    for j in range(l):
      x[j] = np.max(np.random.normal(size=10) + 1.0)
    sample_means[i] = np.mean(x)

  np.mean(sample_means)

  np.std(sample_means)
  """