#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

def vanilla_gradient(theta, F, sigma=1, N=100):
  epsilons = np.random.standard_normal(size=(N, theta.size))
  fn = lambda x: F(theta + sigma * x) * x
  return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma

def FD_gradient(theta, F, sigma=1, N=100):
  #epsilons = np.random.standard_normal(size=(N, theta.size))
  epsilons=orthogonal_epsilons(theta)
  fn = lambda x: (F(theta + sigma * x) - F(theta)) * x
  return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma

def AT_gradient(theta, F, sigma=1, N=100):
  #epsilons = np.random.standard_normal(size=(N, theta.size))
  epsilons=orthogonal_epsilons(theta)
  fn = lambda x: (F(theta + sigma * x) - F(theta - sigma * x)) * x
  return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2

def orthogonal_epsilons(theta):
    epsilons = np.random.standard_normal(size=(theta.size, theta.size))
    Q, _ = np.linalg.qr(epsilons)#orthogonalize epsilons
    Q_normalize=np.copy(Q)
    fn = lambda x, y: np.linalg.norm(x) * y
    #renormalize rows of Q by multiplying it by length of corresponding row of epsilons
    Q_normalize = np.array(list(map(fn, epsilons, Q_normalize)))
    #for i in range(theta.size):
    #  norm=np.linalg.norm(epsilons[i])
    #Q_normalize[i]=Q_normalize[i]*norm
    return Q_normalize@Q

def hessian_gaussian_smoothing(theta, F, sigma=1, N=5):
  epsilons = np.random.standard_normal(size=(N, theta.size))
  fn = lambda x: F(theta + sigma * x) 
  second_term=np.mean(np.array(list(map(fn, epsilons))), axis=0)/(sigma**2)
  fn = lambda x: F(theta + sigma * x)*np.outer(x,x)/(N*sigma**2)
  hessian = np.sum(np.array(list(map(fn, epsilons))), axis = 0)
  # hessian=np.zeros((theta.size,theta.size))
  # for i in range(N):
  #   hessian+=F(theta + sigma * epsilons[i])*np.outer(epsilons[i],epsilons[i])/(N*sigma**2)
  hessian -=np.identity(theta.size)*second_term
  return hessian

def choose_covariate(theta,F,sigma=1,N=theta.size):
    grad=vanilla_gradient(theta, F, sigma=1, N=5)
    hessian=hessian_gaussian_smoothing(theta, F, sigma=1, N=5)
    MSE_AT=((theta.size+2)/N-1)*np.linalg.norm(grad)**2
    MSE_FD=((theta.size+4)/N)*np.sum(np.outer(epsilons[0],epsilons[0]))
    MSE_FD+=((theta.size*2+8)/N-1)*np.linalg.norm(grad)**2
    MSE_FD+=((theta.size+4)*sigma**2/(4*N))*np.linalg.norm(hessian, ord='fro')**2
    MSE_FD+=(5*sigma**2/2)*np.diagonal(hessian)@np.diagonal(hessian)
    choice='AT'
    if (2*N/(N+1))*MSE_AT>MSE_FD:
        choice='FD'
    return choice,MSE_FD,MSE_AT
    
    
def gradascent(theta0, F, method=None, eta=1e-2, max_epoch=200, N=100):
  theta = np.copy(theta0)
  theta_len = theta.size
  accum_rewards = np.zeros(max_epoch)
  choice, MSE_FD, MSE_AT=choose_covariate(theta,F,sigma=1,N=theta.size)
  print('best method is ',choice,', MSE of FD is ',MSE_FD,', MSE OF AT is ', MSE_AT)
  for i in range(max_epoch):
    accum_rewards[i] = F(theta)
    if i%50==0:
      print("The return for episode {0} is {1}".format(i, accum_rewards[i]))
    if method == "FD":
      theta += eta * FD_gradient(theta, F, N=N)
    elif method == "AT":
      theta += eta * AT_gradient(theta, F, N=N)
    else: #vanilla
      theta += eta * vanilla_gradient(theta, F, N=N)
  return theta, accum_rewards

"""The cell below applies your CMA-ES implementation to the RL objective you've defined in the cell above."""
if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  theta_dim = 5
  fn_with_env = functools.partial(rl_fn, env=env)
  num_seeds = 5
  max_epoch = 151
  N = 5
  res = np.zeros((num_seeds, max_epoch))
  method = "FD"

  for k in tqdm.tqdm(range(num_seeds)):
    theta0 = np.random.standard_normal(size=theta_dim)

    # epsilons = np.random.multivariate_normal(np.zeros(5), np.identity(5), size=2)
    # print(epsilons)
    # fn = lambda x: fn_with_env(theta0 + x) * x
    # print(fn_with_env(theta0 + epsilons[0]) * epsilons[0])
    # print(np.array(list(map(fn, epsilons))))
  
    theta, accum_rewards = gradascent(theta0, fn_with_env, method=method, max_epoch=max_epoch, N=N)
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
    plt.title("FD ES", fontsize = 24)
    plt.savefig("plots/FD ES.png")
  elif method == "AT":
    plt.title("AT ES", fontsize = 24)
    plt.savefig("plots/AT ES.png")
  else:
    plt.title("Vanilla ES", fontsize = 24)
    plt.savefig("plots/Vanilla ES.png")


# In[ ]:





# In[ ]:




