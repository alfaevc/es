import os 
import tqdm
import numpy as np
from networks import NN
import matplotlib.pyplot as plt
import gym
import pybullet_envs
import time

import math
import multiprocessing as mp
from itertools import repeat
import tensorflow as tf
from itertools import product
from multiprocessing import Process, Value, Array

def collect_result(result):
    result_list.append(result[0])
    steps_list.append(result[1])    
    
def AT_gradient_parallel(useParallel, theta, sigma=1, N=100):
    numCPU=mp.cpu_count()
    pool=mp.Pool(numCPU)
    jobs=math.ceil(N/numCPU)
    if math.floor(N/numCPU) >= 0.8*N/numCPU:
        jobs = math.floor(N/numCPU)
    N=jobs*numCPU

    epsilons=orthogonal_epsilons(N,theta.size)
    global result_list#must be global for callback function to edit
    result_list = []
    global steps_list
    steps_list = []
    if useParallel==1:
        for i in range(numCPU):
            pool.apply_async(F_arr,args = (epsilons[i*jobs:(i+1)*jobs], sigma, theta),callback=collect_result)
        pool.close()
        pool.join()
        #print('AT grad: ',np.mean(result_list))
    else:
        result_list = F_arr(epsilons,sigma,theta)
    #results=pool.starmap(F_arr,zip(list(repeat(epsilons[0],numCPU)),list(repeat(sigma,numCPU)),list(repeat(theta,numCPU))))
    print('add_time_steps ',sum(steps_list))
    global time_step_count
    time_step_count+=sum(steps_list)
    return np.mean(result_list)

  
def orthogonal_epsilons(N,dim):
    epsilons_N=np.zeros((math.ceil(N/dim)*dim,dim))    
    for i in range(0,math.ceil(N/dim)):
      epsilons = np.random.standard_normal(size=(dim, dim))
      Q, _ = np.linalg.qr(epsilons)#orthogonalize epsilons
      Q_normalize=np.copy(Q)
      fn = lambda x, y: np.linalg.norm(x) * y
      #renormalize rows of Q by multiplying it by length of corresponding row of epsilons
      Q_normalize = np.array(list(map(fn, epsilons, Q_normalize)))
      epsilons_N[i*dim:(i+1)*dim] = Q_normalize@Q
    return epsilons_N[0:N]

def gradascent(useParallel, theta0, filename, method=None, sigma=1, eta=1e-3, max_epoch=200, N=100):
  theta = np.copy(theta0)
  accum_rewards = np.zeros(max_epoch)
  t1=time.time()
  global time_step_count
  for i in range(max_epoch): 
    accum_rewards[i] = eval(theta)
    if i%1==0:
      print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))    
      with open(filename, "a") as f:
        f.write("%.d %.2f %.d \n" % (i, accum_rewards[i],time_step_count))
    if i%5==0:
        print('runtime until now: ',time.time()-t1, ' time step: ',time_step_count)
    if time_step_count>= 10**7:
        sys.exit()
    theta += eta * AT_gradient_parallel(useParallel, theta, sigma, N=N)
  return theta, accum_rewards

def get_output(output):
    nA=int(round(output.shape[1]/2))
    min_logvar=1
    max_logvar=3
    means = output[:, 0:nA]
    raw_vs = output[:, nA:]
    logvars = max_logvar - tf.nn.softplus(max_logvar - raw_vs)
    logvars = min_logvar + tf.nn.softplus(logvars - min_logvar)
    return means, tf.exp(logvars).numpy()

def F(theta , gamma=1, max_step=5e3):
    env = gym.make(env_name)#this takes no time
    nA, = env.action_space.shape
    G = 0.0
    done = False
    discount = 1
    steps = 0
    state = env.reset()
    a_dim = np.arange(nA)
    state_dim = state.size
    steps_count=0#cannot use global var here because subprocesses do not have access to global var
    # while not done:
    while not done and (steps < max_step):
        # WRITE CODE HERE
        fn = lambda a: [theta[2*a*(state_dim+1)] + state @ theta[2*a*(state_dim+1)+1: (2*a+1)*(state_dim+1)], 
                        theta[(2*a+1)*(state_dim+1)] + state @ theta[(2*a+1)*(state_dim+1)+1: (2*a+2)*(state_dim+1)]]
        mvs = np.array(list(map(fn, a_dim))).flatten()
        a_mean, a_v  = get_output(np.expand_dims(mvs, 0))
        action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
        # action = np.random.normal(a_mean[0], a_v[0])

        state, reward, done, _ = env.step(action)
        steps_count+=1
        G += reward * discount
        discount *= gamma
        steps += 1
    return G,steps_count

def F_arr(epsilons, sigma, theta):
    grad = np.zeros(epsilons.shape)
    steps_count = 0
    for i in range(epsilons.shape[0]):#for loop inefficient. to be improved
        output1, time1 = F(theta + sigma * epsilons[i])
        output2, time2 = F(theta - sigma * epsilons[i])
        grad[i] = (output1 - output2) * epsilons[i]
        steps_count += time1+time2
    grad = np.average(grad,axis=0)/sigma/2
    return [grad,steps_count]
        
    #fn = lambda x: (F(theta + sigma * x) - F(theta - sigma * x)) * x
    #return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2

def eval(theta):
    env = gym.make(env_name)#this takes no time
    nA, = env.action_space.shape
    G = 0.0
    done = False
    steps = 0
    state = env.reset()
    a_dim = np.arange(nA)
    state_dim = state.size
    global time_step_count
    while not done:
        # WRITE CODE HERE
        fn = lambda a: [theta[2*a*(state_dim+1)] + state @ theta[2*a*(state_dim+1)+1: (2*a+1)*(state_dim+1)], 
                        theta[(2*a+1)*(state_dim+1)] + state @ theta[(2*a+1)*(state_dim+1)+1: (2*a+2)*(state_dim+1)]]
        mvs = np.array(list(map(fn, a_dim))).flatten()
        a_mean, a_v  = get_output(np.expand_dims(mvs, 0))
        action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
        # action = np.random.normal(a_mean[0], a_v[0])

        state, reward, done, _ = env.step(action)
        time_step_count+=1
        G += reward
        steps += 1
    return G


##########################################################################
global env_name
env_name = 'InvertedPendulumBulletEnv-v0'
# env_name = 'FetchPush-v1'
# env_name = 'HalfCheetah-v2'
# env_name = 'Swimmer-v2'
# env_name = 'LunarLanderContinuous-v2'
# env_name = 'Humanoid-v2'
global time_step_count
time_step_count=0

if __name__ == '__main__':
    useParallel=1#if parallelize
    print("number of CPUs: ",mp.cpu_count())
    env = gym.make(env_name)
    state_dim = env.reset().size
    nA, = env.action_space.shape
    theta_dim = (state_dim + 1) * 2 * nA
    outfile = "Gaus_{}.txt".format(env_name+str(time.time()))
    with open(outfile, "w") as f:
        f.write("")
    b = 1
    num_seeds = 1
    max_epoch = 501
    res = np.zeros((num_seeds, max_epoch))
    method = "AT_parallel"

    t_start=time.time()
    for k in tqdm.tqdm(range(num_seeds)):
        N = theta_dim#make n larger to show effect of parallelization on pendulum
        theta0 = np.random.standard_normal(size=theta_dim)
        time_elapsed = int(round(time.time()-t_start))
        with open(outfile, "a") as f:
            f.write("Seed {}:\n".format(k))
        theta, accum_rewards = gradascent(useParallel, theta0, outfile, method=method, sigma=1, eta=1e-2, max_epoch=max_epoch, N=N)
        res[k] = np.array(accum_rewards)
    ns = range(1, len(accum_rewards)+1)

    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ns, mins, maxs, alpha=0.1)
    plt.plot(ns, avs, '-o', markersize=1, label=env_name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    plt.title("Gaussian {0} ES".format(method), fontsize = 24)
    plt.savefig("plots/Gaussian {0} ES {1}".format(method, env_name))

