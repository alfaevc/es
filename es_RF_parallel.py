import os 
import tqdm
import numpy as np
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
from sklearn.preprocessing import PolynomialFeatures

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
    global time_step_count
    if useParallel==1:
        for i in range(numCPU):
            pool.apply_async(F_arr,args = (epsilons[i*jobs:(i+1)*jobs], sigma, theta),callback=collect_result)
        pool.close()
        pool.join()
        result_list = np.average(result_list,axis=0)
        time_step_count+=sum(steps_list)
    else:
        result_list = F_arr(epsilons,sigma,theta)
        result_list = result_list[0]
        time_step_count+=result_list[1]
    return result_list

  
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
        f.write("%.d %.2f \n" % (i, accum_rewards[i]))
        #f.write("%.d %.2f %.d \n" % (i, accum_rewards[i],time_step_count))
    if i%5==0:
        print('runtime until now: ',time.time()-t1)#, ' time step: ',time_step_count)
    #if time_step_count>= 10**7: #terminate at given time step threshold.
    #    sys.exit()
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

def energy_action(nA,table, latent_state, all_actions):
    max_depth = 2*nA
    left,right = 0,len(table)-1#left end and right end of search region. we iteratively refine the search region
    currentDepth = 0
    multiplier=3
    while currentDepth < max_depth:
        #go to next level of depth
        mid = (left+right)/2#not an integer
        left_latent_action_sum = table[math.ceil(mid)]
        if left>0:
            left_latent_action_sum = table[math.ceil(mid)] - table[left-1]
        left_prob = np.exp(multiplier*np.tanh(left_latent_action_sum@latent_state))#may cause overflow or underflow. need some normalization
        right_latent_action_sum = table[right]-table[math.floor(mid)]
        right_prob = np.exp(multiplier*np.tanh(right_latent_action_sum@latent_state))

        p = left_prob/(left_prob+right_prob)
        #print('depth: ',currentDepth,'p: ',p,'left: ',left,'right: ',right)
        coin_toss = np.random.binomial(1, p)
        if coin_toss == 1:#go left
            right=math.floor(mid)
        else:#go right
            left = math.ceil(mid)
        currentDepth+=1
    return all_actions[left]

def get_latent_action(nA, action, theta):
    latent_action = action@theta[:nA**2].reshape((nA,nA))
    return latent_action

def get_latent_state(nA, state, theta):
    state_dim=len(state)
    latent_state = state@theta[nA**2:].reshape((state_dim,nA))
    return latent_state

def F(theta, gamma=1, max_step=5e3):
    gym.logger.set_level(40)
    env = gym.make(env_name)#this takes no time
    nA, = env.action_space.shape
    G = 0.0
    done = False
    discount = 1
    state = env.reset()
    a_dim = np.arange(nA)
    state_dim = state.size
    steps_count=0#cannot use global var here because subprocesses do not have access to global var
    #preprocessing
    # all_actions = np.array([i for i in product([-1,-1/3,1/3,1],repeat=nA)])#need to make the number of actions some power of 2
    all_actions = np.array([i for i in product([-1,-5/7, -3/7,-1/7,0,3/7,5/7,1],repeat=nA)])#num of actions must be some power of 2
    fn = lambda a: get_latent_action(nA, a, theta)
    table = np.cumsum(np.array(list(map(fn, all_actions))), axis=0)
##    table = np.zeros((all_actions.shape))
##    for i in range(all_actions):#need to vectorize
##        table[i] = get_latent_action(action, theta)
##    for i in range(all_actions-1):#need to make more efficient
##        table[i+1]+=table[i]
    while not done:
        latent_state = get_latent_state(nA,state,theta)
        action = energy_action(nA, table, latent_state, all_actions)
        state, reward, done, _ = env.step(action)
        steps_count+=1
        G += reward * discount
        discount *= gamma
    return G,steps_count

def F_arr(epsilons, sigma, theta):
    grad = np.zeros(epsilons.shape)
    steps_count = 0
    for i in range(epsilons.shape[0]):
        #can be made more efficient. but would not improve runtime, since only loop <=8 times
        output1, time1 = F(theta + sigma * epsilons[i])
        output2, time2 = F(theta - sigma * epsilons[i])
        grad[i] = (output1 - output2) * epsilons[i]
        steps_count += time1+time2
    grad = np.average(grad,axis=0)/sigma/2
    return [grad, steps_count]
        
    #fn = lambda x: (F(theta + sigma * x) - F(theta - sigma * x)) * x
    #return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2

def eval(theta):
    gym.logger.set_level(40)
    env = gym.make(env_name)#this takes no time
    nA, = env.action_space.shape
    G = 0.0
    done = False
    state = env.reset()
    a_dim = np.arange(nA)
    state_dim = state.size
    global time_step_count
    #preprocessing
    # all_actions = np.array([i for i in product([-1,-1/3,1/3,1],repeat=nA)])#need to make the number of actions some power of 2
    fn = lambda a: get_latent_action(nA, a, theta)
    table = np.cumsum(np.array(list(map(fn, all_actions))), axis=0)
##    table = np.zeros((all_actions.shape))
##    for i in range(all_actions):#need to vectorize
##        table[i] = get_latent_action(action, theta)
##    for i in range(all_actions-1):#need to make more efficient
##        table[i+1]+=table[i]
    while not done:
        latent_state = get_latent_state(nA,state,theta)
        action = energy_action(nA, table, latent_state,all_actions)
        state, reward, done, _ = env.step(action)
        time_step_count+=1
        G += reward
    return G


##########################################################################
global env_name
# env_name = 'InvertedPendulumBulletEnv-v0'
# env_name = 'FetchPush-v1'
env_name = 'HalfCheetah-v2'
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
    all_actions = np.array([i for i in product([-1,-2/3,-1/3,0,1/3,2/3,1],repeat=nA)])
    theta_dim=nA*(state_dim+nA)
    #theta_dim = (state_dim + 1) * 2 * nA #gaus
    outfile = "files/RF_{}.txt".format(env_name+str(time.time()))
    with open(outfile, "w") as f:
        f.write("")
    b = 1
    num_seeds = 1
    max_epoch = 1001
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


