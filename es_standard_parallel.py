import os 
import tqdm
import numpy as np
import gym
import pybullet_envs
import time
import math
import multiprocessing as mp
from itertools import repeat
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as torchF

import matplotlib.pyplot as plt
import re

def update_nn_params(input_nn,new_params):
    params = list(input_nn.parameters())
    current_index= 0
    for i in range(len(params)):
        shape = params[i].data.detach().numpy().shape
        if len(shape)>1:#params[i] is 2d tensor
            arr = new_params[current_index:current_index+shape[0]*shape[1]].reshape(shape)
            params[i].data = (torch.from_numpy(arr)).float()
            current_index+=shape[0]*shape[1]
        else:#params[i] is 1d tensor
            arr = new_params[current_index:current_index+shape[0]]
            params[i].data = (torch.from_numpy(arr)).float()
            current_index+=shape[0]
    for param in params:#freeze params
        param.requires_grad = False

def get_nn_dim(input_nn):
    params = list(input_nn.parameters())
    counter = 0
    for i in range(len(params)):
        shape = params[i].data.detach().numpy().shape
        if len(shape)>1:#params[i] is 2d tensor
            counter+=shape[0]*shape[1]
        else:#params[i] is 1d tensor
            counter+=shape[0]
    return counter

class state_tower(nn.Module):
    def __init__(self):
        super(state_tower, self).__init__()
        state_dim = env.reset().size
        nA, = env.action_space.shape
        self.fc1 = nn.Linear(state_dim, nA, bias=False)  
        self.fc2 = nn.Linear(nA, nA, bias=False)
        self.fc3 = nn.Linear(nA, nA, bias=False)
        self.fc4 = nn.Linear(nA, nA, bias=False)
        self.fc5 = nn.Linear(nA, nA, bias=False)
        self.fc6 = nn.Linear(nA, nA, bias=False)

def state_feed_forward(state_net,state):#have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
    x = (torch.from_numpy(state)).float()
    x = torchF.relu(state_net.fc1(x))
    x = torchF.relu(state_net.fc2(x))
    x = torchF.relu(state_net.fc3(x))
    x = torchF.relu(state_net.fc4(x))
    x = torchF.relu(state_net.fc5(x))
    #x = state_net.fc1(x)
    #x = state_net.fc2(x)
    #x = state_net.fc3(x)
    #x = state_net.fc4(x)
    #x = state_net.fc5(x)
    x = state_net.fc6(x)
    action = x.detach().numpy()
    action = np.tanh(action)
    return action

def get_state_net(theta):
    state_net = state_tower()
    update_nn_params(state_net, theta)
    return state_net
    
def get_theta_dim():
    state_net = state_tower()
    state_nn_dim = get_nn_dim(state_net)
    return state_nn_dim

#############################################################################################################################################

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
        grad = np.average(result_list,axis=0)
        time_step_count+=sum(steps_list)
    else:
        result_list = F_arr(epsilons,sigma,theta)
        grad = result_list[0]
        time_step_count+=result_list[1]
    return grad

  
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

def gradascent(useParallel, theta0, filename, sigma=1, eta=1e-3, max_epoch=200, N=100, t=0):
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
    if i%5==0:
        print('runtime until now: ',time.time()-t1)#, ' time step: ',time_step_count)
    #if time_step_count>= 10**7: #terminate at given time step threshold.
    #    sys.exit()
    theta += eta * AT_gradient_parallel(useParallel, theta, sigma, N=N)
    out_theta_file = "files/{0}_theta_{1}.txt".format(policy, env_name+t)
    np.savetxt(out_theta_file, theta, delimiter=' ', newline=' ')
        
  return theta, accum_rewards

def energy_action(actions_arr, latent_actions, latent_state):
    energies = latent_actions@latent_state
    return actions_arr[np.argmin(energies)]

def F(theta , gamma=1, max_step=5e3):
    gym.logger.set_level(40)
    env = gym.make(env_name)#this takes no time
    nA, = env.action_space.shape
    G = 0.0
    done = False
    discount = 1
    state = env.reset()
    a_dim = np.arange(nA)
    state_dim = state.size
    steps_count=0#cannot use global var here because subprocesses cannot edit global var
    state_net = get_state_net(theta)
    while not done:
        action = state_feed_forward(state_net,state)
        state, reward, done, _ = env.step(action)
        steps_count+=1
        G += reward * discount
        discount *= gamma
    return G,steps_count

def F_arr(epsilons, sigma, theta):
    grad = np.zeros(epsilons.shape)
    steps_count = 0
    for i in range(epsilons.shape[0]):
        #can be made more efficient. but would not improve runtime, since only loop <=20 times
        output1, time1 = F(theta + sigma * epsilons[i])
        output2, time2 = F(theta - sigma * epsilons[i])
        grad[i] = (output1 - output2) * epsilons[i]
        steps_count += time1+time2
    grad = np.average(grad,axis=0)/sigma/2
    return [grad,steps_count]

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
    state_net = get_state_net(theta)
    while not done:
        action = state_feed_forward(state_net,state)
        state, reward, done, _ = env.step(action)
        time_step_count+=1
        G += reward
    return G


##########################################################################
global env_name
global policy
global time_step_count
# env_name = 'InvertedPendulumBulletEnv-v0'
# env_name = 'FetchPush-v1'
env_name = 'HalfCheetah-v2'
# env_name = 'Swimmer-v2'
# env_name = 'LunarLanderContinuous-v2'
# env_name = 'Humanoid-v2'
# env_name = 'Walker2d-v2'
policy = "standard"

time_step_count=0

if __name__ == '__main__':
    import_theta = False
    useParallel=0#if parallelize
    print("number of CPUs: ",mp.cpu_count())
    gym.logger.set_level(40)
    env = gym.make(env_name)
    state_dim = env.reset().size
    nA, = env.action_space.shape
    theta_dim = get_theta_dim()
    num_seeds = 1
    max_epoch = 4001
    res = np.zeros((num_seeds, max_epoch))
    method = "AT_parallel"

    old_t = ""

    t = str(time.time())

    if import_theta:
        t = old_t

    # existing logged file
    theta_file = "files/{0}_theta_{1}.txt".format(policy, env_name+t)
    outfile = "files/{0}_{1}.txt".format(policy, env_name+t)

    t_start=time.time()
    for k in tqdm.tqdm(range(num_seeds)):
        N = theta_dim#make n larger to show effect of parallelization on pendulum
        theta0 = np.random.standard_normal(size=theta_dim)
        if import_theta:
            with open(theta_file, "r") as g:
                l = list(filter(len, re.split(' |\*|\n', g.readlines()[0])))
                theta0 = np.array(l, dtype=float)
                print(theta0)
        else: #New experiment
            with open(outfile, "w") as g:
                g.write("Seed {}:\n".format(k))
        time_elapsed = int(round(time.time()-t_start))
        theta, accum_rewards = gradascent(useParallel, theta0, outfile, sigma=1, eta=1e-2, max_epoch=max_epoch, N=N, t=t)
        res[k] = np.array(accum_rewards)