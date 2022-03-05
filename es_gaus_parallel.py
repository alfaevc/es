import os 
import tqdm
import numpy as np
import gym
# import pybullet_envs
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

def get_output(output, nA, min_logvar=1, max_logvar=3):
    sp = nn.Softplus()
    mu, raw_v = output[:nA], output[nA:]
    logvar = max_logvar - sp(max_logvar - raw_v)
    logvar = min_logvar + sp(logvar - min_logvar)

    return mu.detach().numpy(), logvar.detach().numpy()

class Gaus(nn.Module):
    def __init__(self, state_dim, nA):
        super(Gaus, self).__init__()
        self.state_dim = state_dim
        self.nA = nA
        self.fc1 = nn.Linear(state_dim, nA*2, bias=False)  
        self.fc2 = nn.Linear(2*nA, 2*nA, bias=False)
        self.fc3 = nn.Linear(2*nA, 2*nA, bias=False)

def gaus_feed_forward(gaus_net, state):#have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
    x = (torch.from_numpy(state)).float()
    #x = torchF.relu(state_net.fc1(x))
    x = gaus_net.fc1(x)
    x = torchF.relu(x)
    x = gaus_net.fc2(x)
    x = torchF.relu(x)
    x = gaus_net.fc3(x)
    
    return get_output(x,gaus_net.nA)

def get_gaus_net(theta, state_dim, nA):
    gaus_net = Gaus(state_dim, nA)
    update_nn_params(gaus_net, theta)
    return gaus_net


def get_theta_dim(state_dim, nA):
    gaus_net = Gaus(state_dim, nA)
    return get_nn_dim(gaus_net)

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

def gradascent(useParallel, theta0, filename, method=None, sigma=1, eta=1e-3, max_epoch=200, N=100, t=0):
  theta = np.copy(theta0)
  accum_rewards = np.zeros(max_epoch)
  t1 = time.time()
  global time_step_count
  for i in range(max_epoch): 
    accum_rewards[i] = eval(theta)
    if i%1==0:
      print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))    
      with open(filename, "a") as f:
        f.write("%.d %.2f \n" % (i, accum_rewards[i]))
        #f.write("%.d %.2f %.d \n" % (i, accum_rewards[i],time_step_count))
    if i%5==0:
        print('runtime until now: ', time.time()-t1)#, ' time step: ',time_step_count)
    #if time_step_count>= 10**7: #terminate at given time step threshold.
    #    sys.exit()
    theta += eta * AT_gradient_parallel(useParallel, theta, sigma, N=N)
    # print(theta)
    out_theta_file = "files/gaus_theta_{}.txt".format(env_name+t)
    np.savetxt(out_theta_file, theta, delimiter=' ', newline=' ')
    # with open(out_theta_file, "w") as h:
    #    for th in theta:
    #        h.write("{} ".format(th))
        
  return theta, accum_rewards


def F(theta , gamma=1, max_step=5e3):
    G = 0.0
    done = False
    discount = 1
    steps = 0
    state = env.reset()
    # a_dim = np.arange(nA)
    steps_count=0#cannot use global var here because subprocesses do not have access to global var
    # while not done:
    gaus_net = get_gaus_net(theta, state_dim, nA)
    while not done and (steps < max_step):
        # WRITE CODE HERE
        # fn = lambda a: [theta[2*a*(state_dim+1)] + state @ theta[2*a*(state_dim+1)+1: (2*a+1)*(state_dim+1)], 
        #                 theta[(2*a+1)*(state_dim+1)] + state @ theta[(2*a+1)*(state_dim+1)+1: (2*a+2)*(state_dim+1)]]
        #mvs = np.array(list(map(fn, a_dim))).flatten()
        a_mean, a_v = gaus_feed_forward(gaus_net, state)
        action = np.tanh(np.random.normal(a_mean, a_v))
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
    for i in range(epsilons.shape[0]):
        #can be made more efficient. but would not improve runtime, since only loop <=8 times
        output1, time1 = F(theta + sigma * epsilons[i])
        output2, time2 = F(theta - sigma * epsilons[i])
        grad[i] = (output1 - output2) * epsilons[i]
        steps_count += time1+time2
    grad = np.average(grad,axis=0)/sigma/2
    return [grad,steps_count]
        
    #fn = lambda x: (F(theta + sigma * x) - F(theta - sigma * x)) * x
    #return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2

def eval(theta):
    G = 0.0
    done = False
    state = env.reset()
    # a_dim = np.arange(nA)
   
    gaus_net = get_gaus_net(theta, state_dim, nA)
    while not done:
        # WRITE CODE HERE
        # fn = lambda a: [theta[2*a*(state_dim+1)] + state @ theta[2*a*(state_dim+1)+1: (2*a+1)*(state_dim+1)], 
        #                 theta[(2*a+1)*(state_dim+1)] + state @ theta[(2*a+1)*(state_dim+1)+1: (2*a+2)*(state_dim+1)]]
        #mvs = np.array(list(map(fn, a_dim))).flatten()
        a_mean, a_v = gaus_feed_forward(gaus_net, state)
        action = np.tanh(np.random.normal(a_mean, a_v))
        # action = np.random.normal(a_mean[0], a_v[0])
        state, reward, done, _ = env.step(action)
        G += reward
    return G


##########################################################################
global env_name
# env_name = 'InvertedPendulumBulletEnv-v0'
# env_name = 'FetchPush-v1'
# env_name = 'HalfCheetah-v2'
env_name = "Walker2d-v2"
# env_name = 'Swimmer-v2'
# env_name = 'LunarLanderContinuous-v2'
# env_name = 'Humanoid-v2'
global time_step_count
time_step_count=0

if __name__ == '__main__':
    import_theta = False
    policy = "gaus"
    useParallel=1#if parallelize
    print("number of CPUs: ", mp.cpu_count())
    gym.logger.set_level(40)
    env = gym.make(env_name)
    state_dim = env.reset().size
    nA, = env.action_space.shape
    theta_dim = get_theta_dim(state_dim, nA)
    old_t = ""
    t = str(time.time())

    import_theta = False
    # existing logged file
    theta_file = "files/{0}_theta_{1}.txt".format(policy, env_name+old_t)
    outfile = "files/{0}_{1}.txt".format(policy, env_name+old_t)
    
    b = 1
    
    num_seeds = 1
    max_epoch = 5001
    res = np.zeros((num_seeds, max_epoch))
    method = "AT_parallel"


    #all_actions = np.random.uniform(low=-1,high=1,size=(max(10,5**nA),nA))
    #all_actions = np.array([i for i in product([-1,-2/3, -1/3,0,1/3,2/3,1],repeat=nA)])
    
    t_start=time.time()
    for k in tqdm.tqdm(range(num_seeds)):
        N = theta_dim#make n larger to show effect of parallelization on pendulum
        theta0 = np.random.standard_normal(size=theta_dim)

        if import_theta: #Continue previous experiment
            with open(theta_file, "r") as f:
                l = list(filter(len, re.split(' |\*|\n', f.readlines()[0])))
                theta0 = np.array(l, dtype=float)
        else: #New experiment
            outfile = "files/{0}_{1}.txt".format(policy, env_name+t)
            with open(outfile, "w") as f:
                f.write("Seed {}:\n".format(k))
        time_elapsed = int(round(time.time()-t_start))
        # with open(outfile, "a") as f:
        #     f.write("Seed {}:\n".format(k))
        theta, accum_rewards = gradascent(useParallel, theta0, outfile, method=method, sigma=0.1, eta=1e-2, max_epoch=max_epoch, N=N, t=t)
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


