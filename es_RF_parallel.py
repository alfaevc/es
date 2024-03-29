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
        env = gym.make(env_name)
        state_dim = env.reset().size
        nA, = env.action_space.shape
        self.fc1 = nn.Linear(state_dim, nA, bias=False)  
        #self.fc2 = nn.Linear(nA, nA, bias=False)

def state_feed_forward(state_net,state):#have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
    x = (torch.from_numpy(state)).float()
    #x = torchF.relu(state_fc1(x))
    x = state_net.fc1(x)
    #x = state_net.fc2(x)
    latent_state = x.detach().numpy()
    #latent_state = latent_state/sum(np.abs(latent_state)) #normalize
    return latent_state

class action_tower(nn.Module):
    def __init__(self):
        super(action_tower, self).__init__()
        env = gym.make(env_name)
        nA, = env.action_space.shape
        self.fc1 = nn.Linear(nA, nA, bias=False)#can automate this. create nn for any given input layer dimensions, instead of fixed dimensions  
        #self.fc2 = nn.Linear(nA, nA, bias=False)

def action_feed_forward(action_net,action):#have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
    x = (torch.from_numpy(action)).float()
    #x = torchF.relu(fc1(x))
    x = action_net.fc1(x)#can automate this. feedforward given nn dimensions
    #x = action_net.fc2(x)
    latent_action = x.detach().numpy()
    return latent_action

def get_state_net(theta):
    action_net = action_tower()
    action_nn_dim = get_nn_dim(action_net)
    state_net = state_tower()
    update_nn_params(state_net, theta[action_nn_dim:])
    return state_net

def get_action_net(theta):
    action_net = action_tower()
    action_nn_dim = get_nn_dim(action_net)
    update_nn_params(action_net,theta[:action_nn_dim])
    return action_net
    
def get_latent_actions(theta):
    action_net = get_action_net(theta)
    return action_feed_forward(action_net,all_actions)

def get_theta_dim():
    state_net = state_tower()
    action_net = action_tower()
    state_nn_dim = get_nn_dim(state_net)
    action_nn_dim = get_nn_dim(action_net)
    return action_nn_dim+state_nn_dim

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
        result_list = np.average(result_list,axis=0)
        time_step_count+=sum(steps_list)
    else:
        result_list = F_arr(epsilons,sigma,theta)
        result_list = result_list[0]
        time_step_count+=result_list[1]
    #print('result list:', result_list)
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

def energy_action(nA,table, latent_state,all_actions):
    max_depth = nA*round(math.log(all_actions.shape[0],2))
    left,right = 0,len(table)-1#left end and right end of search region. we iteratively refine the search region
    currentDepth = 0
    while currentDepth < max_depth:
        #go to next level of depth
        mid = (left+right)/2#not an integer
        left_latent_action_sum = table[math.ceil(mid)]
        if left>0:
            left_latent_action_sum = table[math.ceil(mid)] - table[left-1]       
        right_latent_action_sum = table[right]-table[math.floor(mid)]
##        #product of energy is better than sum of energy
##        normalizing_constant = (0.1+max(abs(left_latent_action_sum@latent_state),
##                                      abs(right_latent_action_sum@latent_state)))/multiplier#test with and w/out multiplier
##        left_prob = np.exp(left_latent_action_sum@latent_state/normalizing_constant)
##        right_prob = np.exp(right_latent_action_sum@latent_state/normalizing_constant) #this is product of energy

        left_prob = max(left_latent_action_sum@latent_state,0.001)#positive random feature has some noise, and sometimes could be negative
        right_prob = max(right_latent_action_sum@latent_state,0.001)
        multiplier = 5
        left_prob= left_prob**multiplier
        right_prob = right_prob**multiplier
        
        p = left_prob/(left_prob+right_prob)
        if p>1 or math.isnan(p)==True:
            print('p: ',p,'left prob: ',left_prob,'right prob: ',right_prob)
        coin_toss = np.random.binomial(1, p)
        if coin_toss ==1:#go left
            right=math.floor(mid)
        else:#go right
            left = math.ceil(mid)
        currentDepth+=1
        #print('depth: ',currentDepth,'left: ',left,'right: ',right,'p ',p)
    return all_actions[left]

def get_table(theta,nA,w_arr):
    latent_action = get_latent_actions(theta)
    #normalize latent actions by the largest L1 norm cross actions
##    largest_L1_norm=100
##    if nA>1:
##        L1_norm = np.linalg.norm(latent_action,ord=1,axis=1)
##        largest_L1_norm = max(L1_norm)
##    else:
##        largest_L1_norm = max(np.abs(latent_action))
##    #print('normalizing factor: ',largest_L1_norm)
##    latent_action = latent_action/largest_L1_norm
    latent_action = positive_random_feature_actions(w_arr,latent_action)#transform using positive random feature
    table = np.cumsum(latent_action,axis=0)
    return table

def get_latent_state_positive_RF(state,state_net,w_arr):
    latent_state = state_feed_forward(state_net,state)
    #normalize latent state by L1 norm. 
    latent_state = latent_state/sum(np.abs(latent_state))
    #transform using positive random feature
    latent_state = positive_random_feature_state(w_arr,latent_state)
    return latent_state

def positive_random_feature_actions(w_arr,a_arr):
    #arr: each row is a latent action
    m=len(w_arr)
    phi_x_arr = np.exp(a_arr@(w_arr.T))#shape (num_actions,m)
    normalizer = np.exp(-np.power(np.linalg.norm(a_arr,axis=1),2)/2)/(m**0.5)
    phi_x_arr = (phi_x_arr.T*normalizer).T
    return phi_x_arr

def positive_random_feature_state(w_arr,s):
    #s is latent state
    m=len(w_arr)
    phi_y = np.exp(w_arr@s)
    phi_y = phi_y*np.exp(-s@s/2)/(m**0.5)#shape m
    return phi_y

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
    #preprocessing
    w_arr = np.random.normal(loc=0,scale=1,size=(200,nA))
    table = get_table(theta,nA,w_arr)
    state_net = get_state_net(theta)
##    fn = lambda a: get_latent_action_nn(nA, a, theta)
##    table = np.cumsum(np.array(list(map(fn, all_actions))), axis=0)
    while not done:
        latent_state = get_latent_state_positive_RF(state,state_net,w_arr)
        action = energy_action(nA, table, latent_state,all_actions)
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
    return [grad,steps_count]
        
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
    w_arr = np.random.normal(loc=0,scale=1,size=(200,nA))
    table = get_table(theta,nA,w_arr)
    state_net = get_state_net(theta)
##    fn = lambda a: get_latent_action_nn(nA, a, theta)
##    table = np.cumsum(np.array(list(map(fn, all_actions))), axis=0)
    while not done:
        latent_state = get_latent_state_positive_RF(state,state_net,w_arr)
        action = energy_action(nA, table, latent_state,all_actions)
        state, reward, done, _ = env.step(action)
        time_step_count+=1
        G += reward
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
    useParallel=0#if parallelize
    print("number of CPUs: ",mp.cpu_count())
    env = gym.make(env_name)
    state_dim = env.reset().size
    nA, = env.action_space.shape
    theta_dim = get_theta_dim()
    outfile = "positive_RF_{}.txt".format(env_name+str(time.time()))
    with open(outfile, "w") as f:
        f.write("")
    b = 1
    num_seeds = 1
    max_epoch = 501
    res = np.zeros((num_seeds, max_epoch))
    method = "AT_parallel"

    #all_actions = np.random.uniform(low=-1,high=1,size=(2**(3*nA),nA))
    all_actions = np.array([i for i in product([-1,-5/7, -3/7,-1/7,0,3/7,5/7,1],repeat=nA)])#num of actions must be some power of 2
    
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


