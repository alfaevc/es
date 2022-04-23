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
from sklearn import preprocessing

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
        self.fc1 = nn.Linear(state_dim, nA, bias=False)  
        self.fc2 = nn.Linear(nA, nA, bias=False)
        self.fc3 = nn.Linear(nA, nA, bias=False)
        self.fc4 = nn.Linear(nA, nA, bias=False)

def state_feed_forward(state_net,state):#have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
    x = (torch.from_numpy(state)).float()
    x = torchF.relu(state_net.fc1(x))
    x = torchF.relu(state_net.fc2(x))
    x = torchF.relu(state_net.fc3(x))
    x = state_net.fc4(x)
    latent_state = x.detach().numpy()
    return latent_state

class action_tower(nn.Module):
    def __init__(self):
        super(action_tower, self).__init__()
        self.fc1 = nn.Linear(nA, nA, bias=False)#can automate this. create nn for any given input layer dimensions, instead of fixed dimensions  
        self.fc2 = nn.Linear(nA, nA, bias=False)

def action_feed_forward(action_net,action):#have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
    x = (torch.from_numpy(action)).float()
    x = torchF.relu(action_net.fc1(x))
    x = action_net.fc2(x)
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
    
def get_theta_dim():
    state_net = state_tower()
    action_net = action_tower()
    state_nn_dim = get_nn_dim(state_net)
    action_nn_dim = get_nn_dim(action_net)
    return action_nn_dim+state_nn_dim,action_nn_dim,state_nn_dim

def action_feed_forward_efficient(action_net,actions_arr):
    #divide the work. else will take too long time once we scale
    latent_actions_arr = np.zeros((sample_size*unit,nA))
    for i in range(sample_size):
        latent_actions_arr[unit*i:unit*(i+1)] = action_feed_forward(action_net,actions_arr[unit*i:unit*(i+1)])
    return latent_actions_arr

#############################################################################################################################################


def collect_result(result):
    result_list.append(result[0])
    steps_list.append(result[1])    
    
def AT_gradient_parallel(useParallel, theta, sigma=1, N=100,update_action=0):
    numCPU=mp.cpu_count()
    pool=mp.Pool(numCPU)
    epsilons=orthogonal_epsilons(N,theta.size)
    global result_list,steps_list,time_step_count,table_all,sorted_actions_arr_all
    result_list = []; steps_list = []
    # grad_multiplier = epsilons[0]/(sigma*2)
    # if update_action==0:
    #     output1, time1 = F(theta + sigma * epsilons[0],table_all,sorted_actions_arr_all,grad_multiplier)
    #     output2, time2 = F(theta - sigma * epsilons[0],table_all,sorted_actions_arr_all,-grad_multiplier)
    # else:
    #     output1, time1 = F(theta + sigma * epsilons[0],table_all[0],sorted_actions_arr_all[0],grad_multiplier)
    #     output2, time2 = F(theta - sigma * epsilons[0],table_all[0],sorted_actions_arr_all[0],-grad_multiplier)
    # grad=output1+output2
    # print('grad ',grad)
    num_jobs = action_nn_dim
    if update_action == 0:
        num_jobs = state_nn_dim
    table_all,sorted_actions_arr_all = get_table_all(theta,sigma,epsilons[:num_jobs],update_action)

    for i in range(num_jobs):
        if update_action==1:
            pool.apply_async(F,args = (theta + sigma * epsilons[i], table_all[i],
                                       sorted_actions_arr_all[i],epsilons[i]/(sigma*2)),callback=collect_result)
            pool.apply_async(F,args = (theta - sigma * epsilons[i], table_all[num_jobs+i],
                                       sorted_actions_arr_all[num_jobs+i],-epsilons[i]/(sigma*2)),callback=collect_result)
        else:#no update to action tower. use the same table
            pool.apply_async(F,args = (theta + sigma * epsilons[i], table_all,
                                       sorted_actions_arr_all,epsilons[i]/(sigma*2)),callback=collect_result)
            pool.apply_async(F,args = (theta - sigma * epsilons[i], table_all,
                                       sorted_actions_arr_all,-epsilons[i]/(sigma*2)),callback=collect_result)
    pool.close()
    pool.join()
    grad = np.average(result_list,axis=0)
    time_step_count+=sum(steps_list)
    # print('result list length: ',len(result_list),'result list[0] shape: ',result_list[0].shape)
    # print('result list[1]:', result_list[1])
    if update_action == 0:
        grad[:action_nn_dim]=np.zeros(action_nn_dim)#actions params are not updated
    else:
        grad[action_nn_dim:] = np.zeros(state_nn_dim)
    # print('final grad ',grad)
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

def update_or_not(i):
    if i%5==0:
        return 1
    else:
        return 0

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
    if i%1==0:
        print('runtime until now: ',time.time()-t1)#, ' time step: ',time_step_count)
    #if time_step_count>= 10**7: #terminate at given time step threshold.
    #    sys.exit()
    theta += eta * AT_gradient_parallel(useParallel, theta, sigma, N,update_or_not(i))
  return theta, accum_rewards


def energy_action(nA, table, latent_state, actions_arr):
    left, right = 0, len(table) - 1  # left end and right end of search region. we iteratively refine the search region
    currentDepth = 0
    while currentDepth < max_depth:
        # go to next level of depth
        mid = (left + right) / 2  # not an integer
        left_latent_action_sum = table[math.ceil(mid)]
        if left > 0:
            left_latent_action_sum = left_latent_action_sum - table[left - 1]
        right_latent_action_sum = table[right] - table[math.floor(mid)]
        # product of energy is better than sum of energy
        multiplier = 5
        left_dp = left_latent_action_sum @ latent_state
        right_dp = right_latent_action_sum @ latent_state
        normalizing_constant = (0.1 + max(abs(left_dp),abs(right_dp))) / multiplier
        left_prob = np.exp(left_dp / normalizing_constant)
        right_prob = np.exp(right_dp / normalizing_constant)  # this is product of energy

        p = left_prob / (left_prob + right_prob)
        if p > 1 or math.isnan(p) == True:
            print('p: ', p, 'left prob: ', left_prob, 'right prob: ', right_prob)
        coin_toss = np.random.binomial(1, p)
        if coin_toss == 1:  # go left
            right = math.floor(mid)
        else:  # go right
            left = math.ceil(mid)
        currentDepth += 1
        # print('depth: ',currentDepth,'left: ',left,'right: ',right,'p ',p)
    return actions_arr[left]


def get_table(theta, actions_arr):
    action_net = get_action_net(theta)
    latent_action = action_feed_forward_efficient(action_net, actions_arr)
    # sort latent actions
    min_max_scaler = preprocessing.MinMaxScaler()
    multiplier = 0.4 # rescale latent actions to [0,multiplier]
    normalized_latent_actions = multiplier*min_max_scaler.fit_transform(latent_action)
    normalized_latent_actions = np.round(normalized_latent_actions, 1)
    index_arr = np.lexsort(normalized_latent_actions.T)  # sort lexicographically

    table = np.cumsum(latent_action[index_arr], axis=0)
    return table, actions_arr[index_arr]

def get_table_all(theta,sigma,epsilons,update_action):
    if update_action==0:#generate one table
        actions_arr=np.random.uniform(low=-1, high=1, size=(sample_size * unit, nA))
        table_all, sorted_actions_arr_all = get_table(theta, actions_arr)
    else:#generate many tables
        table_all=np.zeros((2*epsilons.shape[0],sample_size * unit, nA))
        sorted_actions_arr_all=np.zeros((2*epsilons.shape[0],sample_size * unit, nA))
        for i in range(epsilons.shape[0]):
            actions_arr = np.random.uniform(low=-1, high=1, size=(sample_size * unit, nA))
            table_all[i], sorted_actions_arr_all[i] = get_table(theta+sigma * epsilons[i], actions_arr)
            actions_arr = np.random.uniform(low=-1, high=1, size=(sample_size * unit, nA))
            table_all[i+epsilons.shape[0]], sorted_actions_arr_all[i+epsilons.shape[0]] = get_table(
                theta-sigma * epsilons[i], actions_arr)
    return table_all,sorted_actions_arr_all

def F(theta, table,sorted_actions_arr,grad_multiplier):
    gym.logger.set_level(40);gamma=1
    env = gym.make(env_name)#this takes no time
    G = 0.0
    done = False
    discount = 1
    state = env.reset()
    steps_count=0#cannot use global var here because subprocesses cannot edit global var
    state_net = get_state_net(theta)
    while not done:
        latent_state = state_feed_forward(state_net,state)
        action = energy_action(nA, table, latent_state,sorted_actions_arr)
        state, reward, done, _ = env.step(action)
        steps_count+=1
        G += reward * discount
        discount *= gamma
    return G*grad_multiplier,steps_count


def eval(theta):
    gym.logger.set_level(40)
    env = gym.make(env_name)#this takes no time
    G = 0.0
    done = False
    state = env.reset()
    global time_step_count
    #preprocessing
    actions_arr = np.random.uniform(low=-1, high=1, size=(sample_size * unit, nA))
    table,sorted_actions_arr = get_table(theta,actions_arr)
    state_net = get_state_net(theta)
    while not done:
        latent_state = state_feed_forward(state_net,state)
        action = energy_action(nA, table, latent_state,sorted_actions_arr)
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
    env = gym.make(env_name); state_dim = env.reset().size; nA, = env.action_space.shape
    sample_size = 64;unit =512 #total sample amount = sampling_size*unit
    theta_dim,action_nn_dim,state_nn_dim = get_theta_dim()
    outfile = "positive_RF_{}.txt".format(env_name+str(time.time()))
    with open(outfile, "w") as f:
        f.write("")
    num_seeds = 1
    max_epoch = 501
    max_depth = nA * round(math.log(sample_size * unit, 2)) #max tree depth
 
    t_start=time.time()
    for k in tqdm.tqdm(range(num_seeds)):
        N = theta_dim
        theta0 = np.random.standard_normal(size=theta_dim)
        time_elapsed = int(round(time.time()-t_start))
        with open(outfile, "a") as f:
            f.write("Seed {}:\n".format(k))
        theta, accum_rewards = gradascent(useParallel, theta0, outfile, sigma=1, eta=1e-2, max_epoch=max_epoch, N=N)


