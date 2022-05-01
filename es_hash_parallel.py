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
import re


def update_nn_params(input_nn, new_params):
    params = list(input_nn.parameters())
    current_index = 0
    for i in range(len(params)):
        shape = params[i].data.detach().numpy().shape
        if len(shape) > 1:  # params[i] is 2d tensor
            arr = new_params[current_index:current_index + shape[0] * shape[1]].reshape(shape)
            params[i].data = (torch.from_numpy(arr)).float()
            current_index += shape[0] * shape[1]
        else:  # params[i] is 1d tensor
            arr = new_params[current_index:current_index + shape[0]]
            params[i].data = (torch.from_numpy(arr)).float()
            current_index += shape[0]
    for param in params:  # freeze params
        param.requires_grad = False


def get_nn_dim(input_nn):
    params = list(input_nn.parameters())
    counter = 0
    for i in range(len(params)):
        shape = params[i].data.detach().numpy().shape
        if len(shape) > 1:  # params[i] is 2d tensor
            counter += shape[0] * shape[1]
        else:  # params[i] is 1d tensor
            counter += shape[0]
    return counter


class state_tower(nn.Module):
    def __init__(self):
        super(state_tower, self).__init__()
        self.fc1 = nn.Linear(state_dim, nA, bias=False)
        self.fc2 = nn.Linear(nA, nA, bias=False)
        self.fc3 = nn.Linear(nA, nA, bias=False)
        self.fc4 = nn.Linear(nA, nA, bias=False)


def state_feed_forward(state_net,
                       state):  # have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
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
        self.fc1 = nn.Linear(nA, nA,
                             bias=False)  # can automate this. create nn for any given input layer dimensions, instead of fixed dimensions
        self.fc2 = nn.Linear(nA, nA, bias=False)


def action_feed_forward(action_net,
                        action):  # have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
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
    update_nn_params(action_net, theta[:action_nn_dim])
    return action_net


def get_theta_dim():
    state_net = state_tower()
    action_net = action_tower()
    state_nn_dim = get_nn_dim(state_net)
    action_nn_dim = get_nn_dim(action_net)
    return action_nn_dim + state_nn_dim, action_nn_dim, state_nn_dim


def action_feed_forward_efficient(action_net, actions_arr):
    # divide the work. else will take too long time once we scale
    latent_actions_arr = np.zeros((sample_size * unit, nA))
    for i in range(sample_size):
        latent_actions_arr[unit * i:unit * (i + 1)] = action_feed_forward(action_net,
                                                                          actions_arr[unit * i:unit * (i + 1)])
    return latent_actions_arr


#############################################################################################################################################


def collect_result(result):
    result_list.append(result[0])
    steps_list.append(result[1])


def AT_gradient_parallel(useParallel, theta, sigma=1, N=100, update_action=0):
    numCPU = mp.cpu_count()
    pool = mp.Pool(numCPU)
    epsilons = orthogonal_epsilons(N, theta.size)
    global result_list, steps_list, time_step_count, table_all, sorted_actions_arr_all
    result_list = [];
    steps_list = []
    num_jobs = theta_dim
    epsilons = orthogonal_epsilons(theta_dim, theta_dim)
    # num_jobs = action_nn_dim
    # if update_action == 0:
    #     num_jobs = state_nn_dim
    #     epsilons = orthogonal_epsilons(state_nn_dim, theta.size)
    #     epsilons[:, :action_nn_dim] = np.zeros((num_jobs, action_nn_dim))
    # else:
    #     epsilons = orthogonal_epsilons(action_nn_dim, theta.size)
    #     epsilons[:, action_nn_dim:] = np.zeros((num_jobs, state_nn_dim))
    actions_arr_all, latent_actions_arr_all, medians_all, intervals_all, projection_vector_all \
        = get_multiple_tables(update_action, sigma, epsilons, theta)
    for i in range(num_jobs):
        # if update_action==1:
        pool.apply_async(F, args=(theta + sigma * epsilons[i], actions_arr_all[i], latent_actions_arr_all[i],
                                  medians_all[i], intervals_all[i], projection_vector_all[i],
                                  epsilons[i] / (sigma * 2)), callback=collect_result)
        pool.apply_async(F, args=(theta - sigma * epsilons[i], actions_arr_all[num_jobs + i] \
                                      , latent_actions_arr_all[num_jobs + i], medians_all[num_jobs + i],
                                  intervals_all[num_jobs + i] \
                                      , projection_vector_all[num_jobs + 1], -epsilons[i] / (sigma * 2)),
                         callback=collect_result)
        # else:
        #     pool.apply_async(F, args=(theta + sigma * epsilons[i], actions_arr_all, latent_actions_arr_all,
        #                               medians_all, intervals_all, projection_vector_all,
        #                               epsilons[i] / (sigma * 2)), callback=collect_result)
        #     pool.apply_async(F, args=(theta - sigma * epsilons[i], actions_arr_all, latent_actions_arr_all,
        #                               medians_all,intervals_all, projection_vector_all, -epsilons[i] / (sigma * 2)),
        #                               callback=collect_result)
    pool.close()
    pool.join()
    grad = np.average(result_list, axis=0)
    time_step_count += sum(steps_list)
    # print('result list length: ',len(result_list),'result list[0] shape: ',result_list[0].shape)
    # print('result list[1]:', result_list[1])
    # if update_action == 0:
    #     grad[:action_nn_dim] = np.zeros(action_nn_dim)  # actions params are not updated
    # else:
    #     grad[action_nn_dim:] = np.zeros(state_nn_dim)
    # print('final grad ',grad)
    return grad


def get_multiple_tables(update_action, sigma, epsilons, theta):
    #if update_action == 1:
    n_tables = epsilons.shape[0]
    actions_arr_all = np.zeros((2 * n_tables, sample_size * unit, nA))
    latent_actions_arr_all = np.zeros((2 * n_tables, sample_size * unit, nA))
    medians_all = [0] * n_tables * 2;
    intervals_all = [0] * n_tables * 2
    projection_vector_all = np.zeros((2 * n_tables, n_proj_vec, nA + 1))
    for i in range(n_tables):
        latent_actions_arr_all[i], actions_arr_all[i], medians_all[i], intervals_all[i], projection_vector_all[i] \
            = construct_hashing_table(theta + sigma * epsilons[i])
        latent_actions_arr_all[n_tables + i], actions_arr_all[n_tables + i], medians_all[n_tables + i] \
            , intervals_all[n_tables + i], projection_vector_all[n_tables + i] = construct_hashing_table(
            theta - sigma * epsilons[i])
    # else:
    #     latent_actions_arr_all, actions_arr_all, medians_all, intervals_all, projection_vector_all \
    #         = construct_hashing_table(theta)
    return actions_arr_all, latent_actions_arr_all, medians_all, intervals_all, projection_vector_all


def orthogonal_epsilons(N, dim):
    epsilons_N = np.zeros((math.ceil(N / dim) * dim, dim))
    for i in range(0, math.ceil(N / dim)):
        epsilons = np.random.standard_normal(size=(dim, dim))
        Q, _ = np.linalg.qr(epsilons)  # orthogonalize epsilons
        Q_normalize = np.copy(Q)
        fn = lambda x, y: np.linalg.norm(x) * y
        # renormalize rows of Q by multiplying it by length of corresponding row of epsilons
        Q_normalize = np.array(list(map(fn, epsilons, Q_normalize)))
        epsilons_N[i * dim:(i + 1) * dim] = Q_normalize @ Q
    return epsilons_N[0:N]


def update_or_not(i):
    if i % 5 == 0:
        return 1
    else:
        return 0


def gradascent(useParallel, theta0, filename, method=None, sigma=1, eta=1e-3, max_epoch=200, N=100):
    theta = np.copy(theta0)
    accum_rewards = np.zeros(max_epoch)
    t1 = time.time()
    global time_step_count
    for i in range(max_epoch):
        accum_rewards[i] = eval(theta)
        if i % 1 == 0:
            print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))
            with open(filename, "a") as f:
                f.write("%.d %.2f \n" % (i, accum_rewards[i]))
                # f.write("%.d %.2f %.d \n" % (i, accum_rewards[i],time_step_count))
        if i % 1 == 0:
            print('runtime until now: ', time.time() - t1)  # , ' time step: ',time_step_count)
        # if time_step_count>= 10**7: #terminate at given time step threshold.
        #    sys.exit()
        theta += eta * AT_gradient_parallel(useParallel, theta, sigma, N, update_or_not(i))
        out_theta_file = "twin_theta_{}.txt".format(env_name)
        np.savetxt(out_theta_file, theta, delimiter=' ', newline=' ')
    return theta, accum_rewards


def construct_hashing_table(theta):
    projection_vector = orthogonal_epsilons(n_proj_vec, nA + 1)
    action_net = get_action_net(theta)
    actions_arr = np.random.uniform(low=-1, high=1, size=(sample_size * unit, nA))
    latent_actions_arr = action_feed_forward_efficient(action_net, actions_arr)
    # add extra dim to make two norms equal. So max dot product is equivalent to max cosine similarity
    latent_actions_norm = np.linalg.norm(latent_actions_arr, axis=1)
    max_norm = max(latent_actions_norm)
    extra_dim = np.sqrt(max_norm - latent_actions_norm)
    aug_latent_actions_arr = np.hstack((latent_actions_arr, extra_dim.reshape((-1, 1))))
    medians = np.median(aug_latent_actions_arr @ projection_vector.T,axis=0)
    # generate keys
    binary_vecs = np.sign(aug_latent_actions_arr @ projection_vector.T-medians)  # shape is (num_actions,n_proj_vec)
    binary_vecs = (binary_vecs + 1) / 2
    powers = 2 ** np.arange(n_proj_vec)
    keys = np.dot(binary_vecs, powers)
    ind = np.argsort(keys)
    keys = keys[ind]
    # make hash table
    keys_val, idx_start, count = np.unique(keys, return_counts=True, return_index=True)
    arr = np.arange(2 ** n_proj_vec).astype(float)
    start_points = abs(arr[:, None] - keys_val[None, :]).argmin(axis=-1)
    start_points = idx_start[start_points]
    end_points = np.minimum(start_points + query_size, sample_size * unit - 1)
    start_points = np.minimum(start_points, sample_size * unit - query_size)
    # just so that we always query the same amount of points
    intervals_all = np.hstack(
        (start_points.reshape((-1, 1)), end_points.reshape((-1, 1))))  # shape should be np.zeros((2**n_proj_vec,2))
    intervals_all = intervals_all.astype(int)
    return latent_actions_arr, actions_arr, medians, intervals_all, projection_vector


def energy_action(latent_actions_arr, actions_arr, latent_state, medians, intervals, projection_vector,powers):
    binary_rep = (np.sign(projection_vector @ latent_state-medians) + 1) / 2
    query = int(round(binary_rep @ powers))
    latent_actions_queried = latent_actions_arr[intervals[query,0]:intervals[query,1]]
    ind = np.argmax(
        latent_actions_queried @ latent_state)  # take argmax here, because hash table gives us largest cosine similarity
    #actions_required = actions_arr[intervals[to_query][0]:endpoint]
    return actions_arr[intervals[query,0]+ind]


def F(theta, actions_arr, latent_actions_arr, medians, intervals, projection_vector, grad_multiplier):
    gym.logger.set_level(40);
    gamma = 1
    env = gym.make(env_name)  # this takes no time
    G = 0.0
    done = False
    discount = 1
    state = env.reset()
    steps_count = 0  # cannot use global var here because subprocesses cannot edit global var
    state_net = get_state_net(theta)
    projection_vector = projection_vector[:,:nA]#last entry is just to normalize actions
    powers = 2 ** np.arange(n_proj_vec)
    while not done:
        latent_state = state_feed_forward(state_net, state)
        action = energy_action(latent_actions_arr, actions_arr, latent_state, medians, intervals,
                                       projection_vector,powers)
        state, reward, done, _ = env.step(action)
        steps_count += 1
        G += reward * discount
        discount *= gamma
    return G * grad_multiplier, steps_count


def eval(theta):
    gym.logger.set_level(40)
    env = gym.make(env_name)  # this takes no time
    G = 0.0
    done = False
    state = env.reset()
    global time_step_count
    latent_actions_arr, actions_arr, medians, intervals, projection_vector = construct_hashing_table(theta)
    state_net = get_state_net(theta)
    #tot_improve = 0
    projection_vector = projection_vector[:,:nA]#last entry is just to normalize actions
    powers = 2 ** np.arange(n_proj_vec)
    while not done:
        latent_state = state_feed_forward(state_net, state)
        action = energy_action(latent_actions_arr, actions_arr, latent_state, medians, intervals,
                                       projection_vector,powers)
        state, reward, done, _ = env.step(action)
        time_step_count += 1
        G += reward
        #tot_improve += improve / 1000
    # print('advantage of using hash ',tot_improve)
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
time_step_count = 0

if __name__ == '__main__':
    useParallel = 1  # if parallelize
    import_theta = False
    theta_file = 'twin_theta_'+env_name+'.txt'
    env = gym.make(env_name);
    state_dim = env.reset().size;
    nA, = env.action_space.shape
    sample_size = 16;
    unit = 1024  # total sample amount = sampling_size*unit
    n_proj_vec = 6  # num of projection vectors to use in hash table
    query_size  = int(round(sample_size*unit/(2**n_proj_vec))) # # of actions to query. query according to hash table.
    print('query size: ',query_size)

    theta_dim, action_nn_dim, state_nn_dim = get_theta_dim()
    outfile = "twin_hash_{}.txt".format(env_name + str(time.time()))
    with open(outfile, "w") as f:
        f.write("")
    num_seeds = 1
    max_epoch = 4001

    t_start = time.time()
    for k in tqdm.tqdm(range(num_seeds)):
        N = theta_dim
        theta0 = np.random.standard_normal(size=theta_dim)
        if import_theta:
            with open(theta_file, "r") as g:
                l = list(filter(len, re.split(' |\*|\n', g.readlines()[0])))
                #theta0 = np.array(l)#theta0 will be strings, not float
            for i in range(len(l)):#convert string to float
                theta0[i] = float(l[i])
        time_elapsed = int(round(time.time()-t_start))
        with open(outfile, "a") as f:
            f.write("Seed {}:\n".format(k))
        theta, accum_rewards = gradascent(useParallel, theta0, outfile, sigma=1, eta=1e-2, max_epoch=max_epoch, N=N)


