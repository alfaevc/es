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

def energy_actions(actor, critic, state, nA, K=10):
    sample_actions = np.random.uniform(low=-1.0, high=1.0, size=(K,nA))
    latent_actions, latent_states = actor(sample_actions).numpy(), np.tile(critic(np.expand_dims(state,0)).numpy().reshape(-1), (K,1))
    energies = np.einsum('ij,ij->i', latent_actions, latent_states)
    return sample_actions[np.argmin(energies)]
     
def energy_min_action(actor, critic, state):
    param1 = actor.get_layer_i_param(0)
    param2 = actor.get_layer_i_param(1)
    latent_state = critic(np.expand_dims(state,0)).numpy()
    return np.dot(np.dot(param1, param2), latent_state.T)

def F(theta, gamma=1, max_step=1e4):
    env = gym.make(env_name)
    nA, = env.action_space.shape
    state = env.reset()
    state_dim = state.size


    actor = NN(state_dim, nA, layers=[2*nA])
    '''
    b=1
    actor.compile(optimizer=actor.optimizer, loss=actor.loss)
    actor.fit(np.random.standard_normal((b,nA)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
    '''
    critic = NN(state_dim, nA, layers=[nA])
    '''
    critic.compile(optimizer=critic.optimizer, loss=actor.loss)
    critic.fit(np.random.standard_normal((b,state_dim)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
    '''
    actor_theta_len = actor.nnparams2theta().size

    steps_count=0
    
    G = 0.0
    done = False
    discount = 1
    actor.update_params(actor.theta2nnparams(theta[:actor_theta_len], nA, nA))
    critic.update_params(critic.theta2nnparams(theta[actor_theta_len:], state_dim, nA))
    while not done:
        action = energy_actions(actor, critic, state, nA, K=nA*10)
        # action = energy_min_action(actor, critic, state)
        state, reward, done, _ = env.step(action)
        G += reward * discount
        discount *= gamma
        steps_count+=1
    return G, steps_count

def eval(theta):
    env = gym.make(env_name)
    nA, = env.action_space.shape
    state = env.reset()
    state_dim = state.size

    b=1
    actor = NN(nA, layers=[2*nA])
    actor.compile(optimizer=actor.optimizer, loss=actor.loss)
    actor.fit(np.random.standard_normal((b,nA)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
    critic = NN(nA, layers=[nA])
    critic.compile(optimizer=critic.optimizer, loss=actor.loss)
    critic.fit(np.random.standard_normal((b,state_dim)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
    actor_theta_len = actor.nnparams2theta().size

    global time_step_count
    G = 0.0
    done = False
    actor.update_params(actor.theta2nnparams(theta[:actor_theta_len], nA, nA))
    critic.update_params(critic.theta2nnparams(theta[actor_theta_len:], state_dim, nA))
    while not done:
        action = energy_actions(actor, critic, state, nA, K=nA*10)
        #action = energy_min_action(actor, critic, state)
        state, reward, done, _ = env.step(action)
        G += reward
        time_step_count+=1
    return G

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
    #fn = lambda x: (F(theta + sigma * x) - F(theta - sigma * x)) * x
    #return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2
    return [grad,steps_count]
        
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
    nA, = env.action_space.shape
    state = env.reset()
    state_dim = state.size

    b=1
    actor = NN(nA, layers=[2*nA])
    actor.compile(optimizer=actor.optimizer, loss=actor.loss)
    actor.fit(np.random.standard_normal((b,nA)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
    critic = NN(nA, layers=[nA])
    critic.compile(optimizer=critic.optimizer, loss=actor.loss)
    critic.fit(np.random.standard_normal((b,state_dim)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
    actor_theta_len = actor.nnparams2theta().size
    critic_theta_len = critic.nnparams2theta().size
    theta_dim = actor_theta_len + critic_theta_len
    
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


