#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
import os 
import re

import numpy as np

import gym

from es_gaus_parallel import get_gaus_net, gaus_feed_forward



def test_video(theta, env_name, method):
    save_path = "./videos-%s-%s" % (env_name, method)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.make(env_name)
    env = gym.wrappers.Monitor(env, save_path, force=True)
    
    G = 0.0
    done = False
    state = env.reset()
    # a_dim = np.arange(nA)
   
    gaus_net = get_gaus_net(theta, state_dim, nA)
    while not done:
        # fn = lambda a: [theta[2*a*(state_dim+1)] + state @ theta[2*a*(state_dim+1)+1: (2*a+1)*(state_dim+1)], 
        #                 theta[(2*a+1)*(state_dim+1)] + state @ theta[(2*a+1)*(state_dim+1)+1: (2*a+2)*(state_dim+1)]]
        #mvs = np.array(list(map(fn, a_dim))).flatten()
        a_mean, a_v = gaus_feed_forward(gaus_net, state)
        action = np.tanh(np.random.normal(a_mean, a_v))
        # action = np.random.normal(a_mean[0], a_v[0])
        state, reward, done, _ = env.step(action)
        G += reward
    
    print("The return is {}".format(G))
    
global env_name
# env_name = 'FetchPush-v1'
# env_name = 'HalfCheetah-v2'
# env_name = 'Swimmer-v2'
env_name = 'Walker2d-v2'
# env_name = 'InvertedPendulumBulletEnv-v0'

  
"""The cell below applies your ES implementation to the RL objective you've defined in the cell above."""
if __name__ == '__main__':
    p = "gaus"
    method = "AT"
    env = gym.make(env_name)
    state_dim = env.reset().size
    nA, = env.action_space.shape

    theta_file = "files/{0}_theta_{1}.txt".format(p, env_name)

    with open(theta_file, "r") as f:
        l = list(filter(len, re.split(' |\*|\n', f.readlines()[0])))
        theta = np.array(l, dtype=float)
        print(theta)
    
    test_video(theta, env_name, method)

    
    
    






