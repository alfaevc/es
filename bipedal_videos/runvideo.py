#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
import os 
import re

import numpy as np

import gym

from es_twin_parallel import get_state_net, state_feed_forward, get_action_net, action_feed_forward, energy_action



def test_video(theta, env_name):
    # save_path = "./videos-%s-%s" % (env_name, method)
    save_path = "./"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    gym.logger.set_level(40)
    env = gym.make(env_name)
    env = gym.wrappers.Monitor(env, save_path, force=True)
    
    nA, = env.action_space.shape
    G = 0.0
    done = False
    state = env.reset()
    # a_dim = np.arange(nA)
    state_dim = state.size
    state_net = get_state_net(theta, state_dim, nA)
    action_net = get_action_net(theta, nA)
    while not done:
        latent_state = state_feed_forward(state_net, state)
        actions_arr = np.random.uniform(-1,1,size=(1000,nA))
        latent_actions = action_feed_forward(action_net, actions_arr)
        action = energy_action(actions_arr, latent_actions, latent_state)
        state, reward, done, _ = env.step(action)
        G += reward

    print("The return is {}".format(G))
    
global env_name
# env_name = 'FetchPush-v1'
# env_name = 'HalfCheetah-v2'
# env_name = 'Swimmer-v2'
# env_name = 'Hopper-v2'
# env_name = 'Walker2d-v2'
# env_name = 'InvertedPendulumBulletEnv-v0'
# env_name = 'Humanoid-v2'
env_name = 'BipedalWalker-v3'

  
"""The cell below applies your ES implementation to the RL objective you've defined in the cell above."""
if __name__ == '__main__':
    method = "twin"
    env = gym.make(env_name)
    state_dim = env.reset().size
    nA, = env.action_space.shape

    time = "1646983240.632449"

    theta_file = "{0}_theta_{1}.txt".format(method, env_name)

    with open(theta_file, "r") as f:
        l = list(filter(len, re.split(' |\*|\n', f.readlines()[0])))
        theta = np.array(l, dtype=float)
        print(theta)
    
    test_video(theta, env_name)

    
    
    






