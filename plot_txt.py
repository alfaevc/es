import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re
import numpy as np

'''
def main():
    n = 10
    # filename = "files/twin_energy_InvertedPendulumBulletEnv-v0.txt"
    filename = "files/energy uniform sampling lunar lander/twin_energy_LunarLanderContnuous-v2(1).txt"
    # filename = "files/energy vertex sampling lunar lander only sample 5/twin_energy_LunarLanderContnuous-v2(1).txt"
    # outfile = "a2c.txt"

    # env_name = 'FetchPush-v1'
    # env_name = 'HalfCheetah-v2'
    # env_name = 'Swimmer-v2'
    env_name = 'InvertedPendulumBullet'

    method = "AT"

    num_seeds, max_epoch = 4, 101

    act_epoch = 101

    res = np.zeros((num_seeds, act_epoch))

    with open(filename, "r") as f:
        lines = f.readlines()
        for k in range(num_seeds):
            ns = []
            evals = []
            for j in range(1, act_epoch+1):
                l = list(filter(len, re.split(' |\*|\n', lines[k*(max_epoch+1)+j])))
                ns.append(j-1)
                evals.append(float(l[-1]))
                # print(l)
            res[k] = np.array(evals)
    
    ns = range(act_epoch)
'''

def main():
    env_name = "InvertedPendulumBulletEnv-v0"
    # env_name = 'LunarLanderContinuous-v2'

    energy_vertex_res = np.zeros((3, 101))

    ev_seed = 3
    ev_epochs = 101

    ev_ns = k = range(ev_epochs)
    # ev_evals = []

    for i in range(ev_seed):
        ev_evals = []
        filename = "files/twin_energy_gradInvertedPendulumBulletEnv-v0.txt"
        #filename = "files/energy_vertex_sampling_lunarlander_sample_9_actions/twin_energy_LunarLanderContinuous-v2({}).txt".format(i)

        method = "AT"

        with open(filename, "r") as f:
            lines = f.readlines()
            for j in range(i*(ev_epochs+1)+1, i*(ev_epochs+1)+ev_epochs+1):
                l = list(filter(len, re.split(' |\*|\n', lines[j])))
                ev_evals.append(float(l[-1]))
        
            energy_vertex_res[i] = np.array(ev_evals)
    
    
    avs = np.mean(energy_vertex_res, axis=0)
    maxs = np.max(energy_vertex_res, axis=0)
    mins = np.min(energy_vertex_res, axis=0)


    plt.fill_between(ev_ns, mins, maxs, alpha=0.1)
    plt.plot(ev_ns, avs, '-o', markersize=1, label='Twin Energy Vertex')

    # plt.fill_between(ns, mins, maxs, alpha=0.1)
    # plt.plot(ns, ns, '-o', markersize=1, label='Shit')

    # plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    plt.title("{0} ES {1}".format(method, env_name), fontsize = 20)
    plt.savefig("plots/Twin Energy {0} ES {1}".format(method, env_name))


if __name__ == '__main__':
    main()