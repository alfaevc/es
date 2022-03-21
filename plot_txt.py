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
    # env_name = "InvertedPendulumBulletEnv-v0"
    # env_name = 'LunarLanderContinuous-v2'
    env_name = "Hopper-v2"

    time = "1646959575.0228987"

    evals_list = []

    ev_seed = 1

    method = "gaus"

    for i in range(ev_seed):
        evals = []
        filename = "files/{}_{}.txt".format(method, env_name + time)
        #filename = "files/energy_vertex_sampling_lunarlander_sample_9_actions/twin_energy_LunarLanderContinuous-v2({}).txt".format(i)

        with open(filename, "r") as f:
            lines = f.readlines()
            N = len(lines)
            for j in range(1,N,1):
                l = list(filter(len, re.split(' |\*|\n', lines[j])))
                evals.append(float(l[-1]))
        
            evals_list.append(np.array(evals))
    
    
    avs = np.mean(evals_list, axis=0)
    maxs = np.max(evals_list, axis=0)
    mins = np.min(evals_list, axis=0)

    ns = np.arange(N-1)

    plt.fill_between(ns, mins, maxs, alpha=0.1)
    plt.plot(ns, avs, '-o', markersize=1)

    # plt.fill_between(ns, mins, maxs, alpha=0.1)
    # plt.plot(ns, ns, '-o', markersize=1, label='Shit')

    # plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    plt.title("{0} ES {1}".format(method, env_name), fontsize = 20)
    plt.savefig("plots/{0} ES {1}.png".format(method, env_name+time))


if __name__ == '__main__':
    main()