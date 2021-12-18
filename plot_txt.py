import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re
import numpy as np

def main():
    n = 10
    filename = "files/twin_energy_InvertedPendulumBulletEnv-v0.txt"
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

    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    # method = "mixed"

    plt.fill_between(ns, mins, maxs, alpha=0.1)
    plt.plot(ns, avs, '-o', markersize=1)

    # plt.fill_between(ns, mins, maxs, alpha=0.1)
    # plt.plot(ns, ns, '-o', markersize=1, label='Shit')

    # plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    plt.title("Energy Twin {0} ES {1}".format(method, env_name), fontsize = 20)
    plt.savefig("plots/Energy twin {0} ES {1}".format(method, env_name))


if __name__ == '__main__':
    main()