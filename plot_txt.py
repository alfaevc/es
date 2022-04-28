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
    # env_name = "MountainCar-v0"
    env_name = "Acrobot-v1"
    # env_name = "MountainCarContinuous-v0"
    # env_name = "InvertedPendulumBulletEnv-v0"
    # env_name = 'LunarLanderContinuous-v2'
    # env_name = "Hopper-v2"
    # env_name = "HalfCheetah-v2"
    # methods = ["twin", "standard"]
    methods = ["twin", "onetower", "standard"]
    # terms = ["ITT", "explicit"]
    terms = ["ITT", "IOT", "explicit"]

    d = {}
    
    twin_ev_seeds = ["1649454729.969542", "1649264146.8901012", "1649267051.6858408",
                     "1649267163.5702064", "1649355160.8114576", "1649355160.4007642",
                     "1649355161.2410011", "1649434333.5052044", "1649454680.8482714",
                     "1649190591.4257033"]
    
    standard_ev_seeds = ["1649837482.846523", "1649837477.2299407", "1649837113.5510046",
                         "1649815887.2139251", "1649815887.2133114", "1649681269.410244",
                         "1649681232.7885218", "1649679362.5310104", "1649659663.1422014",
                         "1649659626.074103"]

    onetower_ev_seeds = ["1649692287.029973", "1649659633.9536998", "1649638066.4576948",
                         "1649638011.3690753", "1649519668.0558302", "1649494375.4381204",
                         "1649475361.8918884", "1649475339.931297", "1649474178.0274796",
                         "1649638013.1301358"]
    
    d["twin"] = [str(i) for i in range(10)]
    d["standard"] = [str(i) for i in range(10)]
    d["onetower"] = [str(i) for i in range(10)]

    N = 100

    for i in range(len(methods)):
        evals_list = []
        for s in d[methods[i]]:
            evals = []
            filename = "files/{0}/{0}_{1}.txt".format(methods[i], env_name+s)
            #filename = "files/energy_vertex_sampling_lunarlander_sample_9_actions/twin_energy_LunarLanderContinuous-v2({}).txt".format(i)

            with open(filename, "r") as f:
                lines = f.readlines()
                for j in range(1,N+1,1):
                    l = list(filter(len, re.split(' |\*|\n', lines[j])))
                    evals.append(float(l[-1]))
            
            evals_list.append(np.array(evals))
    
    
        avs = np.mean(evals_list, axis=0)
        maxs = np.max(evals_list, axis=0)
        mins = np.min(evals_list, axis=0)

        ns = np.arange(N)

        plt.fill_between(ns, mins, maxs, alpha=0.35)
        plt.plot(ns, avs, '-o', markersize=1, label=terms[i])


    plt.legend(loc='lower right', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.grid(True)
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)
    plt.ylim([-500, 0])

    plt.title("ES {0}".format(env_name), fontsize = 20)
    plt.savefig("plots/{0}.png".format(env_name))


if __name__ == '__main__':
    main()