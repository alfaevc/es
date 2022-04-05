import numpy as np
import statistics
import csv
#input filename and nFiles
filename = 'twin_HalfCheetah-v2 (1).txt'
nFiles=3
def single_file_output(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def toArr(lines):
    n_ep = len(lines)-1
    reward_arr = []
    for i in range(1,len(lines)):
        for j in range(len(lines[i])):
            if lines[i][j]==' ':
                reward_arr.append(float(lines[i][j+1:]))
    return np.array(reward_arr)

def all_files_output(filename,nFiles):
    prefix=filename[0:len(filename)-6]
    postfix=').txt'
    all_reward_arr = []
    for i in range(nFiles):
        filename=prefix+str(i+1)+postfix
        lines = single_file_output(filename)
        reward_arr = toArr(lines)
        all_reward_arr = np.append(all_reward_arr,reward_arr)
        print('i = ',i+1)
    return all_reward_arr

all_reward_arr = all_files_output(filename,nFiles)
numIter = len(all_reward_arr)
#plot 
from matplotlib import pyplot
pyplot.plot(np.arange(numIter), all_reward_arr)
title='Half Cheetah Twin Tower'
pyplot.title(title)
pyplot.xlabel('epoch')
pyplot.ylabel('reward')
#pyplot.legend(loc='best')
pyplot.show()
