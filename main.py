import torch
import os
import argparse
import math
import os
import dgl
import statistics
import json
import random

from torch import nn
from torch import Tensor
from torch.nn import Parameter
from tqdm import tqdm


import dgl.nn as dglnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


import evaluate
import inc_exp_sto
# import dist_exp



args = {
   "lr" : 1e-3,
   "weight_decay" : 5e-4,
   "dropout" : 0,
   "epoches" : 1,
   "num_layer" : 2,
   "num_hidfeat" : 64,
   "random_seed":1,
   # "replay_epoch":5,
   "output_act":"log_softmax",
   'cmd_r':1,
   'replay_size':2,
   "loss": "cross_entropy",
   "iid_size":50
}

def fix_random_seed(seed):
   torch.manual_seed(seed)
   random.seed(seed)
   np.random.seed(seed)
   return


def main():
    parser = argparse.ArgumentParser(description = 'Continual Learning')
    parser.add_argument('-d', '--data', type = str, default = 'cora',
                        help = 'dataset used, can be chosen from cora, coau_cs, reddit')
    a = parser.parse_args()
    data = a.data

    strategies = ['uniform','degree','cluster','center','max_cover','centroid']
    for method in strategies:
        file_dir = 'test_result/'+data+'_it_'+method+'.txt'
        args['epoches'] = 10
        args['num_hidfeat'] = 64
        args['replay_size'] = 2
        with open(file_dir, 'w') as file:
            for random_seed in range(5):
                fix_random_seed(random_seed)
                phase_accuracy = inc_exp_sto.incremental_trainig_replay(args, data, 'SAGE', method)
                print(phase_accuracy)
                last_phase = phase_accuracy[-1]

                class_acc_list = last_phase['prev_class_acc']
                class_forget_list = last_phase['prev_class_forget']

                class_acc_list.append(last_phase['test_accuracy'])
                print(class_acc_list)
                print(class_forget_list)

                atp = sum(class_acc_list)/len(class_acc_list)
                atf = sum(class_forget_list)/(len(class_forget_list))

                randomseed_line = 'random seed: ' + str(random_seed)
                atpline  = ' atp: ' + str(atp)
                atfline  = ' atf: ' + str(atf) + '\n'


                file.write(randomseed_line)
                file.write(atpline)
                file.write(atfline)

def test():
    method = 'uniform'
    file_dir = 'test_result/test_cora_it_'+method+'.txt'
    args['epoches'] = 10
    args['num_hidfeat'] = 64
    args['replay_size'] = 2
    with open(file_dir, 'w') as file:
        for random_seed in range(2):
            fix_random_seed(random_seed)
            phase_accuracy = inc_exp_sto.incremental_trainig_replay(args, 'cora', 'SAGE', method)
            print(phase_accuracy)
            last_phase = phase_accuracy[-1]

            class_acc_list = last_phase['prev_class_acc']
            class_forget_list = last_phase['prev_class_forget']

            class_acc_list.append(last_phase['test_accuracy'])
            print(class_acc_list)
            print(class_forget_list)

            atp = sum(class_acc_list)/len(class_acc_list)
            atf = sum(class_forget_list)/(len(class_forget_list))

            randomseed_line = 'random seed: ' + str(random_seed)
            atpline  = ' atp: ' + str(atp)
            atfline  = ' atf: ' + str(atf) + '\n'


            file.write(randomseed_line)
            file.write(atpline)
            file.write(atfline)

if __name__ == '__main__':
    main()

    
