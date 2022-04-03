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
 

if __name__ == '__main__':
   # inc_exp_sto.regular_training(args,'reddit','SAGE')


   # file_dir = 'result/cora_it_nor.txt'
   # file = open(file_dir, "w")
   # args['epoches'] = 30
   # args['num_hidfeat'] = 16
   # for random_seed in range(50):
   #    fix_random_seed(random_seed)
   #    phase_accuracy = inc_exp_sto.incremental_trainig(args,'cora','SAGE')
   #    last_phase = phase_accuracy[-1]

   #    class_acc_list = last_phase['prev_class_acc']
   #    class_forget_list = last_phase['prev_class_forget']

   #    class_acc_list.append(last_phase['test_accuracy'])

   #    atp = sum(class_acc_list)/len(class_acc_list)
   #    atf = sum(class_forget_list)/(len(class_forget_list))

   #    randomseed_line = 'random seed: ' + str(random_seed) 
   #    atpline  = ' atp: ' + str(atp) 
   #    atfline  = ' atf: ' + str(atf) + '\n'
      
   #    file.write(randomseed_line)
   #    file.write(atpline)
   #    file.write(atfline)
   # file.close()

   # file_dir = 'result/cs_it_nor.txt'
   # file = open(file_dir, "w")
   # args['epoches'] = 15
   # args['num_hidfeat'] = 32
   # for random_seed in range(20):
   #    fix_random_seed(random_seed)
   #    phase_accuracy = inc_exp_sto.incremental_trainig(args,'coau_cs','SAGE')
   #    last_phase = phase_accuracy[-1]

   #    class_acc_list = last_phase['prev_class_acc']
   #    class_forget_list = last_phase['prev_class_forget']

   #    class_acc_list.append(last_phase['test_accuracy'])

   #    atp = sum(class_acc_list)/len(class_acc_list)
   #    atf = sum(class_forget_list)/(len(class_forget_list))

   #    randomseed_line = 'random seed: ' + str(random_seed) 
   #    atpline  = ' atp: ' + str(atp) 
   #    atfline  = ' atf: ' + str(atf) + '\n'
      
   #    file.write(randomseed_line)
   #    file.write(atpline)
   #    file.write(atfline)
   # file.close()

      

   # phase_accuracy = inc_exp_sto.incremental_trainig(args,'reddit','SAGE')
   # phase_accuracy = inc_exp_sto.incremental_trainig(args,'coau_cs','SAGE')

   # strategy list
   # strategies = ['uniform','degree','cluster','center','max_cover','centroid']
   # # strategies = ['max_cover','centroid']
   # # strategies = ['centroid']
   # for method in strategies:
   #    file_dir = 'result/cora_it_'+method+'.txt'
   #    file = open(file_dir, "w")
   #    args['epoches'] = 30
   #    args['num_hidfeat'] = 16
   #    args['replay_size'] = 1
   #    for random_seed in range(5):
   #       fix_random_seed(random_seed)
   #       phase_accuracy = inc_exp_sto.incremental_trainig_replay(args,'cora','SAGE',method)
   #       last_phase = phase_accuracy[-1]

   #       class_acc_list = last_phase['prev_class_acc']
   #       class_forget_list = last_phase['prev_class_forget']

   #       class_acc_list.append(last_phase['test_accuracy'])

   #       atp = sum(class_acc_list)/len(class_acc_list)
   #       atf = sum(class_forget_list)/(len(class_forget_list))

   #       randomseed_line = 'random seed: ' + str(random_seed) 
   #       atpline  = ' atp: ' + str(atp) 
   #       atfline  = ' atf: ' + str(atf) + '\n'
         
   #       file.write(randomseed_line)
   #       file.write(atpline)
   #       file.write(atfline)
   #    file.close()

   # for method in strategies:
   #    file_dir = 'result/cs_it_'+method+'_cmd.txt'
   #    file = open(file_dir, "w")
   #    args['epoches'] = 20
   #    args['num_hidfeat'] = 32
   #    args['replay_size'] = 2
   #    for random_seed in range(20):
   #       fix_random_seed(random_seed)
   #       phase_accuracy = inc_exp_sto.incremental_trainig_replay(args,'coau_cs','SAGE',method)
   #       last_phase = phase_accuracy[-1]

   #       class_acc_list = last_phase['prev_class_acc']
   #       class_forget_list = last_phase['prev_class_forget']

   #       class_acc_list.append(last_phase['test_accuracy'])

   #       atp = sum(class_acc_list)/len(class_acc_list)
   #       atf = sum(class_forget_list)/(len(class_forget_list))

   #       randomseed_line = 'random seed: ' + str(random_seed) 
   #       atpline  = ' atp: ' + str(atp) 
   #       atfline  = ' atf: ' + str(atf) + '\n'
         
   #       file.write(randomseed_line)
   #       file.write(atpline)
   #       file.write(atfline)
   #    file.close()


   strategies = ['uniform','degree','cluster','center','max_cover','centroid']
   for method in strategies:
      file_dir = 'result/reddit_it_'+method+'.txt'
      args['epoches'] = 10
      args['num_hidfeat'] = 64
      args['replay_size'] = 2
      file = open(file_dir, "w")
      for random_seed in range(5):
         fix_random_seed(random_seed)
         phase_accuracy = inc_exp_sto.incremental_trainig_replay(args,'reddit','SAGE',method)
         last_phase = phase_accuracy[-1]

         class_acc_list = last_phase['prev_class_acc']
         class_forget_list = last_phase['prev_class_forget']

         class_acc_list.append(last_phase['test_accuracy'])

         atp = sum(class_acc_list)/len(class_acc_list)
         atf = sum(class_forget_list)/(len(class_forget_list))

         randomseed_line = 'random seed: ' + str(random_seed) 
         atpline  = ' atp: ' + str(atp) 
         atfline  = ' atf: ' + str(atf) + '\n'
         
         file.write(randomseed_line)
         file.write(atpline)
         file.write(atfline)
         file.close()


      # phase_accuracy = inc_exp_sto.incremental_trainig_replay(args,'cora','SAGE',method)
      # last_phase = phase_accuracy[-1]
      # print(phase_accuracy)
      # print('method:', method)
      # class_acc_list = last_phase['prev_class_acc']
      # class_acc_list.append(last_phase['test_accuracy'])
      # print("average task accuracy: ",sum(class_acc_list)/len(class_acc_list))
      # class_forget_list = last_phase['prev_class_forget']
      # print("average task accuracy: ",sum(class_forget_list)/(len(class_forget_list)+1))
      # input('checking')
