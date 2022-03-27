import networkx as nx
import dgl
import unittest
from sampling import *
from data import *
from data_process_factored import *

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

# cora = load_data('cora', args)



class Test(unittest.TestCase):

    def test_create_trainset(self):
        training_dict_list, graph, data_info = create_trainset('cora', debug=True)
        self.assertEqual(data_info['class_train_size'], 20)
        self.assertEqual(data_info['task_number'], 3)
        self.assertEqual(data_info['class_per_task'], 2)
        self.assertTrue('task_dict' in data_info)
        self.assertTrue('task_labels' in data_info)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])






if __name__ == '__main__':
    unittest.main()
