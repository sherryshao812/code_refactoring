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

    def test_create_trainset_cora(self):
        training_dict_list, graph, data_info = create_trainset('cora', debug=True)
        self.assertEqual(data_info['class_train_size'], 20)
        self.assertEqual(data_info['task_number'], 3)
        self.assertEqual(data_info['class_per_task'], 2)
        self.assertTrue('task_dict' in data_info)
        self.assertTrue('task_labels' in data_info)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'])
            self.assertEqual(len(test_dataloader), 1)
            for j in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(j[1])]) in range(i*data_info['class_per_task'], (i+1)*data_info['class_per_task']) )
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for k in test_dataloader:
                self.assertEqual(len(k[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])



    def test_create_trainset_coau_cs(self):
        training_dict_list, graph, data_info = create_trainset('coau_cs', debug=True)
        self.assertEqual(data_info['class_train_size'], 40)
        self.assertEqual(data_info['task_number'], 5)
        self.assertEqual(data_info['class_per_task'], 3)
        self.assertTrue('task_dict' in data_info)
        self.assertTrue('task_labels' in data_info)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'])
            self.assertEqual(len(test_dataloader), 1)
            for j in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(j[1])]) in range(i*data_info['class_per_task'], (i+1)*data_info['class_per_task']) )
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for k in test_dataloader:
                self.assertEqual(len(k[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])


    def test_create_trainset_reddit(self):
        training_dict_list, graph, data_info = create_trainset('reddit', debug=True)
        self.assertEqual(data_info['class_train_size'], 50)
        self.assertEqual(data_info['task_number'], 8)
        self.assertEqual(data_info['class_per_task'], 5)
        self.assertTrue('task_dict' in data_info)
        self.assertTrue('task_labels' in data_info)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'])
            self.assertEqual(len(test_dataloader), 1)
            for j in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(j[1])]) in range(i*data_info['class_per_task'], (i+1)*data_info['class_per_task']) )
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for k in test_dataloader:
                self.assertEqual(len(k[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])


    def test_create_trainset_replay_cora_uniform(self):
        training_dict_list, graph, data_info = create_trainset_replay(args, 'cora', 'uniform', debug=False)
        self.assertEqual(data_info['class_train_size'], 20)
        self.assertEqual(data_info['task_number'], 3)
        self.assertEqual(data_info['class_per_task'], 2)
        self.assertEqual(len(data_info['task_dict']), 2708 - list(graph.ndata['label']).count(6))
        self.assertEqual(len(data_info['task_labels']), 3)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])
        self.assertTrue('replay_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']
            replay_dataloader = task_train_dict['replay_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] + args['replay_size'] * data_info['class_per_task'] * i)
            self.assertEqual(len(task_train_dict['replay_list']), args['replay_size'] * data_info['class_per_task'])
            self.assertEqual(len(replay_dataloader), 1)
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])
            for k in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(k[1])]) in range(i * data_info['class_per_task'], (i+1) * data_info['class_per_task']) )



    def test_create_trainset_replay_cora_max_cover(self):
        training_dict_list, graph, data_info = create_trainset_replay(args, 'cora', 'max_cover', debug=False)
        self.assertEqual(data_info['class_train_size'], 20)
        self.assertEqual(data_info['task_number'], 3)
        self.assertEqual(data_info['class_per_task'], 2)
        self.assertEqual(len(data_info['task_dict']), 2708 - list(graph.ndata['label']).count(6))
        self.assertEqual(len(data_info['task_labels']), 3)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])
        self.assertTrue('replay_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']
            replay_dataloader = task_train_dict['replay_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] + args['replay_size'] * data_info['class_per_task'] * i)
            self.assertEqual(len(task_train_dict['replay_list']), args['replay_size'] * data_info['class_per_task'])
            self.assertEqual(len(replay_dataloader), 1)
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])
            for k in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(k[1])]) in range(i * data_info['class_per_task'], (i+1) * data_info['class_per_task']) )


    def test_create_trainset_replay_cora_degree(self):
        training_dict_list, graph, data_info = create_trainset_replay(args, 'cora', 'degree', debug=False)
        self.assertEqual(data_info['class_train_size'], 20)
        self.assertEqual(data_info['task_number'], 3)
        self.assertEqual(data_info['class_per_task'], 2)
        self.assertEqual(len(data_info['task_dict']), 2708 - list(graph.ndata['label']).count(6))
        self.assertEqual(len(data_info['task_labels']), 3)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])
        self.assertTrue('replay_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']
            replay_dataloader = task_train_dict['replay_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] + args['replay_size'] * data_info['class_per_task'] * i)
            self.assertEqual(len(task_train_dict['replay_list']), args['replay_size'] * data_info['class_per_task'])
            self.assertEqual(len(replay_dataloader), 1)
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])
            for k in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(k[1])]) in range(i * data_info['class_per_task'], (i+1) * data_info['class_per_task']) )


    def test_create_trainset_replay_cora_cluster(self):
        training_dict_list, graph, data_info = create_trainset_replay(args, 'cora', 'cluster', debug=False)
        self.assertEqual(data_info['class_train_size'], 20)
        self.assertEqual(data_info['task_number'], 3)
        self.assertEqual(data_info['class_per_task'], 2)
        self.assertEqual(len(data_info['task_dict']), 2708 - list(graph.ndata['label']).count(6))
        self.assertEqual(len(data_info['task_labels']), 3)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])
        self.assertTrue('replay_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']
            replay_dataloader = task_train_dict['replay_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] + args['replay_size'] * data_info['class_per_task'] * i)
            self.assertEqual(len(task_train_dict['replay_list']), args['replay_size'] * data_info['class_per_task'])
            self.assertEqual(len(replay_dataloader), 1)
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])
            for k in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(k[1])]) in range(i * data_info['class_per_task'], (i+1) * data_info['class_per_task']) )


    def test_create_trainset_replay_cora_center(self):
        training_dict_list, graph, data_info = create_trainset_replay(args, 'cora', 'center', debug=False)
        self.assertEqual(data_info['class_train_size'], 20)
        self.assertEqual(data_info['task_number'], 3)
        self.assertEqual(data_info['class_per_task'], 2)
        self.assertEqual(len(data_info['task_dict']), 2708 - list(graph.ndata['label']).count(6))
        self.assertEqual(len(data_info['task_labels']), 3)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])
        self.assertTrue('replay_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']
            replay_dataloader = task_train_dict['replay_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] + args['replay_size'] * data_info['class_per_task'] * i)
            self.assertEqual(len(task_train_dict['replay_list']), args['replay_size'] * data_info['class_per_task'])
            self.assertEqual(len(replay_dataloader), 1)
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])
            for k in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(k[1])]) in range(i * data_info['class_per_task'], (i+1) * data_info['class_per_task']) )


    def test_create_trainset_replay_cora_centroid(self):
        training_dict_list, graph, data_info = create_trainset_replay(args, 'cora', 'centroid', debug=False)
        self.assertEqual(data_info['class_train_size'], 20)
        self.assertEqual(data_info['task_number'], 3)
        self.assertEqual(data_info['class_per_task'], 2)
        self.assertEqual(len(data_info['task_dict']), 2708 - list(graph.ndata['label']).count(6))
        self.assertEqual(len(data_info['task_labels']), 3)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])
        self.assertTrue('replay_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']
            replay_dataloader = task_train_dict['replay_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] + args['replay_size'] * data_info['class_per_task'] * i)
            self.assertEqual(len(task_train_dict['replay_list']), args['replay_size'] * data_info['class_per_task'])
            self.assertEqual(len(replay_dataloader), 1)
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])
            for k in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(k[1])]) in range(i * data_info['class_per_task'], (i+1) * data_info['class_per_task']) )



    def test_create_trainset_replay_coau_cs_uniform(self):
        training_dict_list, graph, data_info = create_trainset_replay(args, 'coau_cs', 'uniform', debug=False)
        self.assertEqual(data_info['class_train_size'], 40)
        self.assertEqual(data_info['task_number'], 5)
        self.assertEqual(data_info['class_per_task'], 3)
        self.assertEqual(len(data_info['task_dict']), len(graph.ndata['label']))
        self.assertEqual(len(data_info['task_labels']), 5)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])
        self.assertTrue('replay_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']
            replay_dataloader = task_train_dict['replay_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] + args['replay_size'] * data_info['class_per_task'] * i)
            self.assertEqual(len(task_train_dict['replay_list']), args['replay_size'] * data_info['class_per_task'])
            self.assertEqual(len(replay_dataloader), 1)
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])
            for k in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(k[1])]) in range(i * data_info['class_per_task'], (i+1) * data_info['class_per_task']) )



    def test_create_trainset_replay_reddit_uniform(self):
        training_dict_list, graph, data_info = create_trainset_replay(args, 'reddit', 'uniform', debug=False)
        self.assertEqual(data_info['class_train_size'], 50)
        self.assertEqual(data_info['task_number'], 8)
        self.assertEqual(data_info['class_per_task'], 5)
        self.assertEqual(len(data_info['task_dict']), len(graph.ndata['label']) - list(graph.ndata['label']).count(40))
        self.assertEqual(len(data_info['task_labels']), 8)
        self.assertTrue('train_dataloader' in training_dict_list[0])
        self.assertTrue('test_dataloader' in training_dict_list[0])
        self.assertTrue('replay_dataloader' in training_dict_list[0])

        for i in range(data_info['task_number']):
            task_train_dict = training_dict_list[i]
            train_dataloader = task_train_dict['train_dataloader']
            test_dataloader = task_train_dict['test_dataloader']
            replay_dataloader = task_train_dict['replay_dataloader']

            self.assertEqual(task_train_dict['task_index'], i)
            self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] + args['replay_size'] * data_info['class_per_task'] * i)
            self.assertEqual(len(task_train_dict['replay_list']), args['replay_size'] * data_info['class_per_task'])
            self.assertEqual(len(replay_dataloader), 1)
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])
            for k in train_dataloader:
                self.assertTrue(int(graph.ndata['label'][int(k[1])]) in range(i * data_info['class_per_task'], (i+1) * data_info['class_per_task']) )



    def test_create_trainset_mix_cora(self):
        training_dict, graph, data_info = create_trainset_mix('cora')
        self.assertEqual(data_info['class_train_size'], 20)
        self.assertEqual(data_info['task_number'], 3)
        self.assertEqual(data_info['class_per_task'], 2)
        self.assertEqual(len(data_info['task_dict']), len(graph.ndata['label']) - list(graph.ndata['label']).count(6))
        self.assertEqual(len(data_info['label_dict']), len(graph.ndata['label']) - list(graph.ndata['label']).count(6))
        self.assertTrue('train_dataloader' in training_dict)
        self.assertTrue('task_test_dataloader' in training_dict)

        train_dataloader = training_dict['train_dataloader']
        test_dataloader_list = training_dict['task_test_dataloader']

        self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] * data_info['task_number'])
        self.assertEqual(len(test_dataloader_list), data_info['task_number'])
        for i in range(data_info['task_number']):
            test_dataloader = test_dataloader_list[i]
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])


    def test_create_trainset_mix_coau_cs(self):
        training_dict, graph, data_info = create_trainset_mix('coau_cs')
        self.assertEqual(data_info['class_train_size'], 40)
        self.assertEqual(data_info['task_number'], 5)
        self.assertEqual(data_info['class_per_task'], 3)
        self.assertEqual(len(data_info['task_dict']), len(graph.ndata['label']))
        self.assertEqual(len(data_info['label_dict']), len(graph.ndata['label']))
        self.assertTrue('train_dataloader' in training_dict)
        self.assertTrue('task_test_dataloader' in training_dict)

        train_dataloader = training_dict['train_dataloader']
        test_dataloader_list = training_dict['task_test_dataloader']

        self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] * data_info['task_number'])
        self.assertEqual(len(test_dataloader_list), data_info['task_number'])
        for i in range(data_info['task_number']):
            test_dataloader = test_dataloader_list[i]
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])



    def test_create_trainset_mix_reddit(self):
        training_dict, graph, data_info = create_trainset_mix('reddit')
        self.assertEqual(data_info['class_train_size'], 50)
        self.assertEqual(data_info['task_number'], 8)
        self.assertEqual(data_info['class_per_task'], 5)
        self.assertEqual(len(data_info['task_dict']), len(graph.ndata['label']) - list(graph.ndata['label']).count(40))
        self.assertEqual(len(data_info['label_dict']), len(graph.ndata['label']) - list(graph.ndata['label']).count(40))
        self.assertTrue('train_dataloader' in training_dict)
        self.assertTrue('task_test_dataloader' in training_dict)

        train_dataloader = training_dict['train_dataloader']
        test_dataloader_list = training_dict['task_test_dataloader']

        self.assertEqual(len(train_dataloader), data_info['class_train_size'] * data_info['class_per_task'] * data_info['task_number'])
        self.assertEqual(len(test_dataloader_list), data_info['task_number'])
        for i in range(data_info['task_number']):
            test_dataloader = test_dataloader_list[i]
            self.assertEqual(len(test_dataloader), 1)
            num_task_node = sum([list(graph.ndata['label']).count(i*data_info['class_per_task']+k) for k in range(data_info['class_per_task'])])
            for j in test_dataloader:
                self.assertEqual(len(j[1]), num_task_node - data_info['class_train_size'] * data_info['class_per_task'])



if __name__ == '__main__':
    unittest.main()
