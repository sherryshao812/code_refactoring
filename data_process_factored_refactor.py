import dgl
import random

#our own code
import utility
import sampling

import networkx as nx

from data import load_data

device = 'cpu'

################################################################################
#  return a training and testing partition of the data based on learning
#  stages without replay strategy. The return training/testing list is in task order.
################################################################################
def create_trainset(args, data, debug = True):
    # input:
    #     args - the arguments list
    #     data - a string specifying the dataset, could be chosen from {'cora',
    #            'cite', 'pubmed', 'cora_full', 'aifb', 'amazon_comp', 'amazon_photo'
    #            'coau_cs', 'coau_phy' and 'reddit'}
    #     debug - whether to print the debug message, defult to True
    # return:
    #     training_dict_list - a list containing a training dictionary for
    #                          each task, which contains the information of
    #                          task_index, test_list, and dataloader for train
    #                          set and test set
    #     graph - the dgl graph of the data

    gnn_layer_num = 2
    task_dict = {}

    # load corresponding dataset
    graph = load_data(data, args)

    class_train_size = args['class_train_size']
    task_number = args['task_number']
    class_per_task = args['class_per_task']

    # sampling method, extract label and determine number of class and number of data
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(gnn_layer_num)
    node_labels = graph.ndata['label']
    node_num = len(node_labels)
    n_class = int(node_labels.max().item() + 1)
    data_population = [i for i in range(node_num)]

    # a nested list to store the node id lists for each class
    class_node_id_list = []
    # a nested list to store the training node id lists for each class
    class_train_id_list = []

    for class_index in range(task_number * class_per_task):
        # extract node id of specific class
        class_node_id = utility.extract_node_id_for_label(node_labels, class_index)
        # sample for training set
        class_train_id = random.sample(class_node_id, k = class_train_size)

        class_node_id_list.append(class_node_id)
        class_train_id_list.append(class_train_id)

    # list of training dictionary
    training_dict_list = []
    task_labels_list = []
    task_index_table = {}
    for task_index in range(task_number):
        # create population for each task and a dict for reference
        task_train_dict = {}
        task_index_dict = {}
        task_population = []
        task_train_list = []
        for task_class_index in range(class_per_task):
            task_population += class_node_id_list[task_index * class_per_task + task_class_index]
            task_train_list += class_train_id_list[task_index * class_per_task + task_class_index]

        # store task index: (node id, task index)
        task_index_list = len(task_population) * [task_index]
        task_index_table.update(zip(task_population, task_index_list))

        # create binary mask for each task
        task_binary_label = utility.create_binary_label(class_node_id_list[task_index * class_per_task], node_num)
        task_labels_list.append(task_binary_label)

        #test list for each task
        task_test_list = [i for i in task_population if not i in task_train_list]

        if debug:
            print('size of test set:', len(task_test_list))


        # create train, test, replay data loader for each task
        task_train_dataloader = dgl.dataloading.NodeDataLoader(graph, task_train_list, sampler,
          device=device, batch_size=1, shuffle=True, drop_last=False, num_workers=0
        )
        task_test_dataloader = dgl.dataloading.NodeDataLoader(graph, task_test_list, sampler,
          device=device, batch_size=len(task_test_list), shuffle=True, drop_last=False, num_workers=0
        )

        # create training dict for each task
        task_train_dict['task_index'] = task_index
        task_train_dict['test_list'] = task_test_list
        task_train_dict['train_dataloader'] = task_train_dataloader
        task_train_dict['test_dataloader'] = task_test_dataloader

        training_dict_list.append(task_train_dict)

    args['task_dict']   = task_index_table
    args['task_labels'] = task_labels_list

    return training_dict_list, graph

################################################################################
#  Return a training and testing partition of the data based on learning stages
#  with replay strategy. The return training/testing list is in task order
################################################################################
def create_trainset_replay(args, data, replay_strategy, debug=False):
    # input:
    #     args - the arguments list
    #     data - a string specifying the dataset, could be chosen from {'cora',
    #            'cite', 'pubmed', 'cora_full', 'aifb', 'amazon_comp', 'amazon_photo'
    #            'coau_cs', 'coau_phy' and 'reddit'}
    #     replay_strategy - a string specifying the replay set selection strategy,
    #                       can be chosen from {'uniform', 'max_cover', 'degree'
    #                       'cluster', 'center', 'centroid'}
    #     debug - whether to print the debug messages, defult to False
    # return:
    #     training_dict_list - a list containing a training dictionary for
    #                          each task, which contains the information of
    #                          task_index, test_list, and dataloader for train,
    #                          test and replay set
    #     graph - the dgl graph for the data

    gnn_layer_num = 2
    task_dict = {}
    replay_size = args['replay_size']

    # checking
    if debug:
        print('replay set size: ', replay_size)

    #load corresponding data
    graph = load_data(data, args)

    class_train_size = args['class_train_size']
    task_number = args['task_number']
    class_per_task = args['class_per_task']

    # sampling method, extract label and determine number of class and number of data
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(gnn_layer_num)
    node_labels = graph.ndata['label']
    node_num = len(node_labels)
    n_class = int(node_labels.max().item() + 1)
    data_population = [i for i in range(node_num)]

    # a nested list to store the node id lists for each class
    class_node_id_list = []
    # a nested list to store the training node id lists for each class
    class_train_id_list = []

    for class_index in range(task_number * class_per_task):
        # extract node id of specific class
        class_node_id = utility.extract_node_id_for_label(node_labels, class_index)
        # sample for training set
        class_train_id = random.sample(class_node_id, k=class_train_size)

        class_node_id_list.append(class_node_id)
        class_train_id_list.append(class_train_id)

    # list of training dictionary
    training_dict_list = []
    task_labels_list = []
    task_index_table = {}
    for task_index in range(task_number):
        # create population for each task and a dict for reference
        task_train_dict = {}
        task_index_dict = {}
        task_population = []
        task_train_list = []
        task_replay_list = []
        for task_class_index in range(class_per_task):

            class_index = task_index * class_per_task + task_class_index
            class_node_id = class_node_id_list[class_index]
            class_train_id = class_train_id_list[class_index]
            task_population += class_node_id
            task_train_list += class_train_id

            if debug:
                print('task class index:', task_class_index)
                print('class index:', class_index)

            # create replay set with given strategy
            if replay_strategy =='uniform':
                class_replay = random.sample(class_node_id, k=replay_size)
                task_replay_list += class_replay
            if replay_strategy == 'max_cover':
                nx_G = graph.to_networkx().to_undirected()
                class_subgraph = nx_G.subgraph(class_node_id)
                class_replay, test_list = sampling.max_cover(class_subgraph, replay_size, 1)
                task_replay_list += class_replay
            if replay_strategy == 'degree':
                nx_G = graph.to_networkx().to_undirected()
                class_subgraph = nx_G.subgraph(class_node_id)
                degree_dict = list(class_subgraph.degree(class_train_id))
                weight_list = [ele[1] for ele in degree_dict]
                class_replay, test_list = sampling.importance_base_sampling(class_train_id, weight_list, replay_size)
                task_replay_list += class_replay
            if replay_strategy == 'cluster':
                nx_G = graph.to_networkx().to_undirected()
                class_subgraph = nx_G.subgraph(class_node_id)
                cluster_dict = nx.square_clustering(class_subgraph)
                weight_list = [ele[1] for ele in cluster_dict.items()]
                class_replay,test_list = sampling.importance_base_sampling(class_node_id, weight_list, replay_size)
                task_replay_list += class_replay
            if replay_strategy == 'center':
                nx_G = graph.to_networkx().to_undirected()
                class_subgraph = nx_G.subgraph(class_node_id)
                center_dict = nx.degree_centrality(class_subgraph)
                weight_list = [ele[1] for ele in center_dict.items()]
                class_replay, test_list = sampling.importance_base_sampling(class_node_id, weight_list, replay_size)
                task_replay_list += class_replay
            if replay_strategy == 'centroid':
                class_replay = sampling.centroid(graph, class_node_id, 1, replay_size)
                task_replay_list += class_replay

            if debug:
                print(task_replay_list)
                input('checking replay list')

        # store task index: (node id, task index)
        task_index_list = len(task_population) * [task_index]
        task_index_table.update(zip(task_population, task_index_list))

        # create binary mask for each task
        task_binary_label = utility.create_binary_label(class_node_id_list[task_index * class_per_task], node_num)
        task_labels_list.append(task_binary_label)

        # test list for each task
        task_test_list = [i for i in task_population if not i in task_train_list]

        # replay list
        task_train_dict['replay_list'] = task_replay_list

        if debug:
            print(task_replay_list)
            input('checking replay list')

        if task_index > 0:
            for i in range(task_index):
                task_train_list += training_dict_list[i]['replay_list']

        # create train, test, replay data loader for each task
        task_train_dataloader = dgl.dataloading.NodeDataLoader(graph, task_train_list, sampler,
          device=device, batch_size=1, shuffle=True, drop_last=False, num_workers=0
        )
        task_test_dataloader = dgl.dataloading.NodeDataLoader(graph, task_test_list, sampler,
        device=device, batch_size=len(task_test_list), shuffle=True, drop_last=False, num_workers=0
        )
        task_replay_dataloader = dgl.dataloading.NodeDataLoader(graph, task_replay_list, sampler,
        device=device, batch_size=len(task_replay_list), shuffle=True, drop_last=False, num_workers=0
        )

        # create training dict for each task
        task_train_dict['task_index'] = task_index
        task_train_dict['test_list']  = task_test_list
        task_train_dict['train_dataloader'] = task_train_dataloader
        task_train_dict['test_dataloader'] = task_test_dataloader
        task_train_dict['replay_dataloader'] = task_replay_dataloader

        training_dict_list.append(task_train_dict)

    args['task_dict']   = task_index_table
    args['task_labels'] = task_labels_list

    return training_dict_list, graph

################################################################################
#  Return a training and testing partition of the data based on learning stages
#  without replay strategy. The return training/testing list is in random order.
################################################################################
def create_trainset_mix(args,data):
    # input:
    #     args - the arguments list
    #     data - a string specifying the dataset, could be chosen from {'cora',
    #            'cite', 'pubmed', 'cora_full', 'aifb', 'amazon_comp', 'amazon_photo'
    #            'coau_cs', 'coau_phy' and 'reddit'}
    # return:
    #     train_dict - a list containing a training dictionary for
    #                  each task, which contains the information of
    #                  dataloaders for train set and test set
    #     graph - the dgl graph for the data

    gnn_layer_num = 2
    task_dict = {}
    label_dict = {}

    # load corresponding data
    graph = load_data(data,args)

    class_train_size = args['class_train_size']
    task_number = args['task_number']
    class_per_task = args['class_per_task']

    # sampling method
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(gnn_layer_num)

    # extract label and determine number of class and number of data
    node_labels = graph.ndata['label']
    node_num = len(node_labels)
    n_class = int(node_labels.max().item() + 1)
    data_population = [i for i in range(node_num)]

    # dictionary contain training information
    train_dict = {}

    # a nested list to store the node id lists for each class
    class_node_id_list = []
    # a nested list to store the training node id lists for each class
    class_train_node_list = []
    for class_index in range(task_number * class_per_task):
        class_node_id = utility.extract_node_id_for_label(node_labels, class_index)
        class_node_id_list.append(class_node_id)

        # sample positive and negative samples
        class_train_node = random.sample(class_node_id, k=class_train_size)
        class_train_node_list.append(class_train_node)

    # create task information
    task_population_list = []
    for task_index in range(task_number):
        task_population = []
        for class_index in range(class_per_task):
            curerrent_class_index = class_per_task * task_index + class_index
            current_class = class_node_id_list[curerrent_class_index]
            task_population += current_class

            # store class label: (node id, class index)
            class_label = class_index
            class_label_list = len(current_class) * [class_label]
            label_dict.update(zip(current_class, class_label_list))

        # store task index: (node id, task index)
        task_population_list.append(task_population)
        task_index_list = len(task_population) * [task_index]
        task_dict.update(zip(task_population, task_index_list))

    args['task_dict'] = task_dict
    args['label_dict'] = label_dict


    # train set
    train_list = sum(class_train_node_list, [])
    trainSet_dataloader = dgl.dataloading.NodeDataLoader(
        graph,              # The graph
        train_list,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    train_dict['train_dataloader'] = trainSet_dataloader

    # test list for each task
    task_test_dataloaderList = []
    for task_index in range(task_number):
        task_test = [i for i in task_population_list[task_index] if not i in train_list]

        task_test_dataloader = dgl.dataloading.NodeDataLoader(
            graph,              # The graph
            task_test,         # The node IDs to iterate over in minibatches
            sampler,            # The neighbor sampler
            device=device,      # Put the sampled MFGs on CPU or GPU
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=len(task_test),    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=False,    # Whether to drop the last incomplete batch
            num_workers=0       # Number of sampler processes
        )

        task_test_dataloaderList.append(task_test_dataloader)

    train_dict['task_test_dataloader'] = task_test_dataloaderList


    return train_dict, graph
