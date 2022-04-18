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
def create_trainset(data, debug = True):
    # input:
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
    #     data_info - a dictionary storing the dataset information

    # load corresponding dataset
    graph, data_info = load_data(data)

    num_layer = 2
    class_train_size = data_info['class_train_size']
    task_number = data_info['task_number']
    class_per_task = data_info['class_per_task']

    # sampling method, extract label and determine number of class and number of data
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layer)
    labels = graph.ndata['label']
    num_node = len(labels)

    # a nested list to store the node id lists for each class
    class_node_id_list = []
    # a nested list to store the training node id lists for each class
    class_train_id_list = []

    for class_index in range(task_number * class_per_task):
        # extract node id of specific class
        class_node_id = utility.extract_node_id_for_label(labels, class_index)
        class_node_id_list.append(class_node_id)

        # randomly sample k training node for each class
        class_train_id = random.sample(class_node_id, k = class_train_size)
        class_train_id_list.append(class_train_id)

    # list of training dictionary
    training_dict_list = []
    task_binary_label_list = []
    task_dict = {}
    for task_index in range(task_number):
        # create population for each task and a dict for reference
        task_train_dict = {}
        task_population = []
        task_train_id_list = []
        # get the task node id list and task train set id list
        for task_class_index in range(class_per_task):
            current_index = task_index * class_per_task + task_class_index
            task_population += class_node_id_list[current_index]
            task_train_id_list += class_train_id_list[current_index]

        # store task index: (node id, task index)
        task_dict.update(((id, task_index) for id in task_population))

        # create binary mask for each task
        task_binary_label = utility.create_binary_label(task_population, num_node)
        task_binary_label_list.append(task_binary_label)

        #test list for each task
        task_test_id_list = [i for i in task_population if not i in task_train_id_list]

        if debug:
            print('size of test set:', len(task_test_id_list))


        # create train, test data loader for each task
        task_train_dataloader = dgl.dataloading.NodeDataLoader(graph, task_train_id_list, sampler,
          device=device, batch_size=1, shuffle=True, drop_last=False, num_workers=0
        )
        task_test_dataloader = dgl.dataloading.NodeDataLoader(graph, task_test_id_list, sampler,
          device=device, batch_size=len(task_test_id_list), shuffle=True, drop_last=False, num_workers=0
        )

        # create training dict for each task
        task_train_dict['task_index'] = task_index
        task_train_dict['test_list'] = task_test_id_list
        task_train_dict['train_dataloader'] = task_train_dataloader
        task_train_dict['test_dataloader'] = task_test_dataloader

        training_dict_list.append(task_train_dict)

    data_info['task_dict']   = task_dict
    data_info['task_labels'] = task_binary_label_list

    return training_dict_list, graph, data_info

################################################################################
#  Return a training and testing partition of the data based on learning stages
#  with replay strategy. The return training/testing list is in task order
################################################################################
def create_trainset_replay(args, data, replay_strategy, debug=False):
    # input:
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
    #     data_info - a dictionary storing the dataset information

    #load corresponding data
    graph, data_info = load_data(data)

    num_layer = args['num_layer']
    replay_size = args['replay_size']
    class_train_size = data_info['class_train_size']
    task_number = data_info['task_number']
    class_per_task = data_info['class_per_task']

    # checking
    if debug:
        print('replay set size: ', replay_size)

    # sampling method, extract label and determine number of class and number of data
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layer)
    labels = graph.ndata['label']
    num_node = len(labels)

    # a nested list to store the node id lists for each class
    class_node_id_list = []
    # a nested list to store the training node id lists for each class
    class_train_id_list = []

    for class_index in range(task_number * class_per_task):
        # extract node id of specific class
        class_node_id = utility.extract_node_id_for_label(labels, class_index)
        class_node_id_list.append(class_node_id)

        # randomly sample k training node for each class
        class_train_id = random.sample(class_node_id, k = class_train_size)
        class_train_id_list.append(class_train_id)

    # list of training dictionary
    training_dict_list = []
    task_binary_label_list = []
    task_dict = {}
    for task_index in range(task_number):
        # create population for each task and a dict for reference
        task_train_dict = {}
        task_population = []
        task_train_id_list = []
        task_replay_list = []
        for task_class_index in range(class_per_task):

            class_index = task_index * class_per_task + task_class_index
            class_node_id = class_node_id_list[class_index]
            class_train_id = class_train_id_list[class_index]
            task_population += class_node_id
            task_train_id_list += class_train_id

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
                class_replay, _ = sampling.max_cover(class_subgraph, replay_size, 1)
                task_replay_list += class_replay
            if replay_strategy == 'degree':
                nx_G = graph.to_networkx().to_undirected()
                class_subgraph = nx_G.subgraph(class_node_id)
                degree_dict = list(class_subgraph.degree(class_train_id))
                weight_list = [ele[1] for ele in degree_dict]
                class_replay, _ = sampling.importance_base_sampling(class_train_id, weight_list, replay_size)
                task_replay_list += class_replay
            if replay_strategy == 'cluster':
                nx_G = graph.to_networkx().to_undirected()
                class_subgraph = nx_G.subgraph(class_node_id)
                cluster_dict = nx.square_clustering(class_subgraph)
                weight_list = [ele[1] for ele in cluster_dict.items()]
                class_replay, _ = sampling.importance_base_sampling(class_node_id, weight_list, replay_size)
                task_replay_list += class_replay
            if replay_strategy == 'center':
                nx_G = graph.to_networkx().to_undirected()
                class_subgraph = nx_G.subgraph(class_node_id)
                center_dict = nx.degree_centrality(class_subgraph)
                weight_list = [ele[1] for ele in center_dict.items()]
                class_replay, _ = sampling.importance_base_sampling(class_node_id, weight_list, replay_size)
                task_replay_list += class_replay
            if replay_strategy == 'centroid':
                class_replay = sampling.centroid(graph, class_node_id, 1, replay_size)
                task_replay_list += class_replay

            if debug:
                print(task_replay_list)
                input('checking replay list')

        # store task index: (node id, task index)
        task_dict.update(((id, task_index) for id in task_population))

        # create binary mask for each task
        #### task_binary_label = utility.create_binary_label(task_population, num_node) ###
        task_binary_label = utility.create_binary_label(class_node_id_list[task_index*class_per_task], num_node)
        task_binary_label_list.append(task_binary_label)

        # test list for each task
        task_test_id_list = [i for i in task_population if not i in task_train_id_list]

        # replay list
        task_train_dict['replay_list'] = task_replay_list

        if debug:
            print(task_replay_list)
            input('checking replay list')

        # combine replay set from (0~n-1)-th task to the n-th task
        if task_index > 0:
            for i in range(task_index):
                task_train_id_list += training_dict_list[i]['replay_list']

        # create train, test, replay data loader for each task
        task_train_dataloader = dgl.dataloading.NodeDataLoader(graph, task_train_id_list, sampler,
          device=device, batch_size=1, shuffle=True, drop_last=False, num_workers=0
        )
        task_test_dataloader = dgl.dataloading.NodeDataLoader(graph, task_test_id_list, sampler,
        device=device, batch_size=len(task_test_id_list), shuffle=True, drop_last=False, num_workers=0
        )
        task_replay_dataloader = dgl.dataloading.NodeDataLoader(graph, task_replay_list, sampler,
        device=device, batch_size=len(task_replay_list), shuffle=True, drop_last=False, num_workers=0
        )

        # create training dict for each task
        task_train_dict['task_index'] = task_index
        task_train_dict['test_list']  = task_test_id_list
        task_train_dict['train_dataloader'] = task_train_dataloader
        task_train_dict['test_dataloader'] = task_test_dataloader
        task_train_dict['replay_dataloader'] = task_replay_dataloader

        training_dict_list.append(task_train_dict)

    data_info['task_dict']   = task_dict
    data_info['task_labels'] = task_binary_label_list

    return training_dict_list, graph, data_info

################################################################################
#  Return a training and testing partition of the data based on learning stages
#  without replay strategy. The return training/testing list is in random order.
################################################################################
def create_trainset_mix(data):
    # input:
    #     data - a string specifying the dataset, could be chosen from {'cora',
    #            'cite', 'pubmed', 'cora_full', 'aifb', 'amazon_comp', 'amazon_photo'
    #            'coau_cs', 'coau_phy' and 'reddit'}
    # return:
    #     training_dict - a list containing a training dictionary for
    #                  each task, which contains the information of
    #                  dataloaders for train set and test set
    #     graph - the dgl graph for the data
    #     data_info - a dictionary storing the dataset information

    # load corresponding data
    graph, data_info = load_data(data)

    num_layer = data_info['num_layer']
    class_train_size = data_info['class_train_size']
    task_number = data_info['task_number']
    class_per_task = data_info['class_per_task']

    # sampling method
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layer)

    # extract label and determine number of nodes
    labels = graph.ndata['label']

    # dictionary contain training information
    training_dict = {}

    # a nested list to store the node id lists for each class
    class_node_id_list = []
    # a nested list to store the training node id lists for each class
    class_train_id_list = []

    for class_index in range(task_number * class_per_task):
        # extract node id of certain class
        class_node_id = utility.extract_node_id_for_label(labels, class_index)
        class_node_id_list.append(class_node_id)

        # randomly sample k training node for each class
        class_train_id = random.sample(class_node_id, k=class_train_size)
        class_train_id_list.append(class_train_id)

    # create task information
    task_population_list = []
    label_dict = {}
    task_dict = {}
    for task_index in range(task_number):
        task_population = []
        for class_index in range(class_per_task):
            current_class_index = class_per_task * task_index + class_index
            current_class_node_id = class_node_id_list[current_class_index]

            # store class label: (node id, class index)
            label_dict.update(((id, class_index) for id in current_class_node_id))
            task_population += current_class_node_id

        # store task index: (node id, task index)
        task_dict.update(((id, task_index) for id in task_population))
        task_population_list.append(task_population)

    data_info['task_dict'] = task_dict
    data_info['label_dict'] = label_dict


    # train set
    train_list = sum(class_train_id_list, [])
    trainSet_dataloader = dgl.dataloading.NodeDataLoader(
        graph,              # The graph
        train_list,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1,       # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    training_dict['train_dataloader'] = trainSet_dataloader

    # test list for each task
    task_test_dataloaderList = []
    for task_index in range(task_number):
        task_test_id_list = [i for i in task_population_list[task_index] if not i in train_list]

        task_test_dataloader = dgl.dataloading.NodeDataLoader(
            graph,              # The graph
            task_test_id_list,  # The node IDs to iterate over in minibatches
            sampler,            # The neighbor sampler
            device=device,      # Put the sampled MFGs on CPU or GPU
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=len(task_test),    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=False,    # Whether to drop the last incomplete batch
            num_workers=0       # Number of sampler processes
        )

        task_test_dataloaderList.append(task_test_dataloader)

    training_dict['task_test_dataloader'] = task_test_dataloaderList


    return training_dict, graph, data_info
