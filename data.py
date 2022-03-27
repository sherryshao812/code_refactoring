import dgl
import numpy
import torch

import utility
import sampling

DEFAULT_TRAIN_PERCENT = 0.1

################################################################################
#  load data from specific dgl dataset
################################################################################
def load_data(name):
    # input:
    #     name - a string specifying the dgl dataset to be loaded
    # return:
    #     graph - a dgl graph
    #     data_info - a dictionary containing the dataset information

    data_info = {}
    if name == 'cora':
        dataset = dgl.data.CoraGraphDataset()
        data_info['class_train_size'] = 20
        data_info['task_number'] = 3
        data_info['class_per_task'] = 2
    if name == 'cite':
        dataset = dgl.data.CiteseerGraphDataset()
        data_info['class_train_size'] = 20
        data_info['task_number'] = 3
        data_info['class_per_task'] = 2
    if name == 'pubmed':
        dataset = dgl.data.PubmedGraphDataset()
    if name == 'cora_full':
        dataset= dgl.data.CoraFullDataset()
    if name == 'aifb':
        dataset= dgl.data.AIFBDataset()
    if name == 'amazon_comp':
        dataset = dgl.data.AmazonCoBuyComputerDataset()
    if name == 'amazon_photo':
        dataset = dgl.data.AmazonCoBuyPhotoDataset()
    if name == 'coau_cs':
        dataset = dgl.data.CoauthorCSDataset()
        data_info['class_train_size'] = 40
        data_info['task_number'] = 5
        data_info['class_per_task'] = 3
    if name == 'coau_phy':
        dataset = dgl.data.CoauthorPhysicsDataset()
    if name == 'reddit':
        dataset = dgl.data.RedditDataset()
        data_info['class_train_size'] = 50
        data_info['task_number'] = 8
        data_info['class_per_task'] = 5

    graph = dataset[0]

    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    # default_train_mask = graph.ndata['train_mask']
    # default_valid_mask = graph.ndata['val_mask']
    # default_test_mask  = graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)

    # data_info['train_mask']      = default_train_mask
    # data_info['test_mask']       = default_test_mask
    # data_info['validation_mask'] =  default_valid_mask
    data_info['num_infeat']     = n_features
    data_info['num_class']       = n_labels
    data_info['graph']           = graph
    data_info['num_outchannel']  = n_labels

    return graph, data_info
