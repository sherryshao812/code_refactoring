import torch
import dgl
import torch.nn.functional as F
import random
import sklearn.metrics

import numpy as np

from tqdm import tqdm

#our code
import data_process_sto
import data_process_factored
import data_process_grow
import models
import train
import evaluate

from data import load_data


################################################################################
# naive incremental learning setting, simply feed each class into the model one by one
# without using replay strategy and the training/testing list is in task order.
################################################################################
def incremental_trainig(args, data, model_name):
    # input:
    #     args - a dictionary containing the training setting information
    #     data - a string specifying the dataset, could be chosen from {'cora',
    #            'cite', 'pubmed', 'cora_full', 'aifb', 'amazon_comp', 'amazon_photo'
    #            'coau_cs', 'coau_phy' and 'reddit'}
    #     model_name - a string specifying the model name, could be chosen from
    #            {'GCN', 'SAGE', 'GAT'}
    # return:
    #     phase_accuracy - a list of phase accuracy

    #create training information
    training_dict_list, graph, data_info = data_process_factored.create_trainset(data)

    #select the corresponding GNN models
    if model_name == 'GCN':
        model = models.GCNStoModel_MultiHead(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'],
            out_feats = args['num_outchannel'],  num_task = args['task_number'],pred_head_out = args['class_per_task'] )
    if model_name == 'SAGE':
        model = models.SAGEStoModel_MultiHead(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'],
            out_feats = args['num_outchannel'], num_task = args['task_number'],pred_head_out = args['class_per_task'])
    if model_name == 'GAT':
        model = models.GATStoModel_MultiHead(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'],
            out_feats = args['num_outchannel'], num_task = args['task_number'], num_heads=2,pred_head_out = args['class_per_task'])

    # create list to store historical accuracy
    phase_accuracy = []

    # train phase index
    train_index = 0

    # iterative training over class
    for training_dict in tqdm(training_dict_list):
        #create dictionary to store training phase information
        train_phase_info = {}

        #the trained model parameter should propogate to the next iteration
        train_loss, test_acc, best_acc, model = train.train_multihead_model(model, args, data_info, training_dict)

        #store the training phase info
        train_phase_info['train_loss'] = train_loss
        train_phase_info['test_accuracy'] = test_acc
        train_phase_info['best_accuracy'] = best_acc

        # input('checking point: complete one task training')
        # Compute and store accuracy on the old test set starting with the second train phase before replay
        if train_index == (args['task_number']-1):
            prev_class_acc = []
            prev_class_forget = []
            for task_index in range(train_index):
                #extract label info and test mask
                task_labels     =  torch.LongTensor(args['task_labels'][task_index])
                test_dataloader = training_dict_list[task_index]['test_dataloader']
                model.eval()
                predictions = []
                labels = []
                with tqdm(test_dataloader) as tq, torch.no_grad():
                    for input_nodes, output_nodes, mfgs in tq:
                        inputs = mfgs[0].srcdata['feat']
                        dstdata_id = mfgs[-1].dstdata[dgl.NID]
                        labels.append(task_labels[dstdata_id])
                        predictions.append(model(mfgs, inputs, task_index).argmax(1).cpu().numpy())
                    predictions = np.concatenate(predictions)
                    labels = np.concatenate(labels)
                    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                    prev_class_acc.append(accuracy)

                    forget = phase_accuracy[task_index]['test_accuracy'] - accuracy
                    prev_class_forget.append(forget)

            #store the new accuracy for the previous class in the train phase info
            train_phase_info['prev_class_acc'] = prev_class_acc
            train_phase_info['prev_class_forget'] = prev_class_forget

        #store the phase information to list
        phase_accuracy.append(train_phase_info)

        train_index = train_index + 1

    return phase_accuracy


################################################################################
# naive incremental learning setting, simply feed each class into the model one by one
# using replay strategy and the training/testing list is in task order.
################################################################################
def incremental_trainig_replay(args, data, model_name, replay_strategy, debug=False):
    # input:
    #     args - a dictionary containing the training setting information
    #     data - a string specifying the dataset, could be chosen from {'cora',
    #            'cite', 'pubmed', 'cora_full', 'aifb', 'amazon_comp', 'amazon_photo'
    #            'coau_cs', 'coau_phy' and 'reddit'}
    #     model_name - a string specifying the model name, could be chosen from
    #            {'GCN', 'SAGE', 'GAT'}
    #     debug - a bollean indicating whether to print debug message, default to be False
    # return:
    #     phase_accuracy - a list of phase accuracy

    #create training information
    training_dict_list, graph, data_info = data_process_factored.create_trainset_replay(data, replay_strategy, args['replay_size'])

    #select the corresponding GNN models
    if model_name == 'GCN':
        model = models.GCNStoModel_MultiHead(in_feats = data_info['num_infeat'], hid_feats = args['num_hidfeat'],
          out_feats = data_info['num_outchannel'], num_task = data_info['task_number'], pred_head_out = data_info['class_per_task'] )
    if model_name == 'SAGE':
        model = models.SAGEStoModel_MultiHead(in_feats = data_info['num_infeat'], hid_feats = args['num_hidfeat'],
          out_feats = data_info['num_outchannel'], num_task = data_info['task_number'], pred_head_out = data_info['class_per_task'])
    if model_name == 'GAT':
        model = models.GATStoModel_MultiHead(in_feats = data_info['num_infeat'], hid_feats = args['num_hidfeat'],
          out_feats = data_info['num_outchannel'], num_task = data_info['task_number'], num_heads=2, pred_head_out = data_info['class_per_task'])

    # create list to store historical accuracy
    phase_accuracy = []

    # train phase index
    train_index = 0

    # iterative training over class
    for training_dict in tqdm(training_dict_list):
        # create dictionary to store training phase information
        train_phase_info = {}

        # compute the representation of the replay set before training
        if train_index > 0 and args['loss'] == 'cmd_reg':
            representation_list = []
            for task_index in range(train_index+1):
                test_list = training_dict_list[task_index]['test_list']
                test_iid_list = random.sample(test_list, k=args['iid_size'])
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num_layer'])
                test_iid_dataloader = dgl.dataloading.NodeDataLoader(graph, test_iid_list, sampler,
                device='cpu', batch_size=len(test_iid_list), shuffle=True, drop_last=False, num_workers=0)
                with tqdm(test_iid_dataloader) as tq, torch.no_grad():
                    for input_nodes, output_nodes, mfgs in tq:
                        inputs = mfgs[0].srcdata['feat']
                        presentation = model.get_embedding(mfgs, inputs)
                        representation_list.append(presentation)
            training_dict['replay_rep'] = torch.cat(representation_list)


        # the trained model parameter should propogate to the next iteration
        train_loss, test_acc, best_acc, model = train.train_multihead_model_onesample(model, args, data_info, training_dict)

        # store the training phase info
        train_phase_info['train_loss'] = train_loss
        train_phase_info['test_accuracy'] = test_acc
        train_phase_info['best_accuracy'] = best_acc


        # Compute and store accuracy on the old test set starting with the second train phase before replay
        if train_index == (data_info['task_number']-1):
            prev_class_acc = []
            prev_class_forget = []
            for task_index in range(train_index):
                #extract label info and test mask
                # the binary label list for task
                task_labels     =  torch.LongTensor(data_info['task_labels'][task_index])
                test_dataloader = training_dict_list[task_index]['test_dataloader']
                model.eval()
                predictions = []
                labels = []
                with tqdm(test_dataloader) as tq, torch.no_grad():
                    for input_nodes, output_nodes, mfgs in tq:
                        inputs = mfgs[0].srcdata['feat']
                        dstdata_id = mfgs[-1].dstdata[dgl.NID]
                        labels.append(task_labels[dstdata_id])
                        pred = model(mfgs, inputs, task_index)
                        predictions.append(pred.argmax(1).cpu().numpy())
                    predictions = np.concatenate(predictions)
                    labels = np.concatenate(labels)
                    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                    prev_class_acc.append(accuracy)

                    forget = phase_accuracy[task_index]['test_accuracy'] - accuracy
                    prev_class_forget.append(forget)


            # store the new accuracy for the previous class in the train phase info
            train_phase_info['prev_class_acc'] = prev_class_acc
            train_phase_info['prev_class_forget'] = prev_class_forget

        train_index += 1
        phase_accuracy.append(train_phase_info)

    return phase_accuracy




################################################################################
# naive incremental learning setting, simply feed each class into the model one by one
# without using replay strategy and the training/testing list is in random order.
################################################################################
def regular_training(args, data, model_name):
    # input:
    #     args - a dictionary containing the training setting information
    #     data - a string specifying the dataset, could be chosen from {'cora',
    #            'cite', 'pubmed', 'cora_full', 'aifb', 'amazon_comp', 'amazon_photo'
    #            'coau_cs', 'coau_phy' and 'reddit'}
    #     model_name - a string specifying the model name, could be chosen from
    #            {'GCN', 'SAGE', 'GAT'}
    # return:
    #     train_phase_info - a dictionary storing the training information

    #create training information
    training_dict, graph, data_info = data_process_factored.create_trainset_mix(data)

    #select the corresponding GNN models
    if model_name == 'GCN':
        model = models.GCNStoModel_MultiHead(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'],
          out_feats = args['num_outchannel'], num_task = args['task_number'],pred_head_out = args['class_per_task'] )
    if model_name == 'SAGE':
        model = models.SAGEStoModel_MultiHead(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'],
          out_feats = args['num_outchannel'], num_task = args['task_number'],pred_head_out = args['class_per_task'])
    if model_name == 'GAT':
        model = models.GATStoModel_MultiHead(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'],
          out_feats = args['num_outchannel'], num_task = args['task_number'], num_heads=2,pred_head_out = args['class_per_task'])


    # create list to store historical accuracy
    train_phase_info = {}
    opt = torch.optim.Adam(model.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])
    for epoch in range(args['epoches']):
        #prepare for training, effect dropout & batchnorm
        model.train()
        train_dataloader = training_dict['train_dataloader']
        with tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata['feat']
                dstdata_id = mfgs[-1].dstdata[dgl.NID]
                node_id = int(dstdata_id[0])
                task_index = args['task_dict'][node_id]
                label = args['label_dict'][node_id]
                label = torch.LongTensor([label])

                logits = model(mfgs, inputs, task_index)
                # logits = F.silu(logits)
                # logits = F.softmax(logits)
                # logits = F.relu(logits)
                logits = F.log_softmax(logits,dim = 1)

                loss = F.cross_entropy(logits,label)

                if args['loss'] == "cross_entropy":
                  loss = F.cross_entropy(logits,label)
                if args['loss'] == "nll_loss":
                  loss = F.nll_loss(logits,label)



                opt.zero_grad()
                loss.backward()
                opt.step()

                # Compute accuracy on training/validation/test
                train_acc = sklearn.metrics.accuracy_score(label.cpu().numpy(), logits.argmax(1).detach().cpu().numpy())
                # test_acc = evaluate.evaluate_multihead(model, args['graph'], args['graph'].ndata['feat'], args['label'], args['test_mask'],task_index)
                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)


    #store the training phase info
    train_phase_info['train_loss'] = loss.item()
    train_phase_info['train_acc'] = train_acc

    model.eval()
    task_test_dataloader_list = training_dict['task_test_dataloader']
    task_num = len(task_test_dataloader_list)
    task_acc_list = []
    for task_index in range(task_num):
        task_test_dataloader = task_test_dataloader_list[task_index]

        predictions = []
        labels = []
        with tqdm(task_test_dataloader) as tq, torch.no_grad():
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata['feat']
                dstdata_id = mfgs[-1].dstdata[dgl.NID]
                labels.append([args['label_dict'][index.item()] for index in dstdata_id])
                pred = model(mfgs, inputs, task_index)
                predictions.append(pred.argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            acc  =  sklearn.metrics.accuracy_score(labels, predictions)

            task_acc_list.append(acc)

    train_phase_info['task_acc_list'] = task_acc_list

    print(sum(task_acc_list)/len(task_acc_list))

    return train_phase_info
