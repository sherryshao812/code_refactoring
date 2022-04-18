import torch
import dgl
import sklearn.metrics

import torch.nn.functional as F
import numpy as np

import evaluate
import dist

from tqdm import tqdm

################################################################################
#  modified training procedure for multi-head GNN models
################################################################################
def train_multihead_model(model, args, data_info, training_dict):
    # input:
    #     model - the initial GNN model
    #     args - a dictionary containing the training setting information
    #     data_info - a dictionary containing the dataset information
    #     training_dict - the training dictionary containing train and test
    #                     dataloader information
    # return:
    #     loss.item(), accuracy, best_accuracy, model - the loss value, latest
    #                       prediction accuracy, best achieved accuracy,
    #                       the trained model

    #use Adam optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    #store the best accuracy during training
    best_accuracy = 0

    #extract label
    task_index        = training_dict ['task_index']
    task_labels       = torch.LongTensor(data_info['task_labels'][task_index])
    train_dataloader  = training_dict['train_dataloader']
    test_dataloader   = training_dict['test_dataloader']

    for epoch in range(args['epoches']):
        #prepare for training, effect dropout & batchnorm
        model.train()
        with tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata['feat']
                dstdata_id = mfgs[-1].dstdata[dgl.NID]
                label = task_labels[dstdata_id]
                logits = model(mfgs, inputs, task_index)

                # Compute prediction
                if args['output_act'] == 'sigmoid':
                    logits = F.sigmoid(logits)
                if args['output_act'] == 'relu':
                    logits = F.relu(logits)
                if args['output_act'] == 'log_softmax':
                    logits = F.log_softmax(logits,dim = 1)
                if args['output_act'] == 'softmax':
                    logits = F.softmax(logits)
                if args['output_act'] == 'swish':
                    logits = F.silu(logits)
                if args['output_act'] == 'null':
                    logits = logits

                if args['loss'] == "cross_entropy":
                    loss = F.cross_entropy(logits,label)
                if args['loss'] == "nll_loss":
                    loss = F.nll_loss(logits,label)
                if args['loss'] == "cmd_reg":
                    loss = F.cross_entropy(logits,label) + args['cmd_r'] * dist.cmd(train_prediction,logits)

                # backward propagation
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Compute accuracy on training/validation/test
                train_acc = sklearn.metrics.accuracy_score(label.cpu().numpy(), logits.argmax(1).detach().cpu().numpy())
                # test_acc = evaluate.evaluate_multihead(model, args['graph'], args['graph'].ndata['feat'], args['label'], args['test_mask'],task_index)
                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

        # input("checking point: training procedure is working")

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
            print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
            if best_accuracy < accuracy:
                best_accuracy = accuracy

    return loss.item(), accuracy, best_accuracy, model



################################################################################
#  modified training procedure for multi-head GNN models
################################################################################
def train_multihead_model_onesample(model, args, data_info, training_dict):
    # input:
    #     model - the initial GNN model
    #     args - a dictionary containing the training setting information
    #     data_info - a dictionary containing the dataset information
    #     training_dict - the training dictionary containing train and test
    #                     dataloader information
    # return:
    #     loss.item(), accuracy, best_accuracy, model - the loss value, latest
    #                       prediction accuracy, best achieved accuracy,
    #                       the trained model

    #use Adam optimizer
    opt = torch.optim.Adam(model.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])

    #store the best accuracy during training and extract information
    best_accuracy = 0
    train_dataloader = training_dict['train_dataloader']
    test_dataloader = training_dict['test_dataloader']

    for epoch in range(args['epoches']):
        #prepare for training, effect dropout & batchnorm
        model.train()
        with tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # input('I can get here')
                inputs = mfgs[0].srcdata['feat']
                dstdata_id = mfgs[-1].dstdata[dgl.NID]
                task_index = data_info['task_dict'][int(dstdata_id[0])]
                label = torch.LongTensor(data_info['task_labels'][task_index])[dstdata_id]

                logits = model(mfgs, inputs, task_index)

                # Compute prediction with different final activation function
                if args['output_act'] == 'sigmoid':
                    logits = F.sigmoid(logits)
                if args['output_act'] == 'relu':
                    logits = F.relu(logits)
                if args['output_act'] == 'log_softmax':
                    logits = F.log_softmax(logits,dim = 1)
                if args['output_act'] == 'softmax':
                    logits = F.softmax(logits)
                if args['output_act'] == 'swish':
                    logits = F.silu(logits)
                if args['output_act'] == 'null':
                    logits = logits

                if args['loss'] == "cross_entropy":
                    loss = F.cross_entropy(logits, label)
                if args['loss'] == "nll_loss":
                    loss = F.nll_loss(logits, label)
                if args['loss'] == "cmd_reg":
                    if 'replay_rep' in training_dict:
                        presentation = model(mfgs, inputs, -1)
                        loss = F.cross_entropy(logits, label) + args['cmd_r'] * dist.cmd(presentation,training_dict['replay_rep'])
                        # loss = F.cross_entropy(logits,label) + args['cmd_r'] * dist.MMD(presentation,training_dict['replay_rep'])
                        # loss = F.cross_entropy(logits,label) + args['cmd_r'] * dist.pairwise_distances(presentation,training_dict['replay_rep'])
                        # print('cmd distance is: ', dist.cmd(presentation,training_dict['replay_rep']))
                        # print('cross entropy loss is: ', F.cross_entropy(logits,label))
                        # print('total loss is: ', loss)
                    else:
                        loss = F.cross_entropy(logits,label)

                # backward propagation
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Compute accuracy on training/validation/test
                train_acc = sklearn.metrics.accuracy_score(label.cpu().numpy(), logits.argmax(1).detach().cpu().numpy())
                # test_acc = evaluate.evaluate_multihead(model, args['graph'], args['graph'].ndata['feat'], args['label'], args['test_mask'],task_index)
                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

        # input("checking point: training procedure is working")

        model.eval()
        predictions = []
        labels = []
        with tqdm(test_dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                inputs = mfgs[0].srcdata['feat']
                dstdata_id = mfgs[-1].dstdata[dgl.NID]
                labels.append(torch.LongTensor(data_info['task_labels'][task_index])[dstdata_id])
                pred = model(mfgs, inputs, task_index)
                predictions.append(pred.argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)
            print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
            if best_accuracy < accuracy:
                best_accuracy = accuracy

    # input("checking point: testing procedure is working")


    return loss.item(), accuracy, best_accuracy, model
