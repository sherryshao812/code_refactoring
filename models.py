##import library
import dgl

from typing import Tuple, Optional
from torch import nn
from torch import Tensor
from torch.nn import Parameter

import dgl.nn as dglnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


################################################################################
#  Adopt GCN model to transform to a multihead model
################################################################################
class GCNStoModel_MultiHead(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_task, pred_head_out):
        super().__init__()

        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, aggregator_type='gcn')
        self.conv2 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='gcn')
        # self.hid_feats = hid_feats

        self.heads = nn.ModuleList([])
        #create prediction heads
        for i in range(num_task):
            self.heads.append(nn.Linear(out_feats, pred_head_out))

    def forward(self, mfgs, x, task_index):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        h = F.relu(h)
        if task_index == -1:
            return h
        else:
            pred = self.heads[task_index](h)
            return pred

    def get_embedding(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        h = F.relu(h)
        return  h


################################################################################
#  Adopt GraphSAGE model to transform to a multihead model
################################################################################
class SAGEStoModel_MultiHead(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_task, pred_head_out):
        super().__init__()

        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='mean')
        # self.hid_feats = hid_feats

        self.heads = nn.ModuleList([])
        #create prediction heads
        for i in range(num_task):
            self.heads.append(nn.Linear(out_feats, pred_head_out))

    def forward(self, mfgs, x, task_index):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        h = F.relu(h)
        if task_index == -1:
            return h
        else:
            pred = self.heads[task_index](h)
            return pred

    def get_embedding(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        h = F.relu(h)
        return  h


################################################################################
#  Adopt GAT model to transform to a multihead model
################################################################################
class GATStoModel_MultiHead(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_task, num_heads, pred_head_out):
        super().__init__()

        self.conv1 = dglnn.GATConv(in_feats, hid_feats, num_heads)
        self.conv2 = dglnn.GATConv(hid_feats*num_heads, out_feats, num_heads)
        # self.hid_feats = hid_feats

        self.heads = nn.ModuleList([])
        #create prediction heads
        for i in range(num_task):
          self.heads.append(nn.Linear(out_feats, pred_head_out))

    def forward(self, mfgs, x, task_index):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        h = F.relu(h)
        if task_index == -1:
            return h
        else:
            pred = self.heads[task_index](h)
            return pred

    def get_embedding(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        h = F.relu(h)
        return  h
