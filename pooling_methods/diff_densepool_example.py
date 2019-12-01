# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:38:06 2019

@author: henry
"""

import os.path as osp
import os
import argparse
import collections

import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, graclus
from torch_geometric.nn import max_pool

from torch_geometric.transforms import Compose,  RemoveIsolatedNodes

import os.path as osp
from math import ceil
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj, \
    dense_to_sparse, from_scipy_sparse_matrix, from_networkx

from networkx import from_numpy_matrix

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes
    
def get_edge_index(adj):
    
    G = from_numpy_matrix(np.squeeze(adj))
    return from_networkx(G)
    
dataset = 'Cora'
path = osp.join(os.getcwd(), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]
data.batch = None

data.adj = to_dense_adj(data.edge_index)


class GCN(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_inputs, n_hidden, cached=True) # a conv layer
        self.conv2 = GCNConv(n_hidden, n_outputs, cached=True) # another conv layer 
        

    def forward(self, x, edge_index, edge_weight=None):
              
        x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        
        return x
        

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_clusters1 = 800
        num_clusters2 = 25
        nh1 = 16
        nh2 = 16
        n_out1 = dataset.num_classes
        n_out2 = 100
        # level 0: original graph
        self.gcn0 = GCN(dataset.num_features, nh1, n_out1)     
        
        # level 1: pooled 1
        self.conv_pool1 = GCNConv(dataset.num_features, num_clusters1, cached=True)
        self.gcn1 = GCN(dataset.num_features, nh2, n_out2)
        
        # level 2: pooled 2
        self.conv_pool2 = GCNConv(dataset.num_features, num_clusters2, cached=True)
        self.gcn2 = GCN(dataset.num_features, nh2, n_out2)
        
        # final layer: concat 
        self.conv_concat = GCNConv(n_out1 + n_out2, dataset.num_classes, cached=True)

    

    def forward(self, mask):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr # unpack the data container
        
        # regurlar gcn
        x0 = self.gcn0(x, edge_index, edge_weight) 
        # pooled1 
        s01 = F.relu(self.conv_pool1(x, edge_index, edge_weight))
        x1, adj1, l1, e1 = dense_diff_pool(x, data.adj, s01, mask)
        
#        data1 = get_edge_index(adj1.cpu().detach().numpy())

        adj1_sparse_tuple = dense_to_sparse(torch.squeeze(adj1))        
#        x2 = self.gcn1(torch.squeeze(x1), data1.edge_index, data1.weight)
        x1 = self.gcn1(torch.squeeze(x1), adj1_sparse_tuple[0], adj1_sparse_tuple[1])
        
        x1_unpooled = torch.matmul(s01, x1)
        
        
        # concat layers
        
        x_final = torch.cat((x0, x1_unpooled), 1)
        
        x_final = self.conv_concat(x_final, edge_index)
        
        
        
        other_loss = l1 + e1
      
        
        
        return F.log_softmax(x_final, dim=1), s01, other_loss
#        return F.log_softmax(x0, dim=1), 0, 0


def train():
    model.train()
    optimizer.zero_grad()
    
    training_output, s, other_loss = model(data.train_mask)
    training_output = training_output[data.train_mask]
    loss = F.nll_loss(training_output, data.y[data.train_mask]) + other_loss
    loss.backward(retain_graph=True) # this does backpropagation
    #F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()
    return s



def test():
    model.eval()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred, s, loss = model(mask)
        pred = pred[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, s



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    
best_val_acc = test_acc = 0
for epoch in range(1, 100):
    s = train()
    accs, s = test()
    train_acc, val_acc, tmp_test_acc = accs
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))




