# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:38:06 2019

@author: henry
"""

import os.path as osp
import os
import sys
import argparse
import collections
import warnings 

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
from datetime import datetime

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj, \
    dense_to_sparse, from_scipy_sparse_matrix, from_networkx

sys.path.append(os.path.join(os.getcwd(),'..'))
    
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    from utils.visual_TB import Visualizer
    
from utils.earlystopping import EarlyStopping
from networkx import from_numpy_matrix
from config_run import config
import json

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes
    
def get_edge_index(adj):
    
    G = from_numpy_matrix(np.squeeze(adj))
    return from_networkx(G)
    


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
        

class diff_pool_net1(torch.nn.Module):
    def __init__(self, num_clusters=100, nh0_1=16, nh1_1=16, n_out0=10, n_out1=100):
        super(diff_pool_net1, self).__init__()
       
        # level 0: original graph
        self.gcn0 = GCN(dataset.num_features, nh0_1, n_out0)     
        
        # level 1: pooled 1
        self.conv_pool1 = GCNConv(dataset.num_features, num_clusters, cached=True)
        self.gcn1 = GCN(dataset.num_features, nh1_1, n_out1)
        
        # final layer: concat 
        self.conv_concat = GCNConv(n_out0 + n_out1, dataset.num_classes, cached=True)

    

    def forward(self, mask):
        # unpack the data container
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr 
        
        # regurlar gcn
        x0 = self.gcn0(x, edge_index, edge_weight) 
        
        # pooled1 
        s01 = F.relu(self.conv_pool1(x, edge_index, edge_weight))
        x1, adj1, l1, e1 = dense_diff_pool(x, data.adj, s01, mask)
        
        # unpool
        adj1_sparse_tuple = dense_to_sparse(torch.squeeze(adj1))        
        x1 = self.gcn1(torch.squeeze(x1), adj1_sparse_tuple[0], adj1_sparse_tuple[1])
        
        x1_unpooled = torch.matmul(s01, x1)
        
        
        # concat predictions
        x_final = torch.cat((x0, x1_unpooled), 1)
        x_final = self.conv_concat(x_final, edge_index)
                
        edge_loss = l1 + e1
        
        output_dict = {'prediction': F.log_softmax(x_final, dim=1), 's01': s01,
                'edge_loss': edge_loss, 'adj1': adj1}
        
        return output_dict
    
def update_config(config, num_clusters=100, nh1_1=16, n_out1=100, lr=0.01, \
                  config_loss='single'):
        
    # model statistics 
    config['model']['num_clusters'] = num_clusters
    config['model']['nh1_1'] = nh1_1
    config['model']['n_out1'] = n_out1
    
    # log name
    config['model_log_info'] =  str(config['model']['num_clusters'])        
    config['optimizer']['lr'] = lr
    
    # loss
    config['loss'] = config_loss
    
    return config
    
    



def train():
    model.train()
    optimizer.zero_grad()
    
    output_dict = model(data.train_mask)
    
    training_output = output_dict['prediction'][data.train_mask]
    if config['loss'] == 'single':
        loss = F.nll_loss(training_output, data.y[data.train_mask])  #+ output_dict['edge_loss']
    else: 
        loss = F.nll_loss(training_output, data.y[data.train_mask])  + output_dict['edge_loss']
    loss.backward(retain_graph=True) # this does backpropagation
    optimizer.step()
    
    if epoch % 10 == 0:
        adj = output_dict['adj1']
        writer.write_hist(adj, epoch)
        writer.write_adj_matrix(adj, epoch)
    return output_dict['s01']

def test():
    model.eval()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        output_dict = model(mask)
        pred = output_dict['prediction']
        pred = pred[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, output_dict['s01']


# define data
dataset = 'Cora'
path = osp.join(os.getcwd(), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]
data.batch = None

data.adj = to_dense_adj(data.edge_index)



# define logging
log_dir = os.path.join('..','runs',config['modelname'] + '_' 
                           + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                           +'_' + config['model_log_info'])
writer = Visualizer(log_dir)
with open(os.path.join(log_dir,'config.json'), 'w') as fp:
        json.dump(config, fp)

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model = diff_pool_net1(**config['model']).to(device)
data = data.to(device)
lr = config['optimizer']['lr']
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# initialize the early_stopping object
early_stopping = EarlyStopping(log_dir, patience=config['patience'], verbose=False)


best_val_acc = test_acc = 0
for epoch in range(1, config['epochs']):
    s = train()
    accs, s = test()
    train_acc, val_acc, tmp_test_acc = accs
    
    writer.write_lr(optimizer, epoch)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        
    writer.write_acc_train(train_acc, epoch)
    writer.write_acc_validation(val_acc, epoch)
    writer.write_acc_test(test_acc, epoch)
    
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
    
    early_stopping(val_acc, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

writer.close()


