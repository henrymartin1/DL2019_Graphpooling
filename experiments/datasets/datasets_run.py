# -*- coding: utf-8 -*-
"""
Created on Jan 17 based on Henry's complexity_run.py implementation

@author: glili
"""

import os
import os.path as osp
import sys
import warnings
from datetime import datetime
import json
import time
import random
import csv
import pickle

sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import seaborn as sns

import torch
from torch_geometric.datasets import Planetoid, CoraFull
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
import torch.nn.functional as F

from networkx import from_numpy_matrix

from experiments.datasets.datasets_config import config


def test(model_name):
    model.eval()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        if model_name in ['diff_pool_net1', 'diff_pool_net2']:
            output_dict = model(data, mask)
            logits = output_dict['prediction']
        else:
             logits = model(data)
            
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def train(model_name):
    model.train()
    optimizer.zero_grad()

    if model_name in ['diff_pool_net1', 'diff_pool_net2']:
        output_dict = model(data, data.train_mask) # predict
        training_output = output_dict['prediction'][data.train_mask]
        loss = F.nll_loss(training_output, data.y[data.train_mask]) + (loss_lambda * output_dict['edge_loss'])
        
    else:
        loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])

        
    
    loss.backward(retain_graph=True)  # this does backpropagation
    optimizer.step()


# define device
device = torch.device(1)

loss_lambda = config['loss_lambda']
lr = config['optimizer']['lr']


# define experiment
acc_tracking_dict = {'dataset': [],
                     'run': [],
                    'n_epochs': [],
                     'method': [],
                     'accuracy': [],
                     'mean': [],
                     'std_dev': []}

for dataset_name in ['PubMed', 'CoraFull', 'Cora']:
    # define data
    #dataset_name = 'Cora' # 'PubMed', 'CoraFull'
    path = osp.join(os.getcwd(), '..', 'data', dataset_name)
    
    if (dataset_name == 'CoraFull'):
        dataset = CoraFull(path, T.NormalizeFeatures())
    else:
        dataset = Planetoid(path, dataset_name, T.NormalizeFeatures())
        
    data = dataset[0]
    data.batch = None
    data.adj = to_dense_adj(data.edge_index)
    # data 
    data.test_mask = torch.empty(size=torch.Size([data.x.shape[0]]), dtype=torch.bool)
    data.val_mask = torch.empty(size=torch.Size([data.x.shape[0]]), dtype=torch.bool)
    data.train_mask = torch.empty(size=torch.Size([data.x.shape[0]]), dtype=torch.bool)

    data = data.to(device)



    training_fraction = 0.05
    if dataset_name == 'CoraFull':
         #set the training mask according to training_fraction:
        for node in range(0, data.num_nodes-1):
            if random.random() < training_fraction: #with probability training_fraction include the node in the training set
                data.train_mask[node] = True
                data.val_mask[node] = False
                data.test_mask[node] = False
            else:
                data.train_mask[node] = False
                if random.random() < 0.5: #split the remaining nodes roughly evenly between validation and test sets
                    data.val_mask[node] = True
                    data.test_mask[node] = False
                else:
                    data.val_mask[node] = False
                    data.test_mask[node] = True
    
    
    for model_this in config['model'].keys():
        if dataset_name in ['CoraFull', 'PubMed'] and model_this in ['diff_pool_net1', 'diff_pool_net2']:
            continue

        model_accuracies = []
        
        for n_repeats in range(config['n_repeats']):
        
            model_constructor = config['model'][model_this]['model_constructor']
            model = model_constructor(dataset, **config['model'][model_this]['parameters']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_val_acc = test_acc = 0
            
            for epoch in range(config['epochs']):

                train(model_this)

                #if epoch in config['tracked_epochs']:
                accs = test(model_this)

                train_acc, val_acc, tmp_test_acc = accs

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                
                if epoch in config['tracked_epochs']:
                    log = 'Method: {}, Run: {:03d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(model_this, n_repeats, epoch, train_acc, best_val_acc, test_acc))
                    
            model_accuracies.append(test_acc)
                
            acc_tracking_dict['dataset'].append(dataset_name)
            acc_tracking_dict['run'].append(n_repeats)
            acc_tracking_dict['n_epochs'].append(config['epochs'])
            acc_tracking_dict['method'].append(model_this)
            acc_tracking_dict['accuracy'].append(test_acc)
                
            #mean and std of the best test_acc's so far
            acc_tracking_dict['mean'].append(np.mean(model_accuracies)) 
            acc_tracking_dict['std_dev'].append(np.std(model_accuracies))
                
           
print('finished')

acc_tracking_df = pd.DataFrame(data=acc_tracking_dict)

#print(acc_tracking_df)

output_dict = {'accuracy': acc_tracking_df}
print('Write to:',  config['output_file'])
pickle.dump(output_dict, open( config['output_file'], "wb" ) )
