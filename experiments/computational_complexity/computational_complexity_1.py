# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:05:42 2020

@author: henry
"""

import os
import os.path as osp
import sys
import warnings
from datetime import datetime
import json
import time

import pandas as pd
import numpy as np
import seaborn as sns

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
import torch.nn.functional as F

from networkx import from_numpy_matrix

from experiments.computational_complexity.comp_complexity_config import config


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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define data
dataset_name = 'Cora'
path = osp.join(os.getcwd(), '..', 'data', dataset_name)
dataset = Planetoid(path, dataset_name, T.NormalizeFeatures())
data = dataset[0]
data.batch = None
data.adj = to_dense_adj(data.edge_index)



loss_lambda = config['loss_lambda']

# data 
data = data.to(device)

lr = config['optimizer']['lr']


# define experiment
time_tracking_dict = {'n_epochs': [],
                      'time': [],
                      'method': [],
                      'accuracy': []}

memory_tracking_dict = {'method': [],
                      'memory': []
                      }


best_val_acc = test_acc = 0

t_start_all = time.time()
for n_repeats in range(10):
    
   
    for model_this in config['model'].keys():
        torch.cuda.reset_max_memory_allocated(device=device)
        model_constructor = config['model'][model_this]['model_constructor']
        model = model_constructor(dataset, **config['model'][model_this]['parameters']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        t_start = time.time()
        for epoch in range(1, 6):
                
            train(model_this)
        
            if epoch in [1, 5, 10, 25, 50, 100]:
                accs = test(model_this)
                t_this = time.time()-t_start
                
                train_acc, val_acc, tmp_test_acc = accs
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                
                time_tracking_dict['n_epochs'].append(epoch)
                time_tracking_dict['time'].append(t_this)
                time_tracking_dict['method'].append(model_this)
                time_tracking_dict['accuracy'].append(test_acc)
                
                log = 'Method: {}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, time: {:.4f}s'
                print(log.format(model_this, epoch, train_acc, best_val_acc, test_acc, t_this))
        
        max_mem = torch.cuda.max_memory_allocated(device)
        
        memory_tracking_dict['method'].append(model_this)
        memory_tracking_dict['memory'].append(max_mem)
        
        print(max_mem)
                


print('finished', 'total time', time.time()-t_start_all)

time_tracking_df = pd.DataFrame(data=time_tracking_dict)
memory_tracking_df = pd.DataFrame(data=memory_tracking_dict)

# plot
sns.set(style="whitegrid")

# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="n_epochs", y="time", hue="method", data=time_tracking_df,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Time in seconds")



g = sns.catplot(x="n_epochs", y="accuracy", hue="method", data=time_tracking_df,
                height=6, kind="point", palette="muted")


g = sns.catplot(x="method", y="memory", data=memory_tracking_df,
                height=6, kind="bar", palette="muted")

