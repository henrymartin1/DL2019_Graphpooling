# -*- coding: utf-8 -*-
"""
Based on Henry's implementation for the computational complexity experiments

@author: christian
"""

import os
import os.path as osp
import sys
import warnings
from datetime import datetime
import json
import csv
import time
import random

import pandas as pd
import numpy as np
import seaborn as sns

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
import torch.nn.functional as F

from networkx import from_numpy_matrix

from varying_training_percentage_config import config


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

        
        
train_percentage_dict = {   'method': [],
                            'train_percentage': [],
                            'best_accuracy': []}


for n_repeats in range(10):
    print("Repetition {} of 10".format(str(n_repeats+1)))
    for model_this in config['model'].keys():
        for training_fraction in config['train_fracs']:
            model_constructor = config['model'][model_this]['model_constructor']
            model = model_constructor(dataset, **config['model'][model_this]['parameters']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
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
            
            best_val_acc = test_acc = 0
            for epoch in range(config['epochs']):
                train(model_this)
                accs = test(model_this)
                train_acc, val_acc, tmp_test_acc = accs
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                print("Epoch {} done".format(str(epoch)))
            
            train_percentage_dict['method'].append(model_this)
            train_percentage_dict['best_accuracy'].append(test_acc)
            train_percentage_dict['train_percentage'].append(training_fraction)
            
            print("Method: " + str(model_this))
            print("Training percentage: " + str(training_fraction))
            print("Best accuracy: " + str(test_acc))
         
w = csv.writer(open("results.csv", "w"))
for key, val in train_percentage_dict.items():
    w.writerow([key, val])
print("All done")


train_percentage_df = pd.DataFrame(data=train_percentage_dict)

# plot
sns.set(style="whitegrid")

                
plot = sns.catplot(x="train_percentage", y="best_accuracy", hue="method", data=train_percentage_df, height=6, kind="point", palette="muted")
plot.savefig("percentage_plot.png")



