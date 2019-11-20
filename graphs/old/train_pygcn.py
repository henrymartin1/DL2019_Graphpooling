from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import os, sys
sys.path.append(os.getcwd())

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN, GCN2

from videoloader import trafic4cast_dataset, test_dataloader


import pickle
import scipy
import numpy
from scipy.sparse import csr_matrix, coo_matrix
from scipy import sparse
import os
import matplotlib.pyplot as plt
import numpy as np
os.chdir(r'E:\Programming\traffic4cast')
import torch
import networkx as nx
from scipy import sparse


def csr_to_torch(A_csr):
    A_csr = coo_matrix(A_csr)    
    values = A_csr.data
    indices = np.vstack((A_csr.row, A_csr.col))
    
    indices = indices.astype(np.int64)
    i = torch.LongTensor(indices)
    
    v = torch.FloatTensor(values.astype('float'))
    shape = A_csr.shape
    A_torch = torch.sparse.IntTensor(i, v, torch.Size(shape))
    
    return A_torch

def coo_to_torch(A_csr):  
    values = A_csr.data
    indices = np.vstack((A_csr.row, A_csr.col))
    
    indices = indices.astype(np.int64)
    i = torch.LongTensor(indices)
    
    v = torch.FloatTensor(values.astype('float'))
    shape = A_csr.shape
    A_torch = torch.sparse.IntTensor(i, v, torch.Size(shape))
    
    return A_torch

def image_to_vector(image, nn_ixs):
    vec = image[...,nn_ixs[0],nn_ixs[1]]

    return vec

def vector_to_image(vec, nn_ixs, n_feat=36, batch_size=1):
    zero_image = np.zeros((batch_size,n_feat,495,436))
    zero_image[...,nn_ixs[0],nn_ixs[1]] = vec
    
    return zero_image
    

def is_adjacent_in_image(image_ix1,image_ix2):
    x1,y1 = image_ix1
    x2,y2 = image_ix2
    
    if (np.abs(x1-x2) <= 1) and (np.abs(y1-y2) <= 1):
        return True
    else:
        return False

# create matrix
mask_dict = pickle.load(open( os.path.join('utils', 'masks.dict'), "rb" ) )
mask = mask_dict['Berlin']['mask']
sum_berlin = mask_dict['Berlin']['sum']
mask2 = sum_berlin > 50000

nn_ixs = np.where(mask2)

# make matrix
mask2_inv = ~mask2
to_delete = np.where(mask2_inv)
to_delete2 = list(zip(to_delete[0],to_delete[1]))

G = nx.grid_graph(dim=[436, 495])
G.remove_nodes_from(to_delete2)
A = nx.to_scipy_sparse_matrix(G)

adj = csr_to_torch(A)

def transform_shape_train(data):
    data = data.reshape((batch_size,36,-1))
    return np.squeeze(data)

def transform_shape_test(data):
    data =  data.reshape((batch_size,9,-1))
    return np.squeeze(data)
    
t = time.time()

def blockify_A(A,batch_size):
    
    A_list = []
    for i in range(batch_size):
        A_list.append(A)
        
    adj_block = scipy.sparse.block_diag((A_list), format='csr')
        
    return csr_to_torch(adj_block)
    
def blockify_data(features, target_red, batch_size):
    n_feats = features.shape[1]
    n_classes = target_red.shape[1]
    features = features.permute(1,2,0)
    features = features.reshape(-1,n_feats,1)
    
    target_red = target_red.permute(1,2,0)
    target_red = target_red.reshape(-1,n_classes,1)
    
    return np.squeeze(features), np.squeeze(target_red)
    
    
    

def train(epoch, data_loader, batch_size):
    
    running_loss = 0
    total_loss = 0
    for batch_ix, (data,target) in enumerate(data_loader):
        

        features = image_to_vector(data/255,nn_ixs)
        target_red = image_to_vector(target/255,nn_ixs)

        
        features_block, target_red_block = blockify_data(features, target_red, batch_size)
                
#        features = features.permute(0,2,1)
#        target_red = target_red.permute(0,2,1)
        
        
        features_block = features_block.to(device)
        target_red_block = target_red_block.to(device)
        
        
        model.train()
        optimizer.zero_grad()
        
        output = model(features_block, adj_block)
        
        loss_train = F.mse_loss(output, target_red_block)
        
        running_loss += loss_train.item()
        total_loss += loss_train.item()
        
        if batch_ix % 50 == 0:
            print('\t batch ix: {:.0f} loss {:.8f}, \t duration {:.2f} s'.format(batch_ix, running_loss/100, time.time() - t))
            running_loss = 0.0
            
        
        loss_train.backward()
        optimizer.step()
            
    return total_loss/(batch_ix)
    
    
    

            
            

if __name__ == '__main__':
    batch_size = 5
    adj_block = blockify_A(A, batch_size)

    # Load data
    source_root = r"C:\Data\traffic4cast"
    target_root = "data"
    
    kwds_dataset = {'compression':None, 'reduce':True, 'filter_test_times':False,
                    'cities': ['Berlin']}
    kwds_loader = {'shuffle': True, 'num_workers':4, 
            'batch_size': batch_size, 'pin_memory': True, 'drop_last': True }
    kwds_tester = {'plot':False}
    
    dataset = trafic4cast_dataset(source_root, target_root,
                                      split_type='training', **kwds_dataset)
    loader = torch.utils.data.DataLoader(dataset, **kwds_loader)    
    
    
    
    
    #adj, features, labels, idx_train, idx_val, idx_test = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_features = 36
    
    # Model and optimizer
#    model = GCN(nfeat=n_features,
#                nhid=24,
#                nclass=9,
#                dropout=0.5)
    
    model = GCN2(nfeat=n_features,
                #nhid1=500, nhid2=200,
                nhid1=10, nhid2=10,
                nclass=9,
                dropout=0.5)
    
#    optimizer = optim.Adam(model.parameters(),
#                           lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    
    model.to(device)
    #adj = adj.to(device)
    adj_block = adj_block.to(device)
    
    best_loss = inf
    # Train model
    t_total = time.time()
    for epoch in range(10):
        loss = train(epoch, loader, batch_size)
        print('epoch: {:i} \t loss: {:.2f}'.format(epoch, loss))
        
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    # Testing

