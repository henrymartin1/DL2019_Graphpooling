import os
import os.path as osp
import warnings
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj, \
    dense_to_sparse, from_scipy_sparse_matrix, from_networkx

sys.path.append(os.path.join(os.getcwd(), '..'))

from networkx import from_numpy_matrix

 
class GCN(torch.nn.Module):
   def __init__(self, n_in, n_out, n_hid=64):
       super(GCN, self).__init__()
       self.conv1 = GCNConv(n_in, n_out, cached=False)  # a conv layer

   def forward(self, x, edge_index, edge_weight=None):
       x = F.relu(self.conv1(x, edge_index, edge_weight))
       return x


class diff_pool_net1(torch.nn.Module):
    def __init__(self, dataset, num_clusters1, num_clusters2, n_hidden=64, hidden_channels=None):
        super(diff_pool_net1, self).__init__()      
            
        # level 0: original graph
        self.gcn0_in = GCN(n_in=dataset.num_features, n_out=hidden_channels, n_hid=n_hidden)
        self.gcn0_out = GCN(n_in=2*hidden_channels, n_out=dataset.num_classes, n_hid=n_hidden)
               
        # level 1: pooled 1
        self.conv_pool1 = GCNConv(hidden_channels, num_clusters1, cached=False)
        self.gcn1_in = GCN(n_in=hidden_channels, n_out=hidden_channels, n_hid=n_hidden)
        self.gcn1_out = GCN(n_in=2*hidden_channels, n_out=hidden_channels, n_hid=n_hidden)
        
        # level 2: pooled 2
        self.conv_pool2 = GCNConv(hidden_channels, num_clusters2, cached=False)
        self.gcn2_in = GCN(n_in=hidden_channels, n_out=hidden_channels, n_hid=n_hidden)
        

    def forward(self, data, mask):
        # unpack the data container
        x0, edge_index0, edge_weight0 = data.x, data.edge_index, data.edge_attr

        # level 0 conv  
        x0_ = self.gcn0_in(x0, edge_index0, edge_weight0)

        # pooled 1 
        s1 = F.relu(self.conv_pool1(x0_, edge_index0, edge_weight0))
        x1, adj1, l1, e1 = dense_diff_pool(x0_, data.adj, s1, mask)
        x1 = torch.squeeze(x1)
        
        # get edge index level 1
        adj1_sparse_tuple = dense_to_sparse(torch.squeeze(adj1))
        edge_index1 = adj1_sparse_tuple[0]
        edge_weight1 = adj1_sparse_tuple[1]
                
        # level 1 conv
        x1_ = self.gcn1_in(x1, edge_index1, edge_weight1)
        
        # pooled 2 
        s2 = self.conv_pool2(x1_, edge_index1, edge_weight1)
        s2 = F.relu(s2)
        x2, adj2, l2, e2 = dense_diff_pool(x1_, adj1, s2)
        x2 = torch.squeeze(x2)
        
        # get edge index level 2
        adj2_sparse_tuple = dense_to_sparse(torch.squeeze(adj2))
        edge_index2 = adj2_sparse_tuple[0]
        edge_weight2 = adj2_sparse_tuple[1]
        
        # level 2 conv
        x2_out = self.gcn2_in(x2, edge_index2, edge_weight2)
        x2_out_up = torch.matmul(s2, x2_out) # unpool level 2
        
        # output level 1
        x1_out = self.gcn1_out(torch.cat((x1_, x2_out_up), 1), edge_index1, edge_weight1)
        x1_out_up = torch.matmul(s1, x1_out) # unpool level 1
        
        # output level 0 
        x0_out = self.gcn0_out(torch.cat((x0_, x1_out_up), 1), edge_index0, edge_weight0)
    
        edge_loss = l1 + e1 +l2 + e2
        
        edges = {'e1' :{'e': edge_index1, 'w': edge_weight1},
                 'e2' :{'e': edge_index2, 'w': edge_weight2}}

        output_dict = {'prediction': F.log_softmax(x0_out, dim=1), 's01': s1,
                       'edge_loss': edge_loss, 'adj1': adj1, 'edges': edges}

        return output_dict



