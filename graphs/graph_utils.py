"""Summary
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torch_geometric.nn.models import GraphUNet
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import pickle
import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'ye'))

from datetime import datetime

import networkx as nx
from scipy import sparse

import scipy
from scipy.sparse import csr_matrix, coo_matrix
from scipy import sparse
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing

def csr_to_torch(A_csr):
    """Summary
    
    Args:
        A_csr (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    A_coo = coo_matrix(A_csr)    
    values = A_coo.data
    indices = np.vstack((A_coo.row, A_coo.col))
    
    indices = indices.astype(np.int64)
    i = torch.LongTensor(indices)
    
    v = torch.FloatTensor(values.astype('float'))
    shape = A_coo.shape
    A_torch = torch.sparse.IntTensor(i, v, torch.Size(shape))
    
    return A_torch

def coo_to_torch(A_coo):  
    """Summary
    
    Args:
        A_coo (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    values = A_coo.data
    indices = np.vstack((A_coo.row, A_coo.col))
    
    indices = indices.astype(np.int64)
    i = torch.LongTensor(indices)
    
    v = torch.FloatTensor(values.astype('float'))
    shape = A_coo.shape
    A_torch = torch.sparse.IntTensor(i, v, torch.Size(shape))
    
    return A_torch

def image_to_vector(image, nn_ixs):
    """Summary
    
    Args:
        image (TYPE): Description
        nn_ixs (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    vec = image[...,nn_ixs[0],nn_ixs[1]]

    return vec

def vector_to_image(vec, nn_ixs, n_feat=36, batch_size=1, n=495, m=436):
    """Summary
    
    Args:
        vec (TYPE): Description
        nn_ixs (TYPE): Description
        n_feat (int, optional): Description
        batch_size (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    zero_image = np.zeros((batch_size,n_feat,n,m))
    zero_image[...,nn_ixs[0],nn_ixs[1]] = vec
    
    return zero_image

def create_adj_matrix(city='Berlin', mask_threshold=0, do_subsample=None):
    """Summary
    
    Args:
        city (str, optional): Description
        mask_threshold (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    # create matrix
    mask_dict = pickle.load(open( os.path.join('utils', 'masks.dict'), "rb" ) )

    if mask_threshold > 0:
        sum_city = mask_dict[city]['sum']
        mask = sum_city > mask_threshold

    else:
        mask = mask_dict[city]['mask']


    if do_subsample is not None:
        i,j = do_subsample
        mask = mask[i:j, i:j]
    

    nn_ixs = np.where(mask)

    # make matrix
    mask_inv = ~mask
    to_delete = np.where(mask_inv)
    to_delete = list(zip(to_delete[0],to_delete[1]))

    m,n = mask.shape
    G = get_grid_graph(m, n)   

    G.remove_nodes_from(to_delete)
    A = nx.to_scipy_sparse_matrix(G)
    A = A + scipy.sparse.identity(A.shape[0], dtype='bool',format='csr')
    return A, nn_ixs, G, mask

def get_grid_graph(n, m):

    G = nx.grid_2d_graph(n, m)
    rows=range(n)
    columns=range(m)
    G.add_edges_from( ((i,j),(i-1,j-1)) for i in rows for j in columns if i>0 and j>0 )
    G.add_edges_from( ((i,j),(i-1,j+1)) for i in rows for j in columns if i>0 and j<max(columns))

    return G
    

def transform_shape_train(data, batch_size, n_channels=36):
    """Summary
    
    Args:
        data (TYPE): Description
        batch_size (TYPE): Description
        n_channels (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    data = data.reshape((batch_size,n_channels,-1))
    return np.squeeze(data)

def transform_shape_test(databatch_size, n_channels=9):
    """Summary
    
    Args:
        databatch_size (TYPE): Description
        n_channels (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    data =  data.reshape((batch_size,n_channels,-1))
    return np.squeeze(data)
    
def blockify_A(A,batch_size):
    """Summary
    
    Args:
        A (TYPE): Description
        batch_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    A_list = []
    for i in range(batch_size):
        A_list.append(A)
        
    adj_block = scipy.sparse.block_diag((A_list), format='csr')
        
    return adj_block
    
def blockify_data(features, target, batch_size):
    """Summary
    
    Args:
        features (TYPE): Description
        target_red (TYPE): Description
        batch_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    n_feats = features.shape[1]
    n_classes = target.shape[1]

    features = features.permute(1,0,2)
    target = target.permute(1,0,2)

    features = features.reshape(n_feats, -1)   
    target = target.reshape(n_classes, -1)

    features = features.permute(1,0)
    target = target.permute(1,0)
    
    return features, target

def unblockify_target(target, batch_size):
    """Summary
    
    Args:
        target_red (TYPE): Description
        batch_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    target_list = np.split(target, batch_size)
    return target_list

def retransform_unblockify_target(target_block, nn_ixs, batch_size, dataset, n_feat=9 ): 
    """input numpy - output numpy
    
    Args:
        target_block (TYPE): Description
        nn_ixs (TYPE): Description
        batch_size (TYPE): Description
        n_feat (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    if dataset.subsample:
        n = np.abs(dataset.n- dataset.m)
        m = n
        
    else:
        n = 495
        m = 436
        
    target_list = unblockify_target(target_block, batch_size)
    target_vector = np.stack(target_list)
    target_vector = np.moveaxis(target_vector, 2,1)
    target_image = vector_to_image(target_vector, nn_ixs, n_feat=n_feat, batch_size=batch_size, n=n, m=m)
    return target_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    class GCNConv(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
            self.lin = torch.nn.Linear(in_channels, out_channels)
    
        def forward(self, x, edge_index):
            # x has shape [N, in_channels]
            # edge_index has shape [2, E]
    
            # Step 1: Add self-loops to the adjacency matrix.
            
    
            # Step 2: Linearly transform node feature matrix.
            
    
            # Step 3-5: Start propagating messages.
            return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        
    
    G = get_grid_graph(10,6)
    
    pos = nx.spring_layout(G, iterations=1000)
    nx.draw(G, pos, font_size=8)
    
    A, nn_ixs, G, mask = create_adj_matrix(city='Istanbul', mask_threshold=400000)
    adj = csr_to_torch(A)
    
    batch_size=2
    A_block = blockify_A(A, batch_size)
    adj_block = csr_to_torch(A_block)
    
    fig, axs = plt.subplots(1,3)
    
    inputs = torch.from_numpy(np.ones((batch_size,36,495,436)))
    target = torch.from_numpy(np.ones((batch_size,9,495,436)))
    
    axs[0].imshow(mask)
#    inputs = torch.from_numpy(np.random.normal(size=(batch_size,36,495,436)))
#    target = torch.from_numpy(np.random.normal(size=(batch_size,9,495,436)))
    
    # transform to vector
    inputs_vec = image_to_vector(inputs, nn_ixs)
    target_vec = image_to_vector(target, nn_ixs)
    
    # create block for batch learning
    
    inputs_block, target_block = blockify_data(inputs_vec, target_vec, batch_size)
    
    # apply model
    inputs_block = inputs_block.type(torch.FloatTensor)
    target_block= target_block.type(torch.FloatTensor)
    
    inputs_block = np.squeeze(inputs_block)
    target_block = np.squeeze(target_block)
    
    outputs_block = torch.matmul(adj_block, inputs_block)
    prediction_block = torch.matmul(adj_block, target_block)
    
      
    prediction_image = retransform_unblockify_target(prediction_block, nn_ixs, batch_size, n_feat=9)
    # unblockify and to_ image transformation
   
    # plot
    axs[1].imshow(prediction_image[0,0,:,:])

    # test edge indices
    
    edge_tuple = A.nonzero()
    edge_array = np.stack(edge_tuple)
    edge_array = edge_array.astype(np.int64)
    edge_index = torch.LongTensor(edge_array)
    
   
    model = GCNConv(1,1)

    prediction_block2 = model.forward(target_block, edge_index)
        
    # unblockify and to_ image transformation
    prediction_image2 = retransform_unblockify_target(prediction_block2, nn_ixs, batch_size, n_feat=9)
    # plot
    axs[2].imshow(prediction_image2[0,0,:,:])
    
    

    
    
# draw a subgraph of G
    plt.figure()
    res = [(i,j) for i in np.arange(100,200) for j in np.arange(100,200)]
    k = G.subgraph(res)  
    Gpos = {}
    kpos = {}
    
    for node_name in G.nodes:
        Gpos[node_name] = node_name
    for node_name in k.nodes:
        kpos[node_name] = node_name
    
    
    
    nx.draw_networkx(k, pos=kpos, with_labels=False, node_size=50)

    
    
    
    
    
    
    
    