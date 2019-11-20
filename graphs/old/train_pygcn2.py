# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:53:55 2019

@author: henry
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import pickle
import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'ye'))


from videoloader import trafic4cast_dataset, test_dataloader
from config_gcn import config
from visual_TB import Visualizer
from earlystopping import EarlyStopping

import networkx as nx
from scipy import sparse
from pygcn.models import GCN, GCN2

import scipy
from scipy.sparse import csr_matrix, coo_matrix
from scipy import sparse
import matplotlib.pyplot as plt

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

def create_adj_matrix(city='Berlin', mask_threshold=0):
    # create matrix
    mask_dict = pickle.load(open( os.path.join('utils', 'masks.dict'), "rb" ) )

    if mask_threshold > 0:
        sum_city = mask_dict[city]['sum']
        mask = sum_city > mask_threshold

    else:
        mask = mask_dict[city]['mask']


    nn_ixs = np.where(mask)

    # make matrix
    mask_inv = ~mask
    to_delete = np.where(mask_inv)
    to_delete = list(zip(to_delete[0],to_delete[1]))

    G = nx.grid_graph(dim=[436, 495])
    G.remove_nodes_from(to_delete)
    A = nx.to_scipy_sparse_matrix(G)

    return A, nn_ixs

def transform_shape_train(data):
    data = data.reshape((batch_size,36,-1))
    return np.squeeze(data)

def transform_shape_test(data):
    data =  data.reshape((batch_size,9,-1))
    return np.squeeze(data)
    
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


def trainNet(model, train_loader, val_loader, device, adj, nn_ixs):
        
    # Print all of the hyper parameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", config['batch_size'])
    print("epochs=", config['num_epochs'])
    print("learning_rate=", config['learning_rate'])
    print("network_depth=", config['depth'])
    print("=" * 30)
    # define the optimizer & learning rate 
    #optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])#
    optim = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, nesterov=True)
    scheduler = StepLR(optim, step_size=config['lr_step_size'], gamma=config['lr_gamma'])
    writer = Visualizer(model_name='PYGCN')

    # Time for printing
    training_start_time = time.time()
    globaliter = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
    adj = adj.to(device)
   
    # Loop for n_epochs
    for epoch in range(config['num_epochs']):
        writer.write_lr(optim, globaliter)

        # train for one epoch
        globaliter = train(model, train_loader, optim, device, writer, epoch, globaliter, adj, nn_ixs)

        # At the end of the epoch, do a pass on the validation set
        val_loss = validate(model, val_loader, device, writer, globaliter, adj, nn_ixs)
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    model.load_state_dict(torch.load('runs/checkpoint.pt'))

    # remember to close
    writer.close()

def train(model, train_loader, optim, device, writer, epoch, globaliter, adj, nn_ixs):
    batch_size = config['batch_size']
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)
 
    # define start time
    start_time = time.time()
    

    for i, data in enumerate(train_loader, 0):
        break
        inputs, Y = data
        globaliter += 1
        # padd the input data with 0 to ensure same size after upscaling by the network

        inputs = image_to_vector(inputs/255,nn_ixs)
        Y = image_to_vector(Y,nn_ixs)
        inputs, Y = blockify_data(inputs, Y, batch_size)
        
        
              
        
        
        
        
        # the Y remains the same dimension
        inputs = inputs.float().to(device) 
        Y = Y.float().to(device) 

        # Set the parameter gradients to zero
        optim.zero_grad()

        # Forward pass, backward pass, optimize
        prediction = model(inputs, adj)



        
        # crop the output for comparing with true Y
        loss_size = torch.nn.functional.mse_loss(prediction, Y)

        loss_size.backward()
        optim.step()

        # Print statistics
        running_loss += loss_size.item()
        if (i+1) % config['print_every_step'] == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, int(100 * (i+1) / n_batches), running_loss / config['print_every_step'], time.time() - start_time))

            # write the train loss to tensorboard
            writer.write_loss_train(running_loss / config['print_every_step'], globaliter)

            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()

    return globaliter


def validate(model, val_loader, device, writer, globaliter, adj, nn_ixs):
    batch_size = config['batch_size'] 
    total_val_loss = 0
    # change to validation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            val_inputs, val_y = data

            val_inputs = image_to_vector(val_inputs/255, nn_ixs)
            val_y = image_to_vector(val_y, nn_ixs)
            val_inputs, val_y = blockify_data(val_inputs, val_y, batch_size)
            
                        
            val_inputs_image, val_y = data
            fig, ax = plt.subplots(1,2)
            val_inputs_vec = image_to_vector(val_inputs_image, nn_ixs)[0,...].numpy()
#            val_inputs_vec_ones = np.ones(val_inputs_vec.shape)
#            val_inputs_vec = val_inputs_vec_ones
            
            val_inputs_vec = np.squeeze(val_inputs_vec) * adj
            val_inputs_image2 = vector_to_image(val_inputs_vec, nn_ixs, batch_size=10)
            ax[1].imshow(val_inputs_image2[0,0,:,:])
            
            ax[0].imshow(val_inputs_image[0,0,:,:])
            
            
            
            # the Y remains the same dimension
            val_inputs = val_inputs.float().to(device) 
            val_y = val_y.float().to(device)

            val_output = model(val_inputs, adj)
            
            # crop the output for comparing with true Y
            val_loss_size = torch.nn.functional.mse_loss(val_output, val_y)
            total_val_loss += val_loss_size.item()

            # each epoch select one prediction set (one batch) to visualize
            if (i+1) % int(len(val_loader)/2+1) == 0:
                pass
                # we have to unblockify the data before we can do the backtransformation from vector to image
                #val_output = vector_to_image(val_output.cpu(), nn_ixs, n_feat=36, batch_size=batch_size)
                #val_output = val_output.to(device)
                #writer.write_image(val_output, globaliter)♦

    val_loss = total_val_loss / len(val_loader)
    print("Validation loss = {:.2f}".format(val_loss))
    # write the validation loss to tensorboard
    writer.write_loss_validation(val_loss, globaliter)
    return val_loss


if __name__ == "__main__":
    
    dataset_train = trafic4cast_dataset(source_root=config['source_dir'],
                                        target_root=config['target_dir'],
                                        split_type='training',
                                        cities=['Berlin'], reduce=True)
    dataset_val = trafic4cast_dataset(source_root=config['source_dir'],
                                      target_root=config['target_dir'],
                                      split_type='validation',
                                      cities=['Berlin'], reduce=True)
#    dataset_test = trafic4cast_dataset(source_root=config['source_dir'],
#                                       target_root=config['target_dir'],
#                                       split_type='test',
#                                       cities=['Berlin'], reduce=True,
#                                       do_subsample=(0,200))
    
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=config['num_workers'],
                                               drop_last=config['drop_last'])
    val_loader = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=config['batch_size'], 
                                             shuffle=True,
                                             num_workers=config['num_workers'],
                                             drop_last=config['drop_last'])
#    test_loader = torch.utils.data.DataLoader(dataset_test,
#                                              batch_size=16, shuffle=True)
    
    # test_dataloader(train_loader)
    # test_dataloader(val_loader)
    # test_dataloader(test_loader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:0')

    # define the network structure -- UNet
    # the output size is not always equal to your input size !!!
    # model = UNet(in_channels=config['in_channels'],
    #              n_classes=config['n_classes'],
    #              depth=config['depth'],
    #              padding=config['if_padding'],
    #              up_mode=config['up_mode'],
    #              batch_norm=True).to(device)
    
    # define the network structure -- partial UNet
    n_features = 36
    model = GCN2(nfeat=n_features,
            nhid1=500, nhid2=100,
            nclass=9,
            dropout=0.5).to(device)

    # load the mask

    adj, nn_ixs = create_adj_matrix(city='Berlin', mask_threshold=50000)
    if config['batch_size'] > 1:
        adj = blockify_A(adj, config['batch_size'])
    else:
        adj = csr_to_torch(adj)
    adj = adj.to(device)
    # define the network structure -- convLSTM
    # model = ConvLSTM(input_size=(192, 192),
    #                 input_dim=3,
    #                 hidden_dim=[64, 128, 64, 3],
    #                 kernel_size=(3, 3),
    #                 num_layers=3,
    #                 batch_first=True,
    #                 bias=True,
    #                 return_all_layers=False).to(device)


    # the architecture overview
    # from torchsummary import summary
    # summary(model, input_size=(1, 12, 3, 192, 192))

    # # need to add the mask parameter when training the partial Unet model
    trainNet(model, train_loader, val_loader, device, adj, nn_ixs)

    # test

