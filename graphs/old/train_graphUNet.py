# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:53:55 2019

@author: henry
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
os.chdir(r"E:\Programming\traffic4cast")
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'ye'))
sys.path.append(os.path.join(os.getcwd(),'henry', 'pygcn'))

from videoloader import trafic4cast_dataset, test_dataloader
from config_gcn import config
from visual_TB import Visualizer
from earlystopping import EarlyStopping
from datetime import datetime

import networkx as nx
from scipy import sparse
from pygcn.models import GCN, GCN2

import scipy
from scipy.sparse import csr_matrix, coo_matrix
from scipy import sparse
from torch_geometric.data import Data

from graph_utils import csr_to_torch, coo_to_torch, image_to_vector,
 vector_to_image, create_adj_matrix, transform_shape_train,
 transform_shape_test, blockify_data, unblockify_target,
 retransform_unblockify_target


def trainNet(model, train_loader, val_loader, device, adj, nn_ixs, edge_index):
        
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
    
    log_dir = 'runs/GraphUNet/' + str(int(datetime.now().timestamp()))
    writer = Visualizer(log_dir)

    # Time for printing
    training_start_time = time.time()
    globaliter = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config['patience'], verbose=True)
#    adj = adj.to(device)
   
    # Loop for n_epochs
    for epoch in range(config['num_epochs']):
        for globaliter in range(5):
            writer.write_lr(optim, globaliter)

        # train for one epoch
        globaliter = train(model, train_loader, optim, device, writer, epoch, globaliter, adj, nn_ixs, edge_index)

        # At the end of the epoch, do a pass on the validation set
        val_loss = validate(model, val_loader, device, writer, globaliter, adj, nn_ixs, edge_index)
        
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
#    writer.close()

def get_graphdata_obj(inputs, edge_index, y):
    graphdata = Data(x=inputs, edge_index=edge_index, y=y)
    
    return graphdata
        

def train(model, train_loader, optim, device, writer, epoch, globaliter, adj, nn_ixs, edge_index):
    batch_size = config['batch_size']
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)
 
    # define start time
    start_time = time.time()
    

    for i, data in enumerate(train_loader, 0):
#        print("validation batch {}/{}".format(i,len(train_loader)))
        inputs, Y = data
        globaliter += 1
        # padd the input data with 0 to ensure same size after upscaling by the network
        break


        inputs = image_to_vector(inputs,nn_ixs)
        Y = image_to_vector(Y,nn_ixs)
        
        
        inputs, Y = blockify_data(inputs, Y, batch_size)
        Y = Y.float()
               
        # the Y remains the same dimension
        inputs = inputs.float().to(device) 
        Y = Y.float().to(device) 

        # Set the parameter gradients to zero
        optim.zero_grad()

        # Forward pass, backward pass, optimize
        prediction = model(inputs, edge_index)

#        print(prediction.shape)

        
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
            print(running_loss, globaliter)
            writer.write_loss_train(running_loss / config['print_every_step'], globaliter)

            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
            

    return globaliter


def validate(model, val_loader, device, writer, globaliter, adj, nn_ixs, edge_index):
    batch_size = config['batch_size'] 
    total_val_loss = 0
    # change to validation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            print("validation batch {}/{}".format(i,len(val_loader)))
            val_inputs, val_y = data

            val_inputs = image_to_vector(val_inputs/255, nn_ixs)
            val_y = image_to_vector(val_y, nn_ixs)
            val_inputs, val_y = blockify_data(val_inputs, val_y, batch_size)
            
            # the Y remains the same dimension
            val_inputs = val_inputs.float().to(device) 
            val_y = val_y.float().to(device)

            val_output = model(val_inputs, edge_index)
            
            # crop the output for comparing with true Y
            val_loss_size = torch.nn.functional.mse_loss(val_output, val_y)
            total_val_loss += val_loss_size.item()


            
            # each epoch select one prediction set (one batch) to visualize
            if i  == 0:
                val_output = retransform_unblockify_target(val_output, batch_size)
                val_y = retransform_unblockify_target(val_y, batch_size)
                # we have to unblockify the data before we can do the backtransformation from vector to image
                
                writer.write_image(val_output.cpu(), globaliter,if_predict=True)
                writer.write_image(val_y.cpu(), globaliter,if_predict=False)
                

    val_loss = total_val_loss / len(val_loader)
    print("Validation loss = {:.2f}".format(val_loss))
    # write the validation loss to tensorboard
    writer.write_loss_validation(val_loss, globaliter)
    return val_loss


if __name__ == "__main__":
    
    dataset_train = trafic4cast_dataset(source_root=config['source_dir'],
                                        target_root=config['target_dir'],
                                        split_type='validation',
                                        cities=config['cities'], reduce=True)
    dataset_val = trafic4cast_dataset(source_root=config['source_dir'],
                                      target_root=config['target_dir'],
                                      split_type='validation',
                                      cities=config['cities'], reduce=True)
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
    
    
    
    model = GraphUNet(in_channels=36,hidden_channels=4,out_channels=9,depth=1).to(device)
    
#    model = GCN2(nfeat=n_features,
#            nhid1=500, nhid2=100,
#            nclass=9,
#            dropout=0.5).to(device)

    # load the mask

    adj, nn_ixs, G = create_adj_matrix(city='Berlin', mask_threshold=50000)
    if config['batch_size'] > 1:
        adj = blockify_A(adj, config['batch_size'])
    
    edge_tuple = adj.nonzero()
    edge_array = np.stack(edge_tuple)
    edge_array = edge_array.astype(np.int64)
    edge_index = torch.LongTensor(edge_array)
    edge_index = edge_index.to(device)
    
#    adj = adj.to(device)
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
#    graphdata_raw = torch_geometric.utils.from_networkx(G)
    # # need to add the mask parameter when training the partial Unet model
    trainNet(model, train_loader, val_loader, device, adj, nn_ixs, edge_index)

    # test

