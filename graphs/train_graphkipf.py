# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:53:55 2019

@author: henry
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import torch
    from torch.optim.lr_scheduler import StepLR
    from datetime import datetime
    import numpy as np
    import time
    import pickle
    import sys, os
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'ye'))
    sys.path.append(os.path.join(os.getcwd(),'henry', 'graphs'))
    from videoloader import trafic4cast_dataset
    from config_gcn import config
    from visual_TB import Visualizer
    from earlystopping import EarlyStopping
    import json

    from graph_models import KipfNet
    from datetime import datetime

    from torch_geometric.data import Data
    import matplotlib.pyplot as plt

    from graph_utils import csr_to_torch, coo_to_torch, image_to_vector, \
     vector_to_image, create_adj_matrix, transform_shape_train, \
     transform_shape_test, blockify_A, blockify_data, unblockify_target, \
     retransform_unblockify_target


def trainNet(model, train_loader, val_loader, val_loader_ttimes, device, adj, nn_ixs, edge_index):
        
    # Print all of the hyper parameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", config['dataloader']['batch_size'])
    print("epochs=", config['num_epochs'])
    print("learning_rate=", config['optimizer']['lr'])
    print("network_depth=", config['model']['depth'])
    print("=" * 30)
    # define the optimizer & learning rate 
    #optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])#
    optim = torch.optim.SGD(model.parameters(), **config['optimizer'])
    scheduler = StepLR(optim, step_size=config['lr_step_size'], gamma=config['lr_gamma'])
    
    log_dir = 'runs/graphs/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + \
            '-'.join(config['dataset']['cities'])
    writer = Visualizer(log_dir)

    # dump config file  
    with open(os.path.join(log_dir,'config.json'), 'w') as fp:
        json.dump(config, fp)

    # Time for printing
    training_start_time = time.time()
    globaliter = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config['patience'], verbose=True)
#    adj = adj.to(device)
   
    # Loop for n_epochs
    for epoch_idx, epoch in enumerate(range(config['num_epochs'])):
        
        writer.write_lr(optim, globaliter)

        # train for one epoch
        globaliter = train(model, train_loader, optim, device, writer, epoch, globaliter, adj, nn_ixs, edge_index)

        # At the end of the epoch, do a pass on the validation set
        val_loss = validate(model, val_loader, device, writer, globaliter, adj, nn_ixs, edge_index)
        val_loss_testtimes = validate(model, val_loader_ttimes, device, writer,
                                globaliter, adj, nn_ixs, edge_index, if_testtimes=True)
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss_testtimes, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        if config['debug'] and epoch_idx >= 0:
            break

        scheduler.step()
        

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

#    model.load_state_dict(torch.load('runs/checkpoint.pt'))

    # remember to close
    writer.close()

def get_graphdata_obj(inputs, edge_index, y):
    graphdata = Data(x=inputs, edge_index=edge_index, y=y)
    
    return graphdata
        

def train(model, train_loader, optim, device, writer, epoch, globaliter, adj, nn_ixs, edge_index):
    batch_size = config['dataloader']['batch_size']
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)
 
    # define start time
    start_time = time.time()
    

    for i, data in enumerate(train_loader, 0):
        
#        print("validation batch {}/{}".format(i,len(train_loader)))
        inputs, Y, features = data
        inputs = inputs/255
        globaliter += 1
        effective_batch_size = inputs.shape[0]
        # padd the input data with 0 to ensure same size after upscaling by the network
        # feature_vec = feature_vec['feature_vector'].float().to(device)

        inputs = image_to_vector(inputs,nn_ixs)
        Y = image_to_vector(Y,nn_ixs)
        
        inputs, Y = blockify_data(inputs, Y, batch_size)
        Y = Y.float()
               
        # the Y remains the same dimension
        inputs = inputs.float().to(device) 
        Y = Y.float().to(device) 


        graphdata = get_graphdata_obj(inputs, edge_index, Y)
        # Set the parameter gradients to zero
        optim.zero_grad()

        # Forward pass, backward pass, optimize
        prediction = model(graphdata)


        
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
            running_loss_norm = running_loss / config['print_every_step']
            writer.write_loss_train(running_loss_norm, globaliter)

            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
            

    return globaliter


def validate(model, val_loader, device, writer, globaliter, adj, nn_ixs, edge_index, if_testtimes=False):
    batch_size = config['dataloader']['batch_size']
    total_val_loss = 0
    if if_testtimes:
        prefix = 'testtimes'
    else:
        prefix = ''

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):

            val_inputs_image, val_y1, feature_vec = data
            val_inputs_image = val_inputs_image/255
            # feature_vec = feature_vec['feature_vector'].float().to(device)

            val_inputs_image = image_to_vector(val_inputs_image,nn_ixs)
            # tested            
            val_y2 = image_to_vector(val_y1,nn_ixs)
            
            val_inputs_block, val_y_block = blockify_data(val_inputs_image, val_y2, batch_size)
                   
            # the Y remains the same dimension
            val_inputs_block = val_inputs_block.float().to(device) 
            val_y_block = val_y_block.float().to(device) 
    
            val_graphdata = get_graphdata_obj(val_inputs_block, edge_index, val_y_block)
            # Set the parameter gradients to zero
            # Forward pass, backward pass, optimize
            prediction_block = model(val_graphdata)
            
            # crop the output for comparing with true Y
            prediction_block = torch.clamp(prediction_block, 0, 255, out=None)
            val_loss_size = torch.nn.functional.mse_loss(prediction_block, val_graphdata.y)
            total_val_loss += val_loss_size.item()


            
            # each epoch select one prediction set (one batch) to visualize
            if i  == 0:
                val_output = retransform_unblockify_target(prediction_block.cpu().detach().numpy(),
                                                           nn_ixs=nn_ixs,
                                                           batch_size=batch_size,
                                                           dataset=val_loader.dataset)
                val_output = torch.from_numpy(val_output)
                
                val_y = retransform_unblockify_target(val_y_block.cpu().detach().numpy(),
                                                      nn_ixs=nn_ixs,
                                                      batch_size=batch_size,
                                                      dataset=val_loader.dataset )
                val_y = torch.from_numpy(val_y)
                
                

                # we have to unblockify the data before we can do the backtransformation from vector to image
                
                writer.write_image(val_output, globaliter,if_predict=True, if_testtimes=if_testtimes)
                writer.write_image(val_y, globaliter,if_predict=False, if_testtimes=if_testtimes)

    val_loss = total_val_loss / len(val_loader)
    print("Validation loss = {:.2f}".format(val_loss))
    # write the validation loss to tensorboard
    writer.write_loss_validation(val_loss, globaliter, if_testtimes=if_testtimes)
    return val_loss


if __name__ == "__main__":
    
    
    dataset_train = trafic4cast_dataset(split_type='training', **config['dataset'],
                                        reduce=True, filter_test_times=True)
    dataset_val = trafic4cast_dataset(split_type='validation', **config['dataset'], 
                                        reduce=True, filter_test_times=True)

    dataset_val_ttimes = trafic4cast_dataset(split_type='validation', **config['dataset'], 
                                        reduce=True, filter_test_times=True)

    
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True,
                                               **config['dataloader'])
    val_loader = torch.utils.data.DataLoader(dataset_val, shuffle=True,
                                             **config['dataloader'])
    val_loader_ttimes = torch.utils.data.DataLoader(dataset_val_ttimes, shuffle=True,
                                             **config['dataloader'])

    device = torch.device(config['device_num'])

  
    # define the network structure -- partial UNet
    n_features = 36
    
    
    model = KipfNet(n_features=n_features, n_classes=9).to(device)
#    model = GraphUNet(in_channels=36,hidden_channels=4,out_channels=9,depth=1).to(device)
    
    adj, nn_ixs, G, mask = create_adj_matrix(city=config['dataset']['cities'][0],
                                mask_threshold=config['mask_threshold'])
    
    
    
    if config['dataloader']['batch_size'] > 1:
        adj = blockify_A(adj, config['dataloader']['batch_size'])
        
    
    
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
    trainNet(model, train_loader, val_loader, val_loader_ttimes, device, adj, nn_ixs, edge_index)

    # test

