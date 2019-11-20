import time
import os
import random


config = dict()


# data set configuration
config['dataset'] = {}

# Where raw data are stored.

config['dataset']['source_root'] = 'data_raw'
config['dataset']['target_root'] = 'data'

config['dataset']['return_features'] = True
config['dataset']['cities'] = ['Berlin']
config['dataset']['compression'] = 'lzf'
##################################################################


config['device_num'] = 0
config['debug']= False

# model statistics 
config['model'] = {}

config['add_coords'] = True
if config['add_coords']:
    config['model']['in_channels'] = 38
else:
    config['model']['in_channels'] = 36


config['model']['depth'] = 1

# Kipfnet
config['model']['KIPF'] = {}
config['model']['KIPF']['nh1'] = 64
config['model']['KIPF']['K'] = 7

# Kipfnet2
config['model']['KIPF2'] = {}
config['model']['KIPF2']['nh1'] = 56
config['model']['KIPF2']['nh2'] = 24
config['model']['KIPF2']['K'] = [12,8,4]

# KipfNet_res
config['model']['KipfNet_res'] = {}
config['model']['KipfNet_res']['nh1'] = 4
config['model']['KipfNet_res']['nh2'] = 4
config['model']['KipfNet_res']['K'] = 4


# Kipfnet3
config['model']['KIPF3'] = {}
config['model']['KIPF3']['nh1'] = 48
config['model']['KIPF3']['nh2'] = 48
config['model']['KIPF3']['nh3'] = 48
config['model']['KIPF3']['K'] = [8,8,8,8]

# ARMANet
config['model']['ARMA'] = {}

# ARMACheb
config['model']['ARMACheb'] = {}
config['model']['ARMACheb']['nh1'] = 36
config['model']['ARMACheb']['K'] = 8

# GUnet
config['model']['GUNET'] = {}
config['model']['GUNET']['nh1'] = 100
config['model']['GUNET']['depth'] = 1
config['model']['GUNET']['pool_ratios'] = [0.5]

config['cont_model_path'] = None  # Use this to continue training a previously started model.

# data loader configuration
config['dataloader'] = {}
config['dataloader']['drop_last'] = True

config['dataloader']['num_workers'] = 4
config['dataloader']['batch_size'] = 4

#Graph creation
# I don't have any idea of an optimal value for the mask threshold. In the past I often used 50'000
# I noticed that 200'000 is probably too much. 
config['mask_threshold'] = 0


# optimizer
config['optimizer_name'] = 'SGD'
config['optimizer'] = {}
config['optimizer']['lr'] = 0.01
# config['optimizer']['weight_decay'] =0.00002
config['optimizer']['momentum'] = 0.8
config['optimizer']['nesterov'] = True

# lr schedule
config['lr_step_size'] = 4
config['lr_gamma'] = 0.1

# early stopping
config['patience'] = 10
config['num_epochs'] = 50
config['print_every_step'] = 1


config['model_name'] = 'KIPF2'