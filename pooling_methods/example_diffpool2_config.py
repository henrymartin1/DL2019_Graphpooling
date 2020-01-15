import time
import os
import random

config = {}

# model statistics 
config['model'] = {}
config['model']['num_clusters1'] = 500
config['model']['num_clusters2'] = 250
config['model']['n_hidden'] = 128
config['model']['wf'] = 7 # should be log_2 (n_features) would be 11 for cora


# model name
config['modelname'] = 'diff_pool_net2' 

# log name
config['model_log_info'] =  str(config['model']['num_clusters1'])        


# optimizer
config['optimizer'] = {}
config['optimizer']['lr'] = 0.01

# loss
config['loss'] = 'multi'
config['loss_lambda'] = 0.0001

# epochs
config['epochs'] = 500

# early stopping
config['patience'] = 100