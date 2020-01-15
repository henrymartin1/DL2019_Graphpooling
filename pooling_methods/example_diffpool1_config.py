import os
import random
import time

config = {}

# model statistics 
config['model'] = {}
config['model']['num_clusters1'] = 600
config['model']['num_clusters2'] = 50
config['model']['wf'] = 5 # should be log_2 (n_features) would be 11 for cora


# model name
config['modelname'] = 'diff_pool_net1' 

# log name
config['model_log_info'] =  str(config['model']['num_clusters1'])        


# optimizer
config['optimizer'] = {}
config['optimizer']['lr'] = 0.001

# loss
config['loss'] = 'multi'
config['loss_lambda'] = 0.01

# epochs
config['epochs'] = 500

# early stopping
config['patience'] = 100