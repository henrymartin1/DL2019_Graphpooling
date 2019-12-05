import time
import os
import random

config = {}

# model statistics 
config['model'] = {}
config['model']['num_clusters'] = 500
config['model']['nh0_1'] = 16
config['model']['nh1_1'] = 16
config['model']['n_out0'] = 10
config['model']['n_out1'] = 100

# model name
config['modelname'] = 'diff_pool_net' 

# log name
config['model_log_info'] =  str(config['model']['num_clusters'])        


# optimizer
config['optimizer'] = {}
config['optimizer']['lr'] = 1e-2

# loss
config['loss'] = 'single'

# epochs
config['epochs'] = 300

# early stopping
config['patience'] = 75