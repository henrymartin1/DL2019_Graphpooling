import time
import os
import random

config = {}

# model statistics 
config['model'] = {}
config['model']['num_clusters1'] = 100
config['model']['num_clusters2'] = 50
config['model']['nh'] = 16
config['model']['wf'] = 2


# model name
config['modelname'] = 'diff_pool_net' 

# log name
config['model_log_info'] =  str(config['model']['num_clusters1'])        


# optimizer
config['optimizer'] = {}
config['optimizer']['lr'] = 1e-2

# loss
config['loss'] = 'single'

# epochs
config['epochs'] = 300

# early stopping
config['patience'] = 75