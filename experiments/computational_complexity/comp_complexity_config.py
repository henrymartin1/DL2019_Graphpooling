import time
import os
import random

from models.gunet_diffpool1 import diff_pool_net1
from models.gunet_diffpool2 import diff_pool_net2
from models.gunet_topk import gunet_topk
from models.gunet_sagpool import sagpool

config = {}




# model parameters
config['model'] = {}

# diffpool 1
config['model']['diff_pool_net1'] = {}
config['model']['diff_pool_net1']['model_constructor'] = diff_pool_net1
config['model']['diff_pool_net1']['parameters'] = {}
config['model']['diff_pool_net1']['parameters']['num_clusters1'] = 600
config['model']['diff_pool_net1']['parameters']['num_clusters2'] = 50
config['model']['diff_pool_net1']['parameters']['wf'] = 5 # should be log_2 (n_features) would be 11 for cora

# diffpool 2
config['model']['diff_pool_net2'] = {}
config['model']['diff_pool_net2']['parameters'] = {}
config['model']['diff_pool_net2']['model_constructor'] = diff_pool_net2
config['model']['diff_pool_net2']['parameters']['num_clusters1'] = 500
config['model']['diff_pool_net2']['parameters']['num_clusters2'] = 250
config['model']['diff_pool_net2']['parameters']['n_hidden'] = 128
config['model']['diff_pool_net2']['parameters']['wf'] = 7

# topk
config['model']['topk'] = {}
config['model']['topk']['parameters'] = {}
config['model']['topk']['model_constructor'] = gunet_topk
config['model']['topk']['parameters']['depth'] = 2
config['model']['topk']['parameters']['pool_ratios'] = [0.5, 0.5]
config['model']['topk']['parameters']['dropout_rate'] = 0.
config['model']['topk']['parameters']['hidden_channels'] = 32

# sagpool
config['model']['sagpool'] = {}
config['model']['sagpool']['parameters'] = {}
config['model']['sagpool']['model_constructor'] = gunet_topk
config['model']['sagpool']['parameters']['depth'] = 2
config['model']['sagpool']['parameters']['pool_ratios'] = [0.5, 0.5]
config['model']['sagpool']['parameters']['dropout_rate'] = 0.
config['model']['sagpool']['parameters']['hidden_channels'] = 32



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