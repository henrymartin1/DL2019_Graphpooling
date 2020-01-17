import time
import os
import random

from gunet_diffpool1 import diff_pool_net1
from gunet_diffpool2 import diff_pool_net2
from gunet_topk import gunet_topk
from gunet_sagpool import sagpool

config = {}


pool_ratios = [0.35, 0.35]

# model parameters
config['model'] = {}

# diffpool 1
config['model']['diff_pool_net1'] = {}
config['model']['diff_pool_net1']['model_constructor'] = diff_pool_net1
config['model']['diff_pool_net1']['parameters'] = {}
config['model']['diff_pool_net1']['parameters']['num_clusters1'] = int(2708*pool_ratios[0])
config['model']['diff_pool_net1']['parameters']['num_clusters2'] = int(2708*pool_ratios[0]*pool_ratios[1])
config['model']['diff_pool_net1']['parameters']['hidden_channels'] = 32

# diffpool 2
config['model']['diff_pool_net2'] = {}
config['model']['diff_pool_net2']['parameters'] = {}
config['model']['diff_pool_net2']['model_constructor'] = diff_pool_net2
config['model']['diff_pool_net2']['parameters']['num_clusters1'] = int(2708*pool_ratios[0])
config['model']['diff_pool_net2']['parameters']['num_clusters2'] = int(2708*pool_ratios[0]*pool_ratios[1])
config['model']['diff_pool_net2']['parameters']['n_hidden'] = 64
config['model']['diff_pool_net2']['parameters']['hidden_channels'] = 32

# sagpool
config['model']['sagpool'] = {}
config['model']['sagpool']['parameters'] = {}
config['model']['sagpool']['model_constructor'] = gunet_topk
config['model']['sagpool']['parameters']['depth'] = 2
config['model']['sagpool']['parameters']['pool_ratios'] = pool_ratios
config['model']['sagpool']['parameters']['dropout_rate'] = 0.
config['model']['sagpool']['parameters']['hidden_channels'] = 32

# topk
config['model']['topk'] = {}
config['model']['topk']['parameters'] = {}
config['model']['topk']['model_constructor'] = gunet_topk
config['model']['topk']['parameters']['depth'] = 2
config['model']['topk']['parameters']['pool_ratios'] = pool_ratios
config['model']['topk']['parameters']['dropout_rate'] = 0.
config['model']['topk']['parameters']['hidden_channels'] = 32



# loss
config['loss_lambda'] = 0

# optimizer
config['optimizer'] = {}
config['optimizer']['lr'] = 0.01

#the different fractions of the nodes to be used for training:
config['train_fracs'] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# epochs
config['epochs'] = 100

# early stopping
config['patience'] = 100
