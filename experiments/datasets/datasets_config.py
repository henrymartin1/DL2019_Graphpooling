import time
import os
import random

from models.gunet_diffpool1 import diff_pool_net1
from models.gunet_diffpool2 import diff_pool_net2
from models.gunet_topk import gunet_topk
from models.gunet_sagpool import sagpool
from models.gcn_plain import gcn_plain

config = {}


pool_ratios = [0.35, 0.35]

# model parameters
config['model'] = {}

config['model']['gcn_plain'] = {}
config['model']['gcn_plain']['model_constructor'] = gcn_plain
config['model']['gcn_plain']['parameters'] = {}

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
config['model']['diff_pool_net2']['parameters']['n_hidden'] = 16
config['model']['diff_pool_net2']['parameters']['hidden_channels'] = 32


# topk
config['model']['topk'] = {}
config['model']['topk']['parameters'] = {}
config['model']['topk']['model_constructor'] = gunet_topk
config['model']['topk']['parameters']['depth'] = 2
config['model']['topk']['parameters']['pool_ratios'] = pool_ratios
config['model']['topk']['parameters']['dropout_rate'] = 0.
config['model']['topk']['parameters']['hidden_channels'] = 32

# sagpool
config['model']['sagpool'] = {}
config['model']['sagpool']['parameters'] = {}
config['model']['sagpool']['model_constructor'] = gunet_topk
config['model']['sagpool']['parameters']['depth'] = 2
config['model']['sagpool']['parameters']['pool_ratios'] = pool_ratios
config['model']['sagpool']['parameters']['dropout_rate'] = 0.
config['model']['sagpool']['parameters']['hidden_channels'] = 32

# loss
config['loss_lambda'] = 0

# optimizer
config['optimizer'] = {}
config['optimizer']['lr'] = 0.01



# epochs and repeats
config['epochs'] = 201
config['tracked_epochs'] = [1, 10, 25, 50, 100, 150, 201]
config['n_repeats'] = 10
config['patience'] = 100

# output
config['output_file'] = os.path.join('experiments', 'datasets', 'datasets_results.pkl')
