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
pool_ratios2 = pool_ratios #[0.5, 0.5]
# model parameters
config['model'] = {}


# sagpool_05_d
config['model']['topk_05_d'] = {}
config['model']['topk_05_d']['parameters'] = {}
config['model']['topk_05_d']['model_constructor'] = gunet_topk
config['model']['topk_05_d']['parameters']['depth'] = 2
config['model']['topk_05_d']['parameters']['pool_ratios'] = [0.5, 0.5]
config['model']['topk_05_d']['parameters']['dropout_rate'] = 0.
config['model']['topk_05_d']['parameters']['hidden_channels'] = 32
config['model']['topk_05_d']['parameters']['p_adj'] = 0.2
config['model']['topk_05_d']['parameters']['p_nodes'] = 0.92

# sagpool_05_no_d
config['model']['topk_05_no_d'] = {}
config['model']['topk_05_no_d']['parameters'] = {}
config['model']['topk_05_no_d']['model_constructor'] = gunet_topk
config['model']['topk_05_no_d']['parameters']['depth'] = 2
config['model']['topk_05_no_d']['parameters']['pool_ratios'] = [0.5, 0.5]
config['model']['topk_05_no_d']['parameters']['dropout_rate'] = 0.
config['model']['topk_05_no_d']['parameters']['hidden_channels'] = 32
config['model']['topk_05_no_d']['parameters']['p_adj'] = 0.
config['model']['topk_05_no_d']['parameters']['p_nodes'] = 0.

# sagpool_035_d
config['model']['topk_035_d'] = {}
config['model']['topk_035_d']['parameters'] = {}
config['model']['topk_035_d']['model_constructor'] = gunet_topk
config['model']['topk_035_d']['parameters']['depth'] = 2
config['model']['topk_035_d']['parameters']['pool_ratios'] = [0.35, 0.35]
config['model']['topk_035_d']['parameters']['dropout_rate'] = 0.
config['model']['topk_035_d']['parameters']['hidden_channels'] = 32
config['model']['topk_035_d']['parameters']['p_adj'] = 0.2
config['model']['topk_035_d']['parameters']['p_nodes'] = 0.92

# sagpool_035_no_d
config['model']['topk_035_no_d'] = {}
config['model']['topk_035_no_d']['parameters'] = {}
config['model']['topk_035_no_d']['model_constructor'] = gunet_topk
config['model']['topk_035_no_d']['parameters']['depth'] = 2
config['model']['topk_035_no_d']['parameters']['pool_ratios'] = [0.35, 0.35]
config['model']['topk_035_no_d']['parameters']['dropout_rate'] = 0.
config['model']['topk_035_no_d']['parameters']['hidden_channels'] = 32
config['model']['topk_035_no_d']['parameters']['p_adj'] = 0.
config['model']['topk_035_no_d']['parameters']['p_nodes'] = 0.


# loss
config['loss_lambda'] = 0

# optimizer
config['optimizer'] = {}
config['optimizer']['lr'] = 0.01



# epochs and repeats
config['epochs'] = 201
config['tracked_epochs'] = [0, 1, 10, 25, 50, 100, 150, 200, 250, 300]
config['n_repeats'] = 5

# output
config['output_file'] = os.path.join('experiments', 'hyperparameter', 'topk_pooling_dropout_results.pkl')