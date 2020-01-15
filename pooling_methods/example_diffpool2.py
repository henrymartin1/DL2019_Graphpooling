import os
import os.path as osp
import sys
import warnings
from datetime import datetime
import json

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from utils.visual_TB import Visualizer

from utils.earlystopping import EarlyStopping
from networkx import from_numpy_matrix

from models.gunet_diffpool2 import diff_pool_net2
from pooling_methods.example_diffpool2_config import config

torch.manual_seed(0)


def train():
    model.train()
    optimizer.zero_grad()

    output_dict = model(data, data.train_mask)

    training_output = output_dict['prediction'][data.train_mask]
    loss_lambda = config['loss_lambda']

    loss = F.nll_loss(training_output, data.y[data.train_mask]) + (loss_lambda * output_dict['edge_loss'])
    loss.backward(retain_graph=True)  # this does backpropagation
    optimizer.step()

    if epoch % 10 == 0:
        adj = output_dict['adj1']
        writer.write_hist(adj, epoch)
        writer.write_adj_matrix(adj, epoch)
    return output_dict


def test():
    model.eval()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        output_dict = model(data, mask)
        pred = output_dict['prediction']
        pred = pred[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, output_dict['s01']


# define data
dataset_name = 'Cora'
path = osp.join(os.getcwd(), '..', 'data', dataset_name)
dataset = Planetoid(path, dataset_name, T.NormalizeFeatures())
data = dataset[0]
data.batch = None
data.adj = to_dense_adj(data.edge_index)

# define logging
log_dir = os.path.join('..', 'runs', config['modelname'] + '_'
                       + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                       + '_' + config['model_log_info'])
writer = Visualizer(log_dir)
with open(os.path.join(log_dir, 'config.json'), 'w') as fp:
    json.dump(config, fp)

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model = diff_pool_net2(dataset, **config['model']).to(device)
data = data.to(device)
lr = config['optimizer']['lr']
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# initialize the early_stopping object
early_stopping = EarlyStopping(log_dir, patience=config['patience'], verbose=False)

best_val_acc = test_acc = 0
for epoch in range(1, config['epochs']):
    output_dict = train()

    accs, s = test()
    train_acc, val_acc, tmp_test_acc = accs

    writer.write_lr(optimizer, epoch)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

    writer.write_acc_train(train_acc, epoch)
    writer.write_acc_validation(val_acc, epoch)
    writer.write_acc_test(test_acc, epoch)
    if epoch % 5 == 0:
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    early_stopping(val_acc, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

writer.close()
