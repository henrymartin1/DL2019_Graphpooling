"""
Created on Sun Dec 2

Based on the original implementation of the paper by the authors

STATUS: Currently adapting to the Cora dataset.

@author: glili
"""

import torch
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.data import DataLoader, DenseDataLoader
from torch_geometric import utils
from torch_geometric.utils.convert import to_networkx
import torch.nn.functional as F
from torch.utils.data import random_split
from networks import Net
from torch_geometric.nn import SAGPooling, GraphConv, TopKPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import argparse
import os
import os.path as osp

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter


class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x, edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

    class Net(torch.nn.Module):
        def __init__(self, args):
            super(Net, self).__init__()
            self.args = args
            self.num_features = args.num_features
            self.nhid = args.nhid
            self.num_classes = args.num_classes
            self.pooling_ratio = args.pooling_ratio
            self.dropout_ratio = args.dropout_ratio

            self.conv1 = GCNConv(self.num_features, self.nhid)
            self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
            self.conv2 = GCNConv(self.nhid, self.nhid)
            self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
            self.conv3 = GCNConv(self.nhid, self.nhid)
            self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
            self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
            self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def test(model, loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


min_loss = 1e10
patience = 0

for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc, val_loss = test(model, val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(), 'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break

model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc, test_loss = test(model, test_loader)
print("Test accuarcy:{}".fotmat(test_acc))