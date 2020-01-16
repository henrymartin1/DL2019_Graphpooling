"""
Based on the graph unet example from torch_geometric
@author: cbohn
"""

import os.path as osp

import sys
import random
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CoraFull
from graph_unet_model import GraphUNet
from torch_geometric.utils import dropout_adj

dataset = 'Cora'#'PubMed' #which dataset to use
patience = 25 #for early stopping
dropout_rate = 0. #the dropout rate in the conv layers
pool_ratios = [0.5, 0.5] #the pooling ratios for each pooling layer
depth = 2  #The depth of the U-Net, i.e., the number of encoding/decoding blocks
#Sets the the fraction of graph nodes to be used for training, the remaining nodes will be split roughly evenly between validation and test data: 
if len(sys.argv) < 2:
    print("Usage: python graph_unet_train.py <train_fraction> \n Setting fraction to default value of 0.2")
    training_fraction = 0.2
else:
    training_fraction = float(sys.argv[1])
print("Dataset: " + dataset)
print("Patience: " + str(patience))
print("Dropout rate: " + str(dropout_rate))
print("Pool ratios: " + str(pool_ratios))
print("Depth: " + str(depth))
print("Training fraction: " + str(training_fraction))


path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
dataset = Planetoid(path, dataset)
#dataset = CoraFull(path)
data = dataset[0]

print("Original training fraction: ")
print(data.train_mask.sum().item()/data.num_nodes)

print("Setting train, val, and test masks according to the fraction specified...")
for node in range(0, data.num_nodes-1):
    if random.random() < training_fraction: #with probability training_fraction include the node in the training set
        data.train_mask[node] = True
        data.val_mask[node] = False
        data.test_mask[node] = False
    else:
        data.train_mask[node] = False
        if random.random() < 0.5: #split the remaining nodes roughly evenly between validation and test sets
            data.val_mask[node] = True
            data.test_mask[node] = False
        else:
            data.val_mask[node] = False
            data.test_mask[node] = True
print("done")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unet = GraphUNet(dataset.num_features, 32, dataset.num_classes, depth=depth, pool_ratios=pool_ratios, dropout_rate=dropout_rate, sum_res=False)

    def forward(self):
        #No dropout happening here anymore, the dropout probabilities are both 0
        edge_index, _ = dropout_adj(
            data.edge_index, p=0, force_undirected=True,
            num_nodes=data.num_nodes, training=self.training)
        x = F.dropout(data.x, p=0, training=self.training)

        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, data = Net().to(device), data.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
num_epochs_since_last_val_improvements = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    num_epochs_since_last_val_improvements += 1
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'saved_model.pth')
        print("Saved model from epoch " + str(epoch))
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        num_epochs_since_last_val_improvements = 0
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
    if num_epochs_since_last_val_improvements >= patience:
        print("Early stopping")
        break
