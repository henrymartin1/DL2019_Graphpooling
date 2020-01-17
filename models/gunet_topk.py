
import torch
from models.graph_unet_model import GraphUNet
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F

class gunet_topk(torch.nn.Module):
    def __init__(self, dataset, depth, pool_ratios, dropout_rate, p_dropout_adj=0.2, p_dropout_nodes=0.92, hidden_channels=32):
        super(gunet_topk, self).__init__()
        self.unet = GraphUNet(in_channels=dataset.num_features, hidden_channels=hidden_channels, out_channels=dataset.num_classes, \
         depth=depth, pool_ratios=pool_ratios, dropout_rate=dropout_rate)
        self.initial_dropout_adj = p_dropout_adj
        self.initial_dropout_nodes = p_dropout_nodes

    def forward(self, data):
        edge_index, _ = dropout_adj(
            data.edge_index, p=self.initial_dropout_adj, force_undirected=True,
            num_nodes=data.num_nodes, training=self.training)
        x = F.dropout(data.x, p=self.initial_dropout_nodes, training=self.training)

        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)
