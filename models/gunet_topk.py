
import torch
from models.graph_unet_model import GraphUNet
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F

class gunet_topk(torch.nn.Module):
    def __init__(self, dataset, depth, pool_ratios, dropout_rate, hidden_channels=32):
        super(gunet_topk, self).__init__()
        self.unet = GraphUNet(in_channels=dataset.num_features, hidden_channels=hidden_channels, out_channels=dataset.num_classes, \
         depth=depth, pool_ratios=pool_ratios, dropout_rate=dropout_rate)

    def forward(self, data):
        edge_index, _ = dropout_adj(
            data.edge_index, p=0, force_undirected=True,
            num_nodes=data.num_nodes, training=self.training)
        x = F.dropout(data.x, p=0, training=self.training)

        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)
