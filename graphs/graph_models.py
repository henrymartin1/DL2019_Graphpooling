import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa


import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, knn_interpolate
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GATConv

from torch_geometric.nn import GraphUNet, SAGEConv
from torch_geometric.utils import dropout_adj
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from sklearn.neighbors import NearestNeighbors


class KipfNet_simple(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, K=8):
        super(KipfNet, self).__init__()
#        self.conv1 = GCNConv(n_features, 60, cached=True)
#        self.conv2 = GCNConv(60, n_classes, cached=True)
        self.conv1 = ChebConv(num_features, num_classes, K=K)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))

        return x

class KipfNet_res(torch.nn.Module): #865
    def __init__(self, num_features, num_classes, nn, device, nh1=36, nh2=36, K=8, \
                            input_pool=2, output_pool=2):
        super(KipfNet_res, self).__init__()
        self.device=device
        # Cheb_net 
        self.conv1 = ChebConv(num_features, nh1, K=K)
        self.conv2 = ChebConv(nh1, nh2, K=K)

        self.bn1 = torch.nn.BatchNorm1d(nh1)
        self.bn2 = torch.nn.BatchNorm1d(nh2)

        # pooling
        self.compress = ChebConv(nh1, input_pool, K=1)
        self.pool1 = TopKPooling(input_pool, ratio=0.1)
        self.conv_p1 = ChebConv(input_pool, output_pool, K=K)
        
        # reconstruct
        self.conv3 = ChebConv(nh1 + nh2 + output_pool, num_classes, K=1)

        # cache unpooling:
        self.nn = nn


    def forward(self, data):
        # Cheb_net 
        x, edge_index = data.x, data.edge_index
        x1 = self.bn1(F.relu(self.conv1(x, edge_index)))
        x2 = self.bn2(F.relu(self.conv2(x1, edge_index)))


        # pool branch
        x1_compressed = self.compress(x1, edge_index)
        print("start pooling")
        xp1, edge_indexp1, _, _, perm, _   = self.pool1(x1_compressed, edge_index)
        print("finished pooling")   
        xp2 = self.conv_p1(xp1, edge_indexp1)

        # unpool
        pos_x = x[perm,-2:].contiguous().detach().cpu().numpy()
        pos_y = x[:,-2:].contiguous()
        
        print("start unpooling")
#        x3 = knn_interpolate2(xp2, pos_x=pos_x, pos_y=pos_y, batch_x=batch_x, device=self.device)
#        x3 = knn_interpolate2(xp2, pos_x=pos_x, pos_y=pos_y, nn=self.nn, device=self.device)
        print("finished unpooling")

        x3 = torch.cat((x1,x2,x3),1)


        x3 = self.conv3(x3, edge_index)
        return x3

import torch
from torch_geometric.nn import knn
from torch_scatter import scatter_add


def knn_interpolate2(x, pos_x, pos_y, nn, device, k=3):

    


    with torch.no_grad():
        dist, ind = nn.kneighbors(pos_x)
        
        y_idx = torch.from_numpy(ind).type(torch.LongTensor).to(device)
        x_idx = torch.from_numpy(np.arange(len(ind))).type(torch.LongTensor).to(device)
        diff = torch.from_numpy(dist).to(device)
        
        
        squared_distance = (diff * diff)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
        weights = weights.type(torch.FloatTensor).to(device)
        print(weights)

    print(type(x[x_idx]), type(weights), print(y_idx))
    y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0))
    y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos_y.size(0))

    return y



class colors_topk_pool(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(colors_topk_pool, self).__init__()

        self.conv1 = GINConv(Sequential(Linear(num_features, 64), ReLU(), Linear(64, 64)))
        self.pool1 = TopKPooling(num_features)
        self.conv2 = GINConv(Sequential(Linear(64, 64), ReLU(), Linear(64, 64)))

        self.lin = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, = data.x, data.edge_index

        out = F.relu(self.conv1(x, edge_index))
        print(out.shape)

        out, edge_index, _, _, _, _ = self.pool1(out, edge_index, attn=x)
        print(out.shape)

        out = F.relu(self.conv2(out, edge_index))
        print(out.shape)
        out = self.lin(out)
        print(out.shape)
        return out






class KipfNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, K=8):
        super(KipfNet, self).__init__()
#        self.conv1 = GCNConv(n_features, 60, cached=True)
#        self.conv2 = GCNConv(60, n_classes, cached=True)
        self.conv1 = ChebConv(num_features, nh1, K=K)
        self.conv2 = ChebConv(nh1, num_classes, K=K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GINNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINNet, self).__init__()
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


class KipfNet_res2(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1_1=36, nh1_2=36, K=8, \
                            nh2=36, nn_hidden=36):
        super(KipfNet_res2, self).__init__()
        #chebconv stream
        self.conv1_1 = ChebConv(num_features, nh1_1, K=K)
        self.conv1_2 = ChebConv(nh1_1, nh1_2, K=K)
        self.conv1 = ChebConv(nh1_1+nh1_2, num_features, K=K)
        self.bn1_1 = torch.nn.BatchNorm1d(nh1_1)
        self.bn1_2 = torch.nn.BatchNorm1d(nh1_2)
        self.bn1 = torch.nn.BatchNorm1d(nh1_1+nh1_2)

        # GINconv stream
        nn2 = Sequential(Linear(num_features, nn_hidden), ReLU(), Linear(nn_hidden, nh2))
        self.conv2 = GINConv(nn2)

        self.bn2 = torch.nn.BatchNorm1d(nh2)

        # pooling
        self.pool1 = TopKPooling(num_features, min_score=0.05)

        # mixer
        self.conv4 = ChebConv(nh1_1+nh1_2+nh2, num_classes, K=1) #nh1_1+nh1_2+nh2_1+nh2_2


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Cheb conv stream  
        x1_1 = F.relu(self.bn1_1(self.conv1_1(x, edge_index)))
        x1_2 = F.relu(self.bn1_2(self.conv1_2(x1_1, edge_index)))

        # x1 = torch.cat((x1_1,x1_2),1)
        # x1 = F.relu(self.conv1(x1, edge_index))

        # GIN Conv stream
        x2 = self.bn2(F.relu(self.conv2(x, edge_index)))

        # mixer
        x4 = torch.cat((x1_1,x1_2,x2),1)
        x4 = self.conv4(x4, edge_index)
        return x4


import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GlobalAttention


class GlobalAttentionNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super(GlobalAttentionNet, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.att = GlobalAttention(Linear(hidden, 1))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.att.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.att(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class colors_topk_pool(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(colors_topk_pool, self).__init__()

        self.conv1 = GINConv(Sequential(Linear(num_features, 64), ReLU(), Linear(64, 64)))
        self.pool1 = TopKPooling(num_features)
        self.conv2 = GINConv(Sequential(Linear(64, 64), ReLU(), Linear(64, 64)))

        self.lin = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, = data.x, data.edge_index

        out = F.relu(self.conv1(x, edge_index))
        print(out.shape)

        out, edge_index, _, _, _, _ = self.pool1(out, edge_index, attn=x)
        print(out.shape)

        out = F.relu(self.conv2(out, edge_index))
        print(out.shape)
        out = self.lin(out)
        print(out.shape)
        return out


class SAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=36):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, nh1)
        self.conv2 = SAGEConv(nh1, num_classes)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x







class KipfNet1b(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=16, K=4):
        super(KipfNet1b, self).__init__()
#        self.conv1 = GCNConv(n_features, 60, cached=True)
#        self.conv2 = GCNConv(60, n_classes, cached=True)
        self.conv1 = ChebConv(num_features, nh1, K=K)
        self.bn1 = torch.nn.BatchNorm1d(nh1)
        self.conv2 = ChebConv(nh1, num_classes, K=K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x






class KipfNet2(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, nh2=18, K=[8,6,4]):
        super(KipfNet2, self).__init__()
#        self.conv1 = GCNConv(n_features, 60, cached=True)
#        self.conv2 = GCNConv(60, n_classes, cached=True)
        self.conv1 = ChebConv(num_features, nh1, K=K[0])
        self.bn1 = torch.nn.BatchNorm1d(nh1)
        self.conv2 = ChebConv(nh1, nh2, K=K[1])
        self.bn2 = torch.nn.BatchNorm1d(nh2)
        self.conv3 = ChebConv(nh2, num_classes, K=K[2])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        return x

class KipfNet3(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, nh2=18, nh3=9, K=[8,6,4,4]):
        super(KipfNet3, self).__init__()
#        self.conv1 = GCNConv(n_features, 60, cached=True)
#        self.conv2 = GCNConv(60, n_classes, cached=True)
        self.conv1 = ChebConv(num_features, nh1, K=K[0])
        self.bn1 = torch.nn.BatchNorm1d(nh1)
        self.conv2 = ChebConv(nh1, nh2, K=K[1])
        self.bn2 = torch.nn.BatchNorm1d(nh2)
        self.conv3 = ChebConv(nh2, nh3, K=K[2])
        self.bn3 = torch.nn.BatchNorm1d(nh3)
        self.conv4 = ChebConv(nh3, num_classes, K=K[3])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, training=self.training)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        return x
    

class spline_net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(spline_net, self).__init__()
        self.conv1 = SplineConv(num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, num_classes, dim=1, kernel_size=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GUNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, pool_ratios=[0.25, 0.25],
        depth=3, nh1=32):
        super(GUNet, self).__init__()
        # pool_ratios = [2000 / num_nodes, 0.5]
        self.unet = GraphUNet(num_features, nh1, num_classes,
                              depth=depth, pool_ratios=pool_ratios)

    def forward(self, data):
        edge_index, _ = dropout_adj(
            data.edge_index, p=0.5, force_undirected=True,
            num_nodes=data.num_nodes, training=self.training)
        x = F.dropout(data.x, p=0.5, training=self.training)

        x = self.unet(x, edge_index)
        return x

class HandleNodeAttention(object):
    def __call__(self, data):
        data.attn = torch.softmax(data.x[:, 0], dim=0)
        data.x = data.x[:, 1:]
        return data




class ARMANet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(ARMANet, self).__init__()

        self.conv1 = ARMAConv(
            num_features,
            16,
            num_stacks=3,
            num_layers=2,
            shared_weights=True,
            dropout=0.25)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        return x

class ARMACheb(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=32, K=8):
        super(ARMACheb, self).__init__()

        self.conv1 = ARMAConv(
            num_features,
            nh1,
            num_stacks=2,
            num_layers=2,
            shared_weights=True,
            dropout=0.5)

        self.conv2 = ChebConv(nh1, num_classes, K=K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x



###############################
class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            8 * 8, dataset.num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)




class topk_pool(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(topk_pool, self).__init__()

        self.conv1 = GraphConv(num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x



