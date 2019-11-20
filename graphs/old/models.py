import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution

class GCN2(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(GCN2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, nclass)
        self.dropout1 = dropout
        self.dropout2 = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout1, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout2, training=self.training)
        return self.gc3(x, adj)
         

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


