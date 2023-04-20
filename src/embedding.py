import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv, GINConv, GATConv
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import dgl


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.1):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
                h = self.dropout(h)
            return self.linears[-1](h)


class GCN(nn.Module):

    def __init__(self, input_dim, type='gcn', hidden_dim=16, layers=3, dropout=0.1, **kwargs):
        super(GCN, self).__init__()
        self.type = type
        self.dropout = nn.Dropout(dropout)
        self.nlayers = layers
        self.input_dim = input_dim

        if type == 'gcn':
            self.layer0 = GraphConv(input_dim, hidden_dim)
            for i in range(layers-1):
                self.add_module('layer{}'.format(i + 1), GraphConv(hidden_dim, hidden_dim))

        elif type == 'gin':
            # removed batch normalisation + linear function for graph pooling + stacking all representations
            #self.nmlp_layers = **num_mlp_layers
            for layer in range(self.nlayers):
                if layer == 0:
                    mlp = MLP(2, input_dim, hidden_dim, hidden_dim, dropout)
                else:
                    mlp = MLP(2, hidden_dim, hidden_dim, hidden_dim, dropout)

                self.add_module('layer{}'.format(layer), GINConv(ApplyNodeFunc(mlp), 'sum', 0, learn_eps=True))
                #self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        elif type == 'gat':
            self.layer0 = GATConv(input_dim, hidden_dim, num_heads=1, residual=False)
            for i in range(layers - 1):
                self.add_module('layer{}'.format(i + 1),
                                GATConv(hidden_dim, hidden_dim, num_heads=1, residual=False))

    def forward(self, graph):

        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        x = graph.ndata['node_attr']

        if self.type == 'identity':
            return x

        for i in range(self.nlayers-1):
            x = torch.relu(self._modules['layer{}'.format(i)](graph,x).squeeze())
            x = self.dropout(x)
        x = self._modules['layer{}'.format(self.nlayers-1)](graph,x).squeeze()

        return x


