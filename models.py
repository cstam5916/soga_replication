import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,):
        super().__init__()
        self.num_layers = num_layers
        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(self.num_layers)])
        self.convs = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(self.num_layers)])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(self.num_layers)])
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.linear_in(x))
        for i in range(self.num_layers):
            x = self.relus[i](self.convs[i](self.batchnorms[i](x), edge_index))
        x = self.linear_out(x)
        return x