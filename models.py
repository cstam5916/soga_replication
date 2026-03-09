import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        dims = [in_channels] + hidden_channels
        self.linears = nn.ModuleList([nn.Linear(dims[0], dims[1])] if hidden_channels else [])
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(1, len(dims) - 1)])
        self.linear_out = nn.Linear(dims[-1], out_channels)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        if self.linears: x = F.relu(self.linears[0](x))
        for conv in self.convs: x = F.relu(conv(x, edge_index))
        return self.linear_out(x)