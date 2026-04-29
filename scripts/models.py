import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn.parameter import Parameter

from scripts.layer import NodeCentricConv, Sparsemax


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        dims = [in_channels] + hidden_channels
        self.linear = nn.Linear(dims[0], dims[1])
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(1, len(dims) - 1)])
        self.linear_out = nn.Linear(dims[-1], out_channels)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.linear(x))
        for conv in self.convs: x = F.relu(conv(x, edge_index))
        return self.linear_out(x)


def extract_gcn_weights(model_list):
    """Extract per-layer GCNConv weights from a list of source GCN models.

    Returns a list of tuples — one tuple per GCNConv layer — where each tuple
    holds the corresponding weight tensor from every source model. This is the
    format expected by NodeCentricConv's model_weights argument.
    """
    weight_list = []
    for model in model_list:
        w_list = []
        for name, param in model.named_parameters():
            if 'convs' in name and name.endswith('lin.weight'):
                w_list.append(param)
        weight_list.append(w_list)
    return list(zip(*weight_list))


class _SOGAClassifierBlend(nn.Module):
    """Sparse-attention blend of source model classifier heads for SOGA GCN models.

    Unlike GraphATA's MLPModule (which calls model.gnn.cls), this accesses
    model.linear_out — the output head in SOGA's GCN.
    """
    def __init__(self, out_channels, model_list):
        super().__init__()
        self.model_list = model_list
        self.att = Parameter(torch.Tensor(out_channels, 1))
        nn.init.xavier_normal_(self.att)
        self.sparse_attention = Sparsemax(dim=1)

    def forward(self, x):
        outputs, weights = [], []
        for model in self.model_list:
            cls_out = model.linear_out(x)
            weights.append(torch.matmul(cls_out, self.att))
            outputs.append(cls_out)
        w = self.sparse_attention(torch.cat(weights, dim=1))  # [N, K]
        outputs = torch.stack(outputs)                         # [K, N, C]
        return torch.sum(w.t().unsqueeze(-1) * outputs, dim=0)


class GraphATANode(torch.nn.Module):
    """GraphATA node-centric adaptation model, adapted for SOGA's GCN architecture.

    Replaces each GCNConv layer with a NodeCentricConv that blends K source models
    via sparse per-node attention. The output classifier is a sparse-attention blend
    of the source models' linear_out heads.

    Args:
        in_channels: Input feature dimension (matches source GCN).
        hidden_channels: Hidden dims list (e.g. [256, 256, 128]). The first element
            is the output of the input linear projection; subsequent elements are
            outputs of NodeCentricConv layers.
        out_channels: Number of classes.
        model_weights: Output of extract_gcn_weights(model_list) — a list of tuples,
            one per GCNConv layer, each tuple containing K weight tensors.
        model_list: List of K pre-trained source GCN models.
        dropout: Dropout probability applied after each layer.
        use_bn: Whether to apply BatchNorm after each NodeCentricConv.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, model_weights, model_list,
                 dropout=0.0, use_bn=False):
        super().__init__()
        dims = hidden_channels
        self.linear = nn.Linear(in_channels, dims[0])
        self.convs = nn.ModuleList(
            NodeCentricConv(dims[i], dims[i + 1], model_weights[i])
            for i in range(len(dims) - 1)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)) if use_bn else None
        self.cls = _SOGAClassifierBlend(out_channels, model_list)
        self.dropout = dropout
        self.use_bn = use_bn

    def feat_bottleneck(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.dropout(F.relu(self.linear(x)), p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn and self.bns is not None:
                x = self.bns[i](x)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x

    def feat_classifier(self, x):
        return self.cls(x)

    def forward(self, data):
        return self.feat_classifier(self.feat_bottleneck(data))
