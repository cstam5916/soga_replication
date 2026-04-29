from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_scatter import scatter

from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.conv import MessagePassing, SimpleConv
from torch_geometric.typing import Adj, OptTensor, PairTensor, OptPairTensor, Size

from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd


def _make_ix_like(input, dim=0):
    d = input.size(dim)  # get size along the target dimension
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)  # 1-indexed rank vector
    view = [1] * input.dim()  # build shape of all ones for broadcasting
    view[0] = -1  # set first dim to -1 so rho can broadcast against input
    return rho.view(view).transpose(0, dim)  # reshape and align rho to target dim


def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold

    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax

    Returns:
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)  # sort values descending to find support
    input_cumsum = input_srt.cumsum(dim) - 1  # cumulative sum shifted by 1 (from sparsemax formula)
    rhos = _make_ix_like(input, dim)  # rank indices aligned to input shape
    support = rhos * input_srt > input_cumsum  # boolean mask: True where element is in the support set

    support_size = support.sum(dim=dim).unsqueeze(dim)  # count how many elements are in the support
    tau = input_cumsum.gather(dim, support_size - 1)  # threshold tau = (cumsum at support boundary) / support_size
    tau /= support_size.to(input.dtype)  # normalize tau by support size
    return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax

        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim  # save dim for use in backward
        max_val, _ = input.max(dim=dim, keepdim=True)  # find max for numerical stability
        input -= max_val  # subtract max to prevent overflow (same trick as softmax)
        tau, supp_size = _threshold_and_support(input, dim=dim)  # compute sparsemax threshold tau
        output = torch.clamp(input - tau, min=0)  # project: zero out elements below threshold
        ctx.save_for_backward(supp_size, output)  # save support size and output for backward pass
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors  # retrieve saved support size and forward output
        dim = ctx.dim  # retrieve the dimension sparsemax was applied along
        grad_input = grad_output.clone()  # start gradient as copy of upstream gradient
        grad_input[output == 0] = 0  # zero out gradients for elements outside the support

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()  # mean gradient over support
        v_hat = v_hat.unsqueeze(dim)  # restore dimension for broadcasting
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)  # subtract mean only within support
        return grad_input, None  # return gradient for input; None for dim (not differentiable)


sparsemax = SparsemaxFunction.apply  # bind the custom autograd function for direct use


class Sparsemax(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim  # dimension along which to apply sparsemax
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)  # apply sparsemax along stored dimension


class NodeCentricConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            model_weights: tuple = (),  # weight matrices from each pre-trained source model
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)  # normalize scalar to (src, dst) tuple

        self.films = list()
        for weight in model_weights:
            self.films.append(weight.t())  # transpose each source weight matrix for projection use

        self.att = Parameter(torch.Tensor(self.out_channels, 1))  # learnable attention vector for scoring sources
        nn.init.xavier_normal_(self.att)  # initialize attention weights with Xavier normal
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))  # residual projection weight
        nn.init.xavier_normal_(self.weight)  # initialize residual weight with Xavier normal

        self.neigh_aggr = SimpleConv(aggr='mean')  # mean neighborhood aggregation for computing context

        self.sparse_attention = Sparsemax(dim=1)  # per-node sparse source selection along source dimension

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_type: OptTensor = None) -> Tensor:
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))  # ensure each node aggregates itself
        neigh_rep = self.neigh_aggr(x, edge_index)  # compute mean neighborhood representation for each node

        atts = []   # will hold per-source attention score for each node
        reps = []   # will hold per-source projected message-passing output for each node

        out = self.propagate(edge_index, x=x, gamma=torch.sigmoid(neigh_rep), edge_weight=None, size=None)
        # propagate: aggregate neighbor features modulated by sigmoid-gated neighborhood context (calls message())

        for i, film in enumerate(self.films):
            rep = torch.matmul(neigh_rep, film)  # project neighborhood context through source i's weight matrix
            res = torch.matmul(rep, self.att)    # compute scalar attention score for source i at each node
            atts.append(res)                     # collect attention score for source i
            rep = torch.matmul(out, film)        # project aggregated messages through source i's weight matrix
            reps.append(rep)                     # collect source-i-projected output
        atts = torch.cat(atts, dim=1)  # shape [N, K]: stack all K source attention scores per node
        w = self.sparse_attention(atts)  # shape [N, K]: sparse per-node weights over K sources (many zeros)
        gamma = torch.stack(reps)  # shape [K, N, out]: stack all source-projected outputs
        w = w.t().unsqueeze(-1)    # shape [K, N, 1]: transpose and expand for broadcasting over feature dim

        wg = torch.matmul(neigh_rep, self.weight)  # residual branch: project neighborhood context directly
        gamma = torch.sum(w * gamma, dim=0)        # weighted sum over sources: each node blends K source outputs

        out = gamma + wg * 0.2  # add small residual to stabilize training and preserve neighborhood info

        return out

    def message(self, x_j: Tensor, gamma_i: Tensor, edge_weight: OptTensor) -> Tensor:
        out = gamma_i * x_j  # modulate neighbor feature x_j by destination node's sigmoid-gated context gamma_i

        return out


class MLPModule(torch.nn.Module):
    def __init__(self, args, model_list):
        super(MLPModule, self).__init__()
        self.args = args
        self.model_list = model_list  # list of pre-trained source models whose classifiers will be blended

        self.att = Parameter(torch.Tensor(args.num_classes, 1))  # learnable vector to score each source classifier
        nn.init.xavier_normal_(self.att)  # initialize with Xavier normal

        self.sparse_attention = Sparsemax(dim=1)  # sparse per-node blending weights over source classifiers

    def forward(self, x):
        outputs = []  # will hold per-source classifier logits
        weights = []  # will hold per-source attention scores
        for i in range(len(self.model_list)):
            cls_output = self.model_list[i].gnn.cls(x)  # run source i's classifier head on shared features
            att = torch.matmul(cls_output, self.att)     # score source i's output via learned attention vector
            outputs.append(cls_output)  # collect source i logits
            weights.append(att)         # collect source i attention score
        weights = torch.cat(weights, dim=1)  # shape [N, K]: concatenate all source scores per node
        w = self.sparse_attention(weights)   # shape [N, K]: sparse weights — each node selects relevant sources

        outputs = torch.stack(outputs)  # shape [K, N, C]: stack all source logits
        w = w.t().unsqueeze(-1)         # shape [K, N, 1]: align weights for broadcasting over class dimension
        x = torch.sum(w * outputs, dim=0)  # weighted blend of source classifier outputs per node

        return x
