"""GraphATA (Aggregate to Adapt) target-domain adaptation setup.

Loads K pre-trained source GCN models, extracts their GCNConv weights,
builds a GraphATANode that blends them via sparse per-node attention, and
returns the algorithm-specific train_step_fn and eval_fn for the generic
training loop in train_transfer.py.
"""
import os

import torch

from scripts.models import GCN, GraphATANode, extract_gcn_weights
from scripts.losses import GraphATALoss


def setup(args, data, device):
    """Build model, optimizer, train_step_fn, and eval_fn for GraphATA.

    Expects args.source_models: list of checkpoint directories, one per source.

    Returns:
        model: GraphATANode on device.
        optimizer: Adam over all trainable parameters (adaptation + source models).
        train_step_fn: callable(model, data) -> loss tensor.
        eval_fn: callable(model, data) -> logit tensor; also updates memory banks.
    """
    if not getattr(args, "source_models", None):
        raise ValueError("--source_models must list at least one source checkpoint directory for GraphATA")

    in_channels = data.x.size(-1)
    out_channels = int(data.y.max().item()) + 1
    hidden_channels = [256, 256, 128]
    nhid = hidden_channels[-1]

    # Load pre-trained source GCN models
    source_models = []
    for path in args.source_models:
        src = GCN(in_channels, hidden_channels, out_channels).to(device)
        ckpt = torch.load(os.path.join(path, "best_model.pt"), map_location=device)
        src.load_state_dict(ckpt["model_state_dict"])
        src.eval()
        source_models.append(src)

    model_weights = extract_gcn_weights(source_models)

    model = GraphATANode(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        model_weights=model_weights,
        model_list=source_models,
    ).to(device)

    # Optimise both the adaptation model and source model parameters jointly
    param_group = list(model.parameters())
    for src in source_models:
        param_group += list(src.parameters())
    optimizer = torch.optim.Adam(param_group, lr=args.learning_rate, weight_decay=5e-4)

    criterion = GraphATALoss(
        num_nodes=data.num_nodes,
        nhid=nhid,
        num_classes=out_channels,
        K=getattr(args, "K", 40),
        momentum=getattr(args, "momentum", 0.9),
    ).to(device)

    def train_step_fn(model, data):
        feat = model.feat_bottleneck(data)
        cls = model.feat_classifier(feat)
        return criterion(feat, cls)

    def eval_fn(model, data):
        feat = model.feat_bottleneck(data)
        cls = model.feat_classifier(feat)
        criterion.update_memory(feat, cls)
        return cls

    return model, optimizer, train_step_fn, eval_fn
