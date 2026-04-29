"""SSFUG (Single-Source Free Unsupervised Graph) adaptation setup.

Encapsulates everything algorithm-specific for the SSFUG mode so that
train_transfer.py only contains generic training-loop boilerplate.
"""
import os

import torch

from scripts.models import GCN
from scripts.losses import SOGALoss
from scripts.utils import parse_mode


def setup(args, data, device):
    """Build model, optimizer, and criterion for SSFUG.

    Loads the pre-trained source GCN, applies layer freezing if requested,
    and constructs the SOGALoss criterion.

    Returns:
        model: GCN on device with source weights loaded.
        optimizer: Adam over unfrozen parameters.
        criterion: SOGALoss on device.
    """
    _, variant, embed_mode = parse_mode(args.mode)

    in_channels = data.x.size(-1)
    out_channels = int(data.y.max().item()) + 1

    model = GCN(
        in_channels=in_channels,
        hidden_channels=[256, 256, 128],
        out_channels=out_channels,
    ).to(device)

    ckpt = torch.load(os.path.join(args.source_model, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if args.freeze_after > -1:
        for param in model.linear.parameters():
            param.requires_grad = False
        for conv_idx in range(args.freeze_after):
            for param in model.convs[conv_idx].parameters():
                param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=5e-4)

    criterion = SOGALoss(data, mode=variant, embed_mode=embed_mode).to(device)

    def train_step_fn(model, data):
        out = model(data)
        return criterion(out, data.edge_index)

    def eval_fn(model, data):
        return model(data)

    return model, optimizer, train_step_fn, eval_fn
