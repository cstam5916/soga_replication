import warnings
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

_SSFUG_VARIANTS = {"SOGA", "IMOnly", "SCOnly"}
_ALGO_TYPES = {"SSFUG", "MSFUG", "MSSG", "MSFU", "GraphATA"}
_IMPLEMENTED_ALGOS = {"SSFUG", "GraphATA"}
_EMBEDMODE_UNUSED_VARIANTS = {"IMOnly"}


@torch.no_grad()
def macro_f1(logits: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    pred = logits.argmax(dim=1)
    if mask is not None:
        pred = pred[mask]
        y = y[mask]
    if pred.numel() == 0:
        return 0.0
    num_classes = int(max(pred.max().item(), y.max().item())) + 1
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(f1s) / len(f1s)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    pred = logits.argmax(dim=1)
    if mask is not None:
        pred = pred[mask]
        y = y[mask]
    if pred.numel() == 0:
        return 0.0
    return (pred == y).float().mean().item()


def parse_mode(mode: str):
    parts = mode.split("_", 2)
    algo_type = parts[0] if len(parts) >= 1 else None
    variant = parts[1] if len(parts) >= 2 else None
    embed_mode = parts[2] if len(parts) >= 3 else None

    if algo_type not in _ALGO_TYPES:
        raise ValueError(f"Unknown algo type {algo_type!r} in --mode {mode!r}")
    if algo_type not in _IMPLEMENTED_ALGOS:
        raise NotImplementedError(f"Algorithm type {algo_type!r} is not yet implemented")

    if algo_type == "SSFUG":
        if variant is not None and variant not in _SSFUG_VARIANTS:
            raise NotImplementedError(f"SSFUG variant {variant!r} is not yet implemented")
        if variant in _EMBEDMODE_UNUSED_VARIANTS:
            if embed_mode is not None:
                warnings.warn(
                    f"EMBEDMODE {embed_mode!r} is ignored when VARIANT is {variant!r}",
                    stacklevel=2,
                )
            embed_mode = None

    return algo_type, variant, embed_mode


def tee(log_path: str, msg: str, end: str = "\n"):
    print(msg, end=end)
    with open(log_path, "a") as f:
        f.write(msg + end)


def save_training_plots(losses: list, f1s: list, accs: list, out_path: str):
    sns.set_theme()
    epochs = range(1, len(losses) + 1)
    fig, (ax_loss, ax_f1, ax_acc) = plt.subplots(1, 3, figsize=(15, 4))

    ax_loss.plot(epochs, losses)
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    ax_f1.plot(epochs, f1s)
    ax_f1.set_title("Macro F1")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1 Score")

    ax_acc.plot(epochs, accs)
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
