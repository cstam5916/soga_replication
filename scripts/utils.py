import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_SSFUG_VARIANTS = {"SOGA", "IMOnly", "SCOnly"}
_ALGO_TYPES = {"SSFUG", "MSFUG", "MSSG", "MSFU"}
_EMBEDMODE_UNUSED_VARIANTS = {"IMOnly"}


def parse_mode(mode: str):
    parts = mode.split("_", 2)
    algo_type = parts[0] if len(parts) >= 1 else None
    variant = parts[1] if len(parts) >= 2 else None
    embed_mode = parts[2] if len(parts) >= 3 else None

    if algo_type not in _ALGO_TYPES:
        raise ValueError(f"Unknown algo type {algo_type!r} in --mode {mode!r}")
    if algo_type != "SSFUG":
        raise NotImplementedError(f"Algorithm type {algo_type!r} is not yet implemented")
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


def save_training_plots(losses: list, accs: list, out_path: str):
    sns.set_theme()
    epochs = range(1, len(losses) + 1)
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, losses)
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    ax_f1.plot(epochs, accs)
    ax_f1.set_title("F1 Score")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1 Score")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
