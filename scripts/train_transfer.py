import argparse
import os

import numpy as np
import torch
from torch import nn

from scripts.data.CitationsData import CitationsData
from scripts.utils import parse_mode, tee, save_training_plots, macro_f1, accuracy


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, acc: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "acc": acc,
        },
        path,
    )


def _build_algorithm(algo_type, args, data, device):
    """Dispatch to the algorithm-specific setup module.

    All setup modules return (model, optimizer, train_step_fn, eval_fn) where:
        train_step_fn(model, data) -> loss tensor  (called under model.train())
        eval_fn(model, data)       -> logit tensor (called under model.eval(), no_grad)
    """
    if algo_type == "SSFUG":
        from scripts import ssfug
        return ssfug.setup(args, data, device)
    if algo_type == "GraphATA":
        from scripts import graphata
        return graphata.setup(args, data, device)
    raise NotImplementedError(f"Algorithm type {algo_type!r} is not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="GCN transfer (PyG)")
    parser.add_argument("--root", type=str, default="./dataset/dblp", help="Target dataset root/cache directory")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--source_model", type=str, default="./checkpoints/source/acm",
                        help="Source model checkpoint directory (SSFUG)")
    parser.add_argument("--source_models", type=str, nargs="+", default=None,
                        help="One or more source model checkpoint directories (GraphATA)")
    parser.add_argument("--base_checkpoints_dir", type=str, default="./checkpoints",
                        help="Base directory for model checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--mode", type=str, default="SSFUG_SOGA_Role2Vec",
        help=(
            "Algorithm mode string. Examples:\n"
            "  SSFUG_SOGA_Role2Vec  — single-source SOGA with Role2Vec embeddings\n"
            "  SSFUG_IMOnly         — single-source information maximisation only\n"
            "  GraphATA             — multi-source node-centric aggregation"
        ),
    )
    parser.add_argument("--freeze_after", type=int, default=-1,
                        help="(SSFUG) Freeze weights up to this GCNConv layer index. -1 = no freezing")
    parser.add_argument("--K", type=int, default=40,
                        help="(GraphATA) Number of nearest neighbours for KNN pseudo-labelling")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="(GraphATA) EMA momentum for memory bank updates")
    args = parser.parse_args()

    algo_type, _, _ = parse_mode(args.mode)

    target_name = os.path.basename(args.root.rstrip("/"))
    if algo_type == "GraphATA" and args.source_models:
        source_tag = "+".join(os.path.basename(p.rstrip("/")) for p in args.source_models)
    else:
        source_tag = os.path.basename(args.source_model.rstrip("/"))
    transfer_dir = f"{source_tag}_to_{target_name}"
    checkpoints_dir = os.path.join(args.base_checkpoints_dir, algo_type, transfer_dir, args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, "best_model.pt")
    loss_path = os.path.join(checkpoints_dir, "loss.npy")
    acc_path = os.path.join(checkpoints_dir, "acc.npy")
    log_path = os.path.join(checkpoints_dir, "training_log.txt")
    plot_path = os.path.join(checkpoints_dir, "training_plots.svg")

    open(log_path, "w").close()

    dataset = CitationsData(args.root, args.root.split("/")[-1])
    data = dataset[0].to(device)

    for k in ("x", "edge_index", "y"):
        if not hasattr(data, k):
            raise ValueError(f"data is missing required attribute: {k}")

    model, optimizer, train_step_fn, eval_fn = _build_algorithm(algo_type, args, data, device)

    # --- pre-training evaluation ---
    model.eval()
    with torch.no_grad():
        out = eval_fn(model, data)
    pretrained_f1 = macro_f1(out, data.y)
    pretrained_acc = accuracy(out, data.y)
    tee(log_path,
        f"Pretrained model evaluation\nf1: {pretrained_f1:.4f} | acc: {pretrained_acc:.4f}\n",
        end="")

    # --- training loop ---
    best_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1
    losses, f1s, accs = [], [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = train_step_fn(model, data)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = eval_fn(model, data)
        f1 = macro_f1(out, data.y)
        acc = accuracy(out, data.y)

        losses.append(float(loss.item()))
        f1s.append(f1)
        accs.append(acc)

        np.save(loss_path, np.array(losses, dtype=np.float32))
        np.save(acc_path, np.array(accs, dtype=np.float32))

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_epoch = epoch
            save_checkpoint(ckpt_path, model, optimizer, epoch, f1)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            tee(log_path,
                f"Epoch {epoch:04d} | loss {loss.item():.4f} | "
                f"f1 {f1:.4f} | acc {acc:.4f} | best_f1 {best_f1:.4f} | best_acc {best_acc:.4f} @ {best_epoch}")

    save_training_plots(losses, f1s, accs, plot_path)
    tee(log_path, f"Done. Best checkpoint saved to: {ckpt_path} (best f1={best_f1:.4f} | best acc={best_acc:.4f} at epoch {best_epoch})")
    tee(log_path, f"Saved logs to: {loss_path} and {acc_path}")
    tee(log_path, f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
