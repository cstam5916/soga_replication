import argparse
import os

import numpy as np
import torch
from torch import nn

from scripts.models import GCN
from scripts.data.CitationsData import CitationsData
from scripts.losses import SOGALoss
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




def main():
    parser = argparse.ArgumentParser(description="GCN transfer (PyG)")
    parser.add_argument("--root", type=str, default="./dataset/dblp", help="Target dataset root/cache directory")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--source_model", type=str, default="./checkpoints/source/acm", help="Source model checkpoint directory")
    parser.add_argument("--base_checkpoints_dir", type=str, default="./checkpoints", help="Base directory for model checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument(
        "--mode", type=str, default="SSFUG_SOGA_Role2Vec",
        help="ALGOTYPE_VARIANT_EMBEDMODE, e.g. SSFUG_SOGA_Role2Vec, SSFUG_IMOnly_Struc2Vec",
    )
    parser.add_argument("--freeze_after", type=int, default=-1, help="Layer to freeze weights after. -1 = no freezing")
    args = parser.parse_args()

    algo_type, variant, embed_mode = parse_mode(args.mode)

    target_name = os.path.basename(args.root.rstrip("/"))
    source_name = os.path.basename(args.source_model.rstrip("/"))
    transfer_dir = f"{source_name}_to_{target_name}"
    checkpoints_dir = os.path.join(args.base_checkpoints_dir, algo_type, transfer_dir, args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = args.learning_rate
    weight_decay = 5e-4

    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, "best_model.pt")
    loss_path = os.path.join(checkpoints_dir, "loss.npy")
    acc_path = os.path.join(checkpoints_dir, "acc.npy")
    log_path = os.path.join(checkpoints_dir, "training_log.txt")
    plot_path = os.path.join(checkpoints_dir, "training_plots.svg")

    open(log_path, "w").close()  # clear/create log file

    dataset = CitationsData(args.root, args.root.split("/")[-1])
    data = dataset[0].to(device)

    required = ["x", "edge_index", "y"]
    for k in required:
        if not hasattr(data, k):
            raise ValueError(f"data is missing required attribute: {k}")

    in_channels = data.x.size(-1)
    out_channels = int(data.y.max().item()) + 1

    hidden_channels = [256, 256, 128]
    model = GCN(
        in_channels=in_channels,
        hidden_channels=[256, 256, 128],
        out_channels=out_channels,
    ).to(device)

    if(args.freeze_after > -1):
        for param in model.linear.parameters():
            param.requires_grad = False
        for conv_idx in range(args.freeze_after):
            for param in model.convs[conv_idx].parameters():
                param.requires_grad = False

    model.load_state_dict(torch.load(os.path.join(args.source_model, "best_model.pt"))['model_state_dict'])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    criterion = SOGALoss(data, mode=variant, embed_mode=embed_mode).to(device)

    model.eval()
    out = model(data)
    pretrained_loss = float(criterion(out, data.edge_index).item())
    pretrained_f1 = macro_f1(out, data.y)
    pretrained_acc = accuracy(out, data.y)

    pretrained_results = (
        f"Pretrained model evaluation\n"
        f"loss: {pretrained_loss:.4f}\n"
        f"f1: {pretrained_f1:.4f} | acc: {pretrained_acc:.4f}\n"
    )
    tee(log_path, pretrained_results, end="")

    best_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1

    losses = []
    f1s = []
    accs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, data.edge_index)
        loss.backward()
        optimizer.step()

        model.eval()
        out = model(data)
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
            tee(
                log_path,
                f"Epoch {epoch:04d} | loss {loss.item():.4f} | "
                f"f1 {f1:.4f} | acc {acc:.4f} | best_f1 {best_f1:.4f} | best_acc {best_acc:.4f} @ {best_epoch}"
            )

    save_training_plots(losses, f1s, accs, plot_path)
    tee(log_path, f"Done. Best checkpoint saved to: {ckpt_path} (best f1={best_f1:.4f} | best acc={best_acc:.4f} at epoch {best_epoch})")
    tee(log_path, f"Saved logs to: {loss_path} and {acc_path}")
    tee(log_path, f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()