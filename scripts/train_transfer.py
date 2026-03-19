import argparse
import os

import numpy as np
import torch
from torch import nn

from scripts.models import GCN
from scripts.DomainData import DomainData
from scripts.losses import SOGALoss


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


@torch.no_grad()
def macro_f1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    if pred.numel() == 0:
        return 0.0
    num_classes = int(max(pred.max().item(), y.max().item())) + 1
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def main():
    parser = argparse.ArgumentParser(description="GCN node classification (PyG)")
    parser.add_argument("--root", type=str, default="./data/acm", help="Dataset root/cache directory")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    # parser.add_argument("--layers", type=int, default=3, help="Number of GCNConv layers (after linear_in)")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--source_model", type=str, default="./checkpoints/dblp_gcn_exact", help="Source model")
    parser.add_argument("--results_dir", type=str, default="./checkpoints/acm_gcn", help="Directory to save outputs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--mode", type=str, default='SOGA', help='SOGA, IMOnly, or SCOnly')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = args.learning_rate
    weight_decay = 5e-4

    results_dir = os.path.join(args.results_dir)

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "best_model.pt")
    loss_path = os.path.join(results_dir, "loss.npy")
    acc_path = os.path.join(results_dir, "acc.npy")
    pretrained_results_path = os.path.join(results_dir, "pretrained_results.txt")

    dataset = DomainData(args.root, args.root.split("/")[-1])
    data = dataset[0].to(device)

    required = ["x", "edge_index", "y"]
    for k in required:
        if not hasattr(data, k):
            raise ValueError(f"data is missing required attribute: {k}")

    in_channels = data.x.size(-1)
    out_channels = int(data.y.max().item()) + 1

    model = GCN(
        in_channels=in_channels,
        hidden_channels=[256, 256, 128],
        out_channels=out_channels,
    ).to(device)

    

    model.load_state_dict(torch.load(os.path.join(args.source_model, "best_model.pt"))['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = SOGALoss(data, mode=args.mode).to(device)

    model.eval()
    out = model(data)
    pretrained_loss = float(criterion(out, data.edge_index).item())
    pretrained_acc = macro_f1(out, data.y)

    pretrained_results = (
        f"Pretrained model evaluation\n"
        f"loss: {pretrained_loss:.4f}\n"
        f"f1: {pretrained_acc:.4f}\n"
    )
    print(pretrained_results, end="")
    with open(pretrained_results_path, "w") as f:
        f.write(pretrained_results)

    best_acc = -1.0
    best_epoch = -1

    losses = []
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
        acc = macro_f1(out, data.y)

        losses.append(float(loss.item()))
        accs.append(acc)

        np.save(loss_path, np.array(losses, dtype=np.float32))
        np.save(acc_path, np.array(accs, dtype=np.float32))

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            save_checkpoint(ckpt_path, model, optimizer, epoch, acc)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d} | loss {loss.item():.4f} | "
                f"f1 {acc:.4f} | best_f1 {best_acc:.4f} @ {best_epoch}"
            )

    print(f"Done. Best checkpoint saved to: {ckpt_path} (best f1={best_acc:.4f} at epoch {best_epoch})")
    print(f"Saved logs to: {loss_path} and {acc_path}")


if __name__ == "__main__":
    main()