import argparse
import os

import numpy as np
import torch
from torch import nn

from models import GCN
from DomainData import DomainData


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_acc: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )


@torch.no_grad()
def macro_f1(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
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
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def main():
    parser = argparse.ArgumentParser(description="GCN node classification (PyG)")
    parser.add_argument("--root", type=str, default="./data/acm", help="Dataset root/cache directory")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=3, help="Number of GCNConv layers (after linear_in)")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--results_dir", type=str, default="./checkpoints/acm_gcn", help="Directory to save outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 1e-2
    weight_decay = 5e-4

    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_path = os.path.join(args.results_dir, "best_model.pt")
    train_loss_path = os.path.join(args.results_dir, "train_loss.npy")
    val_loss_path = os.path.join(args.results_dir, "val_loss.npy")

    dataset = DomainData(args.root, args.root.split("/")[-1])
    data = dataset[0].to(device)

    required = ["x", "edge_index", "y", "train_mask", "val_mask", "test_mask"]
    for k in required:
        if not hasattr(data, k):
            raise ValueError(f"data is missing required attribute: {k}")

    in_channels = data.x.size(-1)
    out_channels = int(data.y.max().item()) + 1

    model = GCN(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        out_channels=out_channels,
        num_layers=args.layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = -1

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        out = model(data)
        train_acc = macro_f1(out, data.y, data.train_mask)
        val_acc = macro_f1(out, data.y, data.val_mask)
        test_acc = macro_f1(out, data.y, data.test_mask)

        train_losses.append(float(train_loss.item()))
        val_loss = float(criterion(out[data.val_mask], data.y[data.val_mask]).item())
        val_losses.append(val_loss)

        np.save(train_loss_path, np.array(train_losses, dtype=np.float32))
        np.save(val_loss_path, np.array(val_losses, dtype=np.float32))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(ckpt_path, model, optimizer, epoch, val_acc)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d} | loss {train_loss.item():.4f} | "
                f"train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f} | "
                f"best_val {best_val_acc:.4f} @ {best_epoch}"
            )

    print(f"Done. Best checkpoint saved to: {ckpt_path} (best val_acc={best_val_acc:.4f} at epoch {best_epoch})")
    print(f"Saved loss logs to: {train_loss_path} and {val_loss_path}")


if __name__ == "__main__":
    main()