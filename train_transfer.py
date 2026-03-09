import argparse
import os

import numpy as np
import torch
from torch import nn

from models import GCN
from DomainData import DomainData
from losses import SOGALoss


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
    parser.add_argument("--source_model", type=str, default="./checkpoints/dblp_gcn", help="Source model")
    parser.add_argument("--results_dir", type=str, default="./checkpoints/acm_gcn", help="Directory to save outputs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--mode", type=str, default='SOGA', help='SOGA, IMOnly, or SCOnly')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = args.learning_rate
    weight_decay = 5e-4

    results_dir = os.path.join(args.results_dir, args.mode, f'seed_{args.seed}')

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "best_model.pt")
    train_loss_path = os.path.join(results_dir, "train_loss.npy")
    val_loss_path = os.path.join(results_dir, "val_loss.npy")
    val_acc_path = os.path.join(results_dir, "val_acc.npy")
    pretrained_results_path = os.path.join(results_dir, "pretrained_results.txt")

    dataset = DomainData(args.root, args.root.split("/")[-1])
    data = dataset[0].to(device)

    # overwrite masks with random 4:1 train/val split
    torch.manual_seed(args.seed)
    num_nodes = data.y.size(0)
    perm = torch.randperm(num_nodes)

    train_size = int(0.8 * num_nodes)  # 4:1 ratio
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True

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

    model.load_state_dict(torch.load(os.path.join(args.source_model, "best_model.pt"))['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = SOGALoss(data, mode=args.mode).to(device)

    model.eval()
    out = model(data)
    pretrained_train_loss = float(criterion(out, data.train_mask, data.edge_index).item())
    pretrained_val_loss = float(criterion(out, data.val_mask, data.edge_index).item())
    pretrained_train_acc = macro_f1(out, data.y, data.train_mask)
    pretrained_val_acc = macro_f1(out, data.y, data.val_mask)
    pretrained_test_acc = macro_f1(out, data.y, data.test_mask)

    pretrained_results = (
        f"Pretrained model evaluation\n"
        f"train_loss: {pretrained_train_loss:.4f}\n"
        f"val_loss: {pretrained_val_loss:.4f}\n"
        f"train_f1: {pretrained_train_acc:.4f}\n"
        f"val_f1: {pretrained_val_acc:.4f}\n"
        f"test_f1: {pretrained_test_acc:.4f}\n"
    )
    print(pretrained_results, end="")
    with open(pretrained_results_path, "w") as f:
        f.write(pretrained_results)

    best_val_acc = -1.0
    best_epoch = -1

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        train_loss = criterion(out, data.train_mask, data.edge_index)
        train_loss.backward()
        optimizer.step()

        model.eval()
        out = model(data)
        train_acc = macro_f1(out, data.y, data.train_mask)
        val_acc = macro_f1(out, data.y, data.val_mask)
        test_acc = macro_f1(out, data.y, data.test_mask)

        train_losses.append(float(train_loss.item()))
        val_loss = float(criterion(out, data.val_mask, data.edge_index).item())
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        np.save(train_loss_path, np.array(train_losses, dtype=np.float32))
        np.save(val_loss_path, np.array(val_losses, dtype=np.float32))
        np.save(val_acc_path, np.array(val_accs, dtype=np.float32))

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