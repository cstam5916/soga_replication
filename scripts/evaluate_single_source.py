import argparse
import os

import torch
import torch.nn.functional as F

from scripts.models import GCN
from scripts.data.CitationsData import CitationsData
from scripts.utils import macro_f1, accuracy


def load_model(ckpt_dir: str, in_channels: int, out_channels: int, device: torch.device) -> torch.nn.Module:
    model = GCN(
        in_channels=in_channels,
        hidden_channels=[256, 256, 128],
        out_channels=out_channels,
    ).to(device)
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluating single source transfer methods through ensemble averaging")
    parser.add_argument("--root", type=str, default="./dataset/dblp", help="Target dataset root/cache directory")
    parser.add_argument("--models", nargs="+", required=True, help="List of checkpoint directories to ensemble")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_name = os.path.basename(args.root.rstrip("/"))
    dataset = CitationsData(args.root, target_name)
    data = dataset[0].to(device)

    in_channels = data.x.size(-1)
    out_channels = int(data.y.max().item()) + 1

    probs_list = []
    with torch.no_grad():
        for ckpt_dir in args.models:
            model = load_model(ckpt_dir, in_channels, out_channels, device)
            logits = model(data)
            probs_list.append(F.softmax(logits, dim=1))

    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)

    f1 = macro_f1(avg_probs, data.y)
    acc = accuracy(avg_probs, data.y)

    result_lines = [
        "Ensemble Evaluation Results",
        f"Target dataset: {args.root} ({target_name})",
        f"Source models ({len(args.models)}):",
    ]
    for m in args.models:
        result_lines.append(f"  {m}")
    result_lines += [
        f"F1 Score (macro): {f1:.4f}",
        f"Accuracy:         {acc:.4f}",
    ]
    result_text = "\n".join(result_lines)
    print(result_text)

    out_dir = "checkpoints/SS_to_MS_ensemble"
    os.makedirs(out_dir, exist_ok=True)
    source_names = [os.path.basename(m.rstrip("/")) for m in args.models]
    fname = f"{'__'.join(source_names)}__{target_name}.txt"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        f.write(result_text + "\n")
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
