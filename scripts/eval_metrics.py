"""Compute Accuracy / Macro-F1 / confusion matrix from a saved AIRHAR checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
AIRHAR = ROOT / "third_party" / "AIRHAR"
sys.path.insert(0, str(AIRHAR))
sys.path.insert(0, str(ROOT / "scripts"))
from ci4r_meta import CLASS_NAMES  # noqa: E402
from models import CoreModel  # noqa: E402
from project import Project  # noqa: E402


def load_model(proj: Project, ckpt: Path) -> torch.nn.Module:
    net = CoreModel(
        hidden_size=proj.Classification_hidden_size,
        num_layers=proj.Classification_num_layers,
        backbone_type=proj.Classification_backbone,
        dim=proj.dim,
        dt_rank=proj.dt_rank,
        d_state=proj.d_state,
        image_height=proj.image_height,
        image_width=proj.frame_length,
        num_classes=proj.num_classes,
        channels=proj.channels,
        dropout=proj.dropout,
        channel_confusion_layer=proj.channel_confusion_layer,
        channel_confusion_out_channels=proj.channel_confusion_out_channels,
        time_downsample_factor=proj.time_downsample_factor,
        optional_avg_pool=proj.optional_avg_pool,
    )
    state = torch.load(ckpt, map_location=proj.device, weights_only=True)
    net.load_state_dict(state)
    return net.to(proj.device).eval()


def collect_preds(net, loader, device):
    ys, ps = [], []
    with torch.no_grad():
        for features, targets in loader:
            logits = net(features.to(device))
            ys.append(targets.argmax(dim=1).cpu().numpy())
            ps.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_name", default="CI4R")
    parser.add_argument("--Classification_backbone", default="radmamba")
    parser.add_argument("--dim", type=int, default=80)
    parser.add_argument("--Classification_hidden_size", type=int, default=64)
    parser.add_argument("--out", default=str(ROOT / "results" / "metrics"))
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0], "--dataset_name", args.dataset_name, "--Classification_backbone", args.Classification_backbone, "--dim", str(args.dim), "--Classification_hidden_size", str(args.Classification_hidden_size), *unknown]
    proj = Project()
    proj.set_device()
    _, test_loader = proj.build_dataloaders()
    net = load_model(proj, Path(args.ckpt))
    y_true, y_pred = collect_preds(net, test_loader, proj.device)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_params": int(sum(p.numel() for p in net.parameters())),
        "ckpt": args.ckpt,
        "backbone": args.Classification_backbone,
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.ckpt).stem
    (out_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / f"{stem}_cm.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, cm):
            writer.writerow([name, *row.tolist()])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_cm.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
