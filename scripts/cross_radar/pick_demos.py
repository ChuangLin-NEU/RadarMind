#!/usr/bin/env python3
"""Held-out timestamp demos. No train samples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_cross_radar import CLASS_NAMES, EmbNet, load_band  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase2_cross_radar"


def preds(run_id, freq):
    d = load_band(freq)
    te = d["split"] == "test"
    net = EmbNet(len(CLASS_NAMES))
    net.load_state_dict(torch.load(OUT / "runs" / run_id / "best.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(d["X"][te]).float())
        pred = logits.argmax(1).numpy()
    return {
        "timestamp": d["timestamp"][te],
        "action": d["action"][te],
        "pred": np.array([CLASS_NAMES[i] for i in pred]),
        "ok": pred == d["y"][te],
    }


def main() -> None:
    w10, w24, w77 = preds("within_10", "10GHz"), preds("within_24", "24GHz"), preds("within_77", "77GHz")
    x7710 = preds("cross_77_to_10", "10GHz")
    coral = preds("coral_uda_77_to_10", "10GHz")
    by77 = {t: i for i, t in enumerate(w77["timestamp"])}
    by10 = {t: i for i, t in enumerate(w10["timestamp"])}
    by24 = {t: i for i, t in enumerate(w24["timestamp"])}
    demos = []
    for t, i77 in by77.items():
        if t in by10 and t in by24:
            i10, i24 = by10[t], by24[t]
            if w77["ok"][i77] and w10["ok"][i10] and w24["ok"][i24] and w77["action"][i77] == w10["action"][i10] == w24["action"][i24]:
                demos.append(
                    {
                        "demo_id": "demo_A",
                        "title": "三频 within 都对",
                        "timestamp": t,
                        "action": w77["action"][i77],
                        "split": "test",
                    }
                )
                break
    for t, i77 in by77.items():
        if t in by10:
            i10 = by10[t]
            j = list(x7710["timestamp"]).index(t) if t in x7710["timestamp"] else None
            if j is None:
                continue
            if w77["ok"][i77] and (not x7710["ok"][j]):
                demos.append(
                    {
                        "demo_id": "demo_B",
                        "title": "77 within 对，77→10 source-only 错",
                        "timestamp": t,
                        "action": w77["action"][i77],
                        "pred_77": w77["pred"][i77],
                        "pred_77_to_10": x7710["pred"][j],
                        "split": "test",
                    }
                )
                break
    for t in coral["timestamp"]:
        j = list(coral["timestamp"]).index(t)
        k = list(x7710["timestamp"]).index(t) if t in x7710["timestamp"] else None
        if k is None:
            continue
        if (not x7710["ok"][k]) and coral["ok"][j]:
            demos.append(
                {
                    "demo_id": "demo_C",
                    "title": "source-only 错，CORAL UDA 对",
                    "timestamp": t,
                    "action": coral["action"][j],
                    "pred_source_only": x7710["pred"][k],
                    "pred_coral": coral["pred"][j],
                    "split": "test",
                    "note": "unsupervised domain adaptation, not domain generalization",
                }
            )
            break
    (OUT / "demo_registry.json").write_text(json.dumps({"n": len(demos), "demos": demos}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(demos, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
