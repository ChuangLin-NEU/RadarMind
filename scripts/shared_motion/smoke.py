#!/usr/bin/env python3
"""CPU/GPU smoke: cache, split, one forward, physics aug, InfoNCE. No training."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from shared_motion.common import (  # noqa: E402
    CLASS_NAMES,
    EmbNet,
    infonce,
    load_packs,
    physics_aug,
    views_for,
)


def main() -> None:
    rows = []
    split = ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"
    rows.append(("split csv", split.is_file(), str(split)))
    packs = load_packs()
    n_ts = None
    for freq, d in packs.items():
        n = len(d["y"])
        rows.append((f"cache {freq}", n == 327 or n >= 320, f"n={n}"))
        if n_ts is None:
            n_ts = set(map(str, d["timestamp"]))
        else:
            rows.append((f"ts overlap {freq}", set(map(str, d["timestamp"])) == n_ts, "same timestamps"))
    tr = views_for(packs, ["10GHz", "24GHz", "77GHz"], "train")
    paired = sum(1 for v in tr.values() if len(v) == 3)
    rows.append(("train triples", paired > 100, f"n={paired}"))
    x = next(iter(next(iter(tr.values())).values()))[0]
    rng = np.random.RandomState(42)
    xa = physics_aug(x, rng)
    rows.append(("physics aug shape", xa.shape == x.shape, str(xa.shape)))
    net = EmbNet(len(CLASS_NAMES))
    xb = torch.from_numpy(np.stack([x, xa])).float()
    logits, emb = net(xb, return_emb=True)
    loss = infonce(emb[:1], emb[1:])
    rows.append(("forward+infonce", torch.isfinite(logits).all().item() and torch.isfinite(loss).item(), f"emb={tuple(emb.shape)}"))
    from radarmind.llm.provider import LLMService

    rows.append(("ui_mode", hasattr(LLMService(), "ui_mode"), LLMService().ui_mode()))
    n_ok = 0
    for name, ok, detail in rows:
        print(("PASS" if ok else "FAIL"), name, detail)
        n_ok += int(bool(ok))
    print(f"passed {n_ok}/{len(rows)}")
    raise SystemExit(0 if n_ok == len(rows) else 1)


if __name__ == "__main__":
    main()
