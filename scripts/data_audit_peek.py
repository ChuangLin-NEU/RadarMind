import json
import pickle
from pathlib import Path

import numpy as np

import os

p = Path(os.environ.get("RADARMIND_MILIPOINT_PKL", ""))
obj = pickle.load(p.open("rb"))
d = obj[0]
print("keys", list(d.keys()))
for k, v in d.items():
    print(
        k,
        type(v).__name__,
        getattr(v, "shape", None),
        getattr(v, "dtype", None),
        (len(v) if hasattr(v, "__len__") and not hasattr(v, "shape") else None),
        str(v)[:120],
    )
print("n", len(obj))

root = Path(os.environ.get("RADARMIND_MRI_RELEASE", os.environ.get("RADARMIND_MRI_ROOT", "")))
radar = list(root.rglob("raw_data/radar/*.csv"))
print("n_radar_csv", len(radar))
if radar:
    t = radar[0].read_text(encoding="utf-8", errors="replace").splitlines()
    print("radar", radar[0])
    print("nlines", len(t))
    print("hdr", t[0][:240])
    print("l1", t[1][:240] if len(t) > 1 else "")

vl = root / "dataset_release/raw_data/videolabels/subject1.json"
j = json.loads(vl.read_text(encoding="utf-8"))
print("subj", j.get("subject"))
labs = j.get("labels")
print("labels type", type(labs).__name__, "len", len(labs) if hasattr(labs, "__len__") else None)
if isinstance(labs, list) and labs:
    print("lab0", labs[0])
elif isinstance(labs, dict):
    k0 = list(labs.keys())[0]
    print("lab0k", k0, labs[k0])
