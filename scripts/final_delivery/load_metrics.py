"""Load metric_snapshot.json plus extra method fields from existing result files."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def _j(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load(root: Path) -> dict:
    snap = _j(root / "results/paper_package_v2/metric_snapshot.json")
    m = {k: rec["value"] for k, rec in snap["metrics"].items()}
    d = {k: rec["display"] for k, rec in snap["metrics"].items()}
    src = {k: rec["source_file"] for k, rec in snap["metrics"].items()}

    pair = _j(root / "results/phase2_cross_radar/pairing_stats.json")
    m["N_CONFLICT"] = int(pair["paired_10_24_77_action_conflict"])
    d["N_CONFLICT"] = str(m["N_CONFLICT"])
    src["N_CONFLICT"] = "results/phase2_cross_radar/pairing_stats.json"

    p8 = _j(root / "results/phase8_physical_representation/answers.json")
    m["TSWEEP_MS"] = float(p8["header_24ghz"]["tsweep_ms"])
    d["TSWEEP_MS"] = str(int(m["TSWEEP_MS"])) if m["TSWEEP_MS"] == int(m["TSWEEP_MS"]) else f"{m['TSWEEP_MS']:g}"
    src["TSWEEP_MS"] = "results/phase8_physical_representation/answers.json header_24ghz.tsweep_ms"
    m["NTS_24"] = int(p8["header_24ghz"]["nts"])
    d["NTS_24"] = str(m["NTS_24"])
    src["NTS_24"] = "results/phase8_physical_representation/answers.json header_24ghz.nts"
    m["BW_HZ"] = float(p8["header_24ghz"]["bw_hz"])
    d["BW_GHZ"] = f"{m['BW_HZ'] / 1e9:.1f}"
    src["BW_GHZ"] = "results/phase8_physical_representation/answers.json header_24ghz.bw_hz"

    grid = _j(root / "results/phase8_physical_representation/common_grid.json")
    m["FD_MIN"] = float(grid["common_fd_min"])
    m["FD_MAX"] = float(grid["common_fd_max"])
    d["FD_MIN"] = f"{m['FD_MIN']:.0f}"
    d["FD_MAX"] = f"{m['FD_MAX']:.3f}"
    src["FD_MIN"] = "results/phase8_physical_representation/common_grid.json common_fd_min"
    src["FD_MAX"] = "results/phase8_physical_representation/common_grid.json common_fd_max"

    chain10 = (root / "results/phase8_physical_representation/preprocessing_chain_10ghz.md").read_text(encoding="utf-8")
    m["N_FLOAT_10"] = int(re.search(r"\*\*(\d+) float\*\*", chain10).group(1))
    m["N_IQ_10"] = int(re.search(r"每帧 \*\*(\d+)\*\*", chain10).group(1))
    d["N_FLOAT_10"] = str(m["N_FLOAT_10"])
    d["N_IQ_10"] = str(m["N_IQ_10"])
    src["N_FLOAT_10"] = "results/phase8_physical_representation/preprocessing_chain_10ghz.md"
    src["N_IQ_10"] = "results/phase8_physical_representation/preprocessing_chain_10ghz.md"

    stft = (root / "results/phase10_velocity_calibration/stft_equivalence_audit.md").read_text(encoding="utf-8")
    win = re.search(r"window \(seconds\)\s*\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*([0-9.]+)\*\*", stft)
    hop = re.search(r"hop \(seconds\)\s*\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*([0-9.]+)\*\*", stft)
    if not win or not hop:
        raise RuntimeError("cannot parse STFT windows from stft_equivalence_audit.md")
    m["STFT_24_WIN"] = float(win.group(1))
    m["STFT_77_WIN"] = float(win.group(2))
    m["STFT_24_HOP"] = float(hop.group(1))
    m["STFT_77_HOP"] = float(hop.group(2))
    d["STFT_24_WIN"] = f"{m['STFT_24_WIN']:.3f}"
    d["STFT_77_WIN"] = f"{m['STFT_77_WIN']:.3f}"
    d["STFT_24_HOP"] = f"{m['STFT_24_HOP']:.3f}"
    d["STFT_77_HOP"] = f"{m['STFT_77_HOP']:.3f}"
    src["STFT_24_WIN"] = "results/phase10_velocity_calibration/stft_equivalence_audit.md"
    src["STFT_77_WIN"] = src["STFT_24_WIN"]
    src["STFT_24_HOP"] = src["STFT_24_WIN"]
    src["STFT_77_HOP"] = src["STFT_24_WIN"]

    u1 = (root / "results/phase10_unified_measurement/unified_stft_definition.md").read_text(encoding="utf-8")
    m["U1_WIN"] = float(re.search(r"target window duration:\s*\*\*([0-9.]+) s\*\*", u1).group(1))
    m["U1_OVERLAP"] = float(re.search(r"overlap fraction:\s*\*\*([0-9.]+)\*\*", u1).group(1))
    m["U1_HOP"] = float(re.search(r"24 hop:.*\*\*([0-9.]+) s\*\*", u1).group(1))
    m["U1_NFFT"] = int(re.search(r"nfft:\s*\*\*(\d+)\*\*", u1).group(1))
    d["U1_WIN"] = f"{m['U1_WIN']:.3f}"
    d["U1_OVERLAP"] = f"{int(m['U1_OVERLAP'] * 100)}"
    d["U1_HOP"] = f"{m['U1_HOP']:.3f}"
    d["U1_NFFT"] = str(m["U1_NFFT"])
    src["U1_WIN"] = "results/phase10_unified_measurement/unified_stft_definition.md"

    p9 = _j(root / "results/phase9_radarbridge_v2/answers.json")
    for k, field in (("PAIR_A0", "pair_A0"), ("PAIR_A1", "pair_A1"), ("PAIR_A2", "pair_A2")):
        m[k] = float(p9[field])
        d[k] = f"{m[k]:.3f}"
        src[k] = f"results/phase9_radarbridge_v2/answers.json {field}"

    p10e = _j(root / "results/phase10_hardware_measurement/answers.json")
    m["VEL_RID_M0"] = float(p10e["M0"]["vel_rid"])
    d["VEL_RID_M0"] = f"{m['VEL_RID_M0']:.3f}"
    src["VEL_RID_M0"] = "results/phase10_hardware_measurement/answers.json M0.vel_rid"

    p10f = _j(root / "results/phase10_motion_conditioned/answers.json")
    m["CAL_VEL_RID"] = float(p10f["calibration"]["before"]["vel_rid"])
    d["CAL_VEL_RID"] = f"{m['CAL_VEL_RID']:.3f}"
    src["CAL_VEL_RID"] = "results/phase10_motion_conditioned/answers.json calibration.before.vel_rid"

    act = root / "results/phase10_conditional_2d_measurement/activity_preservation.csv"
    with act.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["direction"] == "24to77" and row["item"] == "true_target":
                m["B2_ACT_TRUE"] = float(row["activity"])
                d["B2_ACT_TRUE"] = f"{m['B2_ACT_TRUE']:.3f}"
                src["B2_ACT_TRUE"] = "results/phase10_conditional_2d_measurement/activity_preservation.csv 24to77 true_target"

    # already in snapshot; aliases used by writers
    return {"raw": m, "display": d, "source": src, "snapshot": snap}


def D(pack: dict, key: str) -> str:
    if key not in pack["display"]:
        raise KeyError(f"missing metric {key}")
    return pack["display"][key]


def V(pack: dict, key: str):
    if key not in pack["raw"]:
        raise KeyError(f"missing metric {key}")
    return pack["raw"][key]


def pct(pack: dict, key: str) -> str:
    x = float(V(pack, key)) * 100
    if abs(x - round(x)) < 1e-6:
        return f"{int(round(x))}%"
    return f"{x:.1f}%"
