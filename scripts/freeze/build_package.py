#!/usr/bin/env python3
"""Phase 6 freeze package. Read existing results only. Do not retrain."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN = ROOT / "results" / "frozen_2026_08"


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _csv(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8")))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("wb" if False else "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(xs):
    xs = [float(x) for x in xs]
    return float(statistics.mean(xs)) if xs else None


def verify_one(spec: dict) -> dict:
    import torch

    p = ROOT / spec["checkpoint"]
    out = {
        "artifact_id": spec["id"],
        "phase": spec["phase"],
        "dataset": spec["dataset"],
        "protocol": spec["protocol"],
        "model": spec["model"],
        "source_path": spec.get("source_path", spec["checkpoint"]),
        "checkpoint_path": spec["checkpoint"],
        "metrics_path": spec.get("metrics_path", ""),
        "sha256": "",
        "status": spec.get("status", "FROZEN"),
        "notes": spec.get("notes", ""),
        "exists": False,
        "bytes": 0,
        "loadable": False,
        "forward_ok": False,
    }
    if not p.is_file():
        out["status"] = spec.get("missing_status", "MISSING_OPTIONAL")
        out["notes"] = (out["notes"] + " checkpoint missing").strip()
        return out
    out["exists"] = True
    out["bytes"] = p.stat().st_size
    out["sha256"] = sha256(p)
    try:
        try:
            state = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(p, map_location="cpu")
        out["loadable"] = isinstance(state, dict)
    except Exception as e:  # noqa: BLE001
        out["notes"] = f"load fail: {type(e).__name__}"
        return out
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        kind = spec["kind"]
        if kind == "resnet18":
            from models.resnet18_spec import ResNet18Spec

            net = ResNet18Spec(spec["ncls"], 1)
            net.load_state_dict(state)
            y = net(torch.zeros(1, 1, 224, 224))
            out["forward_ok"] = bool(y.shape[-1] == spec["ncls"])
        elif kind == "pointnet_gru":
            from models.point_encoder import PointNetGRU

            net = PointNetGRU(spec["ncls"], in_ch=spec["in_ch"])
            net.load_state_dict(state)
            x = torch.zeros(1, 32, 64, spec["in_ch"])
            m = torch.ones(1, 32, 64, dtype=torch.bool)
            y = net(x, m)
            out["forward_ok"] = bool(y.shape[-1] == spec["ncls"])
        elif kind == "shared":
            sys.path.insert(0, str(ROOT / "scripts" / "cross_radar"))
            from train_cross_radar import EmbNet

            net = EmbNet(spec["ncls"])
            net.load_state_dict(state)
            y = net(torch.zeros(1, 1, 224, 224))
            if isinstance(y, tuple):
                y = y[0]
            out["forward_ok"] = bool(y.shape[-1] == spec["ncls"])
        else:
            out["forward_ok"] = out["loadable"]
    except Exception as e:  # noqa: BLE001
        out["notes"] = (out["notes"] + f" forward fail: {type(e).__name__}").strip()
    return out


def checkpoint_specs() -> list[dict]:
    specs = [
        {"id": "ci4r_resnet18_random", "phase": "1A", "dataset": "CI4R 77 GHz", "protocol": "random-record", "model": "ResNet18", "checkpoint": "results/phase1_ci4r/baselines/random_record/resnet18/best.pt", "metrics_path": "results/phase1_ci4r/baselines/random_record/resnet18/metrics.json", "kind": "resnet18", "ncls": 11, "in_ch": 1, "status": "FROZEN"},
        {"id": "ci4r_resnet18_prefix", "phase": "1A", "dataset": "CI4R 77 GHz", "protocol": "prefix-group-disjoint", "model": "ResNet18", "checkpoint": "results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/best.pt", "metrics_path": "results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/metrics.json", "kind": "resnet18", "ncls": 11, "in_ch": 1, "status": "FROZEN", "notes": "prefix-group-disjoint ≠ cross-subject"},
        {"id": "mri_si_gru_xyz", "phase": "1B", "dataset": "mRI", "protocol": "subject-independent", "model": "PointNet-GRU XYZ", "checkpoint": "results/phase1_pointcloud/baselines/mri_si_gru_xyz/best.pt", "metrics_path": "results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json", "kind": "pointnet_gru", "ncls": 13, "in_ch": 3, "status": "FROZEN"},
        {"id": "radhar_gru_xyzv", "phase": "1B", "dataset": "RadHAR", "protocol": "official Train/Test", "model": "PointNet-GRU XYZV", "checkpoint": "results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/best.pt", "metrics_path": "results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/metrics.json", "kind": "pointnet_gru", "ncls": 5, "in_ch": 4, "status": "FROZEN"},
        {"id": "phase3_M0", "phase": "3", "dataset": "CI4R official PNG 7-class", "protocol": "multi_radar_shared", "model": "M0", "checkpoint": "results/phase3_shared_motion/runs/M0_shared/best.pt", "metrics_path": "results/phase3_shared_motion/runs/M0_shared/metrics.json", "kind": "shared", "ncls": 7, "status": "FROZEN"},
        {"id": "phase3_M1", "phase": "3", "dataset": "CI4R official PNG 7-class", "protocol": "multi_radar_shared", "model": "M1 λ=0.1", "checkpoint": "results/phase3_shared_motion/runs/M1_l01/best.pt", "metrics_path": "results/phase3_shared_motion/runs/M1_l01/metrics.json", "kind": "shared", "ncls": 7, "status": "FROZEN", "notes": "canonical M1 is λ=0.1"},
        {"id": "phase3_M2", "phase": "3", "dataset": "CI4R official PNG 7-class", "protocol": "multi_radar_shared", "model": "M2", "checkpoint": "results/phase3_shared_motion/runs/M2_shared/best.pt", "metrics_path": "results/phase3_shared_motion/runs/M2_shared/metrics.json", "kind": "shared", "ncls": 7, "status": "FROZEN"},
    ]
    for act in ["pose_1", "pose_4", "walk", "T_pose", "pose_2", "pose_3", "pose_5", "pose_6", "pose_7", "pose_8", "pose_9", "pose_10", "free_form"]:
        specs.append(
            {
                "id": f"phase5_{act}",
                "phase": "5",
                "dataset": "mRI",
                "protocol": "LOAO + SI",
                "model": "PointNet-GRU 12-class",
                "checkpoint": f"results/phase5_open_set/folds/{act}/best.pt",
                "metrics_path": f"results/phase5_open_set/folds/{act}/eval.json",
                "kind": "pointnet_gru",
                "ncls": 12,
                "in_ch": 3,
                "status": "FROZEN",
            }
        )
    specs.append(
        {
            "id": "coral_77_to_10",
            "phase": "2B",
            "dataset": "CI4R official PNG",
            "protocol": "CORAL 77→10",
            "model": "CORAL",
            "checkpoint": "results/phase2_cross_radar/runs/coral_uda_77_to_10/best.pt",
            "metrics_path": "results/phase2_cross_radar/runs/coral_uda_77_to_10/metrics.json",
            "kind": "other",
            "ncls": 7,
            "status": "NEGATIVE_RESULT",
            "missing_status": "MISSING_OPTIONAL",
            "notes": "negative baseline; do not retune",
        }
    )
    return specs


def write_manifest(rows: list[dict]) -> None:
    fields = ["artifact_id", "phase", "dataset", "protocol", "model", "source_path", "checkpoint_path", "metrics_path", "sha256", "status", "notes", "exists", "bytes", "loadable", "forward_ok"]
    dest = FROZEN / "artifact_manifest.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (FROZEN / "checkpoint_verify.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def master_results() -> list[dict]:
    rows = []

    def add(**kw):
        base = {k: "" for k in (
            "task", "dataset", "representation", "protocol", "train_domain", "test_domain",
            "model", "features", "temporal_length", "accuracy", "macro_f1", "auroc", "aupr", "fpr95", "notes", "source_file",
        )}
        base.update(kw)
        rows.append(base)

    a = _j(ROOT / "results" / "phase1_ci4r" / "baselines" / "random_record" / "resnet18" / "metrics.json") or {}
    b = _j(ROOT / "results" / "phase1_ci4r" / "baselines" / "prefix_tail4_disjoint" / "resnet18" / "metrics.json") or {}
    add(task="within-dataset", dataset="CI4R 77 GHz", representation="micro-Doppler", protocol="random-record", train_domain="CI4R77", test_domain="CI4R77", model="ResNet18", accuracy=a.get("accuracy"), macro_f1=a.get("macro_f1"), source_file="results/phase1_ci4r/baselines/random_record/resnet18/metrics.json")
    add(task="within-dataset", dataset="CI4R 77 GHz", representation="micro-Doppler", protocol="prefix-group-disjoint", train_domain="CI4R77 prefix", test_domain="CI4R77 held groups", model="ResNet18", accuracy=b.get("accuracy"), macro_f1=b.get("macro_f1"), notes="not cross-subject", source_file="results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/metrics.json")
    p1b = _csv(ROOT / "results" / "phase1_pointcloud" / "training_runs.csv")
    for r in p1b:
        if r["dataset"] == "mRI" and r["split"] == "random_segment":
            add(task="within-dataset", dataset="mRI", representation="point cloud", protocol="random-segment", train_domain="mRI", test_domain="mRI", model=r["model"], features=r["features"], temporal_length=r["temporal_length"], accuracy=r["accuracy"], macro_f1=r["macro_f1"], source_file="results/phase1_pointcloud/training_runs.csv")
        if r["dataset"] == "mRI" and r["split"] == "subject_independent" and r["features"] == "XYZ" and r["temporal_length"] == "32":
            add(task="within-dataset", dataset="mRI", representation="point cloud", protocol="subject-independent", train_domain="mRI train subjects", test_domain="mRI test subjects", model="pointnet_gru", features="XYZ", temporal_length=32, accuracy=r["accuracy"], macro_f1=r["macro_f1"], source_file="results/phase1_pointcloud/training_runs.csv")
        if r["dataset"] == "RadHAR" and r["features"] == "XYZV" and r["model"] == "pointnet_gru":
            add(task="within-dataset", dataset="RadHAR", representation="point cloud", protocol="official Train/Test", train_domain="RadHAR Train", test_domain="RadHAR Test", model="pointnet_gru", features="XYZV", temporal_length=60, accuracy=r["accuracy"], macro_f1=r["macro_f1"], source_file="results/phase1_pointcloud/training_runs.csv")
        if r["dataset"] == "mRI" and r["temporal_length"] == "1":
            add(task="temporal", dataset="mRI", representation="point cloud", protocol="subject-independent", model=r["model"], features=r["features"], temporal_length=1, accuracy=r["accuracy"], macro_f1=r["macro_f1"], source_file="results/phase1_pointcloud/training_runs.csv")
        if r["dataset"] == "mRI" and r["features"] == "XYZVI" and r["temporal_length"] == "32" and r["split"] == "subject_independent":
            add(task="temporal", dataset="mRI", representation="point cloud", protocol="subject-independent", model="pointnet_gru", features="XYZVI", temporal_length=32, accuracy=r["accuracy"], macro_f1=r["macro_f1"], source_file="results/phase1_pointcloud/training_runs.csv")
        if r["dataset"] == "RadHAR" and r["temporal_length"] == "1":
            add(task="temporal", dataset="RadHAR", representation="point cloud", protocol="official Train/Test", model=r["model"], features=r["features"], temporal_length=1, accuracy=r["accuracy"], macro_f1=r["macro_f1"], source_file="results/phase1_pointcloud/training_runs.csv")
        if r["dataset"] == "RadHAR" and r["features"] == "XYZV" and r["temporal_length"] == "60":
            add(task="temporal", dataset="RadHAR", representation="point cloud", protocol="official Train/Test", model="pointnet_gru", features="XYZV", temporal_length=60, accuracy=r["accuracy"], macro_f1=r["macro_f1"], source_file="results/phase1_pointcloud/training_runs.csv")
        if r["dataset"] in ("mRI", "RadHAR") and r["model"] == "pointnet_gru" and r["split"] in ("subject_independent", "official_train_test"):
            add(task="feature-ablation", dataset=r["dataset"], representation="point cloud", protocol=r["split"], model="pointnet_gru", features=r["features"], temporal_length=r["temporal_length"], accuracy=r["accuracy"], macro_f1=r["macro_f1"], source_file="results/phase1_pointcloud/training_runs.csv")
    p2 = _csv(ROOT / "results" / "phase2_cross_radar" / "training_runs.csv")
    for r in p2:
        task = "cross-radar" if r["train_radar"] != r["test_radar"] else "within-radar"
        if r["method"] == "coral_uda":
            task = "coral"
        add(task=task, dataset="CI4R official PNG 7-class", representation="official PNG", protocol=r["protocol"], train_domain=r["train_radar"], test_domain=r["test_radar"], model=r["method"], accuracy=r["accuracy"], macro_f1=r["macro_f1"], notes=r.get("note", ""), source_file="results/phase2_cross_radar/training_runs.csv")
    p3 = _csv(ROOT / "results" / "phase3_shared_motion" / "training_runs.csv")
    for r in p3:
        if r["protocol"] == "multi_radar_shared":
            add(task="shared-motion", dataset="CI4R 10/24/77", representation="official PNG", protocol="Task A", train_domain=r["train_radars"], test_domain=r["test_radar"], model=r["model"], accuracy=r["activity_accuracy"], macro_f1=r["macro_f1"], notes=r.get("note", ""), source_file="results/phase3_shared_motion/training_runs.csv")
        else:
            add(task="loro", dataset="CI4R 10/24/77", representation="official PNG", protocol="leave-one-radar-out", train_domain=r["train_radars"], test_domain=r["test_radar"], model=r["model"], accuracy=r["activity_accuracy"], macro_f1=r["macro_f1"], source_file="results/phase3_shared_motion/training_runs.csv")
    mets = _csv(ROOT / "results" / "phase3_shared_motion" / "representation_metrics.csv")
    for r in mets:
        if r["protocol"] != "multi_radar_shared":
            continue
        add(task="activity-probe", dataset="CI4R 10/24/77", protocol="linear probe", model=r["model"], accuracy=r["activity_probe_accuracy"], macro_f1=r["activity_probe_macro_f1"], source_file="results/phase3_shared_motion/representation_metrics.csv")
        add(task="radar-id-probe", dataset="CI4R 10/24/77", protocol="linear probe", model=r["model"], accuracy=r["radar_probe_accuracy"], macro_f1=r["radar_probe_macro_f1"], notes="1.000 = radar identity still readable", source_file="results/phase3_shared_motion/representation_metrics.csv")
        add(task="retrieval", dataset="CI4R 10/24/77", protocol="cross-radar retrieval", model=r["model"], accuracy=r["retrieval_action_top1"], notes=f"exact_top1={r['retrieval_exact_top1']}", source_file="results/phase3_shared_motion/representation_metrics.csv")
    cr = _csv(ROOT / "results" / "phase4_continuous" / "continuous_runs.csv")
    sg = _csv(ROOT / "results" / "phase4_continuous" / "segment_metrics.csv")
    bd = _csv(ROOT / "results" / "phase4_continuous" / "boundary_metrics.csv")
    for mode in ("raw", "isolated_flip", "confidence_aware"):
        wacc = mean([r["window_accuracy"] for r in cr if r["mode"] == mode])
        wf1 = mean([r["window_macro_f1"] for r in cr if r["mode"] == mode])
        sf1 = mean([r["segment_f1"] for r in sg if r["mode"] == mode])
        bmed = mean([r["median_frames"] for r in bd if r["mode"] == mode])
        add(task="continuous", dataset="mRI labeled-interval", representation="point cloud", protocol=mode, model="mri_si_gru_xyz", accuracy=wacc, macro_f1=wf1, notes=f"window metrics; segment_f1={sf1}; median_boundary_frames={bmed}", source_file="results/phase4_continuous/continuous_runs.csv")
    ev = _j(ROOT / "results" / "phase5_open_set" / "eval_summary.json") or []
    for method in ("MSP", "Energy", "Prototype"):
        auroc = [next(m for m in r["methods"] if m["method"] == method)["AUROC"] for r in ev]
        aupr = [next(m for m in r["methods"] if m["method"] == method)["AUPR"] for r in ev]
        fpr = [next(m for m in r["methods"] if m["method"] == method)["FPR95"] for r in ev]
        add(task="open-set", dataset="mRI", protocol="LOAO 13-fold mean", model=method, auroc=mean(auroc), aupr=mean(aupr), fpr95=mean(fpr), notes="mean over 13 held-out actions; SI subjects", source_file="results/phase5_open_set/eval_summary.json")
    dest = FROZEN / "master_results.csv"
    fields = list(rows[0].keys()) if rows else []
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def write_findings() -> int:
    findings = [
        {"finding_id": "F1", "claim": "Less-strict evaluation can overestimate generalization. CI4R random-record Acc=0.742 vs prefix-group-disjoint Acc=0.448. The latter is an acquisition-group split, not cross-subject. True cross-subject evidence is mRI SI Acc=0.478 vs random-segment Acc=0.511.", "evidence": "phase1_ci4r resnet18 metrics; phase1_pointcloud training_runs.csv", "supporting_experiments": ["phase1a_random", "phase1a_prefix", "phase1b_mri_si", "phase1b_mri_random"], "strength": "STRONG", "limitations": "CI4R has no reliable official subject ID. Single seed."},
        {"finding_id": "F2", "claim": "Temporal modeling matters for point-cloud radar HAR. mRI SI XYZVI T=1 Acc=0.254 vs T=32 Acc=0.465. RadHAR official XYZVI T=1 Acc=0.607 vs XYZV T=60 Acc=0.949.", "evidence": "training_runs.csv temporal rows", "supporting_experiments": ["mri_si_pool_t1", "mri_si_gru_xyzvi", "radhar_pool_t1", "radhar_gru_xyzv"], "strength": "STRONG", "limitations": "T=1 uses PointNetPool, T>1 uses PointNet-GRU; architecture is not identical."},
        {"finding_id": "F3", "claim": "Velocity contribution is dataset-dependent. mRI SI XYZ Acc=0.478 ≥ XYZV 0.471 ≥ XYZVI 0.465. RadHAR XYZV Acc=0.949 >> XYZ 0.868.", "evidence": "training_runs.csv feature ablation", "supporting_experiments": ["mri_si_gru_xyz/v/vi", "radhar_gru_xyz/v/vi"], "strength": "STRONG", "limitations": "Single seed. RadHAR official split is not subject-independent."},
        {"finding_id": "F4", "claim": "Cross-radar generalization is much harder than within-radar recognition on CI4R official PNG 7-class. Within 10/24/77 Acc=0.843/0.898/0.922. Zero-shot transfers drop to 0.039–0.373.", "evidence": "phase2_cross_radar/training_runs.csv", "supporting_experiments": ["within_*", "cross_*"], "strength": "STRONG", "limitations": "Official PNG; 327 agreed triples. Single seed."},
        {"finding_id": "F5", "claim": "24 GHz shows particularly severe transfer difficulty. 24→77 Acc=0.039, 10→24 Acc=0.143, 77→24 Acc=0.163. 24 GHz official PNG native size is 128×128 vs much larger 10/77 images, a possible representation confound.", "evidence": "phase2 training_runs; Phase 3 report on PNG sizes", "supporting_experiments": ["cross_24_to_77", "cross_10_to_24", "cross_77_to_24"], "strength": "MODERATE", "limitations": "Confound between sensor physics and PNG preprocessing is not isolated."},
        {"finding_id": "F6", "claim": "Simple CORAL alignment does not solve cross-radar generalization. 77→10 source-only Acc=0.275, CORAL Acc=0.196.", "evidence": "phase2 coral_uda_77_to_10 vs cross_77_to_10", "supporting_experiments": ["cross_77_to_10", "coral_uda_77_to_10"], "strength": "STRONG", "limitations": "One CORAL setting, not a search. Negative result only for this baseline."},
        {"finding_id": "F7", "claim": "Paired cross-radar contrastive learning improves the multi-radar shared task and retrieval. Task A Acc M0=0.769 → M1=0.932. Action retrieval top-1 0.718 → 0.935. This is a shared-task result, not leave-one-radar-out success.", "evidence": "phase3 training_runs + representation_metrics", "supporting_experiments": ["M0_shared", "M1_l01"], "strength": "STRONG", "limitations": "All three radars seen in Task A training. Does not imply sensor invariance."},
        {"finding_id": "F8", "claim": "Current embeddings remain strongly radar-identifiable. Radar-ID linear probe = 1.000 for M0, M1, and M2.", "evidence": "representation_metrics.csv radar_probe_accuracy", "supporting_experiments": ["M0", "M1", "M2 probes"], "strength": "STRONG", "limitations": "Linear probe only."},
        {"finding_id": "F9", "claim": "Shared Motion v0.1 is not sensor-invariant. F7 and F8 together: activity/retrieval can improve while radar identity remains fully readable. LORO M2 does not beat M0.", "evidence": "Radar-ID=1.0; LORO rows in training_runs.csv", "supporting_experiments": ["probes", "LORO M0/M2"], "strength": "STRONG", "limitations": "v0.1 only. No adversarial training in this freeze."},
        {"finding_id": "F10", "claim": "Labeled-interval continuous classification works at window level (confidence-aware Acc=0.494 / F1=0.419) but segment boundaries remain poor (segment F1=0.152, mean of median boundary error 190.8 frames).", "evidence": "phase4 continuous_runs / segment_metrics / boundary_metrics", "supporting_experiments": ["phase4 test subjects 3/10/20"], "strength": "STRONG", "limitations": "Not full-stream. UNLABELED intervals have no radar inference."},
        {"finding_id": "F11", "claim": "Closed-set confidence is a poor unknown-motion detector. 13-fold MSP AUROC 0.476 ± 0.156, FPR95 0.903 ± 0.138.", "evidence": "phase5 eval_summary.json", "supporting_experiments": ["LOAO MSP"], "strength": "STRONG", "limitations": "Threshold from val-known 5th percentile. Single seed per fold."},
        {"finding_id": "F12", "claim": "Embedding prototype distance gives a more useful open-set score than MSP (AUROC 0.628 ± 0.125) but FPR95 remains high (0.791). This is open-set recognition, not open-world or open-vocabulary.", "evidence": "phase5 13-fold Prototype vs MSP/Energy", "supporting_experiments": ["LOAO Prototype"], "strength": "MODERATE", "limitations": "Insufficient for open-world. pose_* held-outs remain hard."},
    ]
    (FROZEN / "scientific_findings.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(findings)


def write_negatives() -> None:
    text = """# Negative results (keep)

这些结果保留在冻结包里，不删、不改数字。

## CORAL 77→10

- Attempted: unofficial CORAL UDA, source 77 GHz, unlabeled target train 10 GHz.
- Happened: Acc 0.275 (source-only) → 0.196 (CORAL).
- Implies: this CORAL setting did not close the radar gap.
- Does not imply: all UDA methods fail, or 10 GHz is unlearnable.

## M2 physics augmentation on LORO

- Attempted: Shared Motion M2 (contrastive + physics aug) vs M0 on leave-one-radar-out.
- Happened: 10+24→77 Acc 0.347 → 0.204 (worse). 10+77→24 both 0.143. 24+77→10 both 0.367.
- Implies: current physics aug did not help unseen-radar transfer.
- Does not imply: contrastive learning is useless on the shared (all-radar) task — Task A M1 still improved.

## Radar-ID Probe

- Attempted: linear probe of radar identity on M0/M1/M2 embeddings.
- Happened: 1.000 / 1.000 / 1.000.
- Implies: embeddings still contain a near-perfect radar signature.
- Does not imply: activity information is absent (activity probe is high).

## 24 GHz transfer

- Attempted: zero-shot to/from 24 GHz official PNG.
- Happened: 24→77 Acc=0.039; several 24 GHz transfers near floor.
- Implies: 24 GHz is the hardest band under this PNG protocol.
- Does not imply: 24 GHz radar cannot do HAR within-band (within-24 Acc=0.898).

## Continuous segmentation

- Attempted: labeled-interval replay + smoothing, no learned segmenter.
- Happened: window Acc≈0.494, segment F1=0.152, median boundary ~191 frames.
- Implies: window HAR ≠ usable action segmentation.
- Does not imply: continuous understanding is impossible with raw CSV or a dedicated segmenter.

## Open-set MSP

- Attempted: max softmax as unknown score, LOAO + SI.
- Happened: mean AUROC 0.476, FPR95 0.903.
- Implies: low closed-set confidence is not unknown detection.
- Does not imply: open-set is impossible; Prototype is better but still not open-world.
"""
    (FROZEN / "negative_results.md").write_text(text, encoding="utf-8")


def write_limitations() -> None:
    (FROZEN / "limitations.md").write_text(
        """# Limitations

1. CI4R 没有可靠 official subject ID。
2. prefix-group-disjoint 不能称 cross-subject。真正 cross-subject 来自 mRI。
3. mRI continuous replay 只覆盖 labeled intervals。
4. 本机没有 mRI raw CSV，因此没有完整 UNLABELED full-stream inference。
5. RadHAR 没有 subject metadata，official split 不能解释成 cross-subject。
6. Cross-environment 当前不可做：没有可用的环境标签协议。
7. CI4R 24 GHz official PNG 原生分辨率与 10/77 明显不同，可能形成 representation confound。
8. Shared Motion Radar-ID probe 仍为 1.0，v0.1 不是 sensor-invariant。
9. Open-set ≠ open-vocabulary ≠ open-world。Phase 5 只做 open-set recognition。
10. 当前实验主要单 seed，不是最终论文统计版本。
""",
        encoding="utf-8",
    )


def write_status() -> None:
    (FROZEN / "project_status.md").write_text(
        """# Project status

## DONE

- CI4R 77 GHz HAR（random / prefix-group）。原因：metrics 与 checkpoint 可加载。
- mRI / RadHAR point-cloud HAR。原因：training_runs.csv 与 SI/official checkpoint 在。
- mRI cross-subject。原因：官方 subject id + SI split。
- CI4R cross-radar within / zero-shot / CORAL。原因：Phase 2B training_runs.csv。
- Shared Motion v0.1 Task A + probes + LORO。原因：Phase 3 数字冻结。
- Labeled-interval continuous inference。原因：Phase 4 CSV；不是 full-stream。
- Open-set LOAO 13-fold。原因：Phase 5 eval_summary。
- Replay Application + Timeline + DeepSeek/fallback。

## PARTIAL

- sensor-invariant representation。原因：Radar-ID probe=1.0。
- continuous full-stream segmentation。原因：UNLABELED 未做雷达推理。
- open-world understanding。原因：只有 open-set MVP。
- open-vocabulary recognition。原因：未做。

## PENDING

- cross-environment。原因：无环境协议与数据划分。
- MM-Fi / MiliPoint / FAU。原因：本阶段禁止新数据。
- Live Radar。原因：未接硬件。

## BLOCKED_BY_DATA

- true full-stream mRI（含 UNLABELED）。原因：本机无 raw CSV。
- CI4R official cross-subject。原因：无可靠 subject ID。
- RadHAR cross-subject。原因：无 subject metadata。
""",
        encoding="utf-8",
    )


def write_backlog() -> None:
    (FROZEN / "research_backlog.md").write_text(
        """# Research backlog（不在本阶段执行）

## P0 真正 sensor-invariant representation

- Question: 能否在 Radar-ID probe 明显低于 1.0 的同时保持活动分类？
- Data: 现有 CI4R official PNG triples；不要先上新数据集。
- Code: 表征约束 / 规范化；不是再调 CORAL。
- Experiment: Task A + Radar-ID + LORO。
- Success: Radar-ID 下降且 LORO 不低于 M0。

## P1 24 GHz representation normalization

- Question: 24 GHz 难迁移是预处理还是传感器？
- Data: 同一批 327 triples；核对 native PNG 尺寸。
- Code: 统一 resample / 频谱审计。
- Experiment: 重做 within-24 与 24↔77，不改标签。
- Success: 能分离 confound 与真实频段差。

## P2 Cross-environment

- Question: 同一动作换房间/布置后还能否保持？
- Data: 需要带 environment 标签的采集。
- Code: environment split。
- Experiment: within-env vs cross-env。
- Success: 有可引用的环境划分和数字。

## P3 Full mRI raw continuous inference

- Question: UNLABELED 区间上模型会输出什么？
- Data: 原始 mRI radar CSV。
- Code: 全流滑窗。
- Experiment: labeled + unlabeled 时间线。
- Success: 不再只是 labeled-interval replay。

## P4 Open-set improvement

- Question: 相似 pose_* 能否被稳定拒识？
- Data: 现有 mRI LOAO。
- Code: 更好的 embedding / score；不改 test 调阈值。
- Experiment: 13-fold mean AUROC / FPR95。
- Success: FPR95 明显低于当前 Prototype 0.791，且不偷看 unseen。

## P5 Open-vocabulary motion representation

- Question: 能否用文本或属性描述未见过的动作，而不是只输出 UNKNOWN？
- Data: 带文本/属性的动作集。
- Code: 对齐层。
- Experiment: unseen-class 名称检索。
- Success: 才可称 open-vocabulary，仍不是 open-world。

## P6 Live Radar

- Question: 在线雷达流上闭环是否稳定？
- Data: 实时设备。
- Code: 采集适配。
- Experiment: 现场延迟与掉线。
- Success: 不依赖预缓存窗口。
""",
        encoding="utf-8",
    )


def main() -> None:
    FROZEN.mkdir(parents=True, exist_ok=True)
    print("[verify] checkpoints", flush=True)
    verified = [verify_one(s) for s in checkpoint_specs()]
    write_manifest(verified)
    print("[master] results", flush=True)
    rows = master_results()
    n_f = write_findings()
    write_negatives()
    write_limitations()
    write_status()
    write_backlog()
    core = [v for v in verified if v["artifact_id"].startswith(("ci4r_", "mri_", "radhar_", "phase3_", "phase5_"))]
    ok = sum(1 for v in core if v["exists"] and v["loadable"])
    summary = {
        "verified_core_loadable": ok,
        "verified_core_total": len(core),
        "master_rows": len(rows),
        "findings": n_f,
    }
    (FROZEN / "freeze_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
