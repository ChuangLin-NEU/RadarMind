#!/usr/bin/env python3
"""Paper-ready tables from result files. No handwritten numbers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_package" / "tables"


def _j(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).is_file() else {}


def _csv(p):
    p = Path(p)
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.is_file() else []


def _f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def dump(name: str, caption: str, headers: list[str], rows: list[list]):
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# {name}", "", caption, "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    cols = " & ".join(headers) + r" \\"
    body = "\n".join(" & ".join(str(x) for x in r) + r" \\" for r in rows)
    tex = f"% {caption}\n\\begin{{tabular}}{{{'l' * len(headers)}}}\n{cols}\n\\hline\n{body}\n\\end{{tabular}}\n"
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    p1a_r = _j(ROOT / "results/phase1_ci4r/baselines/random_record/resnet18/metrics.json")
    p1a_p = _j(ROOT / "results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/metrics.json")
    p1b = _csv(ROOT / "results/phase1_pointcloud/training_runs.csv")
    p2 = _csv(ROOT / "results/phase2_cross_radar/training_runs.csv")
    p3 = _csv(ROOT / "results/phase3_shared_motion/training_runs.csv")
    mets = _csv(ROOT / "results/phase3_shared_motion/representation_metrics.csv")
    ev = _j(ROOT / "results/phase5_open_set/eval_summary.json") or []

    def p1(ds, split, feat=None, t=None, model=None):
        for r in p1b:
            if r["dataset"] != ds or r["split"] != split:
                continue
            if feat and r["features"] != feat:
                continue
            if t and str(r["temporal_length"]) != str(t):
                continue
            if model and r["model"] != model:
                continue
            return r
        return {}

    dump(
        "table1_datasets",
        "Table 1. Datasets and radar representations used in this freeze. Counts from existing caches/splits.",
        ["Dataset", "Radar", "Representation", "Protocol used", "Notes"],
        [
            ["CI4R", "77 GHz IQ", "micro-Doppler spectrogram", "random-record; prefix-group-disjoint", "no reliable official subject ID"],
            ["CI4R official PNG", "10 / 24 / 77 GHz", "official PNG", "within; zero-shot; CORAL; shared; LORO", "327 agreed 7-class triples"],
            ["mRI", "point cloud", "sampled T×P×C", "random-segment; subject-independent; LOAO; labeled-interval", "20 subjects, 13 actions"],
            ["RadHAR", "point cloud", "sampled T×P×C", "official Train/Test", "no subject metadata"],
        ],
    )
    dump(
        "table2_within",
        "Table 2. Within-dataset HAR. Numbers from metrics.json / training_runs.csv.",
        ["Dataset", "Protocol", "Model", "Acc", "Macro-F1", "Source"],
        [
            ["CI4R 77 GHz", "random-record", "ResNet18", _f(p1a_r.get("accuracy")), _f(p1a_r.get("macro_f1")), "phase1_ci4r"],
            ["CI4R 77 GHz", "prefix-group-disjoint", "ResNet18", _f(p1a_p.get("accuracy")), _f(p1a_p.get("macro_f1")), "phase1_ci4r"],
            ["mRI", "random-segment", "PointNet-GRU XYZVI", _f(p1("mRI", "random_segment").get("accuracy")), _f(p1("mRI", "random_segment").get("macro_f1")), "training_runs"],
            ["mRI", "subject-independent", "PointNet-GRU XYZ", _f(p1("mRI", "subject_independent", "XYZ", 32).get("accuracy")), _f(p1("mRI", "subject_independent", "XYZ", 32).get("macro_f1")), "training_runs"],
            ["RadHAR", "official Train/Test", "PointNet-GRU XYZV", _f(p1("RadHAR", "official_train_test", "XYZV", 60).get("accuracy")), _f(p1("RadHAR", "official_train_test", "XYZV", 60).get("macro_f1")), "training_runs"],
        ],
    )
    dump(
        "table3_cross_subject",
        "Table 3. Cross-subject / acquisition-group. Only mRI is true cross-subject.",
        ["Dataset", "Split A", "Acc A", "Split B", "Acc B", "Note"],
        [
            ["CI4R 77 GHz", "random-record", _f(p1a_r.get("accuracy")), "prefix-group-disjoint", _f(p1a_p.get("accuracy")), "B ≠ cross-subject"],
            ["mRI", "random-segment", _f(p1("mRI", "random_segment").get("accuracy")), "subject-independent", _f(p1("mRI", "subject_independent", "XYZ", 32).get("accuracy")), "official subject IDs"],
            ["RadHAR", "official Train/Test", _f(p1("RadHAR", "official_train_test", "XYZV", 60).get("accuracy")), "—", "—", "no subject metadata"],
        ],
    )
    dump(
        "table4_ablation",
        "Table 4. Temporal and feature ablation from training_runs.csv.",
        ["Dataset", "Protocol", "Features", "T", "Acc", "Macro-F1"],
        [
            ["mRI", "SI", "XYZVI", "1", _f(p1("mRI", "subject_independent", "XYZVI", 1).get("accuracy")), _f(p1("mRI", "subject_independent", "XYZVI", 1).get("macro_f1"))],
            ["mRI", "SI", "XYZ", "32", _f(p1("mRI", "subject_independent", "XYZ", 32).get("accuracy")), _f(p1("mRI", "subject_independent", "XYZ", 32).get("macro_f1"))],
            ["mRI", "SI", "XYZV", "32", _f(p1("mRI", "subject_independent", "XYZV", 32).get("accuracy")), _f(p1("mRI", "subject_independent", "XYZV", 32).get("macro_f1"))],
            ["mRI", "SI", "XYZVI", "32", _f(p1("mRI", "subject_independent", "XYZVI", 32).get("accuracy")), _f(p1("mRI", "subject_independent", "XYZVI", 32).get("macro_f1"))],
            ["RadHAR", "official", "XYZVI", "1", _f(p1("RadHAR", "official_train_test", "XYZVI", 1).get("accuracy")), _f(p1("RadHAR", "official_train_test", "XYZVI", 1).get("macro_f1"))],
            ["RadHAR", "official", "XYZ", "60", _f(p1("RadHAR", "official_train_test", "XYZ", 60).get("accuracy")), _f(p1("RadHAR", "official_train_test", "XYZ", 60).get("macro_f1"))],
            ["RadHAR", "official", "XYZV", "60", _f(p1("RadHAR", "official_train_test", "XYZV", 60).get("accuracy")), _f(p1("RadHAR", "official_train_test", "XYZV", 60).get("macro_f1"))],
            ["RadHAR", "official", "XYZVI", "60", _f(p1("RadHAR", "official_train_test", "XYZVI", 60).get("accuracy")), _f(p1("RadHAR", "official_train_test", "XYZVI", 60).get("macro_f1"))],
        ],
    )
    xr = []
    for r in p2:
        if r["method"] != "source_only":
            continue
        xr.append([r["train_radar"], r["test_radar"], _f(r["accuracy"]), _f(r["macro_f1"])])
    dump("table5_cross_radar", "Table 5. Cross-radar matrix (source-only). CORAL 77→10 Acc=0.196 is a negative result.", ["Train", "Test", "Acc", "Macro-F1"], xr)

    sm = []
    for r in p3:
        if r["protocol"] == "multi_radar_shared":
            sm.append([r["model"], "Task A", _f(r["activity_accuracy"]), _f(r["macro_f1"]), "classification"])
    for r in mets:
        if r["protocol"] == "multi_radar_shared":
            sm.append([r["model"], "Activity probe", _f(r["activity_probe_accuracy"]), _f(r["activity_probe_macro_f1"]), "linear"])
            sm.append([r["model"], "Radar-ID probe", _f(r["radar_probe_accuracy"]), _f(r["radar_probe_macro_f1"]), "linear"])
            sm.append([r["model"], "Retrieval action top-1", _f(r["retrieval_action_top1"]), "—", "6 directed pairs"])
    dump("table6_shared", "Table 6. Shared representation. Radar-ID probe remains 1.000.", ["Model", "Eval", "Acc / score", "Macro-F1", "Note"], sm)

    os_rows = []
    if ev:
        import statistics

        for method in ("MSP", "Energy", "Prototype"):
            a = [next(m for m in r["methods"] if m["method"] == method)["AUROC"] for r in ev]
            b = [next(m for m in r["methods"] if m["method"] == method)["AUPR"] for r in ev]
            c = [next(m for m in r["methods"] if m["method"] == method)["FPR95"] for r in ev]
            os_rows.append([method, f"{statistics.mean(a):.3f} ± {statistics.stdev(a):.3f}", f"{statistics.mean(b):.3f} ± {statistics.stdev(b):.3f}", f"{statistics.mean(c):.3f} ± {statistics.stdev(c):.3f}", "13-fold LOAO + SI"])
    dump("table7_openset", "Table 7. Open-set recognition (mean ± std over 13 held-out actions). Not open-world.", ["Method", "AUROC", "AUPR", "FPR95", "Protocol"], os_rows)
    print("[tables]", OUT, flush=True)


if __name__ == "__main__":
    main()
