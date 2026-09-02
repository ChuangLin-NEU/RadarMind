"""Emit Table 1–7 and the failure ladder from metric_snapshot only."""
from __future__ import annotations

import csv
from pathlib import Path


def D(snap: dict, key: str) -> str:
    return snap["metrics"][key]["display"]


def V(snap: dict, key: str):
    return snap["metrics"][key]["value"]


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _md(path: Path, title: str, fields: list[str], rows: list[dict], note: str = "") -> None:
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |")
    if note:
        lines += ["", note]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tex(path: Path, title: str, fields: list[str], rows: list[dict], note: str = "") -> None:
    def esc(s: str) -> str:
        return str(s).replace("_", "\\_").replace("→", "$\\rightarrow$").replace("%", "\\%")

    cols = "l" * len(fields)
    lines = [
        f"% {title}",
        "\\begin{tabular}{" + cols + "}",
        "\\hline",
        " & ".join(esc(c) for c in fields) + " \\\\",
        "\\hline",
    ]
    for r in rows:
        lines.append(" & ".join(esc(r.get(k, "")) for k in fields) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    if note:
        lines += ["", "% " + note.replace("\n", " ")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_triple(tab: Path, stem: str, title: str, fields: list[str], rows: list[dict], note: str = "") -> list[str]:
    _csv(tab / f"{stem}.csv", fields, rows)
    _md(tab / f"{stem}.md", title, fields, rows, note)
    _tex(tab / f"{stem}.tex", title, fields, rows, note)
    return [f"{stem}.csv", f"{stem}.md", f"{stem}.tex"]


def build_tables(snap: dict, tab: Path) -> list[dict]:
    tab.mkdir(parents=True, exist_ok=True)
    for old in tab.iterdir():
        if old.is_file():
            old.unlink()
    audit = []

    t1 = [
        {"item": "corpus", "value": "CI4R multi-frequency HAR"},
        {"item": "radars", "value": "10 / 24 / 77 GHz, timestamp-paired"},
        {"item": "n_acquisitions", "value": D(snap, "N_PAIRED_ACQ")},
        {"item": "actions", "value": D(snap, "N_ACTIONS")},
        {"item": "split_train_val_test", "value": f"{D(snap,'SPLIT_TRAIN')}/{D(snap,'SPLIT_VAL')}/{D(snap,'SPLIT_TEST')}"},
        {"item": "official_representation", "value": "public plotted PNG"},
        {"item": "numerical_doppler", "value": "S(normalized time, Doppler Hz)"},
        {"item": "velocity_maps", "value": f"24/77 only; common v in [{D(snap,'VEL_LO')}, {D(snap,'VEL_HI')}] m/s"},
    ]
    emit_triple(tab, "table1_dataset", "Table 1. Dataset / radar / representation / protocol", ["item", "value"], t1)
    audit.append({"table": "1", "n_rows": len(t1), "formats": "csv/md/tex"})

    pairs = [
        ("10→10", "R0_10_WITHIN", "R1_10_WITHIN"),
        ("24→24", "R0_24_WITHIN", "R1_24_WITHIN"),
        ("77→77", "R0_77_WITHIN", "R1_77_WITHIN"),
        ("10→24", "R0_10_TO_24", "R1_10_TO_24"),
        ("10→77", "R0_10_TO_77", "R1_10_TO_77"),
        ("24→10", "R0_24_TO_10", "R1_24_TO_10"),
        ("24→77", "R0_24_TO_77", "R1_24_TO_77"),
        ("77→10", "R0_77_TO_10", "R1_77_TO_10"),
        ("77→24", "R0_77_TO_24", "R1_77_TO_24"),
    ]
    t2 = [{"pair": a, "R0": D(snap, b), "R1": D(snap, c)} for a, b, c in pairs]
    emit_triple(
        tab,
        "table2_within_cross",
        "Table 2. Within vs cross-radar 7-class HAR (official vs cropped plotted images)",
        ["pair", "R0", "R1"],
        t2,
        note="R0 and R1 are source-only 7-class HAR on plotted images. Not Radar-ID.",
    )
    audit.append({"table": "2", "n_rows": len(t2), "formats": "csv/md/tex"})

    t3 = [
        {"stage": "official plotted", "radar_id_3class": D(snap, "RID_R0_3CLASS"), "activity_or_within": f"{D(snap,'R0_10_WITHIN')}/{D(snap,'R0_24_WITHIN')}/{D(snap,'R0_77_WITHIN')}", "note": "7-class HAR diagonal; Radar-ID is 3-radar probe"},
        {"stage": "cropped plotted", "radar_id_3class": D(snap, "RID_R1_3CLASS"), "activity_or_within": D(snap, "ACT_R1_REP"), "note": "activity is representation-level probe, not the HAR diagonal"},
        {"stage": "numerical Doppler", "radar_id_3class": D(snap, "RID_R2D_3CLASS"), "activity_or_within": D(snap, "ACT_R2D_REP"), "note": "activity is representation-level probe on R2-D"},
    ]
    emit_triple(
        tab,
        "table3_representation_evolution",
        "Table 3. Official → cropped → numerical Doppler (3-radar Radar-ID)",
        ["stage", "radar_id_3class", "activity_or_within", "note"],
        t3,
        note="Radar-ID column is the 3-radar probe only. Activity column is not a single benchmark across rows.",
    )
    audit.append({"table": "3", "n_rows": len(t3), "formats": "csv/md/tex"})

    t4 = [
        {"stage": "R2-D 24/77", "radar_id_2class": D(snap, "RID_R2D_2477"), "activity_2477": D(snap, "ACT_R2D_2477"), "pair_sep": D(snap, "SEP_R2D")},
        {"stage": "velocity-aligned 24/77", "radar_id_2class": D(snap, "RID_R2V_2477"), "activity_2477": D(snap, "ACT_R2V_2477"), "pair_sep": D(snap, "SEP_R2V")},
        {"stage": "unified STFT", "radar_id_2class": D(snap, "U1_RID"), "activity_2477": D(snap, "U1_ACT"), "pair_sep": D(snap, "U1_SEP")},
        {"stage": "unified MTI", "radar_id_2class": D(snap, "U2_RID"), "activity_2477": D(snap, "U2_ACT"), "pair_sep": D(snap, "U2_SEP")},
    ]
    emit_triple(
        tab,
        "table4_velocity_processing",
        "Table 4. 24/77 two-radar diagnostics only",
        ["stage", "radar_id_2class", "activity_2477", "pair_sep"],
        t4,
        note="Do not compare radar_id_2class to Table 3 radar_id_3class as one series.",
    )
    audit.append({"table": "4", "n_rows": len(t4), "formats": "csv/md/tex"})

    t5 = [
        {"method": "shared encoder (known-radar)", "har_or_activity": D(snap, "A0_ACC"), "radar_id": D(snap, "RID_A0_INPUT"), "gate": "known-radar baseline; not zero-shot"},
        {"method": "radar-specific adapters", "har_or_activity": D(snap, "A1_ACC"), "radar_id": D(snap, "RID_A1_ADAPTER"), "gate": "NOT IMPROVED"},
        {"method": "adapters + alignment", "har_or_activity": D(snap, "A2_ACC"), "radar_id": D(snap, "RID_A2_EMBED"), "gate": "NOT IMPROVED; known-radar joint task"},
        {"method": "C0–C4 calibration", "har_or_activity": D(snap, "CBEST_ACT"), "radar_id": D(snap, "CBEST_RID"), "gate": "VAL gate FAIL; 3-radar Radar-ID"},
        {"method": "per-velocity affine", "har_or_activity": D(snap, "M2_ACT"), "radar_id": D(snap, "M2_RID"), "gate": "FAIL; 24/77 Radar-ID"},
    ]
    emit_triple(
        tab,
        "table5_alignment_negative",
        "Table 5. Alignment / adapter / calibration (negative results)",
        ["method", "har_or_activity", "radar_id", "gate"],
        t5,
        note="A0–A2 HAR is known-radar joint accuracy. Radar-ID protocols differ by row; see gate column.",
    )
    audit.append({"table": "5", "n_rows": len(t5), "formats": "csv/md/tex"})

    t6 = [
        {"direction": "24→77", "B0_R2": D(snap, "D2477_B0_R2"), "B1_R2": D(snap, "D2477_B1_R2"), "B1_MAE": D(snap, "D2477_B1_MAE"), "B1_corr": D(snap, "D2477_B1_CORR")},
        {"direction": "77→24", "B0_R2": D(snap, "D7724_B0_R2"), "B1_R2": D(snap, "D7724_B1_R2"), "B1_MAE": D(snap, "D7724_B1_MAE"), "B1_corr": D(snap, "D7724_B1_CORR")},
    ]
    emit_triple(
        tab,
        "table6_motion_conditioned",
        "Table 6. Motion-conditioned velocity-marginal D(v|q)",
        ["direction", "B0_R2", "B1_R2", "B1_MAE", "B1_corr"],
        t6,
        note="R2 is prediction of paired velocity-marginal transfer, not HAR accuracy.",
    )
    audit.append({"table": "6", "n_rows": len(t6), "formats": "csv/md/tex"})

    t7 = [
        {"item": "selected_K", "value": D(snap, "K_SELECTED")},
        {"item": "TRAIN_explained_var", "value": D(snap, "K20_TRAIN_VAR")},
        {"item": "TEST_oracle_R2", "value": D(snap, "ORACLE_R2_2D")},
        {"item": "B2_R2_24to77", "value": D(snap, "B2_R2_2477")},
        {"item": "B2_R2_77to24", "value": D(snap, "B2_R2_7724")},
    ]
    emit_triple(
        tab,
        "table7_lowrank_2d",
        "Table 7. Low-rank 2D paired-difference model",
        ["item", "value"],
        t7,
        note="TEST oracle R2 is true TEST Delta projected onto TRAIN PCA; not deployment performance.",
    )
    audit.append({"table": "7", "n_rows": len(t7), "formats": "csv/md/tex"})

    ladder = [
        {"Stage": "Official representation", "Input_or_Method": "plotted PNG", "Radar_ID": D(snap, "RID_R0_3CLASS"), "Activity_metric": f"{D(snap,'R0_10_WITHIN')}/{D(snap,'R0_24_WITHIN')}/{D(snap,'R0_77_WITHIN')}", "Generalization_or_Pair": f"24→77 {D(snap,'R0_24_TO_77')}", "Interpretation": "within high, cross collapse", "Footnote": "Radar-ID: 3-radar. Activity: 7-class HAR diagonal."},
        {"Stage": "Cropped representation", "Input_or_Method": "cropped PNG", "Radar_ID": D(snap, "RID_R1_3CLASS"), "Activity_metric": D(snap, "ACT_R1_REP"), "Generalization_or_Pair": f"10→24 {D(snap,'R0_10_TO_24')}→{D(snap,'R1_10_TO_24')}", "Interpretation": "plot confound, not main cause", "Footnote": "Radar-ID: 3-radar. Activity probe ≠ HAR diagonal."},
        {"Stage": "Numerical Doppler", "Input_or_Method": "S(t, f_D)", "Radar_ID": D(snap, "RID_R2D_3CLASS"), "Activity_metric": D(snap, "ACT_R2D_REP"), "Generalization_or_Pair": f"same {D(snap,'R2D_SAME_COS')} vs diff {D(snap,'R2D_DIFF_COS')}", "Interpretation": "identity remains", "Footnote": "Radar-ID: 3-radar representation probe."},
        {"Stage": "Velocity alignment", "Input_or_Method": "24/77 common v", "Radar_ID": D(snap, "RID_R2V_2477"), "Activity_metric": D(snap, "ACT_R2V_2477"), "Generalization_or_Pair": f"sep {D(snap,'SEP_R2D')}→{D(snap,'SEP_R2V')}", "Interpretation": "weak Doppler-scaling effect", "Footnote": "Radar-ID: 2-class 24/77 only. Not the 3-radar probe."},
        {"Stage": "Unified STFT", "Input_or_Method": "shared STFT", "Radar_ID": D(snap, "U1_RID"), "Activity_metric": D(snap, "U1_ACT"), "Generalization_or_Pair": f"sep {D(snap,'U1_SEP')}", "Interpretation": "no Radar-ID drop", "Footnote": "24/77 two-radar protocol."},
        {"Stage": "Unified MTI", "Input_or_Method": "transplanted MTI", "Radar_ID": D(snap, "U2_RID"), "Activity_metric": D(snap, "U2_ACT"), "Generalization_or_Pair": f"sep {D(snap,'U2_SEP')}", "Interpretation": "identity stronger", "Footnote": "24/77 two-radar protocol."},
        {"Stage": "Learned alignment", "Input_or_Method": "adapters + align", "Radar_ID": D(snap, "RID_A2_EMBED"), "Activity_metric": D(snap, "A2_ACC"), "Generalization_or_Pair": "known-radar NOT IMPROVED", "Interpretation": "no invariance", "Footnote": "A2 accuracy is known-radar joint HAR, not zero-shot. Radar-ID is embedding probe."},
        {"Stage": "Motion factors", "Input_or_Method": f"{D(snap,'N_FACTORS')} factors", "Radar_ID": D(snap, "FAC_RID"), "Activity_metric": D(snap, "FAC_ACT"), "Generalization_or_Pair": f"stable RID {D(snap,'FAC_RID_STABLE')}", "Interpretation": "activity and radar coexist", "Footnote": "Probes on factor vectors, not spectrogram pixels."},
        {"Stage": "Calibration", "Input_or_Method": "C0–C4 / per-v affine", "Radar_ID": D(snap, "M2_RID"), "Activity_metric": D(snap, "M2_ACT"), "Generalization_or_Pair": f"vel-RID {D(snap,'M2_VEL_RID')}", "Interpretation": "gates fail", "Footnote": "C-best Radar-ID is 3-radar; M2 is 24/77. Not one series."},
        {"Stage": "D(v|q)", "Input_or_Method": "Ridge velocity transfer", "Radar_ID": f"{D(snap,'CAL_RID_BEFORE')}→{D(snap,'CAL_RID_AFTER')}", "Activity_metric": D(snap, "U1_ACT"), "Generalization_or_Pair": f"24→77 R2 {D(snap,'D2477_B1_R2')}", "Interpretation": "predictable; calibration fails", "Footnote": "R2 is velocity-marginal transfer prediction, not HAR."},
        {"Stage": "2D low-rank", "Input_or_Method": "PCA+Ridge Delta", "Radar_ID": "n/a (not a comparable probe)", "Activity_metric": f"src {D(snap,'B2_ACT_SRC')} → pred {D(snap,'B2_ACT_PRED')}", "Generalization_or_Pair": f"oracle R2 {D(snap,'ORACLE_R2_2D')}", "Interpretation": "low ceiling; stop translation", "Footnote": "Oracle is TEST true Delta on TRAIN basis; not deployment."},
    ]
    fields = ["Stage", "Input_or_Method", "Radar_ID", "Activity_metric", "Generalization_or_Pair", "Interpretation", "Footnote"]
    emit_triple(
        tab,
        "cross_radar_failure_ladder",
        "Cross-Radar Failure Ladder",
        fields,
        ladder,
        note="Protocols differ across rows. Empty or n/a cells are intentional; do not force a common Radar-ID series.",
    )
    audit.append({"table": "failure_ladder", "n_rows": len(ladder), "formats": "csv/md/tex"})
    return audit
