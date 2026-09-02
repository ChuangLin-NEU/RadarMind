from __future__ import annotations

import csv
import re
from pathlib import Path

CLAIM_ROWS = [
    {
        "claim_id": "C01",
        "claim": "Within-radar HAR on official plotted spectrograms is substantially stronger than several cross-radar transfers",
        "strength": "STRONG",
        "supporting_file": "results/phase2_cross_radar/runs/within_24/metrics.json",
        "supporting_metric": "R0_24_WITHIN vs R0_24_TO_77",
        "manuscript_location": "Abstract; §3.1; Conclusions",
    },
    {
        "claim_id": "C02",
        "claim": "Three-radar Radar-ID remains recoverable after numerical Doppler reconstruction",
        "strength": "STRONG",
        "supporting_file": "results/phase8_physical_representation/answers.json",
        "supporting_metric": "RID_R2D_3CLASS",
        "manuscript_location": "Abstract; §3.3",
    },
    {
        "claim_id": "C03",
        "claim": "On the 24/77 GHz pair, velocity-domain alignment only weakly reduces two-class Radar-ID",
        "strength": "STRONG",
        "supporting_file": "results/phase10_velocity_calibration/answers.json",
        "supporting_metric": "RID_R2D_2477 → RID_R2V_2477",
        "manuscript_location": "Abstract; §3.4",
    },
    {
        "claim_id": "C04",
        "claim": "Unified STFT/MTI and first-order calibration are insufficient under their stated gates",
        "strength": "STRONG",
        "supporting_file": "results/phase10_unified_measurement/answers.json",
        "supporting_metric": "U1_RID; U2_RID; CBEST_RID; M2_RID",
        "manuscript_location": "§3.5; Discussion",
    },
    {
        "claim_id": "C05",
        "claim": "Motion state explains part of paired velocity-marginal radar-response variation",
        "strength": "LIMITED",
        "supporting_file": "results/phase10_motion_conditioned/answers.json",
        "supporting_metric": "D2477_B1_R2",
        "manuscript_location": "Abstract; §3.7",
    },
    {
        "claim_id": "C06",
        "claim": "Low-rank 2D modeling of paired differences has a limited TEST oracle ceiling",
        "strength": "LIMITED",
        "supporting_file": "results/phase10_conditional_2d_measurement/answers.json",
        "supporting_metric": "ORACLE_R2_2D; K20_TRAIN_VAR",
        "manuscript_location": "Abstract; §3.8; Conclusions",
    },
    {
        "claim_id": "C07",
        "claim": "Current public paired data lack geometry / acquisition metadata needed for spatial models",
        "strength": "STRONG",
        "supporting_file": "results/phase10_motion_conditioned/geometry_feasibility_audit.md",
        "supporting_metric": "range/distance/aspect UNAVAILABLE",
        "manuscript_location": "Discussion; Conclusions",
    },
    {
        "claim_id": "U01",
        "claim": "A universal radar encoder has been achieved",
        "strength": "UNSUPPORTED",
        "supporting_file": "",
        "supporting_metric": "",
        "manuscript_location": "MUST NOT APPEAR",
    },
    {
        "claim_id": "U02",
        "claim": "Geometry causes the residual domain shift",
        "strength": "UNSUPPORTED",
        "supporting_file": "results/phase10_motion_conditioned/geometry_feasibility_audit.md",
        "supporting_metric": "",
        "manuscript_location": "MUST NOT APPEAR",
    },
    {
        "claim_id": "U03",
        "claim": "10 GHz is on a reliable common velocity axis",
        "strength": "UNSUPPORTED",
        "supporting_file": "results/phase9_radarbridge_v2/answers.json",
        "supporting_metric": "EFFECTIVE_FC_UNRESOLVED",
        "manuscript_location": "MUST NOT APPEAR",
    },
    {
        "claim_id": "U04",
        "claim": "We achieve radar invariance / we solve cross-radar HAR",
        "strength": "UNSUPPORTED",
        "supporting_file": "",
        "supporting_metric": "",
        "manuscript_location": "MUST NOT APPEAR",
    },
    {
        "claim_id": "U05",
        "claim": "Learn Motion, Not Sensor has been achieved",
        "strength": "UNSUPPORTED",
        "supporting_file": "",
        "supporting_metric": "",
        "manuscript_location": "MUST NOT APPEAR",
    },
]

LEAK_PATTERNS = [
    (r"universal radar encoder has been achieved", "U01"),
    (r"we achieve radar invariance", "U04"),
    (r"we solve cross-radar", "U04"),
    (r"geometry causes the residual", "U02"),
    (r"10 ghz (is |was )?velocity[- ]align", "U03"),
    (r"learn motion, not sensor achieved", "U05"),
    (r"cross-environment (has been )?demonstrated", "U04"),
]


def write_claims(out: Path) -> list[dict]:
    fields = ["claim_id", "claim", "strength", "supporting_file", "supporting_metric", "manuscript_location"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in CLAIM_ROWS:
            w.writerow(r)
    return CLAIM_ROWS


def scan_unsupported_leakage(text: str) -> list[str]:
    low = text.lower()
    hits = []
    for pat, cid in LEAK_PATTERNS:
        if re.search(pat, low):
            hits.append(cid)
    return hits


def extract_display_numbers(text: str) -> list[str]:
    return re.findall(r"\b0\.\d{3}\b", text)


def write_number_audit(out: Path, snap: dict, locations: list[tuple[str, str]]) -> list[dict]:
    """locations: (manuscript_location, raw text)"""
    known: dict[str, str] = {}
    for key, rec in snap["metrics"].items():
        if rec["format_rule"] in ("f3", "f2"):
            known.setdefault(rec["display"], key)
    rows = []
    nid = 1
    for loc, text in locations:
        for disp in extract_display_numbers(text):
            key = known.get(disp, "")
            rec = snap["metrics"].get(key, {})
            rows.append(
                {
                    "number_id": f"N{nid:03d}",
                    "display_value": disp,
                    "raw_value": rec.get("value", ""),
                    "context": loc,
                    "metric_key": key,
                    "source_file": rec.get("source_file", ""),
                    "source_field": rec.get("source_field", ""),
                    "manuscript_location": loc,
                }
            )
            nid += 1
    fields = ["number_id", "display_value", "raw_value", "context", "metric_key", "source_file", "source_field", "manuscript_location"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def table_consistency(tab: Path) -> list[dict]:
    rows = []
    for csvp in sorted(tab.glob("*.csv")):
        stem = csvp.stem
        mdp = csvp.with_suffix(".md")
        texp = csvp.with_suffix(".tex")
        csv_nums = set(extract_display_numbers(csvp.read_text(encoding="utf-8")))
        md_nums = set(extract_display_numbers(mdp.read_text(encoding="utf-8"))) if mdp.is_file() else set()
        tex_nums = set(extract_display_numbers(texp.read_text(encoding="utf-8"))) if texp.is_file() else set()
        ok = csv_nums == md_nums == tex_nums
        rows.append(
            {
                "table": stem,
                "csv_n": len(csv_nums),
                "md_n": len(md_nums),
                "tex_n": len(tex_nums),
                "consistent": "PASS" if ok else "FAIL",
                "only_csv": sorted(csv_nums - md_nums - tex_nums),
                "only_md": sorted(md_nums - csv_nums),
                "only_tex": sorted(tex_nums - csv_nums),
            }
        )
    out = tab.parent / "table_consistency_audit.csv"
    fields = ["table", "csv_n", "md_n", "tex_n", "consistent", "only_csv", "only_md", "only_tex"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] if not isinstance(r[k], list) else ";".join(r[k]) for k in fields})
    return rows
