"""Paper figures: Fig. 1 retained; Fig. 2–11 from visual_style / figures_v2."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paper.visual_style import apply_style  # noqa: E402
from paper_figures import figures_v2 as F  # noqa: E402
from paper_figures.io import MetricBook, load_snapshot  # noqa: E402

MAIN = ROOT / "figures" / "main"
SUPP = ROOT / "figures" / "supplementary"

CAPTIONS = {
    "fig01_radarmind_framework": (
        "Fig. 1. RadarMind three-layer framework. Layer I maps radar representations to recognition, "
        "subject generalization, and cross-radar tests. Layer II builds a shared representation, "
        "labeled-interval continuous inference, open-set scores, Motion Intelligence, and a timeline. "
        "Layer III exposes structured JSON to a language layer. The language layer does not receive raw radar. "
        "The dashed box lists problems that remain outside the current evidence."
    ),
    "fig02_dataset_protocol_map": (
        "Fig. 2. Datasets and experimental protocols used for complementary RadarMind evaluations. "
        "CI4R paired 10/24/77 GHz supports cross-radar tests. CI4R 77 GHz 11-class compares random-record "
        "and prefix-group-disjoint splits and is not subject-independent. mRI is the only local corpus with "
        "official subject IDs, labeled-interval continuous replay, and LOAO open-set. RadHAR uses the official "
        "train/test listing. The three sources are complementary, not one unified benchmark."
    ),
    "fig03_baseline_temporal": (
        "Fig. 3. (a) Closed-set accuracy under each dataset’s own protocol. Groups are not a shared leaderboard. "
        "(b) Temporal-context ablation on mRI (T=1 vs T=32) and RadHAR (T=1 vs T=60). "
        "The comparison illustrates temporal context, not architecture superiority."
    ),
    "fig04_cross_radar_matrix": (
        "Fig. 4. Official-representation 7-class HAR on CI4R paired acquisitions. "
        "Rows are train radars and columns are test radars. The split is by acquisition timestamp. "
        "Filename prefix is not a subject. Diagonal cells are within-radar; off-diagonal cells are zero-shot cross-radar."
    ),
    "fig05_shared_representation": (
        "Fig. 5. Known-radar shared representation. (a) Activity accuracy and same-action retrieval for M0–M2. "
        "(b) 3-radar Radar-ID on the same embeddings. InfoNCE improves semantic correspondence among radars seen in training; "
        "it does not remove sensor identity. This is not a leave-one-radar-out ranking."
    ),
    "fig06_radar_identity_ladder": (
        "Fig. 6. Radar-ID and activity probes along the physical processing chain. "
        "(a) 3-radar probe. (b) 24/77 probe. Filled markers are Radar-ID; open markers are activity probes. "
        "The two panels are different probes and are not a single continuous trajectory. "
        "A stage without a stored activity probe is left blank."
    ),
    "fig07_measurement_difference": (
        "Fig. 7. Motion-conditioned and low-rank 2D difference models. "
        "Values are R² of the difference field, not HAR accuracy. "
        "Colour marks transfer direction; marker style marks model type. "
        "TEST oracle is the true TEST Δ projected onto the TRAIN basis; it is an upper bound, not a deployable translator. "
        "Measured 2D difference maps are not stored in the frozen Phase 10 files, so field panels are omitted."
    ),
    "fig08_continuous_inference": (
        "Fig. 8. Labeled-interval continuous replay on mRI recording subject20_0513. "
        "(a) Ground-truth labeled intervals, raw window predictions, and confidence-aware events. "
        "Hatched intervals are unlabeled gaps and are not scored. "
        "(b) Segment F1. (c) Median boundary error. This is labeled-interval replay, not full-stream inference."
    ),
    "fig09_open_set": (
        "Fig. 9. Thirteen-fold leave-one-action-out open-set rejection on mRI. "
        "(a) Per-fold AUROC for MSP, Energy, and Prototype. The dashed line is chance (0.500). "
        "(b) Mean FPR95. Prototype is better than MSP and remains weak in absolute terms. "
        "The protocol is LOAO rejection, not open-world recognition."
    ),
    "fig10_motion_intelligence_llm": (
        "Fig. 10. Language interaction on a stored RadHAR boxing record. "
        "The language layer reads Motion Intelligence fields copied from llm_online_test.json. "
        "Raw radar is isolated from the language layer. Velocity descriptors exist in the JSON and are reported separately from the event list."
    ),
    "fig11_capability_summary": (
        "Fig. 11. Evidence map from closed-set recognition to semantic interaction. "
        "Node colour marks supported, partial, or unresolved status against frozen scores. "
        "The working path is radar → recognition → Motion Intelligence → timeline → language. "
        "Unseen-radar transfer remains unresolved."
    ),
}

SECTIONS = {
    "fig01_radarmind_framework": "2.1",
    "fig02_dataset_protocol_map": "2.2",
    "fig03_baseline_temporal": "3.1",
    "fig04_cross_radar_matrix": "3.4",
    "fig05_shared_representation": "3.5",
    "fig06_radar_identity_ladder": "3.6",
    "fig07_measurement_difference": "3.6",
    "fig08_continuous_inference": "3.7",
    "fig09_open_set": "3.8",
    "fig10_motion_intelligence_llm": "3.10",
    "fig11_capability_summary": "3.11",
}

PROTOCOLS = {
    "fig01_radarmind_framework": "system schematic; not an accuracy claim",
    "fig02_dataset_protocol_map": "dataset roles; not a unified benchmark",
    "fig03_baseline_temporal": "closed-set under each dataset protocol; temporal ablation",
    "fig04_cross_radar_matrix": "CI4R paired 7-class; acquisition-timestamp split",
    "fig05_shared_representation": "known-radar joint encoder; not LORO",
    "fig06_radar_identity_ladder": "3-radar Radar-ID and 24/77 Radar-ID kept separate",
    "fig07_measurement_difference": "difference-field R2; oracle is TEST upper bound",
    "fig08_continuous_inference": "mRI labeled-interval replay; not full-stream",
    "fig09_open_set": "13-fold LOAO open-set rejection; not open-world",
    "fig10_motion_intelligence_llm": "stored MI JSON; LLM does not read raw radar",
    "fig11_capability_summary": "implemented pipeline vs unresolved generalization",
}

METRICS = {
    "fig01_radarmind_framework": [],
    "fig02_dataset_protocol_map": ["N_PAIRED_ACQ", "MRI_N_SUBJECTS", "MRI_N_ACTIONS", "RADHAR_N_ACTIONS"],
    "fig03_baseline_temporal": ["MRI_SI_ACC", "RADHAR_XYZV_ACC", "R0_10_WITHIN", "R0_24_WITHIN", "R0_77_WITHIN", "MRI_T1_ACC", "RADHAR_T1_ACC"],
    "fig04_cross_radar_matrix": ["R0_10_WITHIN", "R0_10_TO_24", "R0_10_TO_77", "R0_24_TO_10", "R0_24_WITHIN", "R0_24_TO_77", "R0_77_TO_10", "R0_77_TO_24", "R0_77_WITHIN"],
    "fig05_shared_representation": ["M0_ACC", "M1_ACC", "M2_ACC", "M0_RET", "M1_RET", "M2_RET", "M0_RID", "M1_RID", "M2_RID_SHARE"],
    "fig06_radar_identity_ladder": ["RID_R0_3CLASS", "RID_R1_3CLASS", "RID_R2D_3CLASS", "RID_R2D_2477", "RID_R2V_2477", "U1_RID", "U2_RID", "CAL_RID_AFTER"],
    "fig07_measurement_difference": ["D2477_B1_R2", "D7724_B1_R2", "ORACLE_R2_2D", "B2_R2_2477", "B2_R2_7724"],
    "fig08_continuous_inference": ["CONT_SEG_F1_RAW", "CONT_SEG_F1_FLIP", "CONT_SEG_F1", "CONT_BND_SEC", "CONT_N_WINDOWS"],
    "fig09_open_set": ["MSP_AUROC", "ENERGY_AUROC", "PROTO_AUROC", "MSP_FPR", "ENERGY_FPR", "PROTO_FPR"],
    "fig10_motion_intelligence_llm": ["BOX_CONF", "BOX_UNC", "BOX_NWIN", "BOX_PEAK_V", "BOX_ENERGY", "LLM_N", "LLM_N_OK"],
    "fig11_capability_summary": ["MRI_SI_ACC", "RADHAR_XYZV_ACC", "R0_24_TO_77", "CONT_SEG_F1", "PROTO_AUROC", "LLM_N", "LLM_N_OK"],
}


def _meta(stem: str, paths: dict, figure: str) -> dict:
    return {
        "figure": figure,
        "stem": stem,
        "file": paths["png"],
        "pdf": paths["pdf"],
        "svg": paths["svg"],
        "section": SECTIONS[stem],
        "purpose": CAPTIONS[stem],
        "data_sources": [],
        "metrics": METRICS[stem],
        "protocol": PROTOCOLS[stem],
        "script": "scripts/paper_figures/figures_v2.py",
        "caption": CAPTIONS[stem],
    }


def _keep_fig01(book: MetricBook) -> dict:
    stem = "fig01_radarmind_framework"
    paths = {ext: str(MAIN / f"{stem}.{ext}").replace("\\", "/") for ext in ("png", "pdf", "svg")}
    book.record_file("Fig. 1", "retained framework", paths["png"], "unchanged", "Fig. 1 not redrawn")
    return _meta(stem, paths, "Fig. 1")


def build_all(snap: dict | None = None) -> tuple[list[dict], MetricBook]:
    apply_style()
    snap = snap or load_snapshot()
    book = MetricBook(snap)
    MAIN.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    metas = [
        _keep_fig01(book),
        _meta("fig02_dataset_protocol_map", F.fig02(book), "Fig. 2"),
        _meta("fig03_baseline_temporal", F.fig03(book), "Fig. 3"),
        _meta("fig04_cross_radar_matrix", F.fig04(book), "Fig. 4"),
        _meta("fig05_shared_representation", F.fig05(book), "Fig. 5"),
        _meta("fig06_radar_identity_ladder", F.fig06(book), "Fig. 6"),
        _meta("fig07_measurement_difference", F.fig07(book), "Fig. 7"),
        _meta("fig08_continuous_inference", F.fig08(book), "Fig. 8"),
        _meta("fig09_open_set", F.fig09(book), "Fig. 9"),
        _meta("fig10_motion_intelligence_llm", F.fig10(book), "Fig. 10"),
        _meta("fig11_capability_summary", F.fig11(book), "Fig. 11"),
    ]
    by_fig: dict[str, list] = {}
    for row in book.rows:
        by_fig.setdefault(row["figure"], []).append(row["source_file"])
    for m in metas:
        srcs = []
        for s in by_fig.get(m["figure"], []):
            if s and s not in srcs:
                srcs.append(s)
        m["data_sources"] = srcs
    return metas, book
