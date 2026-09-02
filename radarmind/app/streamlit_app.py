"""RadarMind v0.1 Replay Application. UI only; inference goes through RadarMindEngine."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

from radarmind.engine import RadarMindEngine
from radarmind.protocols import ProtocolRegistry
from radarmind.results_io import ablation_block, generalization_table, research_status, temporal_rows

try:
    st.set_page_config(page_title="RadarMind", layout="wide")
except Exception:
    pass


def engine() -> RadarMindEngine:
    from radarmind.llm.provider import LLMService

    offline = bool(st.session_state.get("offline_demo"))
    if "rm_engine" not in st.session_state:
        st.session_state.rm_engine = RadarMindEngine(offline_demo=offline)
    eng = st.session_state.rm_engine
    want_off = offline
    have_off = bool(getattr(eng.llm, "force_offline", False))
    if want_off != have_off or not callable(getattr(getattr(eng, "llm", None), "ui_mode", None)):
        eng.llm = LLMService(force_offline=want_off)
    return eng


def _fmt(v, digits=3):
    if v is None or v == "UNKNOWN":
        return "UNKNOWN"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _tv(field):
    if isinstance(field, dict):
        return field.get("value", "UNKNOWN"), field.get("source", "UNKNOWN"), field.get("unit", "")
    return field, "UNKNOWN", ""


def radar_figure(view: dict, plane: str):
    import plotly.graph_objects as go

    if view.get("kind") == "micro_doppler":
        img = np.asarray(view["image"])
        fig = go.Figure(data=go.Heatmap(z=img, colorscale="Viridis", showscale=True))
        fig.add_vline(x=view.get("cursor_t", 0), line_color="red", line_width=2)
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), title="micro-Doppler")
        fig.update_xaxes(title="time bin")
        fig.update_yaxes(title="freq bin")
        return fig
    if view.get("kind") == "point_cloud":
        if plane == "3D":
            xyz = np.asarray(view.get("xyz"))
            fig = go.Figure()
            if xyz.size:
                vel = np.asarray(view.get("velocity"))
                fig.add_trace(
                    go.Scatter3d(
                        x=xyz[:, 0],
                        y=xyz[:, 1],
                        z=xyz[:, 2],
                        mode="markers",
                        marker=dict(size=3, color=vel if vel.size == len(xyz) else None, colorscale="Turbo"),
                    )
                )
            fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), title="point cloud XYZ")
            return fig
        key = {"XY": "xy", "XZ": "xz", "YZ": "yz"}[plane]
        pts = np.asarray(view.get(key))
        fig = go.Figure()
        if pts.size:
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", marker=dict(size=6)))
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), title=f"point cloud {plane}")
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        return fig
    return go.Figure()


def page_replay(eng: RadarMindEngine):
    demos = eng.demo_registry().get("demos", [])
    demo_labels = [f"{d.get('demo_id', '?')}  {d.get('title', d.get('sample_id', ''))}" for d in demos]
    c0, c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.3, 1.3, 2.2])
    with c0:
        dataset = st.selectbox("Dataset", ["ci4r", "ci4r10", "ci4r24", "ci4r77", "mri", "radhar"], key="ds")
    adapter = eng.load_dataset(dataset)
    proto_opts = {
        "ci4r": ["within_distribution", "prefix_group_disjoint"],
        "ci4r10": ["multi_radar_shared"],
        "ci4r24": ["multi_radar_shared"],
        "ci4r77": ["multi_radar_shared"],
        "mri": ["subject_independent", "within_distribution", "open_set_loao"],
        "radhar": ["official_split"],
    }[dataset]
    with c2:
        protocol = st.selectbox("Protocol", proto_opts, key="proto")
    models = eng.models.names_for_dataset(dataset)
    with c3:
        model_name = st.selectbox("Model", models or [eng.default_model(dataset)], key="model")

    demo_pick = None
    with c4:
        if demos:
            lab = st.selectbox("Demo", ["（按 Dataset 自选）"] + demo_labels, key="demo")
            if lab != "（按 Dataset 自选）":
                demo_pick = demos[demo_labels.index(lab)]

    if demo_pick and demo_pick.get("dataset") == dataset:
        sid = demo_pick.get("sample_id")
        seq = demo_pick.get("sequence_id") or sid
        wins = demo_pick.get("window_ids")
    else:
        ids = adapter.sample_ids(split="test", protocol=protocol)
        if not ids:
            ids = adapter.sample_ids()
        with c1:
            sid = st.selectbox("Sample", ids[:400] if len(ids) > 400 else ids, key="sid")
        seq = sid
        wins = None

    if "loaded_key" not in st.session_state:
        st.session_state.loaded_key = ""
    if "fi" not in st.session_state:
        st.session_state.fi = 0
    if "wi" not in st.session_state:
        st.session_state.wi = 0

    def _step_prev():
        st.session_state.fi = max(int(st.session_state.get("fi", 0)) - 1, 0)

    def _step_next():
        st.session_state.fi = int(st.session_state.get("fi", 0)) + 1

    b1, b2, b3, b4, b5 = st.columns(5)
    load_clicked = b1.button("载入")
    b2.button("Play", on_click=_step_next)
    b3.button("Pause")
    b4.button("Previous", on_click=_step_prev)
    b5.button("Next", on_click=_step_next)

    target_ids = list(wins) if wins else [sid]
    already = [s.sample_id for s in eng.replay.samples] == target_ids
    if load_clicked or not already:
        with st.spinner("正在推理…"):
            if wins:
                eng.replay_load(dataset=dataset, sequence_id=seq, protocol=protocol, model_name=model_name, window_ids=wins)
            elif dataset in ("mri", "radhar") and demo_pick and demo_pick.get("sequence_id"):
                eng.replay_load(dataset=dataset, sequence_id=demo_pick["sequence_id"], protocol=protocol, model_name=model_name)
            else:
                eng.replay_load(dataset=dataset, sample_id=sid, protocol=protocol, model_name=model_name)
        st.session_state.fi = 0
        st.session_state.wi = 0

    if eng.replay.samples:
        n_win = len(eng.replay.samples)
        wi = min(int(st.session_state.wi), n_win - 1)
        n_frames = max(eng.replay.samples[wi].n_frames(), 1)
        if st.session_state.fi >= n_frames:
            if wi + 1 < n_win:
                st.session_state.wi = wi + 1
                st.session_state.fi = 0
                wi = st.session_state.wi
                n_frames = max(eng.replay.samples[wi].n_frames(), 1)
            else:
                st.session_state.fi = n_frames - 1
        if st.session_state.fi < 0:
            st.session_state.fi = 0
        eng.replay.seek(window_index=int(st.session_state.wi), frame_index=int(st.session_state.fi))

    stt = eng.replay.state()
    if stt.sample is None:
        st.warning("还没有载入样本。")
        return

    plane = "XY"
    left, mid, right = st.columns([1.15, 0.95, 1.1])
    with left:
        st.subheader("Radar View")
        if stt.view.get("kind") == "point_cloud":
            plane = st.radio("投影", ["XY", "XZ", "YZ", "3D"], horizontal=True)
        st.plotly_chart(radar_figure(stt.view, plane), width="stretch")
        new_fi = st.slider("帧", 0, max(stt.n_frames - 1, 0), value=int(st.session_state.fi), key="frame_slider")
        if int(new_fi) != int(st.session_state.fi):
            st.session_state.fi = int(new_fi)
            eng.replay.seek(window_index=int(st.session_state.wi), frame_index=int(st.session_state.fi))
            stt = eng.replay.state()
        st.caption(f"window {stt.window_index + 1}/{stt.n_windows}  frame {stt.frame_index + 1}/{stt.n_frames}  gt={stt.sample.action_gt}")

    with mid:
        st.subheader("Motion Intelligence")
        intel = stt.intelligence
        rows = [
            ("Activity", "activity"),
            ("Confidence", "confidence"),
            ("Velocity", "mean_velocity"),
            ("Direction", "direction"),
            ("Uncertainty", "uncertainty"),
        ]
        pred = stt.prediction
        if pred is not None and getattr(pred, "open_set_status", None) == "UNKNOWN":
            st.write("**Activity**  Unknown motion")
            st.caption("open_set_status = UNKNOWN。主结果不是 closed-set 标签。")
            st.write(f"nearest known class = {getattr(pred, 'nearest_known_class', pred.activity)}")
            st.write(f"confidence / similarity = {_fmt(pred.confidence)}")
            st.write(f"open_set_method = {pred.open_set_method}  score={_fmt(pred.open_set_score)}  threshold={_fmt(pred.threshold)}")
        else:
            for lab, keyn in rows:
                v, src, unit = _tv(intel.get(keyn))
                extra = f" {unit}" if unit and unit != "UNKNOWN" else ""
                st.write(f"**{lab}**  {_fmt(v) if keyn != 'activity' and keyn != 'direction' else v}{extra}")
                st.caption(f"source = {src}")
        st.write("**Motion statistics**")
        stats = (intel.get("evidence") or {}).get("motion_statistics") or (intel.get("evidence") or {}).get("spectrogram_stats") or {}
        if stats:
            st.json(stats)
        else:
            st.caption("当前 representation 没有额外运动统计。")

    with right:
        st.subheader("Timeline")
        events = (stt.timeline or {}).get("events") or []
        if events:
            import pandas as pd

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "start": e.get("start_time"),
                            "end": e.get("end_time"),
                            "activity": e.get("activity"),
                            "confidence": e.get("confidence"),
                        }
                        for e in events
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("没有事件。")
        st.caption(f"window {stt.window_index + 1}/{stt.n_windows}")

    st.subheader("AI Understanding")
    a, b, c = st.columns(3)
    cur = eng.explain(intelligence=stt.intelligence, timeline=stt.timeline, task="current")
    sm = eng.explain(intelligence=stt.intelligence, timeline=stt.timeline, task="summary")
    un = eng.explain(intelligence=stt.intelligence, timeline=stt.timeline, task="uncertainty")
    a.markdown("**当前动作解释**")
    a.write(cur["text"])
    b.markdown("**Timeline summary**")
    b.write(sm["text"])
    c.markdown("**不确定说明**")
    c.write(un["text"])
    q = st.text_input("Q&A", placeholder="只根据时间线和 MotionIntelligence 提问")
    if q:
        ans = eng.explain(intelligence=stt.intelligence, timeline=stt.timeline, task="qa", question=q)
        st.write(ans["text"])
        st.caption(f"AI mode: {eng.llm.ui_mode()}")
    else:
        st.caption(f"AI mode: {eng.llm.ui_mode()}")


def page_generalization():
    st.subheader("Generalization")
    rows = generalization_table()
    if not rows:
        st.write("没有读到 Phase 1 结果文件。")
        return
    import pandas as pd

    show = []
    for r in rows:
        show.append(
            {
                "dataset": r["dataset"],
                "A": r["protocol_a"],
                "B": r["protocol_b"],
                "note": r["protocol_b_note"],
                "model": r["model"],
                "Acc A": r["acc_a"],
                "F1 A": r["f1_a"],
                "Acc B": r["acc_b"],
                "F1 B": r["f1_b"],
                "gap Acc": r.get("gap_acc"),
                "gap F1": r.get("gap_f1"),
            }
        )
    st.dataframe(pd.DataFrame(show), width="stretch", hide_index=True)
    st.caption("数字来自 results/phase1_ci4r/.../metrics.json 与 results/phase1_pointcloud/training_runs.csv。prefix-group-disjoint 不是 subject-independent。")
    st.markdown("**Temporal contribution**")
    trows = temporal_rows()
    if trows:
        st.dataframe(pd.DataFrame(trows), width="stretch", hide_index=True)
    st.markdown("**Protocol API**")
    st.json(ProtocolRegistry().as_dict())


def page_ablation():
    st.subheader("Representation / Ablation")
    for ds, split in (("mRI", "subject_independent"), ("RadHAR", "official_train_test")):
        blk = ablation_block(ds, split)
        st.markdown(f"**{ds}**")
        if blk["rows"]:
            import pandas as pd

            st.dataframe(pd.DataFrame(blk["rows"]), width="stretch", hide_index=True)
        st.write(blk["text"])


def _csv_rows(path):
    import csv

    if not path.is_file():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def page_cross_radar():
    st.subheader("Cross-Radar")
    st.caption("Phase 2B 是单源 zero-shot。Phase 3 LORO 是多源留一雷达。CORAL 是 UDA，不是 DG。划分按 timestamp。")
    import pandas as pd

    model = st.selectbox("Model", ["Source-only", "CORAL", "Shared Motion v0.1"], key="xr_model")
    p2 = ROOT / "results" / "phase2_cross_radar" / "training_runs.csv"
    p3 = ROOT / "results" / "phase3_shared_motion" / "training_runs.csv"
    p3m = ROOT / "results" / "phase3_shared_motion" / "representation_metrics.csv"
    rows2 = _csv_rows(p2)
    rows3 = _csv_rows(p3)
    mets = _csv_rows(p3m)

    st.markdown("**Within Radar**（Phase 2B，未重训）")
    within = [r for r in rows2 if r.get("protocol") and r.get("train_radar") == r.get("test_radar") and r.get("method") == "source_only"]
    if not within:
        within = [r for r in rows2 if str(r.get("run_id", "")).startswith("within_")]
    st.dataframe(pd.DataFrame(within) if within else pd.DataFrame(), width="stretch", hide_index=True)

    st.markdown("**Single-source Cross Radar**（Phase 2B zero-shot；不是 LORO）")
    if model == "CORAL":
        cross = [r for r in rows2 if "coral" in str(r.get("method", "")).lower() or str(r.get("run_id", "")).startswith("coral_")]
    elif model == "Shared Motion v0.1":
        st.caption("Shared Motion 不走单源 77→10 协议。单源数字仍是 Phase 2B source-only。")
        cross = [r for r in rows2 if str(r.get("run_id", "")).startswith("cross_")]
    else:
        cross = [r for r in rows2 if str(r.get("run_id", "")).startswith("cross_")]
    st.dataframe(pd.DataFrame(cross) if cross else pd.DataFrame(), width="stretch", hide_index=True)

    st.markdown("**Multi-source Leave-One-Radar-Out**（Phase 3；held-out 雷达全程未用）")
    if model == "CORAL":
        st.caption("CORAL 没有 LORO 实验。下面是 Phase 3 M0/M2。")
        loro = [r for r in rows3 if r.get("protocol") == "leave_one_radar_out"]
    elif model == "Shared Motion v0.1":
        loro = [r for r in rows3 if r.get("protocol") == "leave_one_radar_out" and r.get("model") == "M2"]
    else:
        loro = [r for r in rows3 if r.get("protocol") == "leave_one_radar_out" and r.get("model") == "M0"]
    st.dataframe(pd.DataFrame(loro) if loro else pd.DataFrame(), width="stretch", hide_index=True)

    st.markdown("**Representation Analysis**")
    if model == "CORAL":
        probe = ROOT / "results" / "phase2_cross_radar" / "representation_probe.csv"
        st.dataframe(pd.DataFrame(_csv_rows(probe)) if probe.is_file() else pd.DataFrame(), width="stretch", hide_index=True)
    elif model == "Shared Motion v0.1":
        show = [r for r in mets if r.get("model") == "M2"]
        st.dataframe(pd.DataFrame(show) if show else pd.DataFrame(), width="stretch", hide_index=True)
    else:
        show = [r for r in mets if r.get("model") == "M0"]
        st.dataframe(pd.DataFrame(show) if show else pd.DataFrame(), width="stretch", hide_index=True)
    demo_p = ROOT / "results" / "phase2_cross_radar" / "demo_registry.json"
    import json

    demos = json.loads(demo_p.read_text(encoding="utf-8"))["demos"] if demo_p.is_file() else []
    ts_list = [d.get("timestamp") for d in demos] if demos else []
    if not ts_list:
        man = ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv"
        if man.is_file():
            agreed = [r for r in csv.DictReader(man.open(encoding="utf-8")) if r.get("pairing_confidence") == "exact_timestamp" and r.get("usable") in ("True", "true", True)]
            ts_list = sorted({r["timestamp"] for r in agreed})[:40]
    if not ts_list:
        st.write("没有可显示的 pair。")
        return
    pick = st.selectbox("timestamp pair", ts_list)
    cols = st.columns(3)
    for col, freq, folder in zip(cols, ("10GHz", "24GHz", "77GHz"), ("Xethru", "24GHz", "77GHz")):
        man = ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv"
        hit = None
        if man.is_file():
            for r in csv.DictReader(man.open(encoding="utf-8")):
                if r["timestamp"] == pick and r["radar_frequency"] == freq:
                    hit = r
                    break
        col.markdown(f"**{freq}**")
        if not hit:
            col.caption("该频段没有这条 pair，不显示空图。")
            continue
        png = ROOT / hit["source_file"]
        if png.is_file():
            col.image(str(png), caption=f"{hit.get('action')}  {hit.get('pairing_confidence')}")
        else:
            col.caption("谱图文件不在本机。")
    if demos:
        st.markdown("**Demo**")
        st.json(demos)


def page_radar_invariance():
    st.subheader("Radar Invariance")
    import json
    import pandas as pd

    p7 = ROOT / "results" / "phase7_radar_invariance"
    ans_p = p7 / "answers.json"
    ans = json.loads(ans_p.read_text(encoding="utf-8")) if ans_p.is_file() else {}
    if not ans:
        st.caption("还没有 phase7 answers.json。")

    st.markdown("**Q1 跨雷达差有多少来自 preprocessing representation？**")
    st.write(ans.get("Q1", ""))
    st.markdown("**Q2 统一 representation 后 Radar-ID 是否仍然容易识别？**")
    st.write(ans.get("Q2", ""))
    st.markdown("**Q3 Domain-adversarial encoder 是否减少 sensor 信息并改善未见雷达 HAR？**")
    st.write(ans.get("Q3", ""))

    c1, c2, c3 = st.columns(3)
    c1.metric("Official Representation", "R0 整图 PNG")
    c2.metric("Fair Representation", "R1" if ans.get("R1_available") == "YES" else "R1 未建立")
    c3.metric("Radar-Invariant Encoder", "M3" if ans.get("radar_id_M3") not in (None, "") else "未训")

    st.markdown("**Representation Evolution**")
    p8 = ROOT / "results" / "phase8_physical_representation"
    a8p = p8 / "answers.json"
    a8 = json.loads(a8p.read_text(encoding="utf-8")) if a8p.is_file() else {}
    e1, e2, e3 = st.columns(3)
    e1.metric("R0", "官方整图 PNG")
    e2.metric("R1", "裁边后 PNG")
    if a8.get("R2_available") == "YES":
        e3.metric("R2-D", "AVAILABLE")
        st.write("物理数值表示：S(归一化时间, Doppler Hz)。**不是按径向速度对齐。** 24/77 GHz 同一速度对应不同 f_D；10 GHz UWB 没有单一可靠 fc。")
    else:
        e3.metric("R2", "BLOCKED BY RAW DATA")
        st.write("本机没有打开 raw_data.zip。zip 清单里 10/24 `.dat` 和 77 `Raw_0.bin` 已与 327 条 timestamp 对上，缺的是抽出文件，不是映射。")
        miss = p8 / "missing_data_request.md"
        if miss.is_file():
            st.caption("缺哪些成员见 missing_data_request.md。")
    cmp8 = _csv_rows(p8 / "r0_r1_r2_cross_radar.csv")
    if cmp8:
        st.markdown("**R0 / R1 / R2-D 同协议对照**")
        st.dataframe(pd.DataFrame(cmp8), width="stretch", hide_index=True)
    evo = _csv_rows(p8 / "representation_evolution.csv")
    if evo:
        st.dataframe(pd.DataFrame(evo), width="stretch", hide_index=True)

    fair = _csv_rows(p7 / "representation_fairness_comparison.csv")
    if fair:
        st.markdown("**R0 vs R1 单源对照**")
        st.dataframe(pd.DataFrame(fair), width="stretch", hide_index=True)
    probes = _csv_rows(p7 / "probe_results.csv")
    if probes:
        st.markdown("**Activity Probe / Radar-ID Probe**")
        st.dataframe(pd.DataFrame(probes), width="stretch", hide_index=True)
    loro = _csv_rows(p7 / "loro_results.csv")
    if loro:
        st.markdown("**Leave-One-Radar-Out**")
        st.dataframe(pd.DataFrame(loro), width="stretch", hide_index=True)

    figdir = p7 / "figures"
    pairs = [
        ("figA_official_triplet.png", "同一 acquisition：官方图"),
        ("figB_r1_triplet.png", "同一 acquisition：R1"),
        ("figE_r0_vs_r1_matrix.png", "R0 vs R1 cross-radar"),
        ("figF_probes.png", "M0 vs M3 probe"),
        ("figG_loro.png", "LORO"),
    ]
    cols = st.columns(2)
    for i, (name, cap) in enumerate(pairs):
        fp = figdir / name
        if fp.is_file():
            cols[i % 2].image(str(fp), caption=cap)
    for tag in ("figH_umap_R1_M0_shared_activity.png", "figH_umap_R1_M0_shared_radar.png", "figH_umap_R1_M3_shared_activity.png", "figH_umap_R1_M3_shared_radar.png"):
        fp = figdir / tag
        if fp.is_file():
            st.image(str(fp))
        else:
            for kind in ("umap", "tsne"):
                alt = figdir / tag.replace("umap", kind)
                if alt.is_file():
                    st.image(str(alt))
                    break

    p9 = ROOT / "results" / "phase9_radarbridge_v2"
    a9p = p9 / "answers.json"
    a9 = json.loads(a9p.read_text(encoding="utf-8")) if a9p.is_file() else {}
    st.markdown("**Sensor Adapter**")
    st.write("RadarBridge v2 · Known-Radar Sensor-Adapter Framework")
    st.write("Input → Radar-specific Adapter → Shared Motion Encoder")
    st.caption("选 Adapter 要先知道是哪部雷达。不是 Universal Radar Encoder。R2-V 未建立。23.328e9 是测距采样，不是载波。")
    if a9:
        k1, k2, k3 = st.columns(3)
        k1.metric("A0", _fmt(a9.get("A0_acc")))
        k2.metric("A1", _fmt(a9.get("A1_acc")))
        k3.metric("A2", _fmt(a9.get("A2_acc")))
        st.caption("A0 0.946 > A1 0.878 > A2 0.782。Adapter 没有提升联合 HAR。")
        c1, c2, c3 = st.columns(3)
        c1.metric("Radar-ID / 输入", _fmt(a9.get("radar_id_input")))
        c2.metric("Radar-ID / Adapter", _fmt(a9.get("radar_id_adapter")))
        c3.metric("Radar-ID / embedding", _fmt(a9.get("radar_id_embedding")))
        d1, d2 = st.columns(2)
        d1.metric("Activity Probe / Adapter", _fmt(a9.get("activity_adapter")))
        d2.metric("Activity Probe / embedding", _fmt(a9.get("activity_embedding")))
        e1, e2, e3 = st.columns(3)
        e1.metric("三频一致 A0", _fmt(a9.get("agree_A0")))
        e2.metric("三频一致 A2", _fmt(a9.get("agree_A2")))
        e3.metric("表示", a9.get("representation", "R2-D"))
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Known-radar", a9.get("known_radar", "NOT IMPROVED"))
        v2.metric("Radar invariance", a9.get("radar_invariance", "NO"))
        v3.metric("Unseen-radar", a9.get("unseen_radar", "NOT IMPROVED"))
        v4.metric("Learn Motion, Not Sensor", a9.get("learn_motion_not_sensor", "NO"))
        flow = _csv_rows(p9 / "radar_information_flow.csv")
        if flow:
            st.dataframe(pd.DataFrame(flow), width="stretch", hide_index=True)
        pair = _csv_rows(p9 / "paired_embedding_distance.csv")
        if pair:
            st.markdown("**同 acquisition 距离**")
            st.dataframe(pd.DataFrame(pair), width="stretch", hide_index=True)
        agr = _csv_rows(p9 / "prediction_agreement_summary.csv")
        if agr:
            st.markdown("**Prediction Agreement（10 / 24 / 77）**")
            st.dataframe(pd.DataFrame(agr), width="stretch", hide_index=True)
        loro9 = _csv_rows(p9 / "loro_adapter_transfer.csv")
        if loro9:
            st.markdown("**LORO Adapter Transfer**")
            st.dataframe(pd.DataFrame(loro9), width="stretch", hide_index=True)
        f9 = p9 / "figures"
        for name, cap in (
            ("fig1_architecture.png", "Adapter → Shared Encoder"),
            ("fig2_r2_triplet.png", "同 acquisition 三频 R2-D"),
            ("fig3_adapter_features.png", "A2 Adapter 输出"),
            ("fig4_embed_action.png", "embedding，颜色=动作"),
            ("fig5_embed_radar.png", "embedding，颜色=雷达"),
            ("fig6_radar_id_flow.png", "Radar-ID：输入 / Adapter / embedding"),
            ("fig7_activity_flow.png", "Activity probe"),
            ("fig8_paired_distance.png", "同 acquisition 距离"),
            ("fig9_prediction_agreement.png", "三频预测是否一致"),
        ):
            fp = f9 / name
            if fp.is_file():
                st.image(str(fp), caption=cap)
    else:
        st.caption("还没有 phase9 answers.json。")

    p10 = ROOT / "results" / "phase10_motion_factors"
    a10p = p10 / "answers.json"
    a10 = json.loads(a10p.read_text(encoding="utf-8")) if a10p.is_file() else {}
    st.markdown("**Motion Factors**")
    if a10:
        st.write(f"CASE {a10.get('case')} · 下一步 {a10.get('next_step')}")
        f1, f2, f3 = st.columns(3)
        f1.metric("因子数", a10.get("n_factors"))
        f2.metric("All Activity", _fmt(a10.get("probe_all_activity")))
        f3.metric("All Radar-ID", _fmt(a10.get("probe_all_radar")))
        g1, g2 = st.columns(2)
        g1.metric("Stable Activity", _fmt(a10.get("probe_stable_activity")))
        g2.metric("Stable Radar-ID", _fmt(a10.get("probe_stable_radar")))
        tr10 = _csv_rows(p10 / "information_tradeoff.csv")
        if tr10:
            st.dataframe(pd.DataFrame(tr10), width="stretch", hide_index=True)
        fp = p10 / "figures" / "fig8_information_tradeoff.png"
        if fp.is_file():
            st.image(str(fp), caption="Information Trade-off：左上角才是低雷达身份、高动作信息")
        st.caption("因子只用 Doppler Hz / 时间 / 能量。不是速度。per-radar 归一化不拿来声称 unseen。")
    else:
        st.caption("还没有 phase10 answers.json。")

    p10b = ROOT / "results" / "phase10_sensor_calibration"
    a10b = json.loads((p10b / "answers.json").read_text(encoding="utf-8")) if (p10b / "answers.json").is_file() else {}
    st.markdown("**Sensor Calibration**")
    st.write("KNOWN-RADAR CALIBRATION。不是 Universal，也不是 unseen-radar DG。")
    if a10b:
        st.write(f"CASE {a10b.get('case')} · Gate {a10b.get('gate')} · HAR { '已训' if a10b.get('har_trained') else '未训'}")
        b1, b2, b3 = st.columns(3)
        b1.metric("C0 Radar-ID", _fmt(a10b.get("c0_radar")))
        b2.metric("C_best Radar-ID", _fmt(a10b.get("best_radar")))
        b3.metric("C_best", a10b.get("c_best", ""))
        c1, c2, c3 = st.columns(3)
        c1.metric("C0 Activity", _fmt(a10b.get("c0_activity")))
        c2.metric("C_best Activity", _fmt(a10b.get("best_activity")))
        c3.metric("C_best 同对相似度", _fmt(a10b.get("best_same_sim")))
        calp = _csv_rows(p10b / "calibration_probe_results.csv")
        if calp:
            st.dataframe(pd.DataFrame(calp), width="stretch", hide_index=True)
        fp = p10b / "figures" / "fig3_information_tradeoff.png"
        if fp.is_file():
            st.image(str(fp), caption="校准后仍不在左上角")
        fp2 = p10b / "figures" / "fig4_c2_curves.png"
        if fp2.is_file():
            st.image(str(fp2), caption="各雷达 TRAIN Doppler 包络")
    else:
        st.caption("还没有 phase10b answers.json。")

    p10c = ROOT / "results" / "phase10_velocity_calibration"
    a10c = json.loads((p10c / "answers.json").read_text(encoding="utf-8")) if (p10c / "answers.json").is_file() else {}
    st.markdown("**Velocity Calibration**")
    st.write("只做 24↔77。10 GHz 不进速度域。")
    if a10c:
        phy = a10c.get("PHYSICS") or {}
        rep = a10c.get("REPRESENTATION") or {}
        st.write(f"R2-V {phy.get('R2_V')} · Gate {a10c.get('GATE')} · HAR {a10c.get('HAR')}")
        v1, v2, v3 = st.columns(3)
        v1.metric("24 fc", phy.get("fc_24", ""))
        v2.metric("77 fc", phy.get("fc_77", ""))
        v3.metric("Doppler scaling", a10c.get("doppler_scaling", ""))
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("R2-D Radar-ID", _fmt(rep.get("R2D_radar_id")))
        w2.metric("R2-V Radar-ID", _fmt(rep.get("R2V_radar_id")))
        w3.metric("R2-D Activity", _fmt(rep.get("R2D_activity")))
        w4.metric("R2-V Activity", _fmt(rep.get("R2V_activity")))
        pr = _csv_rows(p10c / "representation_probe.csv")
        if pr:
            st.dataframe(pd.DataFrame(pr), width="stretch", hide_index=True)
        fp = p10c / "figures" / "fig4_radar_id.png"
        if fp.is_file():
            st.image(str(fp), caption="24/77 两分类 Radar-ID")
        fp2 = p10c / "figures" / "fig3_r2v_pair.png"
        if fp2.is_file():
            st.image(str(fp2), caption="同一 acquisition 的 R2-V")
        st.caption("下一步：" + str(a10c.get("next_step", "")))
    else:
        st.caption("还没有 phase10c answers.json。")

    p10d = ROOT / "results" / "phase10_unified_measurement"
    a10d = json.loads((p10d / "answers.json").read_text(encoding="utf-8")) if (p10d / "answers.json").is_file() else {}
    st.markdown("**Unified Measurement**")
    st.write("只统一 24/77 的 STFT / MTI。不是 per-radar 消指纹，也不训网络。")
    if a10d:
        st.write(f"CASE {a10d.get('case')} · {a10d.get('main_source')} · U3 {a10d.get('U3')}")
        u0, u1, u2 = a10d.get("U0") or {}, a10d.get("U1") or {}, a10d.get("U2") or {}
        d1, d2, d3 = st.columns(3)
        d1.metric("U0 Radar-ID", _fmt(u0.get("radar_id")))
        d2.metric("U1 Radar-ID", _fmt(u1.get("radar_id")))
        d3.metric("U2 Radar-ID", _fmt(u2.get("radar_id")))
        e1, e2, e3 = st.columns(3)
        e1.metric("U0 pair sep", _fmt(u0.get("sep_cosine")))
        e2.metric("U1 pair sep", _fmt(u1.get("sep_cosine")))
        e3.metric("U2 pair sep", _fmt(u2.get("sep_cosine")))
        ab = _csv_rows(p10d / "operator_ablation_radar_id.csv")
        if ab:
            st.dataframe(pd.DataFrame(ab), width="stretch", hide_index=True)
        fp = p10d / "figures" / "fig6_radar_id.png"
        if fp.is_file():
            st.image(str(fp), caption="各处理层 Radar-ID")
        st.caption("下一步：" + str(a10d.get("next_step", "")))
    else:
        st.caption("还没有 phase10d answers.json。")

    p10e = ROOT / "results" / "phase10_hardware_measurement"
    a10e = json.loads((p10e / "answers.json").read_text(encoding="utf-8")) if (p10e / "answers.json").is_file() else {}
    st.markdown("**Hardware Measurement**")
    st.write("经验测量响应，不是正演模型。输入 U1。")
    if a10e:
        st.write(f"CASE {a10e.get('case')} · 选定 {a10e.get('best_model')} · Gate {'过' if a10e.get('gate') else '未过'}")
        h1, h2, h3 = st.columns(3)
        h1.metric("Transfer 稳定分", _fmt(a10e.get("transfer_stability_score")))
        h2.metric("动作 η²", _fmt(a10e.get("mean_eta2_action")))
        h3.metric("M2 速度边际 Radar-ID", _fmt((a10e.get("M2") or {}).get("vel_rid")))
        cal = _csv_rows(p10e / "measurement_calibration_results.csv")
        if cal:
            st.dataframe(pd.DataFrame(cal), width="stretch", hide_index=True)
        fp = p10e / "figures" / "fig6_radar_id.png"
        if fp.is_file():
            st.image(str(fp), caption="M0–M3 Radar-ID")
        st.caption("下一步：" + str(a10e.get("next_step", "")))
    else:
        st.caption("还没有 phase10e answers.json。")

    p10f = ROOT / "results" / "phase10_motion_conditioned"
    a10f = json.loads((p10f / "answers.json").read_text(encoding="utf-8")) if (p10f / "answers.json").is_file() else {}
    st.markdown("**Motion-conditioned measurement response**")
    st.write("q 只从当前 source 谱提取。不用动作标签。不是 geometry-aware，也不是 unseen-radar DG。")
    if a10f:
        st.write(f"CASE {a10f.get('case')} · Gate {a10f.get('gate')} · geometry {a10f.get('geometry_aware')}")
        f1, f2 = st.columns(2)
        b0 = ((a10f.get("24to77") or {}).get("B0") or {})
        b1 = ((a10f.get("24to77") or {}).get("B1") or {})
        f1.metric("24→77 B0 MAE", _fmt(b0.get("mae")))
        f2.metric("24→77 B1 MAE", _fmt(b1.get("mae")))
        pred = _csv_rows(p10f / "transfer_predictability.csv")
        if pred:
            st.dataframe(pd.DataFrame([r for r in pred if r.get("model") in ("B0", "B1")]), width="stretch", hide_index=True)
        fp = p10f / "figures" / "fig3_b0_b1_mae.png"
        if fp.is_file():
            st.image(str(fp), caption="B0 与 B1 的 D(v) MAE")
        st.caption("下一步：" + str(a10f.get("next_step", "")))
    else:
        st.caption("还没有 phase10f answers.json。")

    p10g = ROOT / "results" / "phase10_conditional_2d_measurement"
    a10g = json.loads((p10g / "answers.json").read_text(encoding="utf-8")) if (p10g / "answers.json").is_file() else {}
    st.markdown("**Motion-conditioned 2D measurement**")
    st.write("低秩 Delta(τ,v)，q 只从 source 来。不是 geometry-aware。")
    if a10g:
        st.write(f"CASE {a10g.get('case')} · Gate {a10g.get('gate')} · {a10g.get('interpretation')}")
        g1, g2, g3 = st.columns(3)
        o24 = ((a10g.get("oracle") or {}).get("24to77") or {})
        b2 = (((a10g.get("transfer") or {}).get("24to77") or {}).get("B2") or {})
        g1.metric("K 24→77", str(((a10g.get("selected_k") or {}).get("24to77") or "")))
        g2.metric("Oracle R² 24→77", _fmt(o24.get("r2")))
        g3.metric("B2 R² 24→77", _fmt(b2.get("r2")))
        fp = p10g / "figures" / "fig8_oracle_vs_b2.png"
        if fp.is_file():
            st.image(str(fp), caption="Oracle 与 q 预测")
        st.caption("下一步：" + str(a10g.get("next_step", "")))
    else:
        st.caption("还没有 phase10g answers.json。")

    p10z = ROOT / "results" / "phase10_closure"
    a10z = json.loads((p10z / "answers.json").read_text(encoding="utf-8")) if (p10z / "answers.json").is_file() else {}
    st.markdown("**Phase 10 收口**")
    if a10z:
        st.write(
            f"经验翻译 {a10z.get('empirical_translation')} · "
            f"瓶颈 {a10z.get('main_bottleneck')} · "
            f"Learn Motion, Not Sensor {a10z.get('learn_motion_not_sensor')}"
        )
        fp = p10z / "figures" / "summary_figure.png"
        if fp.is_file():
            st.image(str(fp), caption="within 高、跨雷达身份仍在")
        st.caption("不在这 327 对上继续堆模型。下一资源是带几何字段的新数据。")
    else:
        st.caption("还没有 phase10z answers.json。")


def page_shared_motion():
    st.subheader("Shared Motion Representation")
    import pandas as pd

    p3 = ROOT / "results" / "phase3_shared_motion"
    mets = _csv_rows(p3 / "representation_metrics.csv")
    runs = _csv_rows(p3 / "training_runs.csv")
    split = _csv_rows(ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv")
    man = ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv"

    ts_list = [r["timestamp"] for r in split if r.get("split") == "test"]
    if not ts_list:
        ts_list = [r["timestamp"] for r in split]
    if ts_list:
        pick = st.selectbox("acquisition timestamp", ts_list, key="sm_ts")
        cols = st.columns(3)
        for col, freq in zip(cols, ("10GHz", "24GHz", "77GHz")):
            hit = None
            if man.is_file():
                import csv

                for r in csv.DictReader(man.open(encoding="utf-8")):
                    if r.get("timestamp") == pick and r.get("radar_frequency") == freq:
                        hit = r
                        break
            col.markdown(f"**{freq}**")
            if not hit:
                col.caption("该频段没有这条 acquisition。")
                continue
            png = ROOT / hit["source_file"]
            if png.is_file():
                col.image(str(png), caption=hit.get("action", ""))
            else:
                col.caption("谱图文件不在本机。")

    st.markdown("**UMAP / 投影**")
    fdir = p3 / "figures"
    c1, c2 = st.columns(2)
    for col, tag, title in (
        (c1, "M0_shared", "M0 Source-only"),
        (c2, "M2_shared", "M2 Shared Motion"),
    ):
        col.markdown(f"**{title}**")
        for kind in ("umap", "tsne"):
            a = fdir / f"{kind}_{tag}_activity.png"
            b = fdir / f"{kind}_{tag}_radar.png"
            if a.is_file():
                col.image(str(a))
            if b.is_file():
                col.image(str(b))

    st.markdown("**Activity Probe / Radar-ID Probe / Retrieval**")
    shared = [r for r in mets if r.get("protocol") == "multi_radar_shared"]
    if shared:
        st.dataframe(pd.DataFrame(shared), width="stretch", hide_index=True)
    else:
        st.caption("还没有 representation_metrics.csv。")

    st.markdown("**Leave-One-Radar-Out**")
    loro = [r for r in mets if r.get("protocol") == "leave_one_radar_out"] or [
        r for r in runs if r.get("protocol") == "leave_one_radar_out"
    ]
    if loro:
        st.dataframe(pd.DataFrame(loro), width="stretch", hide_index=True)
    else:
        st.caption("还没有 LORO 结果。")


def page_continuous(eng: RadarMindEngine):
    st.subheader("Continuous Motion")
    st.caption("labeled-interval continuous replay。只在已标注段上推理。UNLABELED 只有 GT，没有雷达推理。不是 full-stream，也不是 open-world。uncertainty ≠ unknown-class。")
    import json

    import pandas as pd

    p4 = ROOT / "results" / "phase4_continuous"
    demo_p = p4 / "demo_registry.json"
    man = _csv_rows(p4 / "mri_recording_manifest.csv")
    demos = json.loads(demo_p.read_text(encoding="utf-8")).get("demos", []) if demo_p.is_file() else []
    test_recs = [r for r in man if r.get("split") == "test"]
    labels = [f"{d.get('role')}  {d.get('recording')}" for d in demos] + [
        f"{r['recording']}  {r['subject']}" for r in test_recs if r["recording"] not in {d.get("recording") for d in demos}
    ]
    if not labels:
        st.write("还没有 Phase 4 结果。运行 python run_continuous_demo.py")
        return
    c0, c1, c2 = st.columns(3)
    c0.write("Dataset: mRI")
    pick = c1.selectbox("Recording", labels, key="cm_rec")
    rec = pick.split()[1] if pick.startswith("Demo") else pick.split()[0]
    if pick.startswith("Demo"):
        rec = next(d["recording"] for d in demos if d["role"] in pick)
    row = next((r for r in test_recs if r["recording"] == rec), None)
    c2.write(f"Subject: {row['subject'] if row else ''}")

    if "cm_key" not in st.session_state:
        st.session_state.cm_key = ""
    if st.session_state.cm_key != rec:
        with st.spinner("载入连续记录…"):
            eng.replay_load(mode="continuous", recording=rec, protocol="subject_independent")
        st.session_state.cm_key = rec
        st.session_state.cm_frame = eng.replay.cursor_frame

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Play", key="cm_play"):
        eng.replay.play()
        eng.replay.next_frame()
    if b2.button("Pause", key="cm_pause"):
        eng.replay.pause()
    if b3.button("Previous", key="cm_prev"):
        eng.replay.prev_frame()
    if b4.button("Next", key="cm_next"):
        eng.replay.next_frame()

    stt = eng.replay.state()
    wins = eng.replay.pred_windows or []
    if wins:
        lo = min(int(w["start_frame"]) for w in wins)
        hi = max(int(w["end_frame"]) for w in wins)
        fr = st.slider("Frame #", lo, hi, value=int(eng.replay.cursor_frame), key="cm_slider")
        if fr != eng.replay.cursor_frame:
            eng.replay.seek_frame(fr)
            stt = eng.replay.state()

    left, mid, right = st.columns([1.15, 0.95, 1.15])
    with left:
        st.markdown("**Radar View**")
        plane = st.radio("投影", ["XY", "XZ", "YZ", "3D"], horizontal=True, key="cm_plane")
        st.plotly_chart(radar_figure(stt.view, plane), width="stretch")
        st.caption(f"frame {stt.view.get('cursor_frame', stt.frame_index)}  pred={stt.prediction.activity if stt.prediction else ''}  gt={stt.view.get('gt_activity')}")
    with mid:
        st.markdown("**Motion Intelligence**")
        pred = stt.prediction.activity if stt.prediction else "UNKNOWN"
        conf = stt.prediction.confidence if stt.prediction else 0
        if stt.view.get("uncertain"):
            st.warning("Uncertain prediction（低置信度指示，不是 open-set）")
        st.write(f"**Current Action**  {pred}")
        st.caption(f"ground truth = {stt.view.get('gt_activity')}")
        st.write(f"**Confidence**  {_fmt(conf)}")
        intel = stt.intelligence
        v, src, unit = _tv(intel.get("mean_velocity"))
        st.write(f"**Velocity**  {_fmt(v)}  source={src}")
        inten, isrc, _ = _tv(intel.get("motion_intensity"))
        st.write(f"**Motion Energy**  {_fmt(inten)}  source={isrc}")
        u, usrc, _ = _tv(intel.get("uncertainty"))
        st.write(f"**Uncertainty**  {_fmt(u)}  source={usrc}")
    with right:
        st.markdown("**Timeline**")
        gt_ev = (stt.timeline or {}).get("gt_events") or []
        pred_ev = (stt.timeline or {}).get("events") or []
        if gt_ev:
            st.caption("GT")
            st.dataframe(pd.DataFrame([{k: e.get(k) for k in ("start_frame", "end_frame", "activity")} for e in gt_ev]), hide_index=True, width="stretch", height=180)
        if pred_ev:
            st.caption("Prediction")
            st.dataframe(pd.DataFrame([{k: e.get(k) for k in ("start_frame", "end_frame", "activity", "mean_confidence")} for e in pred_ev]), hide_index=True, width="stretch", height=180)

    ser = (stt.timeline or {}).get("series") or []
    if ser:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[s["time"] for s in ser], y=[s["confidence"] for s in ser], name="confidence"))
        fig.add_vline(x=eng.replay.cursor_frame / max(float(eng.replay.pred_pack.get("frame_rate") or 10), 1e-6), line_color="red")
        fig.update_layout(height=180, margin=dict(l=20, r=20, t=20, b=20), title="confidence vs time")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**AI Understanding**")
    intel = dict(stt.intelligence or {})
    if ser:
        confs = [float(s["confidence"]) for s in ser]
        intel["series_stats"] = {
            "n_windows": len(ser),
            "mean_confidence": sum(confs) / len(confs),
            "min_confidence": min(confs),
            "min_confidence_time": min(ser, key=lambda s: s["confidence"]).get("time"),
            "min_confidence_activity": min(ser, key=lambda s: s["confidence"]).get("activity"),
            "peak_abs_velocity": max((s.get("mean_velocity") for s in ser if isinstance(s.get("mean_velocity"), (int, float))), default=None, key=abs)
            if any(isinstance(s.get("mean_velocity"), (int, float)) for s in ser)
            else None,
            "peak_velocity_time": None,
        }
    sm = eng.explain(intelligence=intel, timeline=stt.timeline, task="summary")
    st.write(sm["text"])
    q = st.text_input("Q&A", key="cm_q", placeholder="刚才发生了什么？动作什么时候变化？是否检测到 walk？")
    if q:
        ans = eng.explain(intelligence=intel, timeline=stt.timeline, task="qa", question=q)
        st.write(ans["text"])
        st.caption(f"AI mode: {eng.llm.ui_mode()}")


def _health_banner():
    from radarmind.health import check

    h = check()
    missing_req = [i["name"] for i in h["items"] if i["required"] and not i["ok"]]
    missing_opt = [i["name"] for i in h["items"] if (not i["required"]) and (not i["ok"])]
    if missing_req:
        st.error("缺必要文件：" + "；".join(missing_req))
    if missing_opt:
        st.caption("未接入 / 不可用：" + "；".join(missing_opt) + "。对应页显示 unavailable，不中断其他页。")


def page_home():
    st.subheader("RadarMind")
    _health_banner()
    cols = st.columns(4)
    blocks = [
        ("DATASETS", ["CI4R", "mRI", "RadHAR"]),
        ("RADAR REPRESENTATIONS", ["Micro-Doppler", "Point Cloud"]),
        (
            "RESEARCH PROTOCOLS",
            [
                "Within-distribution",
                "Cross-subject",
                "Cross-radar",
                "Shared representation",
                "Continuous motion",
                "Open-set recognition",
            ],
        ),
        ("AI / APPLICATION", ["Motion Intelligence", "Timeline", "DeepSeek Understanding", "Replay Mode"]),
    ]
    for col, (title, items) in zip(cols, blocks):
        col.markdown(f"**{title}**")
        for x in items:
            col.write(x)
    st.caption("数字和状态见各实验页。未接入的协议不写成已完成。")


def page_open_set(eng: RadarMindEngine):
    st.subheader("Open-Set Recognition")
    st.caption("Open-Set Motion Recognition MVP。不是 open-vocabulary，也不是 open-world。")
    p5 = ROOT / "results" / "phase5_open_set"
    evp = p5 / "eval_summary.json"
    if not evp.is_file():
        st.write("还没有 Phase 5 评测文件。")
        return
    import json
    import pandas as pd

    summary = json.loads(evp.read_text(encoding="utf-8"))
    helds = [r["held_out"] for r in summary]
    held = st.selectbox("Held-out action", helds)
    row = next(r for r in summary if r["held_out"] == held)
    method_names = [m["method"] for m in row["methods"]]
    method = st.selectbox("Method", method_names)
    m = next(x for x in row["methods"] if x["method"] == method)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUROC", f"{m['AUROC']:.3f}")
    c2.metric("AUPR", f"{m['AUPR']:.3f}")
    c3.metric("FPR95", f"{m['FPR95']:.3f}")
    c4.metric("Known Acc", f"{m['known_accuracy']:.3f}")
    c5.metric("Open-set Acc", f"{m['open_set_accuracy']:.3f}")
    show_cols = ["method", "AUROC", "AUPR", "FPR95", "known_accuracy", "known_macro_f1", "open_set_accuracy"]
    st.dataframe(pd.DataFrame([{k: m.get(k) for k in show_cols} for m in row["methods"]]), width="stretch", hide_index=True)
    scores_p = p5 / "folds" / held.replace(" ", "_") / f"scores_{method.lower()}.json"
    if scores_p.is_file():
        sc = json.loads(scores_p.read_text(encoding="utf-8"))
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_histogram(x=sc["score_known_test"], name="known", opacity=0.6)
        fig.add_histogram(x=sc["score_unknown_test"], name="unknown", opacity=0.6)
        fig.update_layout(barmode="overlay", title="score distribution  known vs unknown", height=320)
        st.plotly_chart(fig, width="stretch")
    fig2 = p5 / "figures" / "fig2_score_distribution.png"
    if fig2.is_file():
        st.image(str(fig2))
    demo_p = p5 / "composite_demo.json"
    if demo_p.is_file():
        demo = json.loads(demo_p.read_text(encoding="utf-8"))
        st.markdown("**composite replay for open-set demonstration**")
        st.caption("不是原始真实 continuous recording。窗口来自 TEST。")
        st.write(" → ".join(e["activity"] for e in demo.get("events") or []))
        if st.button("载入 open-set demo"):
            if "mri_openset" in eng.models.names():
                eng.replay_load(
                    dataset="mri",
                    sequence_id="openset_composite",
                    protocol="open_set_loao",
                    model_name="mri_openset",
                    window_ids=demo["window_ids"],
                )
                st.session_state.fi = 0
                st.session_state.wi = 0
            else:
                st.write("mri_openset checkpoint 未登记。")
        stt = eng.replay.state()
        if stt.sample is not None and stt.sequence_id == "openset_composite":
            intel = stt.intelligence
            act = (intel.get("activity") or {}).get("value")
            st.write(f"当前窗口：{act}")
            sm = eng.explain(intelligence=intel, timeline=stt.timeline, task="summary")
            st.write(sm["text"])
            st.caption(f"AI mode: {eng.llm.ui_mode()}")
    cases_p = p5 / "failure_cases.json"
    if cases_p.is_file():
        st.markdown("**Failure cases (TEST)**")
        st.json(json.loads(cases_p.read_text(encoding="utf-8")))


def page_pipeline():
    st.subheader("RadarMind Pipeline")
    blocks = [
        ("Radar Data", "DONE"),
        ("Sensor Adapter", "DONE"),
        ("Motion Encoder", "DONE"),
        ("Motion Representation", "PARTIAL"),
        ("HAR / Generalization / Open-set", "DONE"),
        ("Motion Intelligence", "DONE"),
        ("Timeline", "DONE"),
        ("DeepSeek", "DONE"),
        ("Application", "DONE"),
    ]
    for name, stt in blocks:
        st.write(f"**{name}**  {stt}")
    st.caption("Motion Representation 标 PARTIAL：Phase 3 Radar-ID probe 仍为 1.000，不是 sensor-invariant。Open-set 是 MVP，不是 open-world。")


def page_status():
    st.subheader("研究进度")
    rs = research_status()
    xr = rs.get("CROSS_RADAR") or {}
    if xr:
        st.markdown("**Cross-Radar**")
        st.write("CI4R research: " + str(xr.get("CI4R research", "CLOSED")))
        st.write("Cross-radar diagnosis: " + str(xr.get("Cross-radar diagnosis", "COMPLETE")))
        st.write("Empirical translation: " + str(xr.get("Empirical translation", "STOPPED")))
        st.write("Main bottleneck: " + str(xr.get("CURRENT BOTTLENECK", "SPATIAL / ACQUISITION METADATA")))
        st.write("Acquisition Toolkit: " + str(xr.get("Acquisition Toolkit", "PARTIAL")))
        st.write("Next: " + str(xr.get("Next", "REAL DATA COLLECTION")))
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**DONE**")
    for x in rs["DONE"]:
        c1.write(f"- {x}")
    c2.markdown("**PARTIAL**")
    for x in rs["PARTIAL"]:
        c2.write(f"- {x}")
    c3.markdown("**BLOCKED_BY_DATA**")
    for x in rs.get("BLOCKED_BY_DATA", []):
        c3.write(f"- {x}")
    c4.markdown("**PENDING**")
    for x in rs["PENDING"]:
        c4.write(f"- {x}")


def main():
    st.title("RadarMind")
    page = st.sidebar.radio(
        "页面",
        [
            "首页",
            "回放",
            "连续运动",
            "Open-Set Recognition",
            "泛化对照",
            "表征消融",
            "Cross-Radar",
            "Shared Motion",
            "Radar Invariance",
            "RadarMind Pipeline",
            "研究进度",
            "采集",
        ],
    )
    import os

    if "offline_demo" not in st.session_state:
        st.session_state.offline_demo = os.environ.get("RADARMIND_OFFLINE_DEMO", "").strip() in ("1", "true", "yes")
    st.sidebar.checkbox("Offline Demo", key="offline_demo")
    st.sidebar.caption("Offline 时说明层走模板。点云、分类器和结果文件仍是本机真实产物。")
    st.sidebar.caption(f"AI mode: {engine().llm.ui_mode()}")
    eng = engine()
    missing = [m for m in eng.missing_checkpoints() if m.get("required")]
    if missing:
        st.sidebar.error("缺必要 checkpoint，见 gpu_jobs/")
    if page == "首页":
        page_home()
    elif page == "回放":
        page_replay(eng)
    elif page == "连续运动":
        page_continuous(eng)
    elif page == "Open-Set Recognition":
        page_open_set(eng)
    elif page == "泛化对照":
        page_generalization()
    elif page == "表征消融":
        page_ablation()
    elif page == "Cross-Radar":
        page_cross_radar()
    elif page == "Shared Motion":
        page_shared_motion()
    elif page == "Radar Invariance":
        page_radar_invariance()
    elif page == "RadarMind Pipeline":
        page_pipeline()
    elif page == "采集":
        from radarmind_acquisition.ui import page as page_acq

        page_acq()
    else:
        page_status()


if __name__ == "__main__":
    main()
