"""SCI manuscript .docx from metric pack. No str.format on leftover LaTeX."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from .load_metrics import D, pct

FIG = None  # set by caller


def _font(run, size=11, bold=False, italic=False, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)


def _p(doc, text, *, size=11, bold=False, italic=False, align="left", space_after=8, first_line=True, space_before=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing = 1.15
    if first_line:
        pf.first_line_indent = Cm(0.74)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Cm(0)
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    _font(run, size=size, bold=bold, italic=italic)
    return p


def _h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    _font(run, size=13, bold=True)


def _h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    _font(run, size=11, bold=True)


def _caption(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.first_line_indent = Cm(0)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    _font(run, size=10, italic=True)


def _set_three_line(table):
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", "12"), ("left", "0"), ("bottom", "12"), ("right", "0"), ("insideH", "0"), ("insideV", "0")):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single" if sz != "0" else "nil")
        el.set(qn("w:sz"), sz)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)
    # header underline
    row = table.rows[0]
    tr = row._tr
    trPr = tr.get_or_add_trPr()
    # cell bottom borders on header
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcB = OxmlElement("w:tcBorders")
        b = OxmlElement("w:bottom")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "6")
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "000000")
        tcB.append(b)
        tcPr.append(tcB)


def _table(doc, headers, rows, caption):
    _caption(doc, caption)
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = 1
    table.autofit = True
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        _font(r, size=10, bold=True)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            _font(r, size=10)
    _set_three_line(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def _figure(doc, path: Path, caption: str, width_cm=14.0):
    if not path.is_file():
        raise FileNotFoundError(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(8)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    _caption(doc, caption)


def _setup(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.54)
    sec.right_margin = Cm(2.54)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    styles = doc.styles["Normal"]
    styles.font.name = "Times New Roman"
    styles.font.size = Pt(11)
    styles._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def build_word(pack: dict, fig_dir: Path, out_path: Path) -> dict:
    doc = Document()
    _setup(doc)
    d = lambda k: D(pack, k)

    # Title
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(6)
    t.paragraph_format.first_line_indent = Cm(0)
    r = t.add_run("Cross-Radar Human Activity Recognition:\nWhen High Within-Radar Accuracy Fails to Generalize")
    _font(r, size=16, bold=True)

    au = doc.add_paragraph()
    au.alignment = WD_ALIGN_PARAGRAPH.CENTER
    au.paragraph_format.first_line_indent = Cm(0)
    au.paragraph_format.space_after = Pt(2)
    _font(au.add_run("[Author names — TODO]"), size=11)
    aff = doc.add_paragraph()
    aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    aff.paragraph_format.first_line_indent = Cm(0)
    aff.paragraph_format.space_after = Pt(14)
    _font(aff.add_run("[Affiliation — TODO]"), size=10, italic=True)

    _h1(doc, "Abstract")
    abstract = (
        "Radar-based human activity recognition (HAR) can observe motion through range, Doppler and phase without a camera [CITATION NEEDED]. "
        "Most benchmarks still measure recognition inside one radar domain [CITATION NEEDED]. "
        "The open question is narrower: if a model is accurate on one radar, does it still work after the sensor is changed? "
        f"We test that question on {d('N_PAIRED_ACQ')} timestamp-paired CI4R recordings at 10, 24 and 77 GHz "
        f"({d('N_ACTIONS')} actions; split {d('SPLIT_TRAIN')}/{d('SPLIT_VAL')}/{d('SPLIT_TEST')}). "
        "The check proceeds from official plotted spectrograms through cropped plots, numerical Doppler, a 24/77 GHz velocity grid, "
        "unified STFT/MTI, calibration, learned alignment and motion-conditioned measurement response. "
        f"Within-radar accuracy is {d('R0_10_WITHIN')} / {d('R0_24_WITHIN')} / {d('R0_77_WITHIN')}; 24→77 transfer is {d('R0_24_TO_77')}. "
        f"Three-radar Radar-ID remains {d('RID_R2D_3CLASS')} after numerical Doppler reconstruction. "
        f"On the 24/77 GHz pair, velocity alignment changes two-class Radar-ID only from {d('RID_R2D_2477')} to {d('RID_R2V_2477')}. "
        f"Motion-conditioned D(v|q) reaches R² = {d('D2477_B1_R2')} (24→77); the 2D TEST oracle is R² = {d('ORACLE_R2_2D')}. "
        "Cross-radar failure cannot be attributed to plotting, Doppler scaling or simple calibration alone. "
        "The public pairs lack the spatial and acquisition observables needed to separate motion from sensor-dependent measurement effects."
    )
    _p(doc, abstract, first_line=False, align="justify", space_after=8)
    kw = doc.add_paragraph()
    kw.paragraph_format.first_line_indent = Cm(0)
    _font(kw.add_run("Keywords: "), size=11, italic=True)
    _font(kw.add_run("radar HAR; cross-radar generalization; micro-Doppler; domain shift; measurement response; paired multi-frequency data"), size=11)

    n_abs = len(re.findall(r"[A-Za-z0-9']+", abstract))

    _h1(doc, "1. Introduction")
    _p(
        doc,
        "Radar has been used for indoor activity recognition, gait analysis and related monitoring tasks because it does not require visible light [CITATION NEEDED]. "
        "A moving body changes range, Doppler and phase; those time series can be arranged into micro-Doppler maps or point clouds and then classified. "
        "The same properties that make radar useful in dark or occluded rooms also make it attractive where a camera is not wanted. "
        "This paper is concerned with what those maps actually contain when more than one radar is present, not with advertising a sensing platform.",
        align="justify",
    )
    _p(
        doc,
        "Typical HAR papers follow a short path: one radar, one processing chain, a micro-Doppler image or point cloud, a CNN, Transformer or PointNet, and a fixed activity label [CITATION NEEDED]. "
        "A large fraction of the literature then reports classification accuracy on the same dataset and the same sensor that produced the training examples. "
        "Under that protocol a high score is expected. It does not, by itself, say whether the score would survive a change of carrier frequency, waveform or front-end processing.",
        align="justify",
    )
    _p(
        doc,
        "High within-radar accuracy therefore does not imply that the model has isolated human motion. "
        "The same array can carry sensor characteristics, the processing pipeline, native resolution and a radar-specific response, all of which are stable inside one instrument and all of which co-occur with the action label. "
        "A classifier that uses those cues will look competent until the instrument changes.",
        align="justify",
    )
    _p(
        doc,
        "Cross-radar shift is not ordinary photographic domain shift. "
        "Carrier frequency, waveform, range processing, Doppler mapping, STFT windowing, MTI and range-bin selection each reshape the observation. "
        "The same physical movement does not produce the same radar representation. "
        "Treating the problem as a style transfer between images skips that measurement step.",
        align="justify",
    )
    _p(
        doc,
        "The present study does not introduce a universal encoder. "
        "It starts from the public paired observations and removes, one layer at a time, the confounds that can be removed from the files: plotting conventions, numerical reconstruction, physical Doppler-to-velocity scaling, STFT/MTI mismatch, first-order calibration and simple alignment. "
        "At each layer we ask whether the cross-radar collapse recedes and whether radar identity remains recoverable.",
        align="justify",
    )
    _p(
        doc,
        "Four contributions follow. "
        f"(1) A timestamp-paired 10/24/77 GHz HAR benchmark on {d('N_PAIRED_ACQ')} acquisitions, showing a large gap between within-radar accuracy and several cross-radar transfers. "
        "(2) A reconstruction path from official plotted spectrograms through cropped images to numerical Doppler maps and a 24/77 GHz radial-velocity grid, with Radar-ID and activity probes kept on labeled protocols. "
        "(3) A systematic check of signal-processing harmonization, known-radar calibration, domain alignment, interpretable motion factors and motion-conditioned measurement response. "
        "(4) Evidence that the public pairs lack spatial and acquisition observables needed for robust sensor-invariant modeling, and a next-recording protocol stated as Outlook rather than as an experimental result. "
        "No claim is made that a known-radar adapter solves transfer.",
        align="justify",
    )

    _h1(doc, "2. Methods")
    _h2(doc, "2.1 Paired multi-radar dataset and acquisition matching")
    _p(
        doc,
        f"CI4R provides simultaneous recordings at 10, 24 and 77 GHz [CITATION NEEDED]. "
        f"We keep {d('N_PAIRED_ACQ')} acquisitions whose timestamp is shared across the three radars and whose action folder agrees "
        f"(Away, Bend, Crawl, Kneel, Pick, Sit, Towards). "
        f"{d('N_CONFLICT')} additional timestamp-matched recordings were excluded because the action labels conflicted across radars. "
        f"The split is {d('SPLIT_TRAIN')} / {d('SPLIT_VAL')} / {d('SPLIT_TEST')} at acquisition level: a timestamp never appears in two splits, so the three radars of one trial stay together. "
        "CI4R filenames do not supply a reliable subject identifier aligned to timestamp. Filename prefixes are therefore not treated as subjects and are not used as a subject-independent split.",
        align="justify",
    )

    _h2(doc, "2.2 Evaluation protocols")
    _p(
        doc,
        "Within-radar evaluation trains and tests on the same sensor. "
        "Single-source cross-radar evaluation trains on one radar and tests on another. "
        "Leave-one-radar-out (LORO) trains on two radars and tests on the held-out radar; that protocol is domain generalization and is reported as supplementary. "
        "CORAL is unsupervised domain adaptation (UDA) that uses unlabeled target training data; it is not DG. "
        "Linear Radar-ID and activity probes on pooled features, together with paired cosine similarity, are diagnostics. "
        "A three-radar probe (10/24/77 GHz) is never treated as interchangeable with a two-radar 24/77 GHz probe. "
        "Representation-level, encoder-embedding and motion-factor probes are reported in separate groups.",
        align="justify",
    )

    _h2(doc, "2.3 Official representation benchmark")
    _p(
        doc,
        "The official representation is the public spectrogram PNG released with CI4R for each radar. "
        "Native image sizes differ across the three sensors. "
        "Source-only 7-class classifiers trained on these images form the within-radar and single-source cross-radar numbers in Section 3.1. "
        "No claim is made that the PNG is a physical measurement.",
        align="justify",
    )

    _h2(doc, "2.4 Representation normalization")
    _p(
        doc,
        "A cropped, common-postprocessed image is used only to test whether display conventions explain the domain gap. "
        "Color bars, borders and empty margins are removed and the remaining field is resampled. "
        "The result is still a plotted PNG, not a physical representation.",
        align="justify",
    )

    _h2(doc, "2.5 Numerical Doppler reconstruction")
    _p(
        doc,
        "Numerical maps are rebuilt from native short-time processing rather than from screenshots. "
        f"The 10 GHz stream is a XeThru range-profile baseband sequence: each frame stores {d('N_FLOAT_10')} floats "
        f"({d('N_IQ_10')} in-phase and {d('N_IQ_10')} quadrature samples). "
        f"The 24 GHz header gives carrier frequency {d('FC_24_GHZ')} GHz, sweep time {d('TSWEEP_MS')} ms, "
        f"{d('NTS_24')} range samples and bandwidth {d('BW_GHZ')} GHz. "
        "The 77 GHz chain starts from raw IQ, applies a range FFT, a range gate and an STFT. "
        f"All three radars are placed on a common Doppler grid S(τ, f_D) with f_D ∈ [{d('FD_MIN')}, {d('FD_MAX')}] Hz. "
        "Invertible normalization is a train-only z-score of clipped decibels.",
        align="justify",
    )

    _h2(doc, "2.6 Physical velocity representation")
    _p(
        doc,
        f"Radial velocity is obtained only for 24 and 77 GHz, where a documented carrier exists: "
        f"v = c f_D / (2 f_c), with f_c = {d('FC_24_GHZ')} GHz and {d('FC_77_GHZ')}, respectively. "
        f"Both maps are resampled onto the physical intersection S(τ, v), v ∈ [{d('VEL_LO')}, {d('VEL_HI')}] m/s. "
        "The interval is the overlap of the two physical axes, not a tuned crop. "
        "The 10 GHz effective carrier is unresolved in the public files; a published range-bin sampling rate is not used as f_c, and 10 GHz is excluded from all velocity maps.",
        align="justify",
    )

    _h2(doc, "2.7 Signal-processing harmonization")
    _p(
        doc,
        f"Native STFT settings differ. The 24 GHz numerical chain uses a Hamming window of {d('STFT_24_WIN')} s with hop {d('STFT_24_HOP')} s; "
        f"the 77 GHz chain uses a Hann window of {d('STFT_77_WIN')} s with hop {d('STFT_77_HOP')} s. "
        f"A unified STFT resets both radars to a Hamming window of {d('U1_WIN')} s, {d('U1_OVERLAP')}% overlap (hop {d('U1_HOP')} s) and nfft = {d('U1_NFFT')}, "
        "while official range selection and official MTI status are left unchanged. "
        "A further audit transplants the 24 GHz Butterworth MTI operator onto 77 GHz at a physical cutoff. "
        "A common range-metre gate is not available from the recovered pipelines.",
        align="justify",
    )

    _h2(doc, "2.8 Representation and information probes")
    _p(
        doc,
        "A logistic Radar-ID probe on pooled 16×16 features estimates how recoverable the sensor is. "
        "An activity probe on the same features estimates motion-related information. "
        "The two readouts are complementary: a high activity score does not imply a low sensor score. "
        "Pair separation is same-timestamp cosine minus different-timestamp cosine.",
        align="justify",
    )

    _h2(doc, "2.9 Calibration and learned alignment tests")
    _p(
        doc,
        "CORAL UDA, a domain-adversarial encoder, radar-specific adapters, and train-only affine or histogram matches (C0–C4) are treated as hypotheses, not as proposed state-of-the-art models. "
        "Gates were preregistered on validation statistics. "
        "Adapter accuracies in the main text are known-radar joint-task numbers, not cross-radar zero-shot scores.",
        align="justify",
    )

    _h2(doc, "2.10 Motion-factor representation")
    _p(
        doc,
        f"{d('N_FACTORS')} energy-weighted factors are computed with pre-registered Doppler bands. "
        "They cover Doppler statistics, temporal dynamics, spectral shape, onset/offset and two-dimensional moments. "
        "The full list is in the Supplementary. Action labels are not used as calibration inputs.",
        align="justify",
    )

    _h2(doc, "2.11 Motion-conditioned measurement response")
    _p(
        doc,
        f"On unified-STFT log-energy, the paired velocity-marginal difference is D_i(v) = log(P_77,i(v) + ε) − log(P_24,i(v) + ε), "
        f"restricted to an energy mask with {d('N_VALID_BINS')} valid bins. "
        "Source-only factors q (no target radar, no action label) predict D(v) by Ridge. "
        "The train-mean D(v) is the baseline. This is paired known-radar regression, not unseen-radar DG.",
        align="justify",
    )

    _h2(doc, "2.12 Low-rank 2D paired measurement difference")
    _p(
        doc,
        "The full-field difference is Δ_i(τ, v) = X_77,i(τ, v) − X_24,i(τ, v) in log-energy. "
        f"TRAIN-only PCA is fit for ranks 2, 5, 10 and 20; validation reconstruction error selects K = {d('K_SELECTED')}. "
        "The TEST oracle projects the true TEST Δ onto the TRAIN basis. "
        "That quantity is a representation ceiling, not a deployable translator. "
        "A second model predicts PCA scores from q.",
        align="justify",
    )

    _h1(doc, "3. Results")
    _h2(doc, "3.1 High within-radar accuracy masks poor cross-radar transfer")
    _p(
        doc,
        f"On official plotted spectrograms, within-radar 7-class accuracy is {d('R0_10_WITHIN')} at 10 GHz, {d('R0_24_WITHIN')} at 24 GHz and {d('R0_77_WITHIN')} at 77 GHz. "
        f"The same source-only protocol yields {d('R0_24_TO_77')} for 24→77 and {d('R0_77_TO_24')} for 77→24 (Table 2, Fig. 2). "
        "The remaining off-diagonal entries are also far below the corresponding diagonal. "
        "The same model family that performs reliably within individual radar domains does not transfer reliably across radar systems.",
        align="justify",
    )
    _figure(
        doc,
        fig_dir / "fig2_within_cross.png",
        "Fig. 1. Official-representation 7-class HAR. Rows are the training radar; columns are the test radar. The diagonal is same-sensor evaluation.",
        12.5,
    )

    _h2(doc, "3.2 Plotting artifacts explain only part of the gap")
    _p(
        doc,
        f"Cropping official spectrograms raises 10→24 accuracy from {d('R0_10_TO_24')} to {d('R1_10_TO_24')}. "
        "Display convention is therefore a real confound. "
        f"A three-radar Radar-ID probe remains {d('RID_R0_3CLASS')} on official images and {d('RID_R1_3CLASS')} after cropping. "
        "Other cross-radar directions stay weak (Table 2). "
        "Plotting artifacts matter, but they are not the dominant explanation.",
        align="justify",
    )

    _h2(doc, "3.3 Numerical Doppler reconstruction retains radar identity")
    _p(
        doc,
        f"After numerical reconstruction, the three-radar Radar-ID probe is {d('RID_R2D_3CLASS')} and the matching representation-level activity probe is {d('ACT_R2D_REP')}. "
        f"Within-radar 7-class accuracy on these maps is {d('R2D_W10')} / {d('R2D_W24')} / {d('R2D_W77')}. "
        "Single-source cross-radar transfer remains weak. "
        "Removing axes, color bars and frames does not remove sensor identity.",
        align="justify",
    )
    _figure(
        doc,
        fig_dir / "fig4_radar_id.png",
        "Fig. 2. Radar-ID by protocol. Panel A: three-radar probe on official, cropped and numerical maps. Panel B: two-radar 24/77 GHz probe. The panels are not one series.",
        15.0,
    )

    _h2(doc, "3.4 Physical velocity alignment produces only limited convergence")
    _p(
        doc,
        "The next numbers are two-radar 24/77 GHz diagnostics and must not be read as a drop of the three-radar probe in Section 3.3. "
        f"On numerical Doppler maps restricted to 24/77 GHz, representation-level Radar-ID is {d('RID_R2D_2477')}. "
        f"After both radars are placed on the common radial-velocity axis, that two-class Radar-ID is {d('RID_R2V_2477')}. "
        f"The matching activity probes are {d('ACT_R2D_2477')} and {d('ACT_R2V_2477')}. "
        f"Pair separation changes from {d('SEP_R2D')} to {d('SEP_R2V')}. "
        "Carrier-frequency-dependent Doppler scaling accounts for only a small part of the shift. 10 GHz cannot join this comparison.",
        align="justify",
    )

    _h2(doc, "3.5 Harmonizing STFT and MTI does not remove sensor signatures")
    _p(
        doc,
        f"Still on the 24/77 GHz pair, unified STFT leaves Radar-ID at {d('U1_RID')}, identical to the velocity-aligned baseline {d('U0_RID')}. "
        f"Transplanting the 24 GHz MTI operator raises Radar-ID to {d('U2_RID')}. "
        f"A probe on the velocity-energy profile of the uncalibrated maps is {d('VEL_RID_M0')}. "
        "The result is not that MTI is harmful. Matching these processing choices was insufficient to produce a common measurement distribution.",
        align="justify",
    )

    _h2(doc, "3.6 Generic alignment and radar-specific adapters do not yield invariance")
    _p(
        doc,
        f"A0–A2 in Table 5 are known-radar joint-task accuracies, not zero-shot transfer. "
        f"Shared-encoder accuracy is {d('A0_ACC')}. Radar-specific adapters give {d('A1_ACC')}; adding a pairing term gives {d('A2_ACC')}. "
        f"Radar-ID on A2 embeddings remains {d('RID_A2_EMBED')}. "
        f"Same-acquisition embedding separation moves from {d('PAIR_A0')} to {d('PAIR_A1')} to {d('PAIR_A2')}. "
        "Explicit adapters and pair alignment did not create a radar-invariant embedding. The pairing term that was meant to hide the sensor made same-trial embeddings less, not more, aligned, and activity accuracy fell.",
        align="justify",
    )

    _h2(doc, "3.7 Motion factors remain strongly sensor dependent")
    _p(
        doc,
        f"A linear probe on the {d('N_FACTORS')} factors yields activity {d('FAC_ACT')} and Radar-ID {d('FAC_RID')}. "
        f"A pre-registered stable subset yields activity {d('FAC_ACT_STABLE')} and Radar-ID {d('FAC_RID_STABLE')}. "
        "These probes live on factor vectors, not on the spectrogram pixels of Sections 3.3–3.5. "
        "Interpretable motion statistics still mix sensor response (Fig. 3).",
        align="justify",
    )
    _figure(
        doc,
        fig_dir / "fig6_tradeoff.png",
        "Fig. 3. Activity or HAR information versus Radar-ID, split by probe family. Panel A: representation-level probes. Panel B: known-radar encoder. Panel C: motion-factor probes. Axes are not pooled across panels.",
        15.5,
    )

    _h2(doc, "3.8 Radar response varies with motion state")
    _p(
        doc,
        f"If the 24/77 GHz difference were a fixed hardware curve, the train-mean D(v) would already explain most paired variation. "
        f"It does not: 24→77 R² = {d('D2477_B0_R2')}. "
        f"Source-only Ridge on motion factors raises that figure to {d('D2477_B1_R2')}; the reverse direction rises from {d('D7724_B0_R2')} to {d('D7724_B1_R2')} (Table 6, Fig. 4). "
        "These R² values measure prediction of paired Doppler-energy differences, not HAR accuracy. "
        "The measurement difference is not a fixed sensor-response curve. It changes with the observed motion state.",
        align="justify",
    )
    _figure(
        doc,
        fig_dir / "fig7_dvq.png",
        "Fig. 4. Motion-conditioned velocity-marginal transfer. B0 is the train-mean D(v). B1 predicts D(v) from source-only motion factors. The vertical axis is not HAR accuracy.",
        11.5,
    )

    _h2(doc, "3.9 Predictable marginal response does not imply successful sensor translation")
    _p(
        doc,
        f"Applying the predicted D(v | q) as a symmetric correction moves 24/77 Radar-ID from {d('CAL_RID_BEFORE')} to {d('CAL_RID_AFTER')}. "
        f"The activity probe moves from {d('CAL_ACT_BEFORE')} to {d('CAL_ACT_AFTER')}; pair separation moves from {d('CAL_SEP_BEFORE')} to {d('CAL_SEP_AFTER')}. "
        "Predicting D(v | q) does not mean that sensor identity can be removed.",
        align="justify",
    )

    _h2(doc, "3.10 Full 2D measurement translation remains limited")
    _p(
        doc,
        f"Rank-{d('K_SELECTED')} PCA of the two-dimensional paired difference explains {d('K20_TRAIN_VAR')} of TRAIN variance. "
        f"The TEST oracle—true TEST Δ projected onto the TRAIN basis—has R² = {d('ORACLE_R2_2D')}. "
        f"The q-driven model reaches {d('B2_R2_2477')} (24→77) and {d('B2_R2_7724')} (77→24). "
        f"Activity on predicted 24→77 maps falls from {d('B2_ACT_SRC')} on the source to {d('B2_ACT_PRED')} on the prediction "
        f"(true target maps: {d('B2_ACT_TRUE')}). "
        "Even the oracle projection captures less than half of unseen paired variation. "
        "The bottleneck is not only regression capacity (Fig. 5).",
        align="justify",
    )
    _figure(
        doc,
        fig_dir / "fig8_lowrank.png",
        "Fig. 5. Low-rank 2D paired-difference analysis. The TEST oracle is an upper bound obtained by projecting the true unseen difference field onto the TRAIN basis. It is not a deployed translator.",
        11.5,
    )
    _figure(
        doc,
        fig_dir / "fig9_ladder.png",
        "Fig. 6. Progressive diagnosis from high within-radar accuracy to missing spatial and acquisition variables. The figure is a reading order, not a scoreboard of failures.",
        13.0,
    )

    # Tables
    _h1(doc, "Tables")
    _table(
        doc,
        ["Item", "Value"],
        [
            ["Corpus", "CI4R multi-frequency HAR"],
            ["Radars", "10 / 24 / 77 GHz, timestamp-paired"],
            ["Paired acquisitions", d("N_PAIRED_ACQ")],
            ["Actions", d("N_ACTIONS")],
            ["Split (train / val / test)", f"{d('SPLIT_TRAIN')} / {d('SPLIT_VAL')} / {d('SPLIT_TEST')}"],
            ["Official representation", "public plotted spectrogram"],
            ["Numerical Doppler", "S(normalized time, Doppler Hz)"],
            ["Velocity maps", f"24/77 only; v in [{d('VEL_LO')}, {d('VEL_HI')}] m/s"],
        ],
        "Table 1. Dataset, radars, representations and split. The split is by acquisition timestamp; a filename prefix is not a subject.",
    )
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
    _table(
        doc,
        ["Pair", "Official (R0)", "Cropped (R1)"],
        [[a, d(b), d(c)] for a, b, c in pairs],
        "Table 2. Source-only 7-class HAR on official and cropped plotted images. These columns are not Radar-ID.",
    )
    _table(
        doc,
        ["Stage", "3-radar Radar-ID", "Activity or within-radar HAR", "Note"],
        [
            ["Official PNG", d("RID_R0_3CLASS"), f"{d('R0_10_WITHIN')} / {d('R0_24_WITHIN')} / {d('R0_77_WITHIN')}", "HAR diagonal"],
            ["Cropped PNG", d("RID_R1_3CLASS"), d("ACT_R1_REP"), "representation activity probe"],
            ["Numerical Doppler", d("RID_R2D_3CLASS"), d("ACT_R2D_REP"), "representation activity probe"],
        ],
        "Table 3. Three-radar Radar-ID along official, cropped and numerical representations. The activity column is not a single benchmark across rows.",
    )
    _table(
        doc,
        ["Stage", "24/77 Radar-ID", "24/77 activity", "Pair separation"],
        [
            ["Numerical Doppler 24/77", d("RID_R2D_2477"), d("ACT_R2D_2477"), d("SEP_R2D")],
            ["Velocity-aligned 24/77", d("RID_R2V_2477"), d("ACT_R2V_2477"), d("SEP_R2V")],
            ["Unified STFT", d("U1_RID"), d("U1_ACT"), d("U1_SEP")],
            ["Unified MTI", d("U2_RID"), d("U2_ACT"), d("U2_SEP")],
        ],
        "Table 4. Two-radar 24/77 GHz diagnostics. Do not compare the Radar-ID column with Table 3 as one series.",
    )
    _table(
        doc,
        ["Method", "HAR or activity", "Radar-ID", "Protocol"],
        [
            ["Shared encoder", d("A0_ACC"), d("RID_A0_INPUT"), "known-radar joint task"],
            ["Radar-specific adapters", d("A1_ACC"), d("RID_A1_ADAPTER"), "known-radar; not improved"],
            ["Adapters + pairing", d("A2_ACC"), d("RID_A2_EMBED"), "known-radar; not improved"],
            ["C0–C4 calibration", d("CBEST_ACT"), d("CBEST_RID"), "VAL gate fail; 3-radar Radar-ID"],
            ["Per-velocity affine", d("M2_ACT"), d("M2_RID"), "gate fail; 24/77 Radar-ID"],
        ],
        "Table 5. Alignment, adapters and calibration. A0–A2 HAR is not cross-radar zero-shot. Radar-ID protocols differ by row.",
    )
    _table(
        doc,
        ["Direction", "B0 R²", "B1 R²", "B1 MAE", "B1 corr."],
        [
            ["24→77", d("D2477_B0_R2"), d("D2477_B1_R2"), d("D2477_B1_MAE"), d("D2477_B1_CORR")],
            ["77→24", d("D7724_B0_R2"), d("D7724_B1_R2"), d("D7724_B1_MAE"), d("D7724_B1_CORR")],
        ],
        "Table 6. Motion-conditioned velocity-marginal D(v | q). R² is prediction of paired transfer, not HAR accuracy.",
    )
    _table(
        doc,
        ["Quantity", "Value"],
        [
            ["Selected rank K", d("K_SELECTED")],
            ["TRAIN explained variance", d("K20_TRAIN_VAR")],
            ["TEST oracle R²", d("ORACLE_R2_2D")],
            ["B2 R², 24→77", d("B2_R2_2477")],
            ["B2 R², 77→24", d("B2_R2_7724")],
        ],
        "Table 7. Low-rank 2D paired-difference model. The TEST oracle is not deployment performance.",
    )

    _h1(doc, "4. Discussion")
    _p(
        doc,
        "High within-radar HAR accuracy does not show that a model mainly encodes human motion. "
        "A representation can carry high activity information and near-perfect radar identity at the same time. "
        "The first quantity is what most papers report; the second is what a Radar-ID probe measures. "
        "Calling a high-accuracy spectrogram encoder motion-centric is a category error unless radar identity has been shown to be unrecoverable under a stated protocol.",
        align="justify",
    )
    _p(
        doc,
        "Cross-radar correspondence is not radar invariance. "
        "Paired learning can pull the same action closer across sensors and still leave a linear probe that names the radar. "
        "A map that makes a 24 GHz array look 77-like to a domain classifier can discard activity cues and still fail to match the two-dimensional measurement field.",
        align="justify",
    )
    _p(
        doc,
        "The layers that can be removed from the public files—plotting, Doppler scaling, STFT mismatch, MTI mismatch and first-order calibration—each contribute something. "
        "None of them accounts for the full domain shift. "
        "After those layers, radar identity remains easy to read on the protocol that applies.",
        align="justify",
    )
    _p(
        doc,
        "The motion-conditioned result changes the interpretation of what remains. "
        "D(v | q) is predictable, so the 24/77 GHz difference is not unstructured noise. "
        "The same difference depends on the observed motion state, so the shift is not a global style that a single filter can undo. "
        "Same action is not the same measurement operator.",
        align="justify",
    )
    _p(
        doc,
        "The public pairs do not record physical range, subject–radar distance, aspect, body orientation, radar pose, or reliable subject and environment identifiers. "
        "A geometry-aware hypothesis is therefore not testable on this corpus. "
        "The residual after velocity alignment is not evidence that geometry caused that residual; those fields were never written down.",
        align="justify",
    )
    _p(
        doc,
        "The next measurement is not a larger network on the same timestamps. "
        "It is a recording design that can hold motion fixed while the radar changes, and hold the radar fixed while geometry changes. "
        "Only then can sensor effect, geometry effect and subject effect be separated. "
        "A synchronized 24/77 GHz protocol that writes distance, aspect, orientation, raw IQ, the full radar configuration and video timestamps is specified as Outlook. "
        "Mock toolkit files produced while designing that protocol are not results in this paper.",
        align="justify",
    )

    _h1(doc, "5. Conclusions")
    _p(
        doc,
        f"Within-radar HAR on official CI4R spectrograms is high ({d('R0_10_WITHIN')} / {d('R0_24_WITHIN')} / {d('R0_77_WITHIN')}), "
        f"and several cross-radar transfers collapse (24→77 = {d('R0_24_TO_77')}). "
        "Radar identity remains recoverable after cropping, numerical Doppler reconstruction and 24/77 GHz velocity alignment. "
        "Harmonized STFT/MTI, first-order calibration and known-radar adapters do not produce radar-invariant features under the protocols used here.",
        align="justify",
    )
    _p(
        doc,
        "The main implication is not that cross-radar HAR is impossible, but that it should not be treated as a conventional image-domain transfer problem. "
        "Future progress requires controlled multi-radar measurements with explicit spatial and acquisition metadata before increasingly complex invariant encoders can be meaningfully evaluated.",
        align="justify",
    )

    _h1(doc, "6. Limitations")
    _p(
        doc,
        f"The paired subset contains {d('N_PAIRED_ACQ')} acquisitions and {d('N_ACTIONS')} actions in one reported laboratory setting. "
        "CI4R subject metadata are incomplete relative to those timestamps, so subject-independent claims are not made. "
        "The 10 GHz effective carrier is unresolved; 24/77 velocity results do not extend to 10 GHz. "
        "A joint physical range axis is not recovered. Distance, aspect and orientation are unavailable, so geometry-aware models are blocked. "
        "Cross-environment transfer is not tested. "
        "The two-dimensional empirical difference model is not a radar forward model.",
        align="justify",
    )

    _h1(doc, "Data availability")
    _p(
        doc,
        "CI4R is a public third-party corpus [CITATION NEEDED]. "
        f"The acquisition-level split used here is {d('SPLIT_TRAIN')}/{d('SPLIT_VAL')}/{d('SPLIT_TEST')}. "
        "A public archive of derived files: TODO (no DOI assigned).",
        align="justify",
        first_line=False,
    )
    _h1(doc, "Code availability")
    _p(
        doc,
        "Research code: repository TODO. No public URL or DOI is claimed in this draft.",
        align="justify",
        first_line=False,
    )
    _h1(doc, "Author contributions")
    _p(doc, "[TODO]", first_line=False)
    _h1(doc, "Conflict of interest")
    _p(doc, "[TODO]", first_line=False)
    _h1(doc, "References")
    _p(
        doc,
        "[CITATION NEEDED] Official CI4R dataset publication. Bibliographic fields to be copied from the publisher; a DOI is not invented here.",
        first_line=False,
        align="justify",
    )
    _p(
        doc,
        "[CITATION NEEDED] Additional radar HAR and domain-adaptation references to be added during manual writing.",
        first_line=False,
        align="justify",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    xml = out_path.read_bytes()
    # page estimate: not exact; report word count of body via paragraphs
    return {"abstract_words": n_abs, "path": str(out_path)}
