# RadarMind PPT Audit

- File: `RadarMind_Presentation.pptx` (opened by PowerPoint 16.0 and python-pptx)
- Builder: `build_radarmind_presentation.js` (PptxGenJS 4.0.1)
- Source of truth: `Draft-RadarMind.pdf` (34 pages)
- Content map: `reports/ppt_content_map.md`
- Renders: `renders/slide_01.png` … `renders/slide_23.png`
- Contact sheet: `renders/contact_sheet.png`
- Slides: **23 / 23**
- Fonts: Aptos (English titles/metrics), Microsoft YaHei (mixed / Chinese). Not embedded.

## Required checklist

1. PASS  slides 23/23
2. PASS  quantitative values verified against Draft-RadarMind.pdf
3. PASS  no unsupported scientific claim (manual review; see phrase note)
4. PASS  no raw-radar-to-LLM arrow — isolation box + red X on slides 6 and 18
5. PASS  CI4R prefix not labeled as subject — slide 7 “Prefix ≠ subject”; slide 11 footer
6. PASS  continuous labeled-interval boundary present — slide 15 footer
7. PASS  open-set vs open-world distinction present — slide 16 Claimed / Not claimed
8. PASS  R² vs HAR distinction present — slide 14 “Reconstruction R² ≠ HAR Accuracy”
9. PASS  TEST oracle warning present — slide 14 “TEST oracle ≠ deployable translator”
10. PASS  unseen-radar generalization marked unresolved — slides 1, 5, 19, 20, 23
11. PASS  no text overflow after visual QA of 23 PNG renders
12. PASS  no damaging element overlap after visual QA
13. PASS  fonts valid (system Aptos / Microsoft YaHei)
14. PASS  pptx successfully opens in PowerPoint

## Phrase-scanner note

An automatic substring scan flagged `open-vocabulary`, `deployable translator`, and a cross-slide concatenation that looked like “subject-independent CI4R”. All three are **boundary statements**, not claims:

- Slide 16: Open-vocabulary recognition is listed under **Not claimed**.
- Slide 14: **TEST oracle ≠ deployable translator**.
- Slides 7–8: “True subject-independent” is attached only to **mRI**. CI4R is labeled **Prefix ≠ subject**. RadHAR is labeled **not subject-independent**.

No slide asserts universal sensor invariance, open-world recognition, full-stream continuous sensing, or that the LLM improves HAR / transfer / open-set / localization.

## Quantitative values checked

| Value | Location in deck | PDF source |
| --- | --- | --- |
| mRI Acc / F1 0.478 / 0.420 | Slide 8 | Abstract; Table 2 |
| RadHAR Acc / F1 0.949 / 0.949 | Slide 8 | Abstract; Table 2 |
| CI4R within 0.843 / 0.898 / 0.922 | Slides 8, 10, 11 | Table 2 / Table 3 |
| mRI T=1 → T=32  0.254 → 0.478 | Slide 9 | Table 2; §3.3 |
| RadHAR T=1 → T=60  0.607 → 0.949 | Slide 9 | Table 2; §3.3 |
| 10→24 0.143; 24→77 0.039; 77→24 0.163 | Slides 10–11 | Table 3 |
| Full 3×3 matrix including 0.353 / 0.373 / 0.275 | Slide 11 | Table 3 |
| Official-plot Radar-ID 1.000 | Slide 11 | §3.4 |
| M0 0.769 / 0.718 / 1.000 | Slide 12 | Table 4 |
| M1 0.932 / 0.935 / 1.000 | Slide 12 | Table 4 |
| M2 0.878 / 0.864 / 1.000 | Slide 12 | Table 4 |
| 3-radar R0 / R1 / R2-D  1.000 / 1.000 / 0.993 | Slide 13 | Table 5 |
| 24/77 R2-D / R2-V / STFT / MTI  1.000 / 0.939 / 0.939 / 0.980 | Slide 13 | Table 5 |
| D(v\|q) R² 0.744 / 0.575 | Slide 14 | Table 5 |
| 2D R² 0.418 / 0.300 / 0.173 | Slide 14 | Table 5 |
| Window 0.494 / 0.419; Frame 0.489 / 0.416 | Slide 15 | Table 6 |
| Segment F1 0.152; 190.8 frames / 20.6 s | Slide 15 | Table 6 |
| MSP / Energy / Prototype AUROC 0.476±0.156 / 0.540±0.144 / 0.628±0.125 | Slide 16 | Table 6 |
| Prototype FPR95 0.791±0.219 | Slide 16 | Table 6 |
| boxing 0.999 / 0.004 / −0.035 / 3.073 / 0.176 | Slide 17 | §3.6; Table 7 |
| 327 paired triples | Slides 7, 21 | Table 1 |

Values are copied as printed in the PDF. No rounding, no merged leaderboard axis.

## Layout / XML checks

- Slide size: 13.333 in × 7.5 in (16:9).
- After removing negative-extent line marks (the only PowerPoint-open blocker), no negative `cx`/`cy` remains.
- Off-slide elements: none at the ±0.04 in tolerance used by `scripts/audit_radarmind_ppt.py`.
- Fonts below 9 pt: none. Footnotes are 10 pt; protocol notes 9–11 pt.
- Charts/heatmaps/timelines are native shapes, not PDF screenshots.
- Slide 13 keeps the **3-radar probe** and the **24/77 probe** in separate panels (not one continuous curve).
- Slide 15 timeline is a schematic of labeled-interval replay, not a dumped recording and not claimed as full-stream sensing.

## Visual QA (PNG renders)

Reviewed all 23 exported slides plus `renders/contact_sheet.png`.

Fixes applied after first render:

1. Cover: `RadarMind` is the 32 pt title; pipeline nodes carry simple line-style glyphs.
2. Slide 19: seven capabilities moved from one cramped row to a 4+3 grid so status labels remain readable.
3. X-marks on slides 6 and 18 rebuilt without negative geometry so PowerPoint can open the file.

Remaining design notes (not defects):

- Cover icons are geometric marks, not clip-art or photographs.
- Continuous timeline is illustrative of GT / raw windows / smoothed events.
- No animation (static deck, as specified).

## Scientific boundaries present in the deck

- No universal sensor-invariant representation.
- Robust unseen-radar generalization remains unresolved.
- CI4R prefix-group-disjoint is not called subject-independent.
- True subject-independent evaluation is stated only for mRI.
- Continuous experiment is labeled-interval replay.
- Open-set is LOAO rejection, not open-world / open-vocabulary.
- LLM does not receive raw IQ, spectrogram, point cloud, or hidden features.
- LLM is not claimed to raise HAR / transfer / open-set / localization.
- Motion-conditioned R² is a reconstruction metric.
- TEST oracle is an upper bound, not a translator.
- Geometry / distance / aspect / orientation are plausible causes only.
- Three datasets are complementary instruments, not one leaderboard.

## Output files

| Path | Role |
| --- | --- |
| `RadarMind_Presentation.pptx` | Deliverable deck |
| `build_radarmind_presentation.js` | Reproducible builder |
| `reports/ppt_content_map.md` | Slide ↔ PDF evidence map |
| `reports/radarmind_ppt_audit.md` | This report |
| `renders/` | 23 slide PNGs + contact sheet |
| `scripts/audit_radarmind_ppt.py` | XML / number audit helper |
