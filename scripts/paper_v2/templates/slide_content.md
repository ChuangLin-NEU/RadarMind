# Slide copy (14 slides)

## Slide 1. Background
Title: Same-sensor radar HAR looks solved.
Spoken: Indoor radar HAR is usually evaluated on one radar and one processing chain. Published within-radar numbers are high. That invites the next claim: the encoder has learned motion.
Do not say: we built a full RadarMind system. Continuous / open-set / LLM are out of this talk.

## Slide 2. Research question
Title: Why does transfer across radars fail?
Question: How much of the 10/24/77 GHz gap is plotting, numerical reconstruction, Doppler-to-velocity scaling, signal processing, or a residual measurement response?
We do not claim a new invariant encoder.

## Slide 3. Dataset
Title: @@N_PAIRED_ACQ@@ timestamp-paired acquisitions.
- CI4R 10 / 24 / 77 GHz, @@N_ACTIONS@@ actions
- Split @@SPLIT_TRAIN@@ / @@SPLIT_VAL@@ / @@SPLIT_TEST@@ by acquisition timestamp
- Prefix is not subject
- No distance, aspect, orientation, or joint range axis
Figure: fig1_benchmark.png

## Slide 4. Within vs cross
Title: High diagonal, collapsed off-diagonal.
- Within official-representation HAR: @@R0_10_WITHIN@@ / @@R0_24_WITHIN@@ / @@R0_77_WITHIN@@
- 24→77: @@R0_24_TO_77@@
Figure: fig2_within_cross.png

## Slide 5. Representation audit
Title: Plotting is a confound, not the cause.
- 3-radar Radar-ID official = @@RID_R0_3CLASS@@, cropped = @@RID_R1_3CLASS@@
- 10→24 HAR: @@R0_10_TO_24@@ → @@R1_10_TO_24@@
Figure: fig3_representation.png

## Slide 6. Numerical Doppler
Title: Numerical maps still name the radar.
- 3-radar Radar-ID @@RID_R2D_3CLASS@@
- Representation activity probe @@ACT_R2D_REP@@
Figure: fig4_radar_id.png Panel A

## Slide 7. Velocity alignment
Title: Carrier scaling is a weak contributor — 24/77 pair only.
- Two-class Radar-ID @@RID_R2D_2477@@ → @@RID_R2V_2477@@
- Pair separation @@SEP_R2D@@ → @@SEP_R2V@@
- 10 GHz: effective fc unresolved
Do not say: the 3-radar probe fell from @@RID_R0_3CLASS@@ to @@RID_R2V_2477@@.

## Slide 8. Signal processing
Title: Unified STFT / MTI is not enough.
- Unified STFT 24/77 Radar-ID @@U1_RID@@
- Transplanted MTI @@U2_RID@@
Figure: fig5_processing.png

## Slide 9. Alignment and calibration
Title: Learned maps do not hide the sensor.
- Known-radar HAR: @@A0_ACC@@ > @@A1_ACC@@ > @@A2_ACC@@
- A2 embedding Radar-ID @@RID_A2_EMBED@@
- C0–C4 and per-velocity affine: gates FAIL
Figure: fig6_tradeoff.png
Line: activity information and radar information are not mutually exclusive. Panels are different probe families.

## Slide 10. Motion-conditioned response
Title: Part of D(v) follows source-only motion factors.
- 24→77 B1 $R^2$ @@D2477_B1_R2@@ (B0 @@D2477_B0_R2@@)
- 77→24 B1 $R^2$ @@D7724_B1_R2@@
- Calibration: 24/77 Radar-ID @@CAL_RID_BEFORE@@ → @@CAL_RID_AFTER@@
Figure: fig7_dvq.png
Line: $R^2$ here is transfer prediction, not HAR. Correspondence is not invariance.

## Slide 11. Low-rank ceiling
Title: Observable 2D translation is limited before any deep model.
- TRAIN explained variance @@K20_TRAIN_VAR@@ (K=@@K_SELECTED@@)
- TEST oracle $R^2$ @@ORACLE_R2_2D@@ (upper bound, not deployment)
- B2 @@B2_R2_2477@@ / @@B2_R2_7724@@
Figure: fig8_lowrank.png

## Slide 12. Scientific conclusion
Title: Stop empirical sensor translation on this corpus.
- High HAR ≠ motion-centric representation
- Same action ≠ a style transfer between radars
- Current public pairs cannot separate motion, sensor, geometry, subject
Figure: fig9_ladder.png

## Slide 13. Next-data design
Title: What must be recorded next.
- Synchronized 24 / 77 GHz
- Distance, aspect, orientation
- Raw IQ and full radar configuration
- Video timestamps
Acquisition Toolkit: protocol only. Mock files are not results.

## Slide 14. RadarMind vision
Title: Learn Motion, Not Sensor
Current status: **NOT YET ACHIEVED**
Next requirement: controlled multi-radar data with spatial metadata.
Do not close with a product roadmap.
