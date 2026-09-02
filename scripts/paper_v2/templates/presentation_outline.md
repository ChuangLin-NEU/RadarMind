# 15–20 min outline (research talk)

Timing: about one minute per slide. Not a software demo.

1. Background — radar HAR, high published same-sensor scores
2. Research question — how much of the cross-radar failure is plot, processing, physics, measurement?
3. Dataset — @@N_PAIRED_ACQ@@ paired, @@N_ACTIONS@@ actions, timestamp split @@SPLIT_TRAIN@@/@@SPLIT_VAL@@/@@SPLIT_TEST@@
4. Within vs cross — diagonal @@R0_10_WITHIN@@/@@R0_24_WITHIN@@/@@R0_77_WITHIN@@ vs 24→77 @@R0_24_TO_77@@
5. Representation audit — 3-radar Radar-ID official/cropped = @@RID_R0_3CLASS@@; 10→24 @@R0_10_TO_24@@→@@R1_10_TO_24@@
6. Numerical Doppler — 3-radar Radar-ID @@RID_R2D_3CLASS@@
7. Velocity alignment — 24/77 two-class @@RID_R2D_2477@@ → @@RID_R2V_2477@@; 10 GHz excluded
8. Signal processing — unified STFT @@U1_RID@@, transplanted MTI @@U2_RID@@
9. Alignment / calibration — known-radar A2 @@A2_ACC@@ with embedding Radar-ID @@RID_A2_EMBED@@; gates fail
10. Motion-conditioned response — D(v|q) $R^2$ @@D2477_B1_R2@@ / @@D7724_B1_R2@@
11. Low-rank ceiling — TRAIN @@K20_TRAIN_VAR@@, TEST oracle @@ORACLE_R2_2D@@
12. Scientific conclusion — stop empirical translation on this corpus
13. Next-data design — 24+77, distance / aspect / orientation, raw IQ
14. RadarMind vision — Learn Motion, Not Sensor: NOT YET ACHIEVED
