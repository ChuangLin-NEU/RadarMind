# Negative results (keep)

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
