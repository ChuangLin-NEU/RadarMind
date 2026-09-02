# Limitations

These limits are part of the RadarMind evidence, not an after-the-fact disclaimer.

1. Robust unseen-radar transfer remains unresolved. Within-radar accuracies on the CI4R paired split are high; several source-only transfers, including 24→77 at 0.039, are not.

2. Shared representation learning improves known-radar correspondence without eliminating radar identity. M1 raises activity accuracy and same-action retrieval on radars seen in training. Radar-ID on those embeddings remains 1.000.

3. Radar-ID remains high after physical harmonization. Numerical Doppler (R2-D) 3-radar Radar-ID is 0.993. Velocity-aligned 24/77 Radar-ID is 0.939. Harmonization is not a sensor-invariant representation.

4. Measurement geometry is insufficiently controlled in the current public data. Distance, aspect, orientation, and scene layout are not fully recorded for the CI4R paired triples used here.

5. Continuous inference is labeled-interval replay on mRI test recordings. Unlabeled gaps are not scored. This is not full-stream continuous sensing, and no background class was added to make the stream look complete.

6. Temporal event boundary localization remains weak. Confidence-aware labeled-interval replay has segment F1 0.152 and median boundary error 20.6 s.

7. Open-set rejection remains limited. Leave-one-action-out on mRI is a rejection protocol, not open-world recognition.

8. Prototype AUROC is 0.628 ± 0.125. That is better than MSP in the same protocol and is not a deployment-grade unknown-action detector.

9. Open-world and open-vocabulary motion recognition are not implemented. A rejected window is `UNKNOWN`. Nearest-prototype identity is not a semantic label for an unseen action.

10. Language interaction does not improve HAR accuracy. The language layer reads structured Motion Intelligence and timeline records. It does not receive raw IQ, spectrograms, point clouds, or hidden neural features, and it is not an evidence source.
