# Limitations

1. CI4R 没有可靠 official subject ID。
2. prefix-group-disjoint 不能称 cross-subject。真正 cross-subject 来自 mRI。
3. mRI continuous replay 只覆盖 labeled intervals。
4. 本机没有 mRI raw CSV，因此没有完整 UNLABELED full-stream inference。
5. RadHAR 没有 subject metadata，official split 不能解释成 cross-subject。
6. Cross-environment 当前不可做：没有可用的环境标签协议。
7. CI4R 24 GHz official PNG 原生分辨率与 10/77 明显不同，可能形成 representation confound。
8. Shared Motion Radar-ID probe 仍为 1.0，v0.1 不是 sensor-invariant。
9. Open-set ≠ open-vocabulary ≠ open-world。Phase 5 只做 open-set recognition。
10. 当前实验主要单 seed，不是最终论文统计版本。
