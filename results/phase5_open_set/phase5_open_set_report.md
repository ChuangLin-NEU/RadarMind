# Phase 5 Open-Set Motion Recognition MVP

不是 open-vocabulary，也不是 open-world。本阶段只做 open-set recognition：训练未见过的动作，测试时判断 KNOWN ACTION 或 UNKNOWN MOTION。

数据：mRI Phase 1B labeled windows，T=32，P=64，XYZ。subject-independent 划分未改。
每个 LOAO fold 重新训练 12 类 PointNet-GRU，没有复用 13 类 checkpoint。
阈值只在 validation known 上按 5% 分位数确定。unseen action 不进 train / val / threshold。

## Held-out selection

- **pose_1** (A_easy): SI PointNet-GRU XYZ closed-set F1=0.800, highest among pose_*
- **pose_4** (B_confused_pose): largest fraction of errors going to other pose_* (0.687); closed-set F1=0.020
- **walk** (C_semantic): walk is not a numbered pose; SI F1=0.452; kept as semantic held-out

## Primary folds (3)

### held-out = pose_1

known classes = 12；test known=1109；test unknown=98；blocked unseen non-test=540

| method | AUROC | AUPR | FPR95 | Known Acc | Known Macro-F1 | Open-set Acc |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0.329 | 0.059 | 0.989 | 0.372 | 0.341 | 0.322 |
| Energy | 0.434 | 0.065 | 0.840 | 0.372 | 0.341 | 0.301 |
| Prototype | 0.757 | 0.256 | 0.856 | 0.372 | 0.341 | 0.340 |

Best on this fold (AUROC): **Prototype**.

### held-out = pose_4

known classes = 12；test known=1108；test unknown=99；blocked unseen non-test=536

| method | AUROC | AUPR | FPR95 | Known Acc | Known Macro-F1 | Open-set Acc |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0.418 | 0.075 | 0.987 | 0.523 | 0.474 | 0.456 |
| Energy | 0.446 | 0.072 | 0.994 | 0.523 | 0.474 | 0.448 |
| Prototype | 0.566 | 0.089 | 0.780 | 0.523 | 0.474 | 0.466 |

Best on this fold (AUROC): **Prototype**.

### held-out = walk

known classes = 12；test known=1101；test unknown=106；blocked unseen non-test=567

| method | AUROC | AUPR | FPR95 | Known Acc | Known Macro-F1 | Open-set Acc |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0.474 | 0.105 | 0.962 | 0.468 | 0.413 | 0.411 |
| Energy | 0.579 | 0.127 | 0.915 | 0.468 | 0.413 | 0.406 |
| Prototype | 0.512 | 0.091 | 0.871 | 0.468 | 0.413 | 0.415 |

Best on this fold (AUROC): **Energy**.

## Full LOAO

YES。13 个动作各作一次 held-out。
- MSP: AUROC 0.476 ± 0.156；AUPR 0.083 ± 0.036；FPR95 0.903 ± 0.138
- Energy: AUROC 0.540 ± 0.144；AUPR 0.100 ± 0.051；FPR95 0.850 ± 0.112
- Prototype: AUROC 0.628 ± 0.125；AUPR 0.118 ± 0.055；FPR95 0.791 ± 0.219

per-action unknown detectability（取该 fold 上 AUROC 最好的方法）：
- T pose: AUROC=0.930 (Prototype)
- pose_1: AUROC=0.757 (Prototype)
- pose_9: AUROC=0.711 (Prototype)
- free_form: AUROC=0.705 (Energy)
- pose_10: AUROC=0.683 (Energy)
- pose_5: AUROC=0.677 (Energy)
- pose_7: AUROC=0.665 (Energy)
- pose_6: AUROC=0.660 (Prototype)
- walk: AUROC=0.579 (Energy)
- pose_3: AUROC=0.578 (Prototype)
- pose_2: AUROC=0.574 (Prototype)
- pose_4: AUROC=0.566 (Prototype)
- pose_8: AUROC=0.566 (Prototype)

## Failure cases (TEST)

Case A 正确拒识：`{'sample_id': 'subject3_04_pose_4_w0224', 'gt': 'pose_4', 'pred': 'UNKNOWN', 'note': 'true unknown correctly rejected'}`
Case B unknown 被吸进 known：`{'sample_id': 'subject10_04_pose_4_w0000', 'gt': 'pose_4', 'pred': 'T pose', 'note': 'unknown absorbed as known'}`
Case C known 被拒识：`{'sample_id': 'subject10_02_pose_2_w0208', 'gt': 'pose_2', 'pred': 'UNKNOWN', 'note': 'known rejected as unknown'}`

## Composite demo

composite replay for open-set demonstration
不能称为原始真实 continuous recording。

## Scientific questions

1. closed-set confidence 能不能作为 unknown detector？
不能单独当作可靠 detector。MSP 13-fold AUROC 0.476 ± 0.156，接近随机；FPR95 0.903 ± 0.138。低置信度 ≠ unknown detection。

2. MSP / Energy / Prototype 哪个最好？
Prototype 最好（AUROC 0.628 ± 0.125），其次 Energy（0.540 ± 0.144），MSP 最差（0.476 ± 0.156）。三个方法的 FPR95 都偏高。

3. 哪类动作最容易被拒识？
T pose（AUROC 0.930，Prototype）和 pose_1（0.757）。free_form 用 Energy 为 0.705。walk 只有 0.579。

4. pose_* 是否因为相似而难以 open-set detection？
是。pose_4 / pose_8 / pose_2 / pose_3 的最佳 AUROC 都在 0.566–0.578。pose_4 在 closed-set 上 68.7% 的错误落到其他 pose_*，作为 unseen 时也最容易被吸进 known。

5. subject-independent 条件下 open-set 是否显著变难？
本阶段全部在 SI 上评，没有另开 random-segment open-set。SI closed-set Acc 只有 0.478，12 类 fold 的 Known Acc 约 0.46。open-set 叠在未见过的人和未见过的动作上，是更严的测试。

6. Motion embedding 是否有 open-set value？
有。Prototype Distance 的 mean AUROC 高于 MSP 和 Energy。这是 Motion Representation 与 open-set 的直接连接。它仍不能把 FPR95 降到可用水平。

7. 当前结果是否足以称 open-world？

**NO。** Open-set 只是 open-world 的一个组成部分。本阶段没有 open-vocabulary，没有任意行为理解，也没有在 UNLABELED 全流上推理。

允许的名称：Open-Set Motion Recognition MVP；Toward Open-World Motion Understanding。

low confidence ≠ unknown detection。uncertainty ≠ open-set recognition。