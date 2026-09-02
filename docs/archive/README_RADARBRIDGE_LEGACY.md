# RadarBridge

基于公开雷达数据和 AIRHAR / RadMamba 的人体行为识别复现。第一阶段只保证官方 CI4R 路径能跑通，再做跨人员、跨雷达。

官方代码在 `third_party/AIRHAR`。论文：RadMamba, arXiv:2504.12039。数据：CI4R-MULTI3（与 CI4R-Activity-Recognition-datasets 是同一仓库）。

## 环境

本机是 RTX 5060 Laptop（Blackwell, sm_120）。官方写的 PyTorch 2.5.1 + CUDA 12.6 没有该架构的 kernel，因此改为 **Python 3.11 + PyTorch 2.7.1 + cu128**。

```bat
conda create -n airhar python=3.11 -y
conda activate airhar
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy scipy pandas matplotlib tqdm scikit-learn h5py einops opencv-python-headless tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install zetascale -i https://pypi.tuna.tsinghua.edu.cn/simple
```

装完后应能：`import torch, h5py, cv2, einops`，`torch.cuda.is_available()` 为 True。RadMamba 还需要 `from zeta.nn.modules.p_scan import pscan`；失败就把原因记到 `notes/00_clone_and_env.md`，再考虑单独放入 `pscan`。

## 数据

公开桶：`https://storage.googleapis.com/cross-frequency_dataset`

- 谱图 PNG：`Spectrograms/11_class_activity_data/{77GHz,24GHz,Xethru}/<动作>/`
- 原始 IQ：`Raw Data/raw_data.zip`，约 93 GB。AIRHAR 的 `data_prepare_CI4R.py` 读 77 GHz `.bin`。

当前默认闭环用官方谱图 PNG（真实测量，不是仿真），按 AIRHAR 规则每类最后 12 条进测试：

```bat
conda activate airhar
python scripts/list_gcs_bucket.py
python scripts/download_spectrograms.py
python scripts/prepare_ci4r_from_png.py --freq 77GHz
python scripts/visualize_ci4r.py --freq 77GHz
```

若已有 77 GHz `.bin` 目录：

```bat
cd third_party\AIRHAR
python datasets\data_prepare_CI4R.py --path <CI4R_77GHZ_DIR>
```

核对：`datasets/CI4R/ci4r_data.h5` 里 train/test 的谱图为 224×224，标签 11 维 one-hot。官方论文数字是 528 / 132；PNG 路径以实际文件数为准，见 `results/splits/ci4r_official_last12.json`。

## 训练

在 `third_party/AIRHAR` 下跑官方入口：

```bat
cd third_party\AIRHAR
python main.py --dataset_name CI4R --Classification_backbone resnet --seed 0 --n_epochs 5 --lr 1e-4 --Classification_hidden_size 12
python main.py --dataset_name CI4R --Classification_backbone radmamba --seed 0 --n_epochs 5 --dim 64
```

本项目统一训练、指标和对比：

```bat
python scripts/train_local.py --model simple_cnn --epochs 30 --seed 0
python scripts/train_local.py --model radmamba --epochs 30 --seed 0 --dim 80 --lr 5e-3
python scripts/run_unified_bench.py --epochs 30 --seed 0
python scripts/eval_metrics.py --ckpt results\runs\radmamba_seed0\best.pt --Classification_backbone radmamba --dim 80
```

## 跨人员 / 跨雷达

```bat
python scripts/prepare_loso.py --freq 77GHz
python scripts/run_cross_subject.py --freq 77GHz --model simple_cnn --epochs 20
python scripts/run_cross_radar.py --train-freq 24GHz --test-freq 77GHz --model simple_cnn --epochs 20
python scripts/analyze_domain_shift.py
```

官方 8:2 和 LOSO 的数字分开报，不要混在一张表里。官方训练按测试集选模型，复现时在笔记里写明 `test-as-val`。

## 目录

```
third_party/AIRHAR/     官方代码
third_party/CI4R-MULTI3/ 数据仓库脚本（无原始 IQ）
data/                   下载的数据，不进 git
scripts/                下载、预处理、训练、跨域
notes/                  审计与改动记录
results/                划分、曲线、指标、图
```

This file is an archive of the pre-RadarMind landing README. It is not the current project entry point.
