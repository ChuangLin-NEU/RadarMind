# CI4R cross-radar schema

依据：`third_party/CI4R-MULTI3/README.md`（三雷达同步采集）、`XeThru_mDopp_bulk.m`、`datToImage_Anchortech.m`、`datToImage_77.m`、`third_party/AIRHAR/datasets/data_prepare_CI4R.py`，以及本地官方谱图 PNG。未全量解压 `raw_data.zip`。文件名前缀数字是 **acquisition group**，不是官方 subject id。

## 1. 可用雷达频率

三个：10 GHz（XeThru UWB）、24 GHz（Ancortek FMCW）、77 GHz（TI IWR1443）。

官方 PNG：`data/CI4R_MULTI3/spectrograms/11_class_activity_data/{Xethru,24GHz,77GHz}/`。

## 2. 真实文件格式

| 频率 | raw | 官方 loader | 本阶段训练用 |
|---|---|---|---|
| 10 GHz | `.dat`（XeThru 帧流：uint32 ContentId/Info/Length + 182 float；同目录 616B `.bin` **不是** 77 GHz 8-lane IQ） | `XeThru_mDopp_bulk.m`：拼 IQ → MTI → STFT（PRF=512，window=150） | 官方 released spectrogram PNG |
| 24 GHz | 文本 `.dat`：前 4 个数 fc / Tsweep / NTS / Bw，其后 I/Q | `datToImage_Anchortech.m`：reshape chirp → range FFT → MTI → STFT | 同上 PNG |
| 77 GHz | `*_Raw_0.bin` int16 8-lane DCA1000 | AIRHAR `bin2np` + `RGB_mD`；MATLAB `datToImage_77.m` | 同上 PNG（与 Phase 1A AIRHAR cache 分开，保证三频同一成品） |

统一 tensor 形状 `1×224×224` **不等于** 物理分辨率相同。metadata 保留 `radar_frequency`、`native_*`。

## 3. Preprocessing（官方，不是自造）

- **10 GHz**：XeThru 帧 IQ → 4 阶 highpass MTI → 第 27 距离元 STFT → isodata 去噪 → 20log / −45 dB。
- **24 GHz**：文本 IQ → range FFT 半边 → MTI → 第 19 距离元 STFT → 去噪。
- **77 GHz**：AIRHAR：Range FFT → 门 23:31 求和 → STFT Hann 256 / overlap 200 → 20log clip → 224×224。Phase 1A 已复现。MATLAB 版另有 MTI 和不同 overlap。

本阶段 cross-radar HAR **不重跑 raw**，使用官方已发布谱图，再灰度缩放到 224×224 并逐图 z-score。

## 4. 同步 pairing

官方 README：三雷达 **synched / co-located**。配对证据是文件名中的 10 位 unix timestamp 相同。

当前统计：

```
{
  "N_10GHz": 653,
  "N_24GHz": 653,
  "N_77GHz": 653,
  "paired_10_77": 558,
  "paired_24_77": 510,
  "paired_10_24": 504,
  "paired_10_24_77": 460,
  "paired_10_24_77_action_agreed": 327,
  "paired_10_24_77_action_conflict": 133,
  "triple_action_counts": {
    "Away": 53,
    "Bend": 49,
    "Crawl": 45,
    "Kneel": 46,
    "ACTION_MISMATCH": 133,
    "Pick": 55,
    "Sit": 45,
    "Towards": 34
  },
  "common_actions_agreed": [
    "Away",
    "Bend",
    "Crawl",
    "Kneel",
    "Pick",
    "Sit",
    "Towards"
  ],
  "conflict_note": "same timestamp but different action folders (often 24GHz Toes vs 10/77 Limp); excluded from fair triple subset",
  "n_pair_rows": 1790,
  "n_pair_ids": 652
}
```

`pair_id` 按 timestamp 分配。若三频 timestamp 对上且动作目录一致，`pairing_confidence=exact_timestamp`。

24 GHz 另有早期补录（README：额外约 180/类）。那些 timestamp 对不上 10/77 的，**不进三频共同子集**，也不假装成同步 pair。

## 5. 泄漏约束

同一 `timestamp` 的 10/24/77 视为一次 acquisition。划分按 timestamp，禁止 77 训练、对应 10 进 val/调参。
