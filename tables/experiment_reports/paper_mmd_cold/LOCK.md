# 官方主系统锁：mmd_cold + O2（2026-10-18/19）

**生成器：** `outputs/p123_mmd_cold/model`  
（← `outputs/methodabl_mmd_cold_20261018_161230/model`；coconut 冷启 + dual_beat + SF-C + λ_MMD=0.05，1500 step）

**打分：** 冻结 O2 `outputs/latentrm_order_pref/best`  
**协议：** dropout=None，`claim_agg=top1`，`score_reduce=sum_logit`

**取代：** 不再以 p2mmd05（dualBeat 暖启 + MMD）为官方主系统；dualBeat（λ_MMD=0）保留为消融基线。

## 主数字（GSM N16 O2）

| | Top1 | Cov |
|--|-----:|----:|
| seed 42 | 36.01 | 53.68 |
| **3-seed mean±std** | **36.32±0.40** | ~53.5 |

相对 dualBeat 3-seed 35.81：**+0.51**；相对旧 p2mmd05 3-seed 35.48：**+0.84**。

## 全矩阵 seed42（Top1 / Cov）

| split | N | O2 | B0 |
|-------|--:|----:|----:|
| GSM N1 det | 1 | 36.39 | — |
| GSM | 4 | 35.71 / 44.28 | 35.48 / 44.28 |
| GSM | 16 | **36.01 / 53.68** | 35.18 / 53.45 |
| GSM | 64 | 34.27 / 62.24 | 33.21 / 62.24 |
| Hard | 4 | 8.42 / 11.23 | 7.97 / 11.23 |
| Hard | 16 | 8.19 / 13.05 | 7.59 / 13.05 |
| Hard | 64 | 7.81 / 14.57 | 7.28 / 14.57 |
| MA | 4 | 82.07 / 88.79 | 81.03 / 88.79 |
| MA | 16 | **86.38 / 93.45** | 84.83 / 93.45 |
| MA | 64 | 85.34 / 95.52 | 83.28 / 95.52 |

明细 dump：`results/full/p123/paper_mmd_cold/`；评测脚本：`scripts/eval_mmd_cold_lock.sh`。
