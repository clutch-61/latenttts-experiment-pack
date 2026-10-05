# Latent TTS：基线 A+B0 与 p2mmd05

本仓库只保留两件事：

1. **基线**：冻结 COCONUT（A）+ 基线 LatentRM（B0）的评测与 B0/O2 训练入口  
2. **方法 p2mmd05**：在 dualBeat 双门 Self-Forcing 之上加轻量 content MMD（λ=0.05），推理用冻结 O2、**dropout=None** 的 Best-of-N  

其它探索配方（coverage / jrank / λ_lat / dropout 续训等）已从本仓库去掉。完整逐步实验记录不在这里。

---

## 系统设定

### 基线 A+B0

| 项 | 设定 |
|----|------|
| 生成器 A | 原始 COCONUT，`checkpoints/coconut` |
| RM B0 | CE LatentRM，无顺序偏好，`outputs/latentrm_baseline/best` |
| 论文口径评测 | seed 42，MC-dropout **p=0.2**，BoN，主列 **top1**（同池报 wvote / 多数票 / coverage） |
| N=1 | 确定性解码，无 RM |

GSM-Test N=16 钉死分母：**top1 28.43** / wvote 29.80 / cov 52.69。不要和 dropout=None 的旧 33.74 混减。

### 方法 p2mmd05（选定最优）

生成器路径：`outputs/p123_p2mmd05/model`（实验目录名 `p123_p2mmd05_20261004_215700`）。

**训练（两段）**

1. **dualBeat（暖启）**：从 COCONUT 起，1500 step。P1 冻结 O2 选优 + B0 健康门（`b0_slack=0.5`）：学生 k=64 采样，老师 greedy；仅当答案对、O2 优于老师、B0 ≥ 老师−slack 时 CE 自己，否则 CE 老师。P3 为 SF-C（可微 latent rollout + 答案 CE）。此段 **λ_mmd=0**。  
2. **P2**：暖启 dualBeat，再 1500 step。其余门控不变，加上 content-view MMD **λ_mmd=0.05**（ramp 100 步），bottleneck 64。H^pos：老师做对用老师 latent，否则 A 成功 cache / 选中 self。**λ_lat=0，O2 不更新。** lr 3e-6，k=64，q_batch=16，grad_accum=8，训练 dropout **0.2**，`max_new_tokens=128`，seed 42。

超参快照：[`recipes/p2mmd05_train_args.json`](recipes/p2mmd05_train_args.json)。

**推理（选定协议）**

冻结 O2 `outputs/latentrm_order_pref/best`，`best_of_n`，**不设 dropout**，seed 42，N∈{1,4,16,64}。N=1 为生成器 greedy、无 RM。

相对 dualBeat none，三集×N×(top1/wvote/多数) 等权平均只高约 0.1–0.3pp，属噪声量级；仍按该版作为本仓库方法。GSM N16 none：top1 **34.80** / wvote **36.77** / cov **52.84**。

---

## 主结果（seed 42）

### 论文口径 p=0.2，GSM / Hard / MultiArith

N=1 无 RM。BoN 列为 top1 / wvote / 多数 / cov。

**GSM-Test**

| N | A+B0 | p2mmd05+O2 |
|---|------|------------|
| 1 | 34.42 | 35.18 |
| 4 | 29.42 / 30.17 / 29.72 / 40.41 | 32.37 / 33.43 / 33.66 / 44.12 |
| 16 | **28.43** / 29.80 / 31.77 / 52.69 | 34.04 / 35.94 / 35.86 / 56.56 |
| 64 | 27.75 / 29.34 / 31.92 / 63.08 | 31.77 / 33.97 / 35.94 / 66.11 |

**GSM-Hard**

| N | A+B0 | p2mmd05+O2 |
|---|------|------------|
| 1 | 7.66 | 7.97 |
| 16 | **5.46** / 5.61 / 6.37 / 12.59 | 7.74 / 7.81 / 8.35 / 13.51 |

**MultiArith**

| N | A+B0 | p2mmd05+O2 |
|---|------|------------|
| 1 | 80.86 | 85.69 |
| 16 | **71.03** / 76.72 / 75.52 / 92.24 | 83.10 / 84.66 / 84.14 / 93.97 |

同口径 GSM N16 top1：方法相对基线约 **+5.6pp**。Hard 空池仍约 85%+。

### 方法主协议 dropout=None（p2 + O2）

| 集 | N | top1 | wvote | 多数 | cov |
|----|---|------|-------|------|-----|
| GSM-Test | 16 | 34.80 | **36.77** | 35.94 | 52.84 |
| GSM-Test | 1 | 35.18 | — | — | — |
| GSM-Hard | 16 | 8.19 | 8.27 | 8.04 | 12.90 |
| MultiArith | 16 | 85.52 | 87.41 | 86.21 | 92.07 |

---

## 怎么跑

环境见 `requirements.txt`。权重与 `data/*.json` 需自备（COCONUT、B0、O2、dualBeat 暖启 ckpt），本仓库不含大文件。

```bash
# 基线 B0 训练
bash scripts/train_baseline_detach.sh

# 顺序偏好 O2（冻结后给 dualBeat / p2 当选优器）
# python -m src.train_order_pref training_args/train_order_pref.yaml

# 论文口径基线评测 A+B0 p=0.2
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_paper_b0_baseline.sh

# p2mmd05 训练（先有 dualBeat ckpt）
CKPT=outputs/p123_dualBeat/model bash scripts/train_p2mmd05.sh

# 选定推理：dropout=None N=16
GEN=outputs/p123_p2mmd05/model bash scripts/eval_p2mmd05_none.sh
```

`scripts/train_p123_system.py` 是 dualBeat 与 p2 的同一入口：`λ_mmd=0` 为双门 SF-C，`λ_mmd=0.05` 为 p2mmd05。

---

## 仓库里有什么

| 路径 | 作用 |
|------|------|
| `src/infer_gpt2.py` / `infer_gpt2_rm.py` | N=1 与 BoN |
| `src/train.py` + `training_args/train_baseline_ce.yaml` | B0 |
| `src/train_order_pref.py` | O2 |
| `src/sf_rollout.py` | SF-C |
| `src/order_pref/set_align.py` / `content_view.py` | P2 MMD |
| `src/system_scoring.py` | top1 / wvote |
| `scripts/train_p123_system.py` | 一体训练 |
| `recipes/p2mmd05_train_args.json` | 最优版训练超参 |

不包含：失败配方脚本、原始 BoN json dump、模型权重。
