# 在潜空间里做推理：基线、续训方法与外部生成器对照

大模型解数学题时，通常先写文字推理再给答案。COCONUT 把中间推理改成模型内部的连续向量（潜变量步），最后再解码答案。人读不到这些中间步，普通文本奖励也很难直接打分。LatentRM 因此再训一个潜空间奖励模型：对同一道题采多条轨迹，按分数挑一条，即 Best-of-N。

本仓库提供：**原始基线**、**自家续训方法（p2mmd05）**，以及按同一协议评测的 **外部生成器（SLPO、CODI）** 和 **在 CODI 上再跑同一套续训（CODIp2）**。权重和完整采样 dump 需自备；数字摘要在 [`tables/`](tables/)。

---

## 1. 系统一览

协议一律：潜变量步数 **T=6**、奖励模型测试时 **关掉 dropout**、种子 **42**、最多 **128** 新 token、N ∈ {1,4,16,64}。N=1 是生成器贪心、不用奖励模型。不要把对方论文里的 Acc（例如 SLPO 开 gate、Tmax=12）写进同表。

| 系统 | 生成器 | 奖励模型 | 说明 |
|------|--------|----------|------|
| **基线** | 冻住的 COCONUT | B0（交叉熵 / 官方对比 LatentRM） | 矩阵左上角 |
| **方法 p2mmd05** | 续训后的 COCONUT | O2（顺序偏好） | 自家主系统 |
| **SLPO+B0** | SLPO-COCONUT（关 gate、固定 T=6） | B0 | 生成器轴外基线 |
| **CODI+B0** | 冻住的 CODI | B0 | 生成器轴外基线 |
| **CODI+O2** | 冻住的 CODI | O2 | 隔离：只换打分器 |
| **CODIp2+O2** | CODI 上跑同一套 p2 续训 | O2 | 策略接到 CODI |

B0 = 官方 LatentTTS 逐步对比 LatentRM。O2 = 本仓库的顺序偏好奖励模型（BCE + 层级偏好），**不是**官方 LatentRM。

---

## 2. 评测指标

同一批候选报四个数（单位 %）：

| 列 | 含义 |
|----|------|
| 选最高分 (top1) | 奖励模型分数最高的那一条是否答对 |
| 加权投票 (wvote) | 相同答案的分数相加，取总分最高的答案 |
| 多数票 (maj) | 出现次数最多的答案 |
| 覆盖率 (cov) | N 条里至少有一条答对的比例 |

机器可读摘要：[`tables/bon_summary.json`](tables/bon_summary.json)、[`tables/bon_summary.tsv`](tables/bon_summary.tsv)。

---

## 3. 主结果：基线 vs 自家方法（COCONUT 族）

关掉 dropout。基线 = 冻住 COCONUT + B0。方法 = p2mmd05 生成器 + O2。

| 集合 | N | 基线 COCONUT+B0 | 方法 p2mmd05+O2 |
|------|---|------|------|
| GSM8K Test | 1 | 34.42 | 35.18 |
| GSM8K Test | 4 | 32.37 / 32.52 / 32.98 / 42.15 | 35.25 / 36.01 / 35.18 / 44.20 |
| GSM8K Test | 16 | 33.74 / 33.66 / 33.97 / 51.10 | 34.80 / 36.77 / 35.94 / 52.84 |
| GSM8K Test | 64 | 31.31 / 33.74 / 33.74 / 59.59 | 34.72 / 36.92 / 36.16 / 60.88 |
| GSM-Hard | 1 | 7.66 | 7.97 |
| GSM-Hard | 4 | 6.68 / 6.98 / 7.51 / 9.71 | 7.66 / 8.35 / 8.19 / 10.62 |
| GSM-Hard | 16 | 7.21 / 7.06 / 7.36 / 12.14 | 8.19 / 8.27 / 8.04 / 12.90 |
| GSM-Hard | 64 | 6.53 / 7.21 / 7.44 / 14.57 | 7.51 / 8.27 / 8.04 / 14.57 |
| MultiArith | 1 | 80.86 | 85.69 |
| MultiArith | 4 | 77.41 / 78.28 / 78.10 / 86.21 | 81.72 / 83.62 / 83.62 / 89.83 |
| MultiArith | 16 | 79.66 / 82.07 / 80.86 / 91.38 | 85.52 / 87.41 / 86.21 / 92.07 |
| MultiArith | 64 | 80.17 / 83.10 / 80.69 / 93.97 | 85.52 / 87.76 / 85.69 / 94.66 |

方法侧对照（同一 p2mmd05+O2：关掉 dropout vs dropout=0.2）见文末附录。

---

## 4. 外部生成器：SLPO、CODI 原版

奖励模型固定为 **B0**，只换生成器。权重：`ModalityDance/slpo-coconut-gpt2`、`ModalityDance/latent-tts-codi`（或镜像）。

### 4.1 GSM8K Test（关掉 dropout）

| 系统 | N=1 | N=4 top1/wvote/maj/cov | N=16 | N=64 |
|------|-----|------------------------|------|------|
| COCONUT+B0 | 34.42 | 32.37 / 32.52 / 32.98 / 42.15 | 33.74 / 33.66 / 33.97 / 51.10 | 31.31 / 33.74 / 33.74 / 59.59 |
| SLPO+B0 | 34.80 | 33.97 / 34.50 / 33.66 / 45.03 | 30.78 / 32.52 / 35.78 / 56.94 | 29.04 / 32.75 / 36.69 / 66.79 |
| **CODI+B0** | **42.08** | 41.24 / 41.70 / 41.77 / 48.14 | **40.03** / 41.09 / 42.46 / 54.66 | 38.67 / 40.86 / 42.08 / 59.51 |
| p2mmd05+O2（对照） | 35.18 | 35.25 / 36.01 / 35.18 / 44.20 | 34.80 / 36.77 / 35.94 / 52.84 | 34.72 / 36.92 / 36.16 / 60.88 |

SLPO 在固定 T=6、关 gate 后，BoN top1 常低于原 COCONUT+B0（覆盖率反而更高）——说明池子更大但 B0 选优变差。CODI+B0 明显高于 COCONUT 族与 p2mmd05。

### 4.2 Hard / MultiArith（N=16）

| 系统 | Hard N16 | MultiArith N16 |
|------|----------|----------------|
| COCONUT+B0 | 7.21 / 7.06 / 7.36 / 12.14 | 79.66 / 82.07 / 80.86 / 91.38 |
| SLPO+B0 | 6.45 / 7.13 / 7.89 / 13.58 | 79.66 / 82.41 / 80.52 / 93.10 |
| CODI+B0 | 8.57 / 8.88 / 9.33 / 12.82 | 90.69 / 91.55 / 93.10 / 98.28 |
| p2mmd05+O2 | 8.19 / 8.27 / 8.04 / 12.90 | 85.52 / 87.41 / 86.21 / 92.07 |

完整格子见 [`tables/bon_summary.tsv`](tables/bon_summary.tsv)。

评测脚本：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_ext_slpo_b0.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/eval_ext_codi_b0.sh
```

---

## 5. CODI 上接自家策略（CODIp2）与隔离

在冻住 **CODI** 上跑与 p2mmd05 相同的配方：dual_beat 门控 + Self-Forcing(C) + λ_MMD=0.05，冻住 O2/B0；推理关掉 dropout。超参见 [`recipes/codi_p2mmd05_train_args.json`](recipes/codi_p2mmd05_train_args.json)。

### 5.1 隔离（GSM N16 top1）

同生成器、同候选池时：

| 系统 | top1 | wvote | maj | cov |
|------|-----:|------:|----:|----:|
| 冻住 CODI+B0 | 40.03 | 41.09 | 42.46 | 54.66 |
| 冻住 CODI+O2 | 40.94 | 42.91 | 42.46 | 54.66 |
| **CODIp2+O2** | **42.00** | **43.29** | 42.91 | 54.36 |

相对 CODI+B0：**+1.97pp**。其中约一半来自 B0→O2（+0.91），一半来自续训（再 +1.06）。覆盖率几乎不动。N=1 greedy 42.08→42.15，生成器确定性几乎没变。

### 5.2 CODIp2 全矩阵（关掉 dropout）

| 集合 | N | CODIp2+O2 |
|------|---|-----------|
| GSM8K Test | 1 | 42.15 |
| GSM8K Test | 4 | 42.00 / 42.15 / 42.23 / 49.28 |
| GSM8K Test | 16 | 42.00 / 43.29 / 42.91 / 54.36 |
| GSM8K Test | 64 | 42.23 / 43.44 / 42.68 / 59.97 |
| GSM-Hard | 1 | 9.33 |
| GSM-Hard | 4 | 9.41 / 9.56 / 9.79 / 11.08 |
| GSM-Hard | 16 | 9.10 / 9.64 / 9.48 / 12.67 |
| GSM-Hard | 64 | 9.41 / 9.86 / 9.56 / 13.96 |
| MultiArith | 1 | 91.90 |
| MultiArith | 4 | 91.55 / 92.07 / 92.24 / 95.86 |
| MultiArith | 16 | 92.41 / 92.76 / 93.10 / 98.45 |
| MultiArith | 64 | 91.90 / 93.45 / 93.97 / 98.79 |

相对原基线 COCONUT+B0，GSM N16 top1 **+8.26pp**；相对自家 p2mmd05 **+7.20pp**。大头是 CODI 骨干更强（冻住 CODI+B0 已是 40.03），不是同一套配方在 COCONUT 上又涨 7 点。COCONUT 官方主系统仍报 p2mmd05+O2，不因 CODI 改锁。

```bash
# 训练 + 杀伤评测（GSM N16）
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_eval_codi_p2mmd05.sh
# 全矩阵：冻住 CODI+O2 与 CODIp2+O2
bash scripts/eval_codi_full_matrix.sh
```

---

## 6. 计划加入的基线（尚未出同协议数字）

设计是两个**一维**替换，不是满网格。详情与取舍见 [`tables/baseline_candidate_survey.md`](tables/baseline_candidate_survey.md)。

**生成器轴（奖励模型固定 = B0）**

| 方法 | 状态 | 说明 |
|------|------|------|
| SLPO | **已评** | 关 gate、T=6；见第 4 节 |
| CODI | **已评** | 冻住 + CODIp2；见第 4–5 节 |
| Latent-SFT | 计划 | 词表叠加式潜推理，不是 GPT-2 COCONUT；需适配到本协议后再报 |

**奖励模型轴（生成器固定 = 原始 COCONUT）**

| 方法 | 状态 | 说明 |
|------|------|------|
| ELHSR / SWIFT (2505.12225) | 计划 | 线性 hidden-state RM；与 HSRM(2608.30841) **不是**同一篇 |
| GenPRM | 计划 | 文本过程奖励；只能作文本侧对照，不能直接当 LatentRM |

**本轮不做 / 不进主表：** PMPS、LTF、RSP、HSRM（缺权重或接口不符）；外方法互相交叉（如 SLPO×ELHSR）；对方论文 Acc 当同协议数字。

---

## 7. 方法简述（p2mmd05 / CODIp2 共用）

1. **门控。** 老师 = 初始检查点贪心；学生每题采 64 条。同时满足：答案对、O2 分高于老师、B0 不低于老师减 0.5，才用学生轨迹做交叉熵，否则模仿老师或跳过。
2. **Self-Forcing (C)。** 潜变量步可微展开后再对答案做交叉熵。
3. **轻量 MMD。** λ=0.05，对内容视图（bottleneck projector）对齐；前 100 步从 0 升到 0.05。不更新 O2。

COCONUT 上：老师/学生都是 coconut。CODI 上：老师/学生都是 `checkpoints/codi`（不要混用 coconut 老师）。

```bash
# COCONUT 第二段
CKPT=outputs/p123_dualBeat/model bash scripts/train_p2mmd05.sh
GEN=outputs/p123_p2mmd05/model bash scripts/eval_p2mmd05_none.sh

# CODI 同配方
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_eval_codi_p2mmd05.sh
```

`scripts/train_p123_system.py --generator_type coconut|codi` 是统一入口。

---

## 8. 代码与脚本

| 路径 | 作用 |
|------|------|
| `src/infer_gpt2.py` | N=1；`model_type=coconut\|codi` |
| `src/infer_gpt2_rm.py` | Best-of-N；`generator_type=coconut\|codi`（CODI 不传 `target_id`） |
| `src/models/codi.py` | CODI GPT-2 + projector |
| `src/paths.py` | 生成器类与答案抽取 |
| `src/train.py` + `training_args/train_baseline_ce.yaml` | 基线奖励模型 B0 |
| `src/train_order_pref.py` | 顺序偏好奖励模型 O2 |
| `src/sf_rollout.py` / `src/order_pref/` | Self-Forcing、内容视图、MMD |
| `scripts/train_p123_system.py` | 生成器续训（含 CODI） |
| `scripts/eval_ext_slpo_b0.sh` | SLPO+B0 矩阵 |
| `scripts/eval_ext_codi_b0.sh` | 冻住 CODI+B0 矩阵 |
| `scripts/eval_codi_full_matrix.sh` | 冻住 CODI+O2 与 CODIp2+O2 全矩阵 |
| `scripts/train_eval_codi_p2mmd05.sh` | CODIp2 训练 + GSM N16 杀伤评测 |
| `tables/bon_summary.*` | 全部系统数字摘要 |
| `tables/baseline_candidate_survey.md` | 外基线调研与计划 |

不含模型权重和逐题采样 JSON（体积过大）。本地完整 dump 在实验机 `results/full/`。

---

## 附录 A. 方法：关掉 dropout vs dropout=0.2

同一 p2mmd05 生成器 + O2。N=1 不用奖励模型，两列相同。

| 集合 | N | 关掉 dropout | dropout=0.2 |
|------|---|--------------|-------------|
| GSM8K Test | 1 | 35.18 | 35.18 |
| GSM8K Test | 4 | 35.25 / 36.01 / 35.18 / 44.20 | 32.37 / 33.43 / 33.66 / 44.12 |
| GSM8K Test | 16 | 34.80 / 36.77 / 35.94 / 52.84 | 34.04 / 35.94 / 35.86 / 56.56 |
| GSM8K Test | 64 | 34.72 / 36.92 / 36.16 / 60.88 | 31.77 / 33.97 / 35.94 / 66.11 |
| GSM-Hard | 1 | 7.97 | 7.97 |
| GSM-Hard | 4 | 7.66 / 8.35 / 8.19 / 10.62 | 7.44 / 7.36 / 7.51 / 9.86 |
| GSM-Hard | 16 | 8.19 / 8.27 / 8.04 / 12.90 | 7.74 / 7.81 / 8.35 / 13.51 |
| GSM-Hard | 64 | 7.51 / 8.27 / 8.04 / 14.57 | 7.21 / 7.59 / 7.89 / 15.71 |
| MultiArith | 1 | 85.69 | 85.69 |
| MultiArith | 4 | 81.72 / 83.62 / 83.62 / 89.83 | 77.59 / 79.14 / 79.48 / 88.62 |
| MultiArith | 16 | 85.52 / 87.41 / 86.21 / 92.07 | 83.10 / 84.66 / 84.14 / 93.97 |
| MultiArith | 64 | 85.52 / 87.76 / 85.69 / 94.66 | 82.24 / 87.59 / 83.62 / 96.03 |

论文口径下基线另有一套 **dropout=0.2** 的 A+B0（GSM N16 top1 **28.43**），与上表关掉 dropout 的 33.74 **不要混比**。
