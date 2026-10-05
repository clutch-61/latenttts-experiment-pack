# 在潜空间里做推理：基线与续训方法

大模型解数学题时，通常先写文字推理再给答案。COCONUT 把中间推理改成模型内部的连续向量（潜变量步），最后再解码答案。人读不到这些中间步，普通文本奖励也很难直接打分。LatentRM 因此再训一个潜空间奖励模型：对同一道题采多条轨迹，按分数挑一条，即 Best-of-N。

本仓库提供两套系统。权重和数据需自备。

1. **基线**：冻住的 COCONUT + 交叉熵训练的潜空间奖励模型 + Best-of-N。
2. **方法**：在生成器上继续训练；用带顺序偏好的奖励模型筛选轨迹，并加一项很轻的分布对齐；测试时用该奖励模型挑样本。

---

## 1. 两套系统

基线：生成器是冻住的 COCONUT；奖励模型只用答案对错、用交叉熵训练；测试做 Best-of-N。检查点：`checkpoints/coconut`，`outputs/latentrm_baseline/best`。

方法：不换题目、不换测试集，继续训练生成器。训练时冻住一个带步骤顺序偏好的奖励模型当裁判；测试时用这个模型挑样本。

主结果一张表：关掉 dropout，基线对方法，三个数据集、N=1/4/16/64。另一张表：同一方法关掉 dropout 对开 dropout=0.2。

---

## 2. 评测

数据放在 `data/`。GSM8K Test 是主集；GSM-Hard 把数字改大；MultiArith 是较简单的算术应用题。

共同设定：种子 42；最多生成 128 个新 token；采样条数 N ∈ {1, 4, 16, 64}。N = 1 时生成器确定性解码，不用奖励模型。N ≥ 4 时采 N 条，再用奖励模型排序。

同一批候选报四个数（单位 %）：

| 列 | 含义 |
|----|------|
| 选最高分 | 奖励模型分数最高的那一条是否答对 |
| 加权投票 | 相同答案的分数相加，取总分最高的答案 |
| 多数票 | 出现次数最多的答案 |
| 覆盖率 | N 条里至少有一条答对的比例 |

主表：基线和方法都关掉 dropout。对照表：方法关掉 dropout，对开 dropout=0.2。格子为选最高分 / 加权投票 / 多数票 / 覆盖率。N=1 不用奖励模型，只有准确率。

---

## 3. 基线

生成器：冻住的 COCONUT。打分器：交叉熵潜空间奖励模型。测试：Best-of-N，关掉 dropout。

```bash
bash scripts/train_baseline_detach.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_paper_b0_baseline.sh
```

---

## 4. 方法

在生成器上继续训练，使轨迹更容易既答对、又被奖励模型打高分。

基线奖励模型只学答对。另训一个带步骤顺序偏好的奖励模型（`outputs/latentrm_order_pref`）：除对错外，还约束潜变量步的先后结构。训练生成器时该模型冻住，只打分、不更新。测试 Best-of-N 用它，不用基线交叉熵模型。

训练两段。

**第一段。** 老师是原始 COCONUT，每题贪心解一次。学生是当前生成器，每题采 64 条。同时满足下面三条，才用学生轨迹做交叉熵，否则仍模仿老师：答案对；偏好奖励模型分高于老师；基线奖励模型不低于老师减 0.5。潜变量步用可微展开（Self-Forcing）滚到答案再做交叉熵。本段不加分布对齐，1500 步。

**第二段。** 接上一段检查点。门控不变。加 MMD（最大均值差异），把学生潜变量的内容表示拉向做过对的轨迹，系数 0.05，前 100 步从 0 升到 0.05。不更新偏好奖励模型。学习率 3e-6，再 1500 步。超参见 [`recipes/p2mmd05_train_args.json`](recipes/p2mmd05_train_args.json)。

测试：偏好奖励模型 + Best-of-N，关掉 dropout，种子 42。N=1 仍是生成器贪心、不用奖励模型。

```bash
CKPT=outputs/p123_dualBeat/model bash scripts/train_p2mmd05.sh
GEN=outputs/p123_p2mmd05/model bash scripts/eval_p2mmd05_none.sh
```

`scripts/train_p123_system.py` 是两段的同一入口：`lambda_mmd=0` 为第一段，`0.05` 为第二段。

---

## 5. 结果（种子 42）

格子为：选最高分 / 加权投票 / 多数票 / 覆盖率。N=1 不用奖励模型，只有准确率。

### 表 1. 基线 vs 方法（关掉 dropout）

基线：冻住的 COCONUT + 交叉熵奖励模型。方法：续训生成器 + 偏好奖励模型。

| 集合 | N | 基线 | 方法 |
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

### 表 2. 方法：关掉 dropout vs dropout=0.2

同一生成器和偏好奖励模型。N=1 不用奖励模型，两列相同。

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

---

## 6. 运行

依赖见 `requirements.txt`。需要 GSM 等 json、COCONUT 权重、两个奖励模型检查点，以及第一段生成器检查点。

```bash
bash scripts/train_baseline_detach.sh
# python -m src.train_order_pref training_args/train_order_pref.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_paper_b0_baseline.sh
CKPT=outputs/p123_dualBeat/model bash scripts/train_p2mmd05.sh
GEN=outputs/p123_p2mmd05/model bash scripts/eval_p2mmd05_none.sh
```

---

## 7. 代码

| 文件 | 作用 |
|------|------|
| `src/infer_gpt2.py` | N=1，只用生成器 |
| `src/infer_gpt2_rm.py` | Best-of-N |
| `src/train.py` + `training_args/train_baseline_ce.yaml` | 基线奖励模型 |
| `src/train_order_pref.py` | 顺序偏好奖励模型 |
| `src/sf_rollout.py` | 潜变量可微展开 |
| `src/order_pref/content_view.py`、`set_align.py` | 内容表示与 MMD |
| `src/system_scoring.py` | 选最高分 / 加权投票 |
| `scripts/train_p123_system.py` | 生成器续训 |
| `recipes/p2mmd05_train_args.json` | 第二段超参 |

不含模型权重和完整采样结果。
