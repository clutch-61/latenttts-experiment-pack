# 在潜空间里做推理：基线与续训方法

大模型解数学题时，通常先写文字推理再给答案。COCONUT 把中间推理改成模型内部的连续向量（潜变量步），最后再解码答案。人读不到这些中间步，普通文本奖励也很难直接打分。LatentRM 因此再训一个潜空间奖励模型：对同一道题采多条轨迹，按分数挑一条，即 Best-of-N。

本仓库提供两套系统。权重和数据需自备。

1. **基线**：冻住的 COCONUT + 交叉熵训练的潜空间奖励模型 + Best-of-N。
2. **方法**：在生成器上继续训练；用带顺序偏好的奖励模型筛选轨迹，并加一项很轻的分布对齐；测试时用该奖励模型挑样本。

---

## 1. 两套系统

基线与原 LatentRM 评测对齐：生成器是冻住的 COCONUT；奖励模型只用答案对错、用交叉熵训练；测试做 Best-of-N，奖励模型上开 MC-dropout = 0.2。检查点：`checkpoints/coconut`，`outputs/latentrm_baseline/best`。

方法不换题目、不换测试集。继续训练生成器。训练时冻住一个带步骤顺序偏好的奖励模型当裁判；测试时用这个模型挑样本，并关掉 dropout。

---

## 2. 评测

数据放在 `data/`。GSM8K Test 是主集；GSM-Hard 把数字改大；MultiArith 是较简单的算术应用题。

共同设定：种子 42；最多生成 128 个新 token；采样条数 N ∈ {1, 4, 16, 64}。N = 1 时生成器确定性解码，不用奖励模型。N ≥ 4 时采 N 条，再用奖励模型排序。

同一批候选报四个数（单位 %）：

| 列 | 含义 |
|----|------|
| 选最高分 | 奖励模型分数最高的那一条是否答对。与原论文对齐时用这一列。 |
| 加权投票 | 相同答案的分数相加，取总分最高的答案 |
| 多数票 | 出现次数最多的答案 |
| 覆盖率 | N 条里至少有一条答对的比例 |

两种测试设定：

| 设定 | 奖励模型 dropout | 用途 |
|------|------------------|------|
| 与原论文对齐 | 0.2 | 基线和方法的公平对比 |
| 方法部署 | 关掉 | 方法实际怎么用 |

基线在 GSM8K Test、N=16、dropout=0.2 下，选最高分为 28.43%。对比涨幅用同一设定。关掉 dropout 的基线数字不能拿来减方法。

---

## 3. 基线

生成器：冻住的 COCONUT。打分器：交叉熵潜空间奖励模型。测试：MC-dropout = 0.2，Best-of-N，主列是选最高分。

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

公平对比用 dropout=0.2。N≥4 的格子为：选最高分 / 加权投票 / 多数票 / 覆盖率。N=1 只有生成器准确率。

### GSM8K Test

| N | 基线（COCONUT + 交叉熵奖励模型） | 方法（续训生成器 + 偏好奖励模型） |
|---|----------------------------------|-----------------------------------|
| 1 | 34.42 | 35.18 |
| 4 | 29.42 / 30.17 / 29.72 / 40.41 | 32.37 / 33.43 / 33.66 / 44.12 |
| 16 | **28.43** / 29.80 / 31.77 / 52.69 | 34.04 / 35.94 / 35.86 / 56.56 |
| 64 | 27.75 / 29.34 / 31.92 / 63.08 | 31.77 / 33.97 / 35.94 / 66.11 |

N=16、选最高分：28.43 → 34.04，约 +5.6 个百分点。覆盖率约一半，许多题的 16 条样本里没有正确答案。N 增大时选最高分有时下降，打分器会把错误轨迹排到前面。

### GSM-Hard

| N | 基线 | 方法 |
|---|------|------|
| 1 | 7.66 | 7.97 |
| 16 | **5.46** / 5.61 / 6.37 / 12.59 | 7.74 / 7.81 / 8.35 / 13.51 |

覆盖率约 13%，多数题的候选里没有正确答案。

### MultiArith

| N | 基线 | 方法 |
|---|------|------|
| 1 | 80.86 | 85.69 |
| 16 | **71.03** / 76.72 / 75.52 / 92.24 | 83.10 / 84.66 / 84.14 / 93.97 |

### 方法关掉 dropout 时

与上一节不是同一设定，不能直接减基线 28.43。

| 集合 | N | 选最高分 | 加权投票 | 多数票 | 覆盖率 |
|------|---|---------|---------|--------|--------|
| GSM8K Test | 1 | 35.18 | — | — | — |
| GSM8K Test | 16 | 34.80 | 36.77 | 35.94 | 52.84 |
| GSM-Hard | 16 | 8.19 | 8.27 | 8.04 | 12.90 |
| MultiArith | 16 | 85.52 | 87.41 | 86.21 | 92.07 |

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
