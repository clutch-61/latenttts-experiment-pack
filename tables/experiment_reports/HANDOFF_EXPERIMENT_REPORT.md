# Phase-1 实测交接报告（供其他智能体复核 / 纠错）

> 生成时间：2026-09-28 ~19:00（Asia/Shanghai）  
> **勘误补丁：2026-09-28 ~19:15** — 见文末 **§13 外部质疑自查**（计数、eval_order 抽样、seed 措辞、分母）。  
> 工作区：`/home/lihourun/ll/LatentTTS-main`  
> 决策依据：`/home/lihourun/ll/科研计划.md`、`/home/lihourun/ll/意见扩展.txt`、本仓库 `results/full/PLAN.md`  
> **用途**：完整记录本轮 agent 接到任务后**实际跑完并落盘**的协议、数字、判读与已知坑。请用原始日志/TSV 复核，不要只信本文件的二次叙述。

---

## 0. 一句话现状

在 COCONUT + 自建 annotate 池上：

- **机制**：O2（hierarchical preference）相对 B0，顺序辨别 / 层级一致性 **明显更好**。
- **任务 BoN**：O2 相对自训 B0/B1，在自然候选上有 **稳定但小的 Acc 增益**（GSM N=16 约 +1～3pt；Hard/MultiArith 同向）。
- **相对官方 LatentRM**：互有胜负，**不足以**写「超过官方」。
- **Claim 边界**：可写「顺序敏感 + 选优不伤 / 小增益」；**不可**写 Latent TTS 大提升 / Phase-2 已验证。
- **进行中**：O1（order only，`λ_hier=0`）仍在训练（约 27%，尚无 `best`）。

Phase-2 Chamfer/MMD、Phase-3 Self-Forcing：**未做**（按计划 hold）。

---

## 1. 实验矩阵 ID（本轮实际含义）

| ID | 配置 | 权重路径 | 状态 |
|----|------|----------|------|
| **B0** | `loss_type=ce`，无 pref | `outputs/latentrm_baseline/best` | 训完 + BoN |
| **B1** | `loss_type=bce`，无 pref | `outputs/latentrm_baseline_bce/best` | 训完 + BoN |
| **O2** | BCE + `L_order` + `L_hier`（`λ_hier=1`） | `outputs/latentrm_order_pref/best` | 训完 + eval_order + BoN |
| **O1** | BCE + `L_order` only（`λ_hier=0`） | `outputs/latentrm_order_only/`（尚无 best） | **训练中** |
| **official** | 发布权重 | `checkpoints/latentRM` | 仅作 BoN 外部对照 |
| **MV / Cov** | 多数投票 / 覆盖率 | 与 PRM **无关**（同生成池） | 已汇总 |

训练 yaml：

- B0：`training_args/train_baseline_ce.yaml`
- B1：`training_args/train_baseline_bce.yaml`
- O2：`training_args/train_order_pref.yaml`（`pref_lambda=0.5`, `order_margin=0.05`, `hierarchy_margin=0.0125`, `lambda_hier=1.0`）
- O1：`training_args/train_order_only.yaml`（同上但 `lambda_hier=0.0`）

分数定义（训练 / `eval_order` 一致）：`pref_score_reduce=mean_log_prob`。  
Corruption ladder：`adjacent_swap` (rank1) / `segment_swap` (2) / `multi_segment` (3) / `full_reverse` (4)。

---

## 2. 共享协议（所有 BoN / MV 对比必须对齐）

| 项 | 值 |
|----|-----|
| Generator | COCONUT（`checkpoints/coconut`） |
| `prm_mode` | `best_of_n` |
| `prm_model_family` | `gpt2` |
| 数据 | `data/gsm_test.json`（1319）/ `gsm_hard.json`（1318）/ `multiarith.json`（580） |
| N | ∈ {4, 16, 64} |
| 主 seed | 42；稳定性另加 GSM N=16 × seed {43, 44} |
| 多卡 | 通常 `CUDA_VISIBLE_DEVICES=0,2,6,7`，`NUM_PROCESSES=4` |
| batch | `batch_size=1024`（infer） |
| Annotate 池 | `latent-data/coconut/{train,valid-4,valid-64}`，annotate `seed=42`，train `n=8` + `n_per_step=64` + dropout `0.2` |

**主对比永远是同 annotate 训出的 O2 vs B0 vs B1。** official 只作外部参照；绝对值不对齐论文数字时，不据此判方法失败（计划：公平第一）。

---

## 3. Phase-0 Annotate（磁盘审计摘要）

来源：`results/full/PLAN.md` Phase-0（2026-09-27）。

| split | shards | unique idx | complete（可训练） | markers | gsm size |
|-------|--------|------------|-------------------|---------|----------|
| train | 11987 | 383517 | **135616** | 247968 | 385620 |
| valid-4 | 16 | 500 | **202** | 310 | 500 |
| valid-64 | 16 | 500 | **346** | 166 | 500 |

- complete：四键齐全；`n_samples`=8/4/64，`num_latents`=6，est/emb fp32。  
- 质量抽样：无 all_correct/all_false；正确轨迹 mean-est ≈0.52 vs 错误 ≈0.37。  
- **注意**：online tqdm 的 pass@1/vot **含丢弃题**，不要当训练集质量指标。

---

## 4. 训练过程数字（valid n64 `recall_at_1`）

| 模型 | 主日志 | train_runtime (s) | 停在 epoch | 日志内 best n64 R@1 | best 落盘时间 |
|------|--------|-------------------|------------|---------------------|---------------|
| B0 | `logs/train_baseline_20260927_214648.log` | 6272.1 | 8.95 | **0.6146** | 2026-09-27 23:31 |
| O2 | `logs/train_pref_20260928_001344.log`（成功那次；此前有中断/重开） | 15719.6 | 6.78 | **0.6146** | 2026-09-28 04:36 |
| B1 | `logs/train_b1_20260928_123528.log` | 6607.6 | 9.54 | **0.6245** | 2026-09-28 14:25 |
| O1 | `logs/train_o1_20260928_160057.log` | （未结束） | ~5.3 @ step~2909/10600 | 目前所见最高 **0.6166** | **尚无** `outputs/latentrm_order_only/best` |

说明：

- 训练时 n64 R@1 **O2≈B0≈0.615**，B1 略高到 **0.6245**。机制优势不体现在这个 R@1 打平上，而体现在 `eval_order` 与下游 BoN。
- **Early-stopping 指标名 bug**：日志反复出现  
  `early stopping required metric_for_best_model, but did not find eval_test_n_64_recall_at_1`  
  yaml 写的是 `metric_for_best_model: test_n_64_recall_at_1`，与 HF 实际 `eval_*` 前缀对不齐 → early stop 实际常被关掉；停止依赖其它逻辑 / 手动或中途结束。复核时不要假设「按 patience 正规早停」。
- O2 曾因 `save_only_model=true` 导致 resume 后 LR 重 warmup；后改为 `save_only_model=false` 并干净重训。以 `001344` 日志 + `outputs/latentrm_order_pref/best` 为准。

---

## 5. 机制评估 `eval_order`（合成 corruption）

日志：`logs/eval_order_20260928_091520.log`  
协议：`mean_log_prob`；prefer `corrects=1`；四种 corruption。

### 5.1 汇总

| split | model | order_pair_acc | hierarchy_consistency |
|-------|-------|----------------|------------------------|
| valid-4（全量 ~804 traj） | **O2** | **0.9657** | **0.8742** |
| valid-4 | B0 | 0.8774 | 0.7977 |
| valid-64（~3200 traj sample） | **O2** | **0.9682** | **0.9008** |
| valid-64 | B0 | 0.8224 | 0.7256 |

### 5.2 分 corruption（pos>neg 率）

**valid-4**

| corruption | O2 | B0 |
|------------|-----|-----|
| adjacent_swap | **0.9909** | 0.7009 |
| segment_swap | **1.0000** | 0.9467 |
| multi_segment | 0.8719 | 0.8788 |
| full_reverse | **1.0000** | 0.9835 |

**valid-64 sample**

| corruption | O2 | B0 |
|------------|-----|-----|
| adjacent_swap | **0.9922** | 0.7084 |
| segment_swap | **0.9685** | 0.8748 |
| multi_segment | **0.9122** | 0.7964 |
| full_reverse | **1.0000** | 0.9099 |

**信号解读：** O2 在最细粒度 adjacent_swap 上相对 B0 拉开最大 → 符合 L_pref 目标。  
**不解读为：** 推理能力提升；这是合成置换排序。

B1 / O1 的 `eval_order`：**本轮未跑**（O1 未训完）。

---

## 6. 任务层 BoN Acc（自然候选）

入口：`src.infer_gpt2_rm`，`prm_mode=best_of_n`。  
**Acc** = RM 选出的那一条是否答对。  
**Coverage** = N 条里是否至少一条答对（与 PRM 无关）。  
**Voting Accuracy** = 答案多数投票（与 PRM 无关）。

### 6.1 GSM8K-Test · seed=42 · 主矩阵

来源：`results/full/bon_gsm_test_20260928_093713.tsv` + 日志 `logs/bon_matrix/*_20260928_093713.log`  
（该次多数 run 在打印 Acc 后 Cov/Vot 收尾 crash；**Acc 可信**。Cov/Vot 见 §7 补跑。）

| PRM | N=4 Acc | N=16 Acc | N=64 Acc |
|-----|---------|----------|----------|
| **O2** | **33.4344%** | **35.0265%** | 31.6149% |
| B0 | 32.5246% | 32.8279% | 31.0083% |
| official | 32.7521% | 33.4344% | **33.7377%** |

O2−B0（pt）：**+0.91 / +2.20 / +0.61**。

### 6.2 B1 · 同协议 · seed=42

来源：`results/full/bon_b1_20260928_152912.tsv`（`logs/bon_b1/`）

| split | N=4 | N=16 | N=64 |
|-------|-----|------|------|
| gsm_test | 32.0697% | 32.1456% | 31.1600% |
| hard | 7.1320% | 6.9803% | 6.6768% |
| multiarith | 79.6552% | 79.3103% | 82.0690% |

同表对比（GSM seed42）：

| N | B1 | B0 | O2 |
|---|-----|-----|-----|
| 4 | 32.07% | 32.52% | **33.43%** |
| 16 | 32.15% | 32.83% | **35.03%** |
| 64 | 31.16% | 31.01% | **31.61%** |

**信号：** O2 > B1 ≈ B0 → **增益不是「换成 BCE」alone 造成的**；pref 有独立贡献。

### 6.3 Hard · seed=42（O2/B0/official + B1）

来源：`logs/bon_stability/*` + `bon_b1`

| PRM | N=4 | N=16 | N=64 |
|-----|-----|------|------|
| O2 | **7.5114%** | **7.2838%** | **7.5114%** |
| B0 | 7.2838% | 6.9044% | 6.7527% |
| B1 | 7.1320% | 6.9803% | 6.6768% |
| official | 7.5873% | **7.6631%** | 7.4355% |

O2−B0：**+0.23 / +0.38 / +0.76** pt（全正、量小）。

### 6.4 MultiArith · seed=42

| PRM | N=4 | N=16 | N=64 |
|-----|-----|------|------|
| O2 | 79.4828% | **80.3448%** | **82.4138%** |
| B0 | 78.2759% | 78.7931% | 78.6207% |
| B1 | 79.6552% | 79.3103% | 82.0690% |
| official | **80.1724%** | **81.5517%** | **84.8276%** |

O2−B0：**+1.21 / +1.55 / +3.79** pt。相对 official 仍整体偏低。

### 6.5 GSM-Test N=16 · 多 seed（稳定性）

| PRM | seed42 | seed43 | seed44 | 三 seed 均值 |
|-----|--------|--------|--------|--------------|
| O2 | 35.0265% | 33.2828% | 33.3586% | **33.8893%** |
| B0 | 32.8279% | 31.9939% | 32.7521% | **32.5246%** |
| official | 33.4344% | 34.1926% | 33.2070% | **33.6113%** |

O2−B0 deltas：**+2.20 / +1.29 / +0.61** pt，均值 **+1.36** pt，**三 seed 全正**。

原始稳定性 TSV：`results/full/bon_stability_20260928_101417.tsv`  
（注意：该 TSV 的 `vot` 列曾被错误解析成字符串 `Accuracy:`——应用日志里的 `Voting Accuracy:` 第三字段；Acc/Cov 列一般正确。）

---

## 7. Coverage / Majority Voting（与 PRM 无关）

同一 `split × N × seed` 下，换 O2/B0/official/B1 **Cov/Vot 应对齐**（已观察到对齐）。

### 7.1 汇总表（推荐引用）

文件：`results/full/mv_coverage_summary.tsv`  
GSM seed42 补跑：`results/full/bon_mv_gsm42_20260928_122615.tsv`（用 official 权重跑生成；只取 Cov/Vot）

| split | N | seed | Coverage | Voting Acc |
|-------|---|------|----------|------------|
| gsm_test | 4 | 42 | 41.0917% | 32.8279% |
| gsm_test | 16 | 42 | 51.0993% | 34.3442% |
| gsm_test | 64 | 42 | 57.9985% | 33.7377% |
| gsm_test | 16 | 43 | 50.7202% | 33.7377% |
| gsm_test | 16 | 44 | 51.4026% | 33.3586% |
| hard | 4 | 42 | 9.7876% | 7.5873% |
| hard | 16 | 42 | 12.2914% | 7.2079% |
| hard | 64 | 42 | 14.5675% | 7.0561% |
| multiarith | 4 | 42 | 86.2069% | 78.2759% |
| multiarith | 16 | 42 | 92.4138% | 79.3103% |
| multiarith | 64 | 42 | 94.1379% | 79.8276% |

### 7.2 与 O2 BoN 对照（GSM seed42）

| N | MV | O2 BoN Acc | O2−MV |
|---|-----|------------|-------|
| 4 | 32.83% | 33.43% | +0.60 |
| 16 | 34.34% | **35.03%** | +0.68 |
| 64 | 33.74% | 31.61% | −2.12 |

**信号：** N=16 上 RM 选优略优于投票；N=64 上 O2 选优反而差于 MV（与「大 N 排序变差」一致）。

本仓库无独立「MC-dropout eval」脚本；dropout 采样已嵌在 annotate/BoN 生成协议里，用 Cov/Vot 对照即可。

---

## 8. 已知工程坑（复核时务必注意）

1. **`infer_gpt2_rm` Cov/Vot 收尾**：曾因 `list.any` / `ndarray.float` 在打印 Acc 后崩溃。已改为 `np.asarray(..., bool)` + `.any(axis=-1).mean()`（见 `src/infer_gpt2_rm.py` 约 277–282 行）。**旧 bon_matrix 多数只有 Acc。**
2. **TSV 解析 Voting**：`Voting Accuracy: xx%` 用 `awk '{print $2}'` 会得到 `Accuracy:`；应用 `$3` 或正则 `Voting Accuracy:\s*([\d.]+%)`。`eval_bon_stability.sh` 的 vot 列不可信；以日志为准。
3. **Early-stopping 指标名**：见 §4；B0/B1/O2/O1 日志均有 warning。
4. **绝对 Acc ~33%（GSM）**：偏低；计划要求先查协议对齐，**不要**用「没复现论文绝对值」单独否定相对增益。
5. **GPU**：共享机器；默认避开别人占用的卡（常用 0,2,6,7）。不要杀他人进程。
6. **O2 训练日志有多份**：以 `train_pref_20260928_001344.log` + `outputs/latentrm_order_pref/best` 为准。

---

## 9. 原始产物索引

| 类型 | 路径 |
|------|------|
| 工作计划 / 决策备忘 | `results/full/PLAN.md` |
| GSM BoN 主矩阵 TSV | `results/full/bon_gsm_test_20260928_093713.tsv` |
| 稳定性 TSV | `results/full/bon_stability_20260928_101417.tsv` |
| B1 BoN TSV | `results/full/bon_b1_20260928_152912.tsv` |
| MV 汇总 | `results/full/mv_coverage_summary.tsv` |
| GSM42 MV 补跑 | `results/full/bon_mv_gsm42_20260928_122615.tsv` |
| eval_order 日志 | `logs/eval_order_20260928_091520.log` |
| BoN 日志目录 | `logs/bon_matrix/`、`logs/bon_stability/`、`logs/bon_b1/`、`logs/bon_mv/` |
| 脚本 | `scripts/eval_bon_matrix.sh`、`eval_bon_stability.sh`、`eval_bon_b1.sh`、`eval_mv_fill_gsm42.sh`、`summarize_mv.py`、`train_*_detach.sh` |

---

## 10. 判读结论（给纠错方：这是「主张」不是「事实」）

对照 `科研计划.md` Go / No-go：

| 主张 | 依据 | 强度 |
|------|------|------|
| L_pref 学到了顺序 / 层级 | eval_order O2≫B0，尤其 adjacent_swap | **强** |
| 自然 BoN 相对自训 CE/BCE 稳定略升 | GSM/Hard/MA；三 seed；O2>B1≈B0 | **中（软正）** |
| 可超过官方 LatentRM | 互有胜负，N=64 常输 | **弱 / 不主张** |
| hierarchy 比 flat order 更必要 | 需 O1 BoN；**尚未完成** | **未决** |
| 应开 Phase-2 set alignment | 计划：仅当 Phase-1 自然 BoN 有效后再议；当前仅软正 | **本轮不做** |

**建议对外叙事（收窄版）：**  
Order-aware hierarchical LatentRM 在合成时序扰动上显著提高顺序辨别与层级一致性；在 COCONUT 自然候选 Best-of-N 上，相对同数据 CE/BCE LatentRM 带来约 1–2pt、难集同向的稳定小增益；尚不足以宣称全面超过官方 LatentRM。

---

## 11. 未完成 / 建议复核清单

- [ ] O1 训完 → `eval_order` + 同协议 BoN（至少 GSM N=16 + Hard）→ 拆 hierarchy  
- [ ] O2-m（gap 标定 margin）、corruption Abl  
- [ ] B1 / O1 的 `eval_order`  
- [ ] 修 early-stop 指标名后是否应重训（可选；当前 best 已用于表）  
- [ ] Beam search / 更多 seed 方差报告  
- [ ] CODI（计划：COCONUT 主表够写后再开）  
- [ ] 用本文件数字 vs 原始 log 做一次独立 diff，抓二次叙述错误  

---

## 12. 本轮 agent 做过的事（便于追责，非流水账）

1. 完成/沿用 Phase-0 annotate 与 B0、O2 训练及 `eval_order`。  
2. 跑 GSM BoN 矩阵（O2/B0/official × N）；修 Cov/Vot crash。  
3. 跑 Hard/MultiArith + GSM 多 seed 稳定性。  
4. 汇总 MV/Cov；补 GSM seed42 Cov/Vot。  
5. 训 B1（BCE 无 pref）并跑全套 BoN → 排除「换 BCE」假说。  
6. 启动 O1（`λ_hier=0`）训练（截至写本文件时未完成）。  
7. **未**开 Phase-2/3；**未**改主方法 corruption / 猛加 `λ_pref`。

若发现本文件与日志冲突：**以日志 / `*.tsv` 为准**，并应在 `PLAN.md` 与本文件顶部标注勘误。

---

## 13. 外部质疑自查（2026-09-28 19:15，磁盘+代码复核）

针对另一智能体提出的 8 点，**本地重新审计**结论如下。总判断：**没有发现「训错数据 / 测错全集 / 协议串台」级错误**；有 **报告表述不透明** 与 **eval_order valid-64 抽样过窄**，必须改正措辞并建议补测。

### 13.1 数据计数「对不上」——已解释，不是损坏

此前 PLAN/报告把 **per-shard 记录数** 写成了像「unique 样本数」：

| split | unique idx | unique complete | unique marker-only | 二者之和 | per-shard complete+marker | 超额 |
|-------|------------|-----------------|---------------------|----------|---------------------------|------|
| train | 383517 | **135597** | **247920** | **=383517** | 135616+247968=383584 | +67 |
| valid-4 | 500 | **201** | **299** | **=500** | 202+310=512 | +12 |
| valid-64 | 500 | **340** | **160** | **=500** | 346+166=512 | +12 |

- **按 `idx` 去重后**：`complete ∪ marker-only` 互斥，和 = unique，**账平**。  
- **超额 67/12**：同一 `idx` 出现在多个 shard（多卡/resume 边界）。其中 train 有 19 个 idx 出现 **>1 份 complete**；另有「先 marker 后 complete」跨文件。  
- Loader（`CachedPickleDatasetV2`）按文件遍历写入 `idx_to_file_name`，**后出现的文件覆盖前者**；训练只认带 `estimations` 的 idx。  
- **结论**：不是「多出来神秘样本」；是报告把 **shard 键计数** 与 **unique 题数** 混写。应改用上表 unique 列。旧文「complete≈135616」应改为 **unique complete≈135597**（键计数 135616 含重复）。

### 13.2 筛掉 ~65% 题——设计如此；切分按题目、无泄漏

- Annotate 默认 `remove_all_correct=True`, `remove_all_false=True`（`src/annotate_data.py`）：全对/全错题只写 `input_ids=0` 占位，不进 RM 训练。  
- **题目级隔离（question 字符串集合）**：`train ∩ valid = 0`，`train ∩ test = 0`，`valid ∩ test = 0`。  
- train/valid/test 来自不同 JSON（`gsm_train` / `gsm_valid` / `gsm_test`），**不是**「先生成轨迹再随机切轨迹」。  
- **R@1 / eval_order**：在 **筛选后的 valid complete** 上（适合「有区分度的候选」）。  
- **BoN / MV**：在 **完整** `gsm_test.json`(1319) / `gsm_hard.json`(1318) / `multiarith.json`(580) 上——**不是** annotate 子集。  
- **结论**：训练分布偏向「有时对有时错」的题，应在论文写明；**未发现** train↔valid 题目泄漏。不算做错，但旧报告没强调「valid 指标 ≠ 全集难度分布」。

### 13.3 valid-64 `eval_order` 确实只覆盖约 50 题——质疑成立

`src/eval_order_pref.py`：`n = min(len(ds), max_batches * batch_size)`，且 `get_single_sample=True`（一条 traj 一个 index）。

日志：

- valid-4：`order-eval` **51/51** batch → 与 `201×4=804` traj、`batch_size=16` 几乎全量一致。  
- valid-64：`order-eval` **200/200** batch → `200×16=3200` traj → **3200/64 = 50 道题**（数据集前 50 个 complete idx 的全部 traj，再在 batch 内滤 `corrects=1`）。

旧报告写「~3200 traj sample」**低估了「独立单位是题」**；应改写为：**约 50 题 × 最多 64 条 traj**。  
O2/B0 用同一脚本、同一 `data_dir` 顺序，**题目集合一致**；指标是 batch 内 pair 再对 batch 平均，**不是**严格的「先按题平均再宏观平均」。  
**结论**：方向性信号仍在（valid-4 全量也 O2≫B0），但 **valid-64 机制数字不能当充分总体估计**；建议全量重跑或题目级 bootstrap。属 **评估覆盖不足 / 报告夸大把握**，不是协议串台。

### 13.4 「三 seed」= 推理采样 seed，不是三次独立训练——质疑成立

GSM N=16 的 seed 42/43/44：固定 **同一个** `outputs/latentrm_* /best` checkpoint，只改 `infer_gpt2_rm --seed=`（生成候选随机性）。  

正确表述：

> 在多个**推理/候选采样 seed** 上，O2−B0 同号。

**不能**写成「三次独立训练均可复现」。旧交接文若暗示训练可复现，视为 **措辞过强**，应收回。

### 13.5 Corruption rank 是约定，不是已证「逻辑严重度」——同意，非实验错误

计划红线已要求称 **permutation-corrupted / 合成顺序扰动**。`hierarchy_consistency` 高 = 学会了**预设 rank 阶梯**。尚未做「扰动后再解码看正确率是否随 rank 下降」。属 **claim 边界**，不是测错。

### 13.6 MultiArith 580、Hard 1318——不是评测时偷偷滤题

磁盘上：

- `data/multiarith.json` **len=580**（不是 600 再滤 20）；  
- `data/gsm_hard.json` **len=1318**（不是 1319 再滤 1）；  
- `data/gsm_test.json` **len=1319**。

BoN 分母即 JSON 长度。与官方别处「600/1319」差异来自**本仓库数据文件本身**；各 PRM 共用同一文件。建议在报告注明「本仓库自带分母」，无需怀疑某模型少评题。

### 13.7 B0/B1/O2 混杂——部分已控，措辞可再收

- O2 vs B0：CE↔BCE **与** pref 混杂。  
- O2 vs B1：同为 BCE，检验 pref —— 更干净的因果。  
- 多 seed 上 O2>B1 已补（GSM N16）；仍是**推理 seed**。  
- O2 墙钟更长（~15.7ks vs B0/B1 ~6.3–6.6ks）应交代。  

不算做错；「排除换 BCE」在推理 seed 上成立，训练复现仍缺。

### 13.8 可先不担心的点——同意

R@1 打平 vs eval_order 高；N=64 BoN 掉点；official 互有胜负；Cov/Vot 与 PRM 无关 —— 均不指示协议错误。

### 13.9–13.14 补测与 O1（摘要）

- valid-64 **全量** eval_order（340 题）：O2 0.972/0.894；B0 0.860/0.769；B1 0.766/0.728。  
- B1 GSM N16 seed 42/43/44：32.15 / 31.99 / 32.45%。  
- O1 训完并评估：pair 0.970、hier 0.736；GSM N16 均值 32.22%≈B1≪O2 33.89%。  

### 13.15 claim 收紧 (2026-09-29 12:04)

推荐表述：hierarchy 是**当前 COCONUT 设定下**下游收益的重要组成，非普遍必要；非 generator 提升；推理 seed≠训练复现。

**Hard Acc@seed42（%）：** O2 7.51/7.28/7.51；O1 7.21/6.60/6.30；B1 7.13/6.98/6.68；B0 7.28/6.90/6.75（N4/16/64）。  

**MultiArith Acc@seed42（%）：** O2 79.48/80.34/82.41；O1 74.48/76.72/78.62；B1 79.66/79.31/82.07；B0 78.28/78.79/78.62。  
MA 上 O2−O1 ≈ +3.6～+5.0pt（O1 甚至低于 B1）；Hard/MA 尚无多推理 seed。

**配对统计（已完成）**：见 §13.17。

### 13.16 磁盘事故 (2026-09-29)

根分区曾 100% 满，导致本文件中段截断；已删 `outputs/*/checkpoint-*`（保留 `best/`）及 ABORT 目录腾出空间后修复本节。

### 13.17 同候选池配对 BoN · GSM N16 (2026-09-29 16:05 完成)

`results/full/bon_paired_gsm16_20260929_155950/pair_summary.json`  
`candidates_identical=true`（answers_sha1 全一致，n=1319）；Cov/MV 三边相同。

| | Acc | 独对/对方独对 | 配对差 (pp) [95% CI] |
|--|-----|---------------|----------------------|
| O2 vs O1 | 35.03 / 31.01 | 94 / 41 | **+4.02 [2.35, 5.84]** |
| O2 vs B1 | 35.03 / 32.15 | 77 / 39 | **+2.88 [1.29, 4.47]** |
| O1 vs B1 | 31.01 / 32.15 | 56 / 71 | −1.14 [−2.88, +0.45] |

**锁定表述**：本设定下 O2 改善 LatentRM 选优；O1 对 B1 无可靠优势；≠ 普遍必要 / ≠ generator 提升；单 checkpoint + 单推理 seed。  
**Caveat**：answers_sha1 ≠ latent trajectory hash（流程上同 seed 应同源；严格核验可选）。  
**Next**：Hard / MultiArith 同协议配对；可选训练重复。 → **已被 Phase-G 覆盖优先级：Hard/MA 配对可后做；先冻 O2，开 SF-style generator pilot（见 PLAN Phase-G）。**

### 13.18 Phase-G 锁定 (2026-09-29 18:05；A/B/C 收紧 21:27)

冻结 Phase-1 O2。称呼：**答案监督的可微 self-rollout pilot**（不做硬凑 TF→SF）。  
A 冻结 / B 缓存 latent+detach+答案 CE / C 可微 self-rollout+同答案 CE；主差=梯度是否穿过 rollout。Smoke 须证非零 grad、参数更新、rollout≡推理。成败看 Pass@1/Coverage，不用 O2 BoN。MMD 后置。详见 `PLAN.md` Phase-G。

### 13.19 SF-style smoke PASS (2026-09-29 21:40)

备份 `checkpoints/_freeze_backup_20260929/`。`src/sf_rollout.py` + `scripts/smoke_sf_rollout.py`：B/C 梯度区分成立（C 非零 latent grad；B=0 仍更新参数）。详见 PLAN Phase-G Smoke。

### 13.20 SF-style 最小 pilot 首跑 (2026-09-29 21:50)

512 train / 256 valid / 200 steps。A Pass@1=0.310 Cov@4=0.457；B 0.238/0.348；C 0.257/0.371。B/C 均低于 A；C≳B 但整体负向。对照与梯度路径成立；先查答案监督格式，不上 MMD。详见 PLAN Phase-G。

### 13.21 首轮 SF pilot 负结果读法 (2026-09-29 23:23)

B/C 均 < A；C 略 > B 但无 CI。链路通 ≠ 目标对。表述：答案监督微调退步；不贴 SF 失败/有效。格式确认前不多训、不加 MMD。诊断顺序见 PLAN。

### 13.22 答案格式诊断 (2026-09-29 23:30)

原模型 end-latent 后为 `### {answer}<eos>`，首轮 CE 误用 `#answer`。loss mask 位置正确。先修正目标再小样本过拟合。

### 13.23 格式修正后读法 (2026-09-29 23:47)

格式非主因；短答案 CE 损害生成。C≳B 无 CI。先逐题迁移+输出结构，不加步数/MMD。

### 13.24 退步主因：短答案 CE 压短推理 (2026-09-30)

after_len A283/B38/C123；非解析失败。C 缓退步 ≠ 恢复 A。Pass@1 两表差=pass@k vs sample0。下一轮先改监督目标（组 D），不加步数/MMD。见 PLAN。

### 13.25 D 读法：形状≠推理；暂停扩训 (2026-09-30)

D 拉长输出但 Acc 更差。先审 steps 标签 → A 正确轨迹伪标签对照。不加步数/MMD。见 PLAN。

### 13.26 steps 审计 + E 伪标签 (2026-09-30)

steps 96.5% 与答案一致。E=A 正确轨迹伪标签：Pass@1≈31.3% Cov≈44.1% ≈A；D 崩溃主因是 dataset steps 不适配，非过程监督无用。见 PLAN。

### 13.27 组合方案锁定 SF+MMD+O2 (2026-09-30)

主线改为组合验证。O2 先冻结于推理选择；D_mmd=SF+A-pseudo+轻量 content-view MMD；C 为 SF-only 消融。生成指标不用 O2。见 PLAN Phase-G2。

### 13.25 Phase-G2 D_mmd claim 收紧 (2026-09-30 20:37)

轻量 MMD 无可见生成提升、未破坏形态；content probe~83% → 非 position-free。不写 31.4>31.3 改善；D+O2 仅可选诊断（非默认）；不加大 λ/步数。三块分写：O2 最有证据 / SF A-pseudo≈A / MMD 无明确收益。见 PLAN。

### 13.26 O2-RFT + D+O2 diag (2026-09-30)

D_mmd+O2 < A+O2（BoN 35.2→33.6）。O2rft-short 塌缩；O2rft-traj(仅<<) 形态≈A 但 Pass@1/Cov/BoN 均未超 A。耦合方向对、本轮无涨点。见 PLAN。

### 13.27 O2-gate + A-pseudo (2026-09-30)

margin_min=0.5 修正负 margin 更新。k4：Pass@1 32.3 Cov 46.9 after≈A；+O2 Acc 34.8 Cov 43.4。弱于/近 A+O2 Acc，生成略高于 A 但 CI 跨 0。k8 生成略升、BoN Acc 降。见 PLAN。

### 13.28 O2-pref best-vs-worst (2026-09-30)

cvw+UL 损害 Cov。cvc λul=0：Pass@1=32.6 Cov=45.7；+O2 34.8/42.2。相对 O2gate 未稳赢。见 PLAN。

### 13.29 beat-A (2026-09-30)

统一 O2 打分；过程轨迹接近/超过 A 则 CE 用 self。self≈19% 更新。Pass@1=32.3 Cov=45.7；+O2 33.6/42.2。未稳超 O2gate。见 PLAN。
