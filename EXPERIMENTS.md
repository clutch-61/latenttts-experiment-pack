# 三主实验方案（论文主线规划）

> 实验章节成稿草稿（可直接对照写论文 §Experiments）：[`CLOSED_LOOP_STORY.md`](CLOSED_LOOP_STORY.md)。

> **2026-10-19 改锁：** G-Ours = **mmd_cold**（取代 p2mmd05）。权威主表/消融：`tables/experiment_reports/paper_tables_mmd_cold/`；锁矩阵：`tables/experiment_reports/paper_mmd_cold/`。


> 锁定时间：2026-10-15。这是论文主线「生成器 ↔ 打分器协同」的顶层实验规划。每个实验统一：潜变量步数 **T=6**、奖励模型测试时**关 dropout**、种子 42、GSM8K-Test / GSM-Hard / MultiArith、主指标 **top1**、主 N∈{16,64}（N=1/4、wvote/maj/cov 进附录）。不跨生成器底座归因绝对分差。

---

## 主线一句话

两个贡献互为因果：

- **贡献 1（生成器）**：p2 续训拓宽潜空间探索——Greedy +0.76、Coverage@16 +1.74，且在 CODI 强底座上依然 +1.06（通用，不是弱模型过拟合）。
- **贡献 2（打分器）**：探索引发表征漂移，官方对比 LatentRM（B0）变现探索收益的能力相对不足；顺序偏好 O2 把同池 O2−B0 gap 从 +0.78 翻倍到 +1.57（3 seed 同向）。

严谨口径：只比**同池 O2−B0 gap**，不引用「B0 变现率下降」（跨口径混比；严格同池下 B0 变现率基本持平）。详见 README 第 0 / 3.1 节。

---

## 主实验 1：生成器训练策略对比（固定 COCONUT 架构）

**核心目的：** 在完全相同的连续潜空间架构与起点下，回答「后训练/微调该用什么策略」。

**统一协议：** 冻结 COCONUT-GPT2 124M 起点；T=6；评测不用早停 gate；打分器统一用 **B0**（官方 LatentRM）或确定性规则（Greedy / Majority Vote），**不引入 O2 偏好绑定**。

| 编号 | 生成器 | 性质 | 状态 |
|------|--------|------|------|
| G0 | 冻结 COCONUT | 起跑线 | ✅ 全矩阵 |
| G1 | COCONUT + Standard SFT | 经典范式对照 | ❌ 需训（纯问答 CE，无门控） |
| G2 | COCONUT + RFT (Self-Training) | 正确轨迹过滤微调 | ❌ 需训（只筛答对样本重训） |
| G3 | COCONUT + SLPO (RLOO) | 2026 外部强基线 | ✅ 同起点同预算（3000 步）已训 |
| G4 | COCONUT + Dropout-GRPO | 2026 外部强基线 | ✅ 同起点同预算（3000 步）已训（同骨干适配版） |
| G-Ours | COCONUT + **mmd_cold**（dual_beat+SF-C+λ_MMD=0.05） | 本文工作 | ✅ 全矩阵 `paper_mmd_cold/`；消融基线 dualBeat |

**汇报指标：** N=1 Greedy、Coverage@16/64、BoN Top1（接 B0）。

**当前读数（GSM8K-Test，B0 打分，N=16，3-seed 均值，统一 NEW 协议）：**

| 生成器 | Cov@16 | B0 Top1 | Δ top1 vs G0 |
|--------|-------:|--------:|-------------:|
| G0 冻结 COCONUT | 50.67 | 32.88 | — |
| G3 SLPO (RLOO 3000) | 50.90 | 32.88 | +0.00 |
| G4 Dropout-GRPO (3000) | 51.23 | 32.85 | −0.03 |
| **G-Ours mmd_cold** | **~54.0** | **35.03** | **+2.15** |

逐 seed B0 top1（mmd_cold）：35.18 / 35.10 / 34.80。旧 p2mmd05≈34.3 已降级。

**读数解读（诚实口径）：** 同起点、同 3000 步预算下，SLPO(RLOO) 与 Dropout-GRPO 的 B0 Top1 与冻结 COCONUT **持平**（−0.03~+0.00），训练时 `eval_reward` 全程 0.352–0.356 无提升——RL 在该预算下没学到东西；p2 同预算下 +1.44pp。**但注意预算口径不可直接等同**：官方 SLPO/GRPO 为 40000/50000 步，这里压到 3000（约 1/15）；且 RL 每步 `num_generations=8`（128 轨迹/步）远小于 p2 的 `k=64`（1024 轨迹/步）。所以严格结论是「**同 3000 步、RL 轨迹数天然劣势**下的对照」，不能宣传成「p2 打赢 SLPO/GRPO 论文分」。

---

## 主实验 2：策略对生成器底座的泛化（COCONUT / CODI / CoLaR）

**核心目的：** 验证 p2 在不同潜空间底座上是否依然带来额外收益，还是只在弱底座起作用。

**统一协议：** 每个底座内部做前后对照；**不拿跨底座绝对分差当策略收益**。
- A/B（GPT-2）：评测可用 O2（零样本迁到 CODI 须标注）。
- C（CoLaR = Llama-1B + LatentHead，d=2048）：**不能**接 GPT-2 B0/O2；主指标 **Greedy / Coverage@16/64 / Majority Vote**；门控用答案正确性（`answer_beat`）。

| 底座 | 训练策略 | 评测 | 状态 |
|------|----------|------|------|
| A (COCONUT) | 原版冻结 G0 | O2 | ✅ 冻 +O2 = 32.07 (GSM N16) |
| A (COCONUT) | + **mmd_cold** | O2 | ✅ **36.01**（3-seed **36.32±0.40**；vs 冻 +3.94 / ~+2.6） |
| B (CODI) | 原版冻结 | O2 | ✅ 40.94 |
| B (CODI) | + CODIp2 | O2 | ✅ 42.00（+1.06，同池） |
| C (CoLaR) | 原版冻结（官方 `###`+max_L=64） | Greedy/Cov/Maj | ✅ Greedy 40.56 / Cov@16 64.97 / Maj@16 45.11 / Cov@64 77.79 / Maj@64 47.16 |
| C (CoLaR) | + Colar-p2 v1（官方同协议） | 同协议 | ❌ 崩塌：Greedy 40.56→1.59 / Cov@16 64.97→7.51 |
| C (CoLaR) | 冻结 fixed_t（T=16，无 `###`） | Greedy/Cov/Maj | ✅ 22.44 / 35.48 / 23.73（**与官方分表**） |
| C (CoLaR) | + Colar-p2v2（fixed_t 自洽） | 同 fixed_t | ✅ **34.42 / 49.58 / 36.85**（底座内 +12.0 / +14.1 / +13.1） |

详情与纪律：[`COLAR_ADAPTATION.md`](COLAR_ADAPTATION.md)。

**写作逻辑：**
- CODI 原版显著高于 COCONUT → 归因底座压缩能力与自蒸馏初始化；
- 重点：在 CODI 极强基线上，CODIp2 依然推高 Top1（40.94 → 42.00），证明 p2 是通用潜空间强化机制，非针对弱模型的过拟合。
- CoLaR 提供 **非 GPT-2** 第三点：v1 证明「盲目移植会崩」；v2 在 **策略自洽 fixed_t** 下底座内正 Δ，**不可**与官方 40.56 混成同一 Δ，也**不可**写成已接 O2 的完整闭环。

**跨池警告：** 底座 A 的冻 O2 与 p2mmd05+O2 的候选池 cov/maj 不等（新协议下 51.02/33.74 vs 52.99/36.32），方向稳但非严格同池，写作须标「同协议、seed 对齐，非逐条同池」。底座 B 侧 cov/maj 全等，干净。


---

## 主实验 3：潜空间轨迹打分器对比（固定候选池）

**核心目的：** 候选轨迹已生成好的前提下，谁能把做对的那条挑出来。

**统一协议：** 固定同一批候选 Dump（禁止不同打分器吃不同采样池）；关 scorer dropout；GSM8K-Test / GSM-Hard / MultiArith；N∈{4,16,64}。

| 编号 | 打分器 | 输入 | 状态 |
|------|--------|------|------|
| S0 | Majority Vote | 文本答案 | ✅ 同池 maj 列 |
| S1 | B0 (LatentRM) | 潜变量向量序列 | ✅ |
| S2 | ELHSR-Latent（=SWIFT，2505.12225） | 潜层隐状态线性奖励 | ✅ 已适配测评（1.5K 参数） |
| S3 | HSRM-Latent（2608.30841） | 步骤边界隐状态 Transformer 排序器 | ✅ 已适配测评（1.79M 参数） |
| S-Abl | BCE Scorer | 潜变量向量序列 | ⚠️ 数字已有（B1 GSM N16=32.15），checkpoint 删、复现需重训 |
| S-Ours | O2 Scorer | 潜变量向量序列 | ✅ |

**汇报指标：** Top1 Accuracy（主要）、Weighted Vote、Majority Vote。

**当前读数（GSM N16 top1，同池，5-seed 均值 ± std，seeds 42–46）：**

| 池 | B0 | O2 | SWIFT-Latent | HSRM-Latent |
|----|----|----|--------------|-------------|
| 冻 COCONUT | 32.72 ± 0.96 | **33.72 ± 1.03** | 33.69 ± 0.75 | 33.60 ± 0.75 |
| p2mmd05 | 34.44 ± 0.38 | 35.92 ± 0.28 | 35.88 ± 0.48 | **36.06 ± 0.37** |

**核心结论 1（O2 vs B0，5-seed）：** 同池配对 top1：冻池 **+1.00pp [0.17, 1.85] 显著**；p2 池 **+1.49pp [0.59, 2.38] 显著**。O2 变现优势在续训后稳定放大，5 seed 后冻结池也显著。

**核心结论 2（轻量 scorer vs B0）：** SWIFT/HSRM 同样可挑出正确轨迹，且在 p2 池上**比 B0 高 ~1.4–1.6pp**、在冻池高 ~0.9pp。→ 漂移后掉队的是 **B0**；轻量 scorer 与 O2 同档。

**核心结论 3（O2 vs SWIFT/HSRM，诚实阴性 + 机制轴优势，5-seed）：** top1 上 O2 **无优势**（冻/p2 配对 CI 全跨 0）；gate 消融亦无下游优势（5 seed，B0 尺 SWIFT−O2 **+0.36 ± 0.66**，方向混杂）。但 **`auc_cw`（正确-错误排序）** 上 O2 显著：冻 O2−SWIFT +2.34 [0.12, 4.57]；p2 O2−SWIFT +3.19 [0.89, 5.49]、O2−HSRM +2.70 [0.48, 4.92]。O2 的**机制轴**优势仍在**顺序/层级指标**：

| 模型 | order_pair_acc | hierarchy_consistency |
|------|---------------:|----------------------:|
| **O2** | **0.972** | **0.894** |
| B0 | 0.860 | 0.769 |
| B1 | 0.766 | 0.728 |
| SWIFT-Latent | 0.449 | 0.242 |
| HSRM-Latent | 0.475 | 0.441 |

SWIFT/HSRM 对潜轨迹顺序**低于随机**（outcome-only 训练，无顺序梯度）；O2 接近满分。此优势**未**转化为下游 BoN/gate 收益（该 trainer 的 correctness 由答案抽取判定，scorer 仅需在正确答案间排序，不需要顺序感知）。

**核心结论 4（可比性审计，2026-10-16 晚）：** 三项针对性检查（详见 `tables/CONDITION_ALIGNMENT_AUDIT.md`）：
1. **条件不对齐**：训练**数据**对齐（同划分/同池/同 `corrects`），但 O2 是 **124M、吃整条轨迹的预训练 GPT-2**，SWIFT/HSRM 是**随机初始化、只看 6 个潜步**的小模型（1.5K / 1.8M 参数，差 ~10³–10⁵×）→ 主表名次**不能**当方法优劣读。
2. **逐题配对（Top1 之外）**：`top1 / top1_ach / mrr` 上 O2 vs SWIFT/HSRM **全部 CI 跨 0**；但**正确-错误配对排序 `auc_cw`** 上 **O2 逐 seed 6/6 同号为更高**（frozen O2−SWIFT **+2.63pp [0.20, 5.07] 显著**；p2 +2.41pp 边缘；对 HSRM +1.9pp 不显著）→ O2 提升了对/错区分度，但未稳定转为 top-1。
3. **无 test 泄漏**：train/valid/test 两两问题级重叠 = 0；选模型只用 valid-64；机制指标也在 valid 上算。配置里 `test_n_64` 是 valid-64 的历史命名。

> 注意：B0/O2 是 COCONUT latent 上训的权重，对 CODI 是零样本迁移；公平对比须标「是否在 CODI latent 重训/校准」，否则不算同条件。`CODIp2+B0` = 测试时换 scorer，非独立于 O2 的训练消融（p2 训练用过 O2 门控）。

**核心结论 5（Order-Decision 受控实验，5-seed）：** 构造「内容/答案相同、仅顺序不同」的候选池，让顺序感知成为唯一判据（详见 `tables/order_decision/`）：
- O2 首次把机制优势**转化为下游决策**：`strict_win@N16` adjacent ~0.96 / segment ~0.92 / full_reverse 1.00（5 seed，方差极小）。
- **但 B0（纯 CE）也高** → 顺序感知**主要来自表征/骨干**；SWIFT/HSRM 的顺序盲来自**置换不变聚合**。
- 同表征同骨干下 **O2 > B0**（adjacent `pair_acc` +0.20，CI 排除 0）→ order-pref 损失加可测正增量。
- 诚实边界：合成诊断任务，**非** GSM8K 准确率。
- **写作口径**：禁止「O2 top-1 更强」；允许「O2 顺序敏感可测，变现条件是任务需要顺序」。

**核心结论 6（五种子扩展，2026-10-16）：** 打分器轴相关实验统一种子 `42–46` 全部齐 5/5（BoN N16 冻/p2 ×4 scorer、p2 N64 ×2、Order-Decision、gate 消融、配对分析）。详见 `tables/scorer_axis_5seed/FIVESEED_STATUS.md`。生成器轴 SLPO/GRPO 仍为 3-seed（本轮不动）。

**产物：** `tables/scorer_axis_3seed.json`（机器可读）、`tables/SCORER_AXIS_REPORT.md`（完整汇报）、`tables/order_metric_comparison.tsv`、`tables/paired_o2_vs_lightweight.tsv`、`tables/gate_ablation_summary.tsv`、`tables/CONDITION_ALIGNMENT_AUDIT.md`、`tables/paired_ranking_*.tsv`、`tables/order_decision/*`。代码：`src/lightweight_scorers.py`、`scripts/train_lightweight_scorer.py`、`scripts/eval_lightweight_scorers.sh`、`scripts/eval_order_lightweight.py`、`scripts/paired_multi_scorer.py`、`scripts/paired_ranking_analysis.py`、`scripts/gate_ablation.sh`、`scripts/order_decision_test.py`、`scripts/order_decision_paired.py`、`scripts/run_order_decision_all.sh`、`scripts/train_p123_system.py --selector_family`。


---

## 当前缺口与启动顺序

**训练类（重）：**
- G1 SFT、G2 RFT（生成器轴，需新实现）

**启动顺序（便宜→贵）：**
1. ✅ 补 B0 评测矩阵、CODIp2+B0 同池（已做）
2. ✅ 3-seed 同池配对（已做）
3. ✅ 同起点同预算 G3 SLPO / G4 Dropout-GRPO 训练 + B0 3-seed 评测（已做，见主实验 1）
4. ✅ S2 ELHSR/SWIFT、S3 HSRM 适配 + 同池评测 + 顺序机制指标 + gate 消融（已做，见主实验 3）
5. ✅ 五种子扩展 seeds 42–46（BoN / Order-Decision / gate / 配对分析全部 5/5）
6. G1 SFT / G2 RFT（经典对照，改动最小）
7. S-Abl BCE 重训（恢复 checkpoint，数字已有）
8. 冻 CODI 四 scorer 同池、O2/p2 组件消融
9. 同起点 CODI-SLPO、G4 GRPO 更大预算复跑（若需对齐官方 40000/50000 步）
