# Scorer 轴汇报：O2 vs SWIFT-Latent / HSRM-Latent vs B0

日期：2026-10-16（2026-10-17 五种子修订）
协议：GSM8K-Test，BoN N=16（部分 N=64），frozen COCONUT 与 p2(续训) 两候选池，**seeds 42/43/44/45/46（5-seed）**。
所有对比均在**同候选池**上（`answers_sha1` 核实候选完全一致）。

---

## 0. 一句话结论

- **相对 B0**：O2 有真实优势——5-seed 配对 top1 冻池 **+1.00pp [0.17, 1.85] 显著**，p2 池 **+1.49pp [0.59, 2.38] 显著**。
- **相对 SWIFT/HSRM**：`top1 / top1_ach / mrr` 均**无显著差异**（配对 CI 全跨 0）；但 **`auc_cw` 上 O2 显著**（冻 O2−SWIFT +2.34 [0.12, 4.57]；p2 O2−SWIFT +3.19、O2−HSRM +2.70）。
- **可比性警告（关键）**：O2 是 124M、吃整条轨迹的预训练 GPT-2 RM；SWIFT/HSRM 是随机初始化、只看 6 个潜步的小模型（1.5K / 1.8M 参数）。**条件不对齐 → 名次不能当方法优劣读**。详见 `CONDITION_ALIGNMENT_AUDIT.md`。
- Gate 消融（5-seed）仍阴性：SWIFT−O2 B0 尺 **+0.36 ± 0.66**。
- Order-Decision 受控实验：当顺序是唯一判据时，机制优势**可转化为决策力**（见 `../order_decision/`）。

---

## 1. BoN Top-1（排序能力，5-seed）

| 池 | scorer | N=16 (5-seed mean±std) | N=64 (5-seed) |
|---|---|---|---|
| frozen | B0 | 32.72 ± 0.96 | — |
| frozen | O2 | **33.72 ± 1.03** | — |
| frozen | SWIFT | 33.69 ± 0.75 | — |
| frozen | HSRM | 33.60 ± 0.75 | — |
| p2 | B0 | 34.44 ± 0.38 | — |
| p2 | O2 | 35.92 ± 0.28 | — |
| p2 | SWIFT | 35.88 ± 0.48 | 35.57 ± 0.50 |
| p2 | **HSRM** | **36.06 ± 0.37** | 35.50 ± 0.28 |

**读数**：SWIFT/HSRM ≈ O2 ≥ B0，差距在 ±0.5pp 噪声带内（O2 对 SWIFT/HSRM 无 top1 优势）。

## 2. Voting（多数投票，与打分器无关的池质量）

| 池 | voting（5-seed cov 均值） |
|---|---|
| frozen | ~50.61（cov） |
| p2 | ~53.75（cov） |

同池内 B0/O2/SWIFT/HSRM 完全相同（池一致，voting 与 scorer 无关）。

## 3. Order / Hierarchy 机制指标（★ O2 机制轴完胜）

valid-64 full，corrects=1 集（n=7021），同一套 permutation corruption：

| model | 训练目标 | order_pair_acc | hierarchy_consistency |
|---|---|---|---|
| **O2** | outcome + order/hierarchy pref | **0.9718** | **0.8936** |
| B0 | 官方 contrastive CE | 0.8600 | 0.7695 |
| B1 | BCE-only | 0.7660 | 0.7284 |
| SWIFT | outcome-only BCE | **0.4487** | **0.2416** |
| HSRM | outcome-only BCE | **0.4751** | **0.4414** |

**读数**：SWIFT/HSRM **低于随机**；O2 接近满分。

## 4. 同池配对（5-seed 逐题平均 + bootstrap）

| pair | top1 Δ [95% CI] | auc_cw Δ [95% CI] |
|---|---|---|
| frozen O2−B0 | **+1.00 [0.17, 1.85] ✅** | +2.21 [−0.16, 4.58] |
| frozen O2−SWIFT | +0.03 [−0.99, 1.02] | **+2.34 [0.12, 4.57] ✅** |
| frozen O2−HSRM | +0.12 [−0.82, 1.05] | +1.45 [−0.67, 3.63] |
| p2 O2−B0 | **+1.49 [0.59, 2.38] ✅** | **+2.62 [0.27, 5.02] ✅** |
| p2 O2−SWIFT | +0.05 [−0.97, 1.06] | **+3.19 [0.89, 5.49] ✅** |
| p2 O2−HSRM | −0.14 [−1.15, 0.89] | **+2.70 [0.48, 4.92] ✅** |

**读数**：O2 相对 B0 的 top1 优势在 5-seed 下**冻/p2 均显著**；相对 SWIFT/HSRM 的 top1 仍跨 0，但 `auc_cw` 多数显著。

## 5. Gate 消融（功能优势检验，5-seed，阴性）

400 步，k=32，λ_mmd=0，只变 gate selector，seeds 42–46：

| arm | B0 尺 top1 | O2 尺 top1 | voting |
|---|---|---|---|
| O2-gate | 33.19 ± 0.61 | 34.34 ± 0.15 | 35.01 |
| SWIFT-gate | **33.56 ± 0.35** | 34.39 ± 0.69 | 35.03 |

同池配对 SWIFT−O2：B0 尺 **+0.36 ± 0.66**（逐 seed：+0.83/+0.38/+0.83/+0.53/**−0.76**），O2 尺 +0.05 ± 0.55。

**读数**：O2 作 gate **无下游优势**；5-seed 后均值更接近 0，方向混杂，阴性更稳。
**原因**：该 trainer 的 correctness 由答案抽取判定，scorer 只负责「correct student 间排序 + teacher fallback」，**不需要**顺序感知。

---

## 汇总表（四轴，5-seed）

| 轴 | O2 vs SWIFT/HSRM | 判定 |
|---|---|---|
| BoN top1（frozen+p2，N=16/64，5-seed） | 平手（CI 跨 0） | 无优势 |
| wvote / voting | 平手 | 无优势 |
| **auc_cw（正确-错误排序）** | **O2 显著 +2.3~+3.2** | **优势** |
| **order/hierarchy 机制** | **O2 完胜 +0.50~0.65** | **优势** |
| gate→下游（400步×5seed） | O2 不占优（+0.36±0.66） | 无优势 |
| Order-Decision（顺序作唯一判据） | O2 strict_win ~0.96 vs SWIFT ~0 | **优势（条件明确）** |

**最终**：O2 相对 SWIFT/HSRM 的确认优势 = **机制指标 + auc_cw + Order-Decision 条件任务**；常规 BoN top1 / gate **无优势**。因条件不对齐，主表名次不能当方法优劣。

---

## 6. 可比性审计 + 五种子扩展

详见 `CONDITION_ALIGNMENT_AUDIT.md`、`../scorer_axis_5seed/FIVESEED_STATUS.md`、`../order_decision/ORDER_DECISION_REPORT.md`。

相对 3-seed 的关键变化：
1. O2−B0 top1：冻池也显著了；
2. O2−SWIFT/HSRM `auc_cw`：p2 两者都显著；
3. Gate 均值从 +0.68 → +0.36，仍不显著；
4. O2 vs SWIFT/HSRM **top1 仍跨 0**——主结论不变。

---

## 产物清单

| 文件 | 内容 |
|---|---|
| `../scorer_axis_5seed/scorer_axis_5seed_table.tsv` | 5-seed BoN 汇总 |
| `../scorer_axis_5seed/paired_ranking_*.tsv` | 5-seed 逐题配对 + bootstrap |
| `../scorer_axis_5seed/FIVESEED_STATUS.md` | 五种子覆盖盘点 |
| `../gate_ablation/gate_ablation_summary.tsv` | gate 消融 5-seed |
| `../order_decision/` | Order-Decision 报告 + 5-seed 表 |
| `CONDITION_ALIGNMENT_AUDIT.md` | 三项检查完整报告 |
| `order_metric_comparison.tsv` | 机制指标 5 模型 |
| `../../PLAN.md` | 全部决策与坑记录 |
