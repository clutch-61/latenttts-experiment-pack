# Scorer 轴汇报：O2 vs SWIFT-Latent / HSRM-Latent vs B0

日期：2026-10-16
协议：GSM8K-Test，BoN N=16（部分 N=4/64），frozen COCONUT 与 p2(续训) 两候选池，seed 42/43/44。
所有对比均在**同候选池**上（`answers_sha1` 核实候选完全一致）。

---

## 0. 一句话结论

- **相对 B0**：O2 有真实但温和的优势（p2 池 top1 +1.57pp，配对显著）。
- **相对 SWIFT/HSRM**：`top1 / top1_ach / mrr` 均**无显著差异**（配对 CI 全跨 0）；但**正确-错误配对排序 `auc_cw` 上 O2 同号一致 +2–3pp**（frozen 对 SWIFT 显著、对 HSRM 不显著）——这是 Top1 掩盖的、唯一稳定的正向信号。
- **可比性警告（关键）**：O2 是 124M、吃整条轨迹的预训练 GPT-2 RM；SWIFT/HSRM 是随机初始化、只看 6 个潜步的小模型（1.5K / 1.8M 参数）。**条件不对齐 → 名次不能当方法优劣读**。详见 `CONDITION_ALIGNMENT_AUDIT.md`。
- SWIFT/HSRM 作为**廉价推理期 scorer**（参数量少 ~10³–10⁵×、与 O2 打平）可用。

---

## 1. BoN Top-1（排序能力）

| 池 | scorer | N=16 (3-seed mean±std) | N=64 | N=4 |
|---|---|---|---|---|
| frozen | B0 | 32.88 ± 1.06 | 31.31 (s42) | 32.37 (s42) |
| frozen | **O2** | 33.66 ± 1.38 | 31.46 (s42) | 33.43 (s42) |
| frozen | **SWIFT** | **34.07 ± 0.49** | — | — |
| frozen | **HSRM** | 33.76 ± 1.01 | — | — |
| p2 | B0 | 34.32 ± 0.36 | 33.21 (s42) | 34.65 (s42) |
| p2 | O2 | 35.89 ± 0.38 | 34.72 (s42) | 35.25 (s42) |
| p2 | **SWIFT** | 35.94 ± 0.27 | **35.53 ± 0.53** | — |
| p2 | **HSRM** | **36.19 ± 0.35** | 35.41 ± 0.35 | — |

**读数**：SWIFT/HSRM ≥ O2 ≥ B0，差距在 ±0.5pp 噪声带内（O2 对 SWIFT/HSRM 无优势）。

## 2. Voting（多数投票，与打分器无关的池质量）

| 池 | voting（3-seed） |
|---|---|
| frozen | 33.71 |
| p2 | 36.42 |

同池内 B0/O2/SWIFT/HSRM 完全相同（池一致，voting 与 scorer 无关）。

## 3. Order / Hierarchy 机制指标（★ O2 唯一完胜轴）

valid-64 full，corrects=1 集（n=7021），同一套 permutation corruption：

| model | 训练目标 | order_pair_acc | hierarchy_consistency |
|---|---|---|---|
| **O2** | outcome + order/hierarchy pref | **0.9718** | **0.8936** |
| B0 | 官方 contrastive CE | 0.8600 | 0.7695 |
| B1 | BCE-only | 0.7660 | 0.7284 |
| SWIFT | outcome-only BCE | **0.4487** | **0.2416** |
| HSRM | outcome-only BCE | **0.4751** | **0.4414** |

**读数**：SWIFT/HSRM **低于随机**（对轨迹顺序零感知，甚至偏好乱序）；O2 接近满分。
O2 − SWIFT/HSRM = **+0.50~0.52 (pair) / +0.45~0.65 (hier)**。

## 4. 同池配对 gap（3-seed mean）

| pair | frozen | p2 |
|---|---|---|
| O2 − B0 | +0.78 | **+1.57** |
| O2 − SWIFT | −0.40 | −0.05 |
| O2 − HSRM | −0.10 | −0.30 |
| SWIFT − B0 | +1.19 | +1.62 |
| HSRM − B0 | +0.89 | +1.87 |

**读数**：O2 的"gap 翻倍"是对 **B0** 而言；SWIFT/HSRM **也**赢 B0（甚至更多）。
→ 真正被漂移放大的是 **B0 的劣势**，O2 与轻量 scorer 都不退化。

## 5. Gate 消融（功能优势检验，阴性）

400 步，k=32，λ_mmd=0，只变 gate selector，3-seed：

| arm | B0 尺 top1 | O2 尺 top1 | voting |
|---|---|---|---|
| O2-gate | 32.80 ± 0.27 | 34.34 ± 0.15 | 35.18 |
| SWIFT-gate | **33.48 ± 0.36** | 34.52 ± 0.78 | 35.15 |

同池配对 SWIFT-gate − O2-gate：B0 尺 **+0.68pp**（CI 全跨 0），O2 尺 +0.18pp。

**读数**：O2 作 gate **无下游优势**，B0 尺上 SWIFT-gate 反而 3 seed 全高（不显著）。
**原因**：该 trainer 的 correctness 由答案抽取判定，scorer 只负责「correct student 间排序 + teacher fallback 分差」，**不需要**顺序感知 → 机制优势用不上。

---

## 汇总表（四轴）

| 轴 | O2 vs SWIFT/HSRM | 判定 |
|---|---|---|
| BoN top1（frozen+p2，N=4/16/64，3-seed） | 平手/略低 | 无优势 |
| wvote / voting | 平手 | 无优势 |
| **order/hierarchy 机制** | **O2 完胜 +0.50~0.65** | **优势** |
| gate→下游（400步×3seed） | O2 不占优（SWIFT 略高 +0.68） | 无优势 |

**最终**：O2 相对 SWIFT/HSRM 的**唯一确认优势 = 机制指标本身**，且**无下游转化证据**。

---

## 6. 可比性审计修订（2026-10-16 晚）

用户要求三项针对性检查，完整报告见 `CONDITION_ALIGNMENT_AUDIT.md`。对上面结论的修订：

1. **条件不对齐（推翻「名次=方法优劣」）**：训练**数据**对齐（同划分/同池/同 `corrects`），但 O2 相对 SWIFT/HSRM 的**输入表征、骨干、参数量、调参预算**全不对齐（124M 预训练吃整轨迹 vs 随机初始化只看 6 潜步、1.5K/1.8M 参数）。→ §1 的「SWIFT/HSRM 略高」**不能**读成方法更强。

2. **Top1 之外的配对指标（新增 auc_cw）**：逐题配对（同池 sha1 校验 6/6）显示
   - `top1 / top1_ach / mrr`：O2 vs SWIFT/HSRM **全部 CI 跨 0**（无显著差异）。
   - `auc_cw`（正确候选 vs 错误候选配对排序）：**O2 逐 seed 6/6 同号为更高**，frozen 池 O2−SWIFT **+2.63pp [0.20, 5.07]（显著）**，p2 池 +2.41pp（边缘）；对 HSRM +1.9pp（不显著）。
   - 即：O2 的 order 训练**提升了对/错区分度**，但**未**稳定转化为 top-1 / MRR。

3. **无 test 泄漏**：train/valid/test 三套问题两两重叠 = 0；选模型只用 valid-64；机制指标也在 valid 上算。配置里 `test_n_64` 是 valid-64 的历史遗留命名。

**修订后的净结论**：O2 相对轻量 scorer 的优势 **不是**「top1 更高」，而是 (a) 机制指标（order/hierarchy，决定性）与 (b) 正确-错误配对排序 `auc_cw`（小、同号一致、部分显著）；且因条件严重不对齐，主表名次只能当「廉价 scorer 追平大模型」的证据，不能当方法优劣排序。

---

## 产物清单

| 文件 | 内容 |
|---|---|
| `scorer_summary.tsv` | 四 scorer top1（frozen，3-seed） |
| `order_metric_comparison.tsv` | 机制指标 5 模型 |
| `paired_o2_vs_lightweight.tsv` | 同池配对 gap 矩阵 |
| `../gate_ablation/gate_ablation_summary.tsv` | gate 消融 3-seed |
| `CONDITION_ALIGNMENT_AUDIT.md` | 三项检查完整报告（条件对齐 / 逐题配对 / 无泄漏） |
| `condition_alignment.tsv` | 各 scorer 训练条件逐项对照 |
| `paired_ranking_stats.tsv` | 逐 (池,seed,scorer) top1/top1_ach/auc_cw/mrr/cov |
| `paired_ranking_pairs.tsv` | 逐 seed 配对 Δ + bootstrap CI + McNemar |
| `paired_ranking_crossseed.tsv` | 3-seed 逐题平均汇总 |
| `paired_ranking_crossseed_pairs.tsv` | 跨 seed 配对 bootstrap CI |
| `../../scripts/paired_ranking_analysis.py` | 上述配对分析脚本 |
| `../../PLAN.md` | 全部决策与坑记录 |
