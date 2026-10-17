# 五种子扩展状态（2026-10-16）

统一种子：`42 / 43 / 44 / 45 / 46`。**全部目标实验已齐 5/5。**

## 覆盖盘点

| 实验 | 现状 | 判定 |
|---|---|---|
| 冻结池 BoN N16（B0/O2/SWIFT/HSRM） | **5/5** | ✅ |
| p2 池 BoN N16（B0/O2/SWIFT/HSRM） | **5/5** | ✅ |
| p2 池 BoN N64（SWIFT/HSRM） | **5/5** | ✅ |
| Order-Decision（4 模型 × N∈{4,16}） | **5/5** | ✅ |
| 配对排序分析（top1/auc_cw/mrr） | **5/5 重算** | ✅ |
| Gate 消融（O2-gate vs SWIFT-gate） | **5/5** | ✅ |
| 冻结池 O2/B0 N4/N64 | 1 seed | ⏭ 不补（N16 为主表） |
| 机制指标 order/hierarchy | 单次 | ⏭ 无 seed 维度 |
| 生成器轴 SLPO/GRPO | 3 seed | ⏭ 本轮不动 |

## 5-seed 结果摘要

### BoN N16 top1（mean ± std）

| 池 | B0 | O2 | SWIFT | HSRM |
|---|---|---|---|---|
| frozen | 32.72 ± 0.96 | **33.72 ± 1.03** | 33.69 ± 0.75 | 33.60 ± 0.75 |
| p2 | 34.44 ± 0.38 | 35.92 ± 0.28 | 35.88 ± 0.48 | **36.06 ± 0.37** |

p2 N64：SWIFT 35.57 ± 0.50；HSRM 35.50 ± 0.28。

### 逐题配对（5-seed 逐题平均 + bootstrap）

| 对比 | top1 Δ [CI] | auc_cw Δ [CI] |
|---|---|---|
| frozen O2−SWIFT | +0.03 [−0.99, 1.02] | **+2.34 [0.12, 4.57] ✅** |
| frozen O2−HSRM | +0.12 [−0.82, 1.05] | +1.45 [−0.67, 3.63] |
| **frozen O2−B0** | **+1.00 [0.17, 1.85] ✅** | +2.21 [−0.16, 4.58] |
| p2 O2−SWIFT | +0.05 [−0.97, 1.06] | **+3.19 [0.89, 5.49] ✅** |
| p2 O2−HSRM | −0.14 [−1.15, 0.89] | **+2.70 [0.48, 4.92] ✅** |
| **p2 O2−B0** | **+1.49 [0.59, 2.38] ✅** | **+2.62 [0.27, 5.02] ✅** |

相对 3-seed：O2−B0 在**冻结池**也显著了；O2−HSRM / O2−SWIFT 的 `auc_cw` 在 p2 更稳。O2 vs SWIFT/HSRM 的 **top1 仍跨 0**。

### Gate 消融（5-seed）

| arm | B0 尺 | O2 尺 | voting |
|---|---|---|---|
| O2-gate | 33.19 ± 0.61 | 34.34 ± 0.15 | 35.01 |
| SWIFT-gate | 33.56 ± 0.35 | 34.39 ± 0.69 | 35.03 |

配对 SWIFT−O2：B0 尺 **+0.36 ± 0.66**（方向混杂：+0.83/+0.38/+0.83/+0.53/**−0.76**）；O2 尺 +0.05 ± 0.55。**仍无显著优势**——5 seed 未改阴性结论。

### Order-Decision（5-seed，N=16 adjacent）

O2 pair_acc ~0.99 / strict_win ~0.96；B0 ~0.79 / ~0.30；SWIFT ~0.49 / ~0；HSRM ~0.61 / ~0.12。方向与单 seed 一致，方差很小。

## 产物

- `results/full/scorer_axis_5seed/`（统一目录 + table/summary/paired_ranking_*.tsv）
- `results/full/order_decision/order_decision_{summary,paired}_5seed.tsv`
- `results/full/gate_ablation/gate_ablation_summary.tsv`（现含 5 seed）
- 脚本：`extend_scorer_axis_seed.sh`、`assemble_scorer_axis_5seed.sh`、`scorer_axis_5seed_summary.py`、`run_order_decision_seeds.sh`、`chain_gate_ablation_seeds.sh`、`summarize_gate_ablation.py`
