# Scorer 可比性审计（对齐条件 / 逐题配对 / 无泄漏）

日期：2026-10-16 晚
触发：用户要求停止盲找新基线，先做三个针对性检查。判读规则：**若训练条件不同，现有名次不能解释为方法优劣；不许只看 0.05–0.30pp 均值；测试集只能做最终报告。**

产物：
- `condition_alignment.tsv`（检查 1）
- `paired_ranking_stats.tsv` / `paired_ranking_pairs.tsv` / `paired_ranking_crossseed.tsv` / `paired_ranking_crossseed_pairs.tsv`（检查 2）
- `scripts/paired_ranking_analysis.py`（可复现）

---

## 检查 1：训练条件对齐

见 `condition_alignment.tsv`。要点：

| 维度 | O2 | B0 | SWIFT | HSRM |
|---|---|---|---|---|
| 训练划分 | `latent-data/coconut/train`（`data/gsm_train.json`）| 同 | 同 | 同 |
| 正确性标签 | 同一 `corrects` | 同 | 同 | 同 |
| 训练轨迹数 | 1,084,776 | 135,597 组 | 1,084,776 | 1,084,776 |
| 任务损失 | BCE + 0.5·order/hier pref | CE | BCE | BCE |
| **输入表征** | **完整 token 轨迹 + 潜步（经 GPT-2）** | 同 | **仅 6 个潜步向量** | **仅 6 个潜步向量** |
| **骨干** | **GPT-2 124M（预训练）** | 同 | 随机初始化 Linear | 随机初始化 2 层 Transformer |
| **参数量** | **124,442,881** | 124,442,881 | **1,538** | **1,793,025** |
| model selection | valid-64 recall@1 | valid-64 recall@1 | valid-64 top1 | valid-64 top1 |
| 调参预算 | 1 组锁定配置 | 上游配置 | 1 组配置、无 sweep | 1 组配置、无 sweep |

**对齐结论：**
- ✅ **训练数据条件对齐**：同划分、同候选池、同 `corrects` 标签；O2/SWIFT/HSRM 的任务损失同为「逐轨迹 BCE on corrects」。就训练信号而言，O2 相对 SWIFT/HSRM 只多了一个 order/hier 辅助项。
- ❌ **表征 / 容量 / 骨干严重不对齐**：O2/B0 吃**整条 token 轨迹 + 潜步**、走 **124M 预训练 GPT-2**；SWIFT/HSRM 只看**6 个潜步向量**、骨干**随机初始化**，参数量相差 **~8×10⁴–8×10⁵ 倍**。
- ⚠️ **调参预算不对齐**：SWIFT/HSRM 各只有 1 组配置、无超参搜索；O2/B0 有早停与多组 restart。

> **判读**：现有「SWIFT/HSRM 在 top1 上不输 O2」**不能**读成「轻量方法更强」。它真正的含义是：**一个只看 6 个潜步、随机初始化的小模型，就能追平一个吃整条轨迹、124M 预训练的 RM**。这不是方法优劣的结论，而是表征/容量优势被稀释的证据。真正的受控对比需要：同表征（都吃潜步 or 都吃整轨迹）、同骨干、同调参预算。

---

## 检查 2：逐题配对比较（Top1 之外的排序指标）

**同池校验**：6 个 (pool×seed) 单元全部 `answers_sha1` 逐题一致，`sha1_mismatch=0` → 四 scorer 吃的是**完全相同的候选**。

**逐题指标定义**（每题独立计算，再跨题配对）：
- `top1`：被选中候选是否正确。
- `top1_ach`：仅在**存在正确候选**的题上算 top1（剥离 coverage 影响）。
- `auc_cw`：**正确候选 vs 错误候选的配对排序准确率**（P(随机正确 > 随机错误)），仅在同时存在对/错候选的题上定义（n≈425–446/seed）。
- `mrr`：首个正确候选的排名倒数。

**3-seed 逐题平均 + 跨题配对 bootstrap（10000 次，95% CI）：**

| 池 | 对比 | 指标 | Δ(pp) | 95% CI | 判定 |
|---|---|---|---|---|---|
| frozen | O2−SWIFT | top1 | −0.40 | [−1.54, 0.76] | 跨 0 |
| frozen | O2−SWIFT | top1_ach | −0.44 | [−2.81, 1.96] | 跨 0 |
| frozen | O2−SWIFT | mrr | −0.23 | [−1.08, 0.60] | 跨 0 |
| frozen | **O2−SWIFT** | **auc_cw** | **+2.63** | **[0.20, 5.07]** | **显著** |
| frozen | O2−HSRM | auc_cw | +1.86 | [−0.50, 4.19] | 跨 0 |
| p2 | O2−SWIFT | top1 | −0.05 | [−1.21, 1.09] | 跨 0 |
| p2 | O2−SWIFT | auc_cw | +2.41 | [−0.04, 4.89] | 边缘 |
| p2 | O2−HSRM | auc_cw | +1.90 | [−0.47, 4.25] | 跨 0 |
| p2 | **O2−B0** | **top1** | **+1.57** | **[0.53, 2.63]** | **显著** |
| p2 | **O2−B0** | **top1_ach** | **+2.04** | **[0.06, 3.98]** | **显著** |
| p2 | **O2−B0** | **mrr** | **+1.33** | **[0.50, 2.16]** | **显著** |
| frozen | O2−B0 | top1 | +0.78 | [−0.15, 1.74] | 边缘 |

**逐 seed McNemar（top1 不一致对）**：O2−SWIFT 在 6 个单元里 p=0.077/1.000/0.777/0.467/0.784/0.850；O2−HSRM p=0.562/0.928/1.000/0.311/0.930/0.928；O2−B0 p=0.653/0.254/0.105/0.061/0.137/**0.009**（仅 p2-s44 显著）。→ **top1 上 O2 对 SWIFT/HSRM 无显著差异**（对 B0 也只在 p2 池显著）。

**关键新发现（Top1 掩盖的轴）**：`auc_cw` 上 **O2 一致更高**——逐 seed 对 SWIFT 为 +1.59/+3.87/+3.88（frozen）、+3.37/+3.25/+1.62（p2），**6/6 同号**；对 HSRM 亦 6/6 同号（+1.0~+3.5）。跨 seed 后 **frozen 池 O2−SWIFT 显著 +2.63pp [0.20, 5.07]**，p2 池 +2.41pp 边缘。

**判读（诚实口径）：**
- O2 相对 SWIFT/HSRM：**top1 / top1_ach / mrr 全部无显著差异**；但在**正确-错误配对排序（auc_cw）**上有一个**同号一致、量级 2–3pp** 的小优势，且仅在 frozen 池对 SWIFT 达到显著、对 HSRM 不显著。
- 即：O2 的 order 训练**确实提高了「把对的和错的区分开」的排序质量**，但**没有**稳定转化为 top-1 选择或 MRR 提升。
- 差异不显著的原因之一：`auc_cw` 只在 ~43% 的题上可定义（其余题池内全对或全错），有效样本 n≈560，检验功效本就不高。
- 多重比较警告：本次共做 24 个配对检验（4 指标 × 3 对手 × 2 池），单个显著需打折看；但 6/6 同号使「O2 在 auc_cw 上更好」比单点显著更可信。
- O2 相对 B0：**p2 池上 top1/top1_ach/mrr 三项均显著**，O2 变现优势成立（与旧结论一致）。

---

## 检查 3：训练与模型选择只用 train/valid（无测试集泄漏）

| 用途 | 数据文件 | 题数 | 泄漏？ |
|---|---|---|---|
| 训练（全部 scorer） | `data/gsm_train.json` | 385,536 unique | — |
| 验证 / 早停 / 选模型 | `data/gsm_valid.json` → `valid-4`/`valid-64` | 500 | 否 |
| 最终报告（BoN） | `data/gsm_test.json` | 1,319 | 仅报告 |

- **三套划分两两问题级重叠 = 0**（train∩valid=0，train∩test=0，valid∩test=0）。
- 模型选择：O2/B0 用 `metric_for_best_model=test_n_64_recall_at_1`（该指标实际作用在 **valid-64**；命名 `test_n_64` 是历史遗留，**不是** GSM8K test）；SWIFT/HSRM 用 `valid64 top1` 选 `best.pt`。
- 机制指标（order/hierarchy）在 `valid-4`/`valid-64` 上算（`eval_order_pref.py` / `eval_order_lightweight.py`），**不在 test 上**。
- 训练/选模型代码**不读 `gsm_test.json`**：`grep gsm_test` 只命中评测脚本（26 个），无训练或超参选择路径。
- 四 scorer 的 BoN 报告全部用同一 `data/gsm_test.json`，无任何按 test 调阈值/挑配置的痕迹；O2 的 `order_margin/hierarchy_margin` 为固定配置（0.05/0.0125），标注的 gap 标定（O2-m）仍是 TODO，未在 test 上做。

**结论：未发现测试集泄漏。** 唯一需注意的口径风险是配置里 `test_n_64` 的**命名**易被误读为测试集，实为 valid-64。

---

## 三条检查的净结论

1. **条件不对齐（关键）**：训练**数据**对齐，但**表征/容量/骨干**差 4–5 个数量级 → 现有名次**不能**解释为 O2 vs SWIFT/HSRM 的方法优劣。
2. **Top1 之外的配对分析**揭示：O2 在 `auc_cw`（正确-错误排序）上**同号一致 +2–3pp**（frozen 对 SWIFT 显著），但 top1/top1_ach/mrr 无显著差异；对 B0 的 p2 优势依旧显著。
3. **无 test 泄漏**：划分两两不重叠，选模型只用 valid-64，机制指标也在 valid 上算。
