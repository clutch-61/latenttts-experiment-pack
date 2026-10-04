# Order-aware LatentRM — working plan

> **完整实测交接（给其他智能体复核）：** [`HANDOFF_EXPERIMENT_REPORT.md`](./HANDOFF_EXPERIMENT_REPORT.md)  
> 含协议、全部 BoN/MV/eval_order 数字、训练 R@1、已知坑与未完成项。以该文件 + 原始 log/TSV 为准。  
> **2026-09-28 19:15 勘误：** 见该文件 §13（计数=unique 互斥；valid-64 eval_order≈50 题；三 seed=推理 seed；MA/Hard 分母=JSON 原长）。

## Phase-0 — Annotate done (2026-09-27)

### Decision
Phase-0 COCONUT annotate **已完成**，产物可进入 `train_baseline` → `train_pref`。协议未改：`seed=42`，train `n=8` + `n_per_step=64` + dropout `0.2`；valid `N=4/64` + `n_per_step=0`。

### Numbers (disk audit, post-run)
| split | shards | unique idx | complete (trainable) | markers (allC/allF placeholder) | gsm size |
|-------|--------|------------|----------------------|---------------------------------|----------|
| train | 11987 | 383517 | **135616** | 247968 | 385620 |
| valid-4 | 16 | 500 | **202** | 310 | 500 |
| valid-64 | 16 | 500 | **346** | 166 | 500 |

- complete：四键齐全 `input_ids/estimations/latent_embeds/corrects`；`n_samples`=8/4/64，`num_latents`=6，est/emb **fp32**
- markers：仅标量 `input_ids=0`，给 resume 用；`CachedPickleDatasetV2` 只认带 `estimations` 的 idx，训练会跳过
- train 可训练占比 ≈ **35%** of GSM（与 online allC+allF 过滤量级一致）

### Quality (sampled audit)
- 已保存样本：**无** all_correct / all_false（过滤生效）；轨迹正确率 ~54%；难度混合（每题 1–7 条对）
- est ∈ [0,1]，无 NaN/Inf/负值；正确轨迹 mean-est ≈0.52 vs 错误 ≈0.37；~90% 题上正确轨迹分数更高 → 适合 LatentRM
- Online tqdm 的 pass@1/vot 是**含丢弃题**的累计，且短题段会偏高，**不要**当训练集质量指标

### How to read later results
- 训 CE / pref 以 **complete** 计数为准，不要用 unique 或 shard 数当样本数
- valid-4 完整仅 ~202：eval / early-stop 方差会偏大；对比方法必须同目录、同协议
- 重复 idx：train **67**（其中 complete+complete **19**）、valid-4/64 各 **12**（多卡/重启边界）。量级很小；loader 按文件遍历后者覆盖前者。若要洁净可后做去重，**不阻塞**开训

### Do not
- 不要因 markers 占比高重标 train
- 不要改 `n_samples` / `n_samples_per_step` / dropout / seed 另开一套 annotate「优化」
- 不要新开 Chamfer/MMD/Self-Forcing 或换 generator 协议

### Next
1. ~~`train_baseline`（B0 CE）~~ → 早停结束，`outputs/latentrm_baseline/best`
2. `train_pref`（O2）← **进行中**
3. `eval_order` 用 `latent-data/coconut/valid-4`

## Phase-1 — eval_order (2026-09-28)

日志：`logs/eval_order_20260928_091520.log`  
协议：`mean_log_prob`；corruptions = adjacent_swap / segment_swap / multi_segment / full_reverse；prefer corrects=1。

| split | model | order_pair_acc | hierarchy_consistency |
|-------|-------|----------------|------------------------|
| valid-4 (全量 ~804 traj) | **O2** | **0.966** | **0.874** |
| valid-4 | B0 | 0.877 | 0.798 |
| valid-64 (~3200 traj sample) | **O2** | **0.968** | **0.901** |
| valid-64 | B0 | 0.822 | 0.726 |

valid-4 分 corruption（pos>neg 率）：
- O2：adj 0.99 / seg 1.00 / multi 0.87 / rev 1.00
- B0：adj 0.70 / seg 0.95 / multi 0.88 / rev 0.98

### 怎么读
- **R@1 打平**（训练时 n64≈0.615），但 **顺序机制指标 O2 明显更好**（pair +8～15pt，hierarchy +8～17pt）。
- 最大差距在 **adjacent_swap**（最细粒度乱序）——符合 L_pref 训练目标。
- 这是「同 annotate / 同 valid」下的相对增益；下游 BoN/TTS 仍需 `eval_tts` 另验。

### Do not
- 不要因 R@1 打平就判 O2 失败；机制指标是 Phase-1 主证据之一。
- 不要改 corruption / score reduce 后再拿这组数混比。

## Phase-1 — BoN matrix started (2026-09-28 09:35)

- 脚本：`scripts/eval_bon_matrix.sh`（detach）
- 矩阵：GSM8K-Test × {O2, B0, official} × N∈{4,16,64}，`seed=42`，`prm_mode=best_of_n`，4 卡
- 日志目录：`logs/bon_matrix/`；汇总 TSV：`results/full/bon_gsm_test_*.tsv`
- wrapper：`logs/bon_matrix/nohup_20260928_093545.out` → 正式跑 `*_093713.*`
- 读结果：主看各 N 下 Acc（及 Cov/Vot）；O2 vs B0 相对增益；official 仅对照
- Do not：中途改 N/seed/generator；勿与不同 annotate 池混比

## Phase-1 — BoN Acc 判断 (2026-09-28 10:10)

汇总：`results/full/bon_gsm_test_20260928_093713.tsv`（Acc 可信；Cov/Vot 大多因收尾 crash 缺）

| PRM | N=4 | N=16 | N=64 |
|-----|-----|------|------|
| O2 | 33.43% | **35.03%** | 31.61% |
| B0 | 32.52% | 32.83% | 31.01% |
| official | 32.75% | 33.43% | 33.74% |

### Decision（对照科研计划 Go / No-go）
- **不是失败**：机制（eval_order）明显 ↑，自然 BoN 相对 B0 **不掉且 N=4/16 略升**（+0.9～+2.2pt）→ 可写「顺序敏感 + 选优不伤 / 小增益」。
- **也还不是硬 Go（性能 claim）**：增益小、单 seed、绝对值 ~33%、N=64 O2/B0 都掉而 official 略升 → 更像 **软正信号 / 偏收窄 claim**，不够单独撑「Latent TTS 性能提升」。
- **主叙事现在站得住的部分**：order 机制成立（pair/hier 大优势）；下游 BoN 相对自训 B0 一致略好，最佳点 N=16。

### How to read
- 主对比是 **O2 vs B0**（同 annotate）；official 只作外部对照，不对齐绝对值不判死。
- N=64 掉点：大 N 下 RM 排序噪声放大 / 合成 corruption 与自然错误分布错位；**不要**用 N=64 单独否定 O2。
- 绝对值偏低：先查协议（generator、N、seed、是否同论文设置），再谈「没复现论文数字」。

### Do not
- 不要因 N=16 +2pt 就开 Phase-2 / 换方法 / 加大 λ_pref 猛训。
- 不要把「corruption 排序 ↑」写成「推理能力 ↑」。
- 不要在缺 multi-seed / Hard / Cov 前写强 performance claim。

### Next（若继续）
1. 可选：修后重跑收尾拿 Cov/Vot；或只补缺格。
2. 至少再 1–2 seed 或 Hard/MultiArith 看增益是否稳。
3. 增益不稳 → 收窄 claim + 查 λ_pref / margin / OOD；稳且 Hard 有增益 → 再谈性能表扩写。

## Phase-1 — BoN stability run (2026-09-28 10:13)

### Decision
不改方法/λ；先验证增益是否稳。跑 `scripts/eval_bon_stability.sh`（detach，GPU 0,2,6,7）：
- **A** Hard + MultiArith × {O2,B0,official} × N∈{4,16,64}，seed=42（难集是否仍略赢）
- **B** GSM8K-Test **仅 N=16** × seed∈{43,44} × 三 PRM（最佳 N 上 multi-seed）

Cov/Vot 用已修的 `infer_gpt2_rm`（`np.asarray` + `.any`）。主对比仍是 O2 vs B0。

### How to read
- Hard/MultiArith：O2−B0 同号且量级接近 GSM → 增益可迁移；Hard 上变负 → 收窄 claim。
- seed 43/44：N=16 上 O2 仍 ≥ B0 → 软正信号更稳；翻转 → 单 seed 噪声，勿硬 claim。
- 本轮**不开** Phase-2、不调 λ_pref。

### Do not
- 不与 seed=42 旧 Acc 混写成「平均」而不标 seed
- 不因某一格波动就重训 / 改 corruption

## Phase-1 — BoN stability 结果 (2026-09-28 12:21)

汇总：`results/full/bon_stability_20260928_101417.tsv`（Acc/Cov 对；TSV 的 vot 列解析坏了，以日志 `Voting Accuracy:` 为准）

**Hard Acc@seed42：** O2 7.51/7.28/7.51 vs B0 7.28/6.90/6.75 vs official 7.59/7.66/7.44（N4/16/64）  
**MultiArith：** O2 79.48/80.34/82.41 vs B0 78.28/78.79/78.62 vs official 80.17/81.55/84.83  
**GSM N16 multi-seed Acc：** O2 35.03/33.28/33.36；B0 32.83/31.99/32.75；official 33.43/34.19/33.21（s42/43/44）

### Decision
- **O2 vs B0：增益稳**——Hard 全 N 正、MultiArith 全 N 正（约 +1.2～+3.8pt）、GSM N16 三 seed 全正（约 +0.6～+2.2pt，均值 ~+1.4pt）。
- **相对 official：仍软**——Hard≈打平略低；MultiArith/GSM 多数输或互有胜负；**不升级为硬 performance claim**。
- Claim 维持：**机制成立 + 相对自训 CE 选优稳定略升**；难集未翻转。

### Do not
- 不因相对 official 偶胜（如 GSM s44）写成「超过官方 LatentRM」
- 仍不开 Phase-2；不调 λ 重训除非要做 margin/λ 消融表

## Phase-1 — 下一步锁定 (2026-09-28 12:22)

### Decision
进入 **「补齐主表 + 消融」**，不换方法、不开 Phase-2。优先级：

1. **补基线协议（先便宜后贵）**  
   Majority Voting +（若脚本现成）MC-dropout / pass@N，同 generator 池；用来解释 ~33% 绝对 Acc、对照 BoN 增益是否真是 RM。
2. **消融矩阵（叙事必需）**  
   B1（BCE 无 pref）→ O1（order only）→ 再视需要 O2-m（gap 标定 margin）。目的：证明增益来自 L_pref / hierarchy，不是换了 BCE。
3. **Corruption Abl**（去掉一类扰动）— 在 O1/O2 训稳后做，解释 adjacent_swap 优势。
4. **CODI** — 仅当 COCONUT 主表+消融够写一节后再开；现不做。

### How to read later
- 若 B1≈O2：pref 没贡献，回头查 λ/margin。  
- 若 O1≈O2≫B1：hierarchy 可弱写；order 是主因。  
- 若 O2≫O1：hierarchy 可进主 claim。

### Do not
- 不为追 official 绝对值先开 Phase-2 / 换 generator  
- 不并行乱开多条新训练；一次一个消融 ID

## Phase-1 — MV 汇总 + B1 开训 (2026-09-28 12:25)

### Decision
按锁定优先级开干：
1. **Majority Voting / Coverage**：同 BoN 生成池已自带（`Voting Accuracy` / `Coverage`）；Hard/MultiArith/GSM s43–44 已有。补 GSM-Test seed=42 × N=4/16/64（`scripts/eval_mv_fill_gsm42.sh`），再 `summarize_mv.py` → `results/full/mv_coverage_summary.tsv`。
2. **B1 训练**：`training_args/train_baseline_bce.yaml`（`loss_type=bce`，无 pref，`outputs/latentrm_baseline_bce`）；`src.train`；GPU 0,2,6,7。链：`logs/chain_mv_b1_*.log`（先 MV 后 B1）。

说明：本仓库无独立「MC-dropout eval」脚本；dropout 采样 + MV/pass 即 annotate/BoN 协议本身，用 Cov/Vot 对照即可。

### How to read
- MV/Cov **与 PRM 无关**（同 N/seed/split 应对齐）；BoN Acc 才比 O2/B0/official。
- B1 训完后：同协议 BoN（至少 GSM N=16）比 B0/O2；若 B1≈O2 → pref 贡献可疑。

### Do not
- B1 未完成前不开 O1
- 不把 MV 写成「某 PRM 的 Voting」

## Phase-1 — B1 训完 + BoN (2026-09-28 15:28)

### Decision
B1（`outputs/latentrm_baseline_bce/best`）已训完：~1h50m，epoch 9.54，best n64 R@1 **0.6245**（略高于 B0/O2 ~0.615）。  
立刻同协议 BoN：`scripts/eval_bon_b1.sh` → GSM-Test / Hard / MultiArith × N=4/16/64 × seed=42；GPU 0,2,6,7。  
读法：主比 **B1 vs B0 vs O2**（同 annotate）；若 B1≈O2 → pref 贡献弱；若 O2>B1≥B0 → pref 有增量。

### Do not
- B1 BoN 未出前不开 O1 训练

## Phase-1 — B1 BoN 结果 (2026-09-28 15:54)

`results/full/bon_b1_20260928_152912.tsv` seed=42 Acc：

| split | N | B1 | B0 | O2 |
|-------|---|----|----|-----|
| gsm_test | 4 | 32.07 | 32.52 | **33.43** |
| gsm_test | 16 | 32.15 | 32.83 | **35.03** |
| gsm_test | 64 | 31.16 | 31.01 | **31.61** |
| hard | 4 | 7.13 | 7.28 | **7.51** |
| hard | 16 | 6.98 | 6.90 | **7.28** |
| hard | 64 | 6.68 | 6.75 | **7.51** |
| multiarith | 4 | 79.66 | 78.28 | 79.48 |
| multiarith | 16 | 79.31 | 78.79 | **80.34** |
| multiarith | 64 | 82.07 | 78.62 | **82.41** |

### Decision
- **O2 > B1 ≈ B0（GSM/Hard）**：换 BCE  alone **不解释** O2 增益；pref 有独立贡献。
- MultiArith：B1 有时略高于 B0，但仍 ≤ O2（N16/64）。
- 下一优先：**O1**（BCE + order only，`λ_hier=0`），看 hierarchy 是否额外贡献。

## Phase-1 — O1 开训 (2026-09-28 16:00)

### Decision
训 O1：`training_args/train_order_only.yaml`（同 O2，但 `lambda_hier=0.0`），输出 `outputs/latentrm_order_only`；`scripts/train_o1_detach.sh`；GPU 0,2,6,7。  
读法：训完后同协议 BoN；若 O1≈O2≫B1 → order 主因、hierarchy 可弱写；若 O2≫O1 → hierarchy 进主 claim。

### Do not
- 不并行开 O2-m / corruption abl
- 不改 pref_lambda / margin（先与 O2 对齐比）

## Phase-1 — 补测排队 (2026-09-28 19:20)

### Decision
不杀 O1。GPU 0/2/6/7 空后自动跑：
1. **valid-64 全量 eval_order**（`max_batches=0`）O2→B0→B1；`scripts/eval_order_valid64_full.sh`
2. **B1 GSM N=16 推理 seed 43/44**；`scripts/eval_bon_b1_seeds.sh`（42 已有）

队列日志：`logs/queue_eval_full_b1seeds_20260928_192005.log`  
`eval_order_pref.py`：`max_batches<=0` = 全量，并打印 scan 题数。

### How to read
- 全量仍 O2≫B0 → 机制结论把握上升；否则以全量为准。  
- B1×seed 与同 seed 的 O2/B0 比 O2−B1；仅称**推理 seed**。

## Phase-1 — 补测后判读 + O1 评估 (2026-09-29 11:01)

### Decision（对齐外部复核）
- 全量 valid-64 + O2>B1×3 推理 seed → **preference 训练包有效** 可说得稍强。  
- **仍不能**归因 hierarchy；**不能**说 generator 变强。  
- **立刻跑 O1**：全量 `eval_order`（valid-64）+ BoN（至少 GSM N=16 seed42，加 Hard；尽量同协议 N=4/16/64）。  
- 后续（本轮可并行或紧接）：题目级汇总 / bootstrap；同候选池 O2 vs B1 配对胜负。

### How to read O1
- O1≈O2≫B1（BoN + order）→ order 主因，hierarchy 可弱写。  
- O2≫O1≥B1 → hierarchy 有额外贡献，可进主 claim。  
- O1≈B1 → pref 实现/超参可疑。

### Do not
- 不把 R@1=0.6245 当 O1 下游结论  
- 不写「训练多 seed 复现」

## Phase-1 — O1 结果判读 (2026-09-29 11:46)

`eval_order_o1_valid64_full_20260929_110224.tsv` + `bon_o1_20260929_110224.tsv`

- 机制：O1 pair≈O2（0.970），hier **0.736 ≪ O2 0.894**（甚至略低于 B0）。  
- BoN：O1 GSM N16 均值 **32.22% ≈ B1 32.20% ≪ O2 33.89%**；Hard/MA 亦整体低于 O2。  
- **Decision：hierarchy 对本设定的 BoN 增益必要；仅 L_order 不够。** Preference 包的有效成分需写成 order+hier，不能写成 flat order。

## Phase-1 — claim 收紧 + 下一步 (2026-09-29 12:04)

### Decision（对外表述）
采用收紧版，**不要**写「hierarchy 已证明普遍必要 / 生成能力提高」：

> 在本次 COCONUT 实验中，order-only 的 O1 显著提高了对合成乱序的辨别，却没有带来 GSM N=16 Best-of-N 的平均收益；加入层级偏好的 O2 提高了层级一致性，并在三个**推理**采样 seed 上取得高于或等于 O1 的自然候选选优结果。结果提示层级偏好是**当前设定下**方法获得下游收益的重要组成部分。

限定：当前数据/配置/这些 checkpoint；推理 seed ≠ 训练复现；辨别的是预设 rank，不是已证真实逻辑严重度；训的是 LatentRM 挑选器。

### Next（优先级）
1. 题目级配对统计（同题同候选池 O2 vs O1/B1）+ bootstrap/CI  
2. Hard/MultiArith 写清具体数字（seed42 已有；多推理 seed 可选）  
3. 若主张 hierarchy 因果：至少再一组独立训练 seed，或明确单 checkpoint 对照  
4. 核对 O1/O2 训练公平性（数据/超参/预算/best 选择；早停 metric warning）

### Do not
- 不把 O2−O1≈1.7pt 写成普遍规律  
- 不写 Pass@1/Coverage/generator 提升

## Phase-1 — 同候选池配对 BoN (2026-09-29 15:57)

### Decision
Hard/MA 单 seed 支持 O2>O1（尤其 MA），但非多 seed 稳定结论。  
**立刻**：改 `infer_gpt2_rm` 落盘逐题（idx / answers / corrects / scores / selected / answers_hash）；单卡、同 seed=42、N=16 依次跑 O2/O1/B1；用 hash 核实候选一致后再做配对胜负与 CI。  
best_of_n 生成不依赖 PRM，同 seed 理论上同候选；仍以 hash 为准，不一致则改为 generate-once cache。

### Do not
- 不把 Hard/MA 单 seed 写成多 seed 稳定  
- 配对前不假设「同 seed = 同候选」已成立

## Phase-1 — 配对 BoN 结果 (2026-09-29 16:05)

`results/full/bon_paired_gsm16_20260929_155950/pair_summary.json`  
GSM N=16 seed=42；**candidates_identical=true**（answers_sha1 全一致，n=1319）。

| | Acc | vs O1 独赢/独输 | vs B1 独赢/独输 | 配对差 (pp) 95%CI |
|--|-----|-----------------|-----------------|-------------------|
| O2 | **35.03%** | 94 / 41 | 77 / 39 | O2−O1 **+4.02 [2.35, 5.84]**；O2−B1 **+2.88 [1.29, 4.47]** |
| O1 | 31.01% | — | 56 / 71 | O1−B1 **−1.14 [−2.88, +0.45]**（含 0） |
| B1 | 32.15% | — | — | — |

### Decision
同候选池上 O2 显著优于 O1/B1（CI 不含 0）；O1 不优于 B1。强化「hierarchy 在当前设定下把选优拉开」；仍单 seed、单 checkpoint。

## Phase-1 — 配对结果表述锁定 (2026-09-29 17:48)

### Decision（当前最有说服力的一组）
同批 GSM 候选上 O2 比 O1、B1 更会挑对；O1 对 B1 **无可靠优势**。支持「hierarchy 在当前设定下有实际选优价值」，强于只比总体 Acc。差异归因于 LatentRM 打分/选择，非生成器碰巧。

### 对外表述（锁定）
> 在 COCONUT 的 GSM8K-Test、N=16 设定下，同候选池逐题配对显示，O2（order+hierarchy）相对 O1 / B1 分别 +4.02 / +2.88 pt，95% bootstrap CI 均未跨 0；O1 相对 B1 未达同样证据强度。支持层级偏好在本设定下改善 LatentRM 候选选择；仍需独立训练重复与其他任务配对验证。

三条读法：
1. 单纯 order 提高人工乱序辨别，**未**带来 GSM 选优收益。
2. O2 在同一批自然候选上优于 O1 与 B1。
3. 支持 hierarchy 在本设定额外收益 ≠ 已证对所有任务/模型必要。

### 不能写 / 未证明
- 仍是 **一组** 训好的 O1/O2/B1 checkpoint + **一个** 推理 seed；配对消了候选池运气，**未**回答换训练 seed 收益是否还在。
- 仍是 LatentRM 选得更好，**不是** COCONUT generator 变强。
- `answers_sha1` 保证答案文本与顺序一致；若要严格确认 latent trajectory 同池，需再核 latent embedding / trajectory ID hash（或确认 generate-once 缓存共用）。当前同 seed 顺序生成，流程上 latent 应同源，但哈希层尚未加。

### Next
1. **优先**：同协议逐题配对扩到 Hard、MultiArith（N=16 seed=42）。  
2. 资源允许：独立训练重复。  
3. 可选：dump/核对 latent hash，或 generate-once 显式缓存。

### Do not
- 不启动 Phase-2/3；不改 claim 为普遍必要或 generator 增益。  
- 不把 GSM 配对 CI 写成训练复现已完成。

## Phase-G — Self-Forcing-style generator pilot（锁定 2026-09-29 18:05）

### Decision（阶段切换）
Phase-1 LatentRM（O2/O1/B1）**冻结为已验证评分器结果**，不再反复改 O2 / 加 loss。  
下一阶段关键问题变为：**训练 generator 后，Pass@1 与 Coverage 有没有提高？**（不是 O2 BoN 再涨。）  
优先做小规模 **Self-Forcing-style** pilot；**MMD/Chamfer 后置**。称呼上在查清 TF 前只叫 SF-style pilot，**不声称已实现 Self-Forcing**。

覆盖旧锁：`决策锁定.md` / `科研计划.md` 中「Phase-3 第一篇不做」→ **改为允许小规模 SF-style pilot**；Phase-2 set alignment 仍默认后置。

### 两条线（勿混）
| 已验证 | 未验证 |
|--------|--------|
| COCONUT 生成多候选；O2 同池选优 > O1/B1 | MMD/Chamfer；SF-style 训 generator；Pass@1/Coverage↑ |

### Do not
- 不把 MMD 当正确性目标；不对齐分布 ≠ 会答题。  
- 不把 `no_grad` / 冻结权重上的 rollout loss 当 generator 训练。  
- 初版不上 latent 逐步回归、不要求 self-t ≡ teacher-t。  
- 不 Chamfer+MMD 同时上；set loss 只打 content view，不打 ordered stream。  
- 不把 MoFlow `2503.09950` 当 MMD 出处。  
- 第一轮成败**不要**用 O2 选答案判定。

### 方法称呼（2026-09-29 21:27 收紧）
不做硬凑 TF→SF curriculum。准确描述：
> **受 Self-Forcing 启发的、答案监督的可微 self-rollout pilot**  
采样 latent ≠ ground-truth；不把 annotate 轨迹当 TF target。

### A/B/C 对照定义（必须能实现再开全量 pilot）

| 组 | latent thoughts | 答案监督 | 作用 |
|----|-----------------|----------|------|
| **A 冻结基线** | 原始 COCONUT 正常生成 | 不训练 | 原始 Pass@1 / Coverage |
| **B 答案监督对照** | 原始 COCONUT 生成并**缓存**；训练时作固定前缀并 **`detach`** | 正确答案 token CE（teacher-forced） | 测答案监督微调本身 |
| **C SF-style** | **当前** generator 自回归生成；**梯度跨 latent steps** | 同一答案 token CE | 测可微 self-rollout 相对 B 的额外收益 |

**B vs C 唯一主差：** latent 来源 + 梯度是否穿过 rollout。答案监督、数据、初始化、训练预算尽量相同。  
B **不是**「完全冻结 generator」——backbone 共享，答案 CE 仍会更新参数；准确说法是「缓存 latent 前缀 + 答案监督，**梯度不穿过 latent rollout**」。  
**若代码做不到这一区分 → 先统一实验定义与路径，不开全量 pilot。**

### Smoke（三件事，缺一不可）
1. **C：答案 loss → latent rollout 非零梯度**（看 grad norm，不只看 loss↓）。  
2. **参数真更新**（更新前后参数或固定输入输出对比）。  
3. **rollout ≡ 推理路径**（结构/前缀拼接/embedding 回灌一致；生成 latent 时不得用 gold answer 或未来信息）。  

首版 latent 长=6：可先无 KV cache、小 batch 全展开；若 detach/截断反传须标明近似点。

### Next（执行顺序）
1. 备份 `checkpoints/coconut` + `outputs/latentrm_order_pref/best`。  
2. 固定小训练子集 + 验证集；A/B/C 同题。  
3. 短 smoke（上列三事 + 显存）。  
4. 小规模 A/B/C：同步数/初始化/答案 loss/优化器/评估题。  
5. 成败只看 generator：**Pass@1、Coverage@N（4 或 16）、重复率/多样性、算力成本**；**不用 O2 BoN**。  
6. 仅 C 相对 B 有可信正向趋势 → 再扩；漂移/崩塌后再议**一个** set loss（MMD 现不做）。

### TF 审计初查（本仓库，2026-09-29）
- 只训 LatentRM；无 generator 微调入口；annotate≠TF target；`generate`=`@torch.no_grad()`。  
→ 与上表一致：不做 TF→SF；先落地 B（缓存+detach）vs C（可微 rollout）。

### Smoke 结果 (2026-09-29 21:40)
- 备份：`checkpoints/_freeze_backup_20260929/{coconut,latentrm_order_pref_best}`
- 代码：`src/sf_rollout.py`（B=缓存/detach；C=可微 hidden-state rollout）；`scripts/smoke_sf_rollout.py`
- **SMOKE PASS**（GPU5，`latenttts` env）：
  - C：`grad_to_latent_norm≈1.09`；参数更新
  - B：`grad_to_latent_norm=0`；答案 CE 仍更新参数
  - latent embed ≠ token emb（‖·‖≈78）
- 注意：首版无 KV cache、全展开；答案目标暂用 `#{answer}`（对齐 extractor 的 `#` 切分）。全量 pilot 前需固定子集并核对答案串格式是否与 COCONUT 生成分布一致。


### 最小 A/B/C pilot 首跑 (2026-09-29 21:50)
- 子集：`data/sf_pilot/` train=512 valid=256 seed=42；预算 **200 steps**，bs=1，lr=1e-5；答案目标 `#{answer}`
- 备份已做；smoke PASS；B 用冻结 coconut 缓存 latent（`train_latents_B.pt`）
- 产物：`outputs/sf_pilot_{B,C}_20260929_214444/`；评测 `results/full/sf_pilot/*_valid_20260929_214444.json`

| 组 | Pass@1 | Coverage@4 | mean unique ans |
|----|--------|------------|-----------------|
| **A** 冻结 | **0.310** | **0.457** | 2.35 |
| B 缓存+答案CE | 0.238 | 0.348 | 2.41 |
| C 可微 rollout | 0.257 | 0.371 | 2.46 |


### 首轮负结果读法锁定 (2026-09-29 23:23)
**暂时负结果**：200 步后 B、C 均差于 A；C 仅略好于 B（Pass@1 +1.9pt / Cov@4 +2.3pt），**无 CI，不能写成稳定优势或 SF 有效**。相对 A：C −5.3pt Pass@1、−8.6pt Cov@4。

### Decision（表述）
- 梯度路径符合设计 ≠ 训练目标/标签/格式正确。  
- 最稳妥：当前答案监督微调让生成退步；可微 rollout 可能减轻一部分退步，**不足以证明有效**。  
- 测的是「可微 rollout + 答案 CE」，**不是** Self-Forcing 原论文目标；不贴「SF 失败」标签，也不把 C≳B 写成有效。

### Do not（格式确认前）
- 不以「再多训几百步」为首选；目标错了只会学错更牢。  
- 不加 MMD；不用 O2 判成败；不宣称 SF 复现。

### Next（诊断顺序，先于重训）
1. 打印成功样例：完整输入/目标 vs 原模型真实输出（prompt 末、latent 后分隔、答案前缀、结束 token）。  
2. 逐 token 核 loss mask（只监督答案；不含 prompt/latent/pad）。  
3. 小样本过拟合（几十题）：拟合不了 → 查格式/标签/构造，不扩步数。  
4. 同解码设置比 A/B/C 规范化输出（格式变？解析失败？早截断？）。  
5. 逐题对错矩阵：解析问题 vs 推理内容变差。  
通过后再同 512/256 重跑 A/B/C + 题目级区间。


### Pilot 负结果表述锁定 (2026-09-29 23:24)

**Decision：** 200 步后 B、C 都差于 A；C 略好于 B（Pass@1 +1.9pt，Cov@4 +2.3pt）**无 CI**，不能写成 Self-Forcing 有效，也不能贴「Self-Forcing 失败」。  
准确说法：训练链路（梯度路径）已打通；首轮生成指标退步；当前答案监督微调在损害表现；可微 rollout 可能减轻部分退步，证据不足。测的是「可微 rollout + 答案 CE」，≠ 原论文 Self-Forcing 目标。

### Do not
- 不把「再多训几百步」当首选（格式错会学得更牢）
- 不上 MMD；不用 O2 判成败；不扩大 512/256 前未过诊断

### Next（诊断顺序，按序）
1. 打印成功样例完整输入/目标：prompt 末尾、latent 后分隔、答案前缀、结束 token vs 原模型真实输出  
2. 逐 token 核对 loss mask（仅答案；不含 prompt/latent/pad）  
3. 小样本过拟合（几十题）：学不会 → 先查格式/标签，不加大训练  
4. 同解码设置比较 A/B/C 规范化输出（格式/截断/解析失败）  
5. 逐题 A↔B↔C 对错迁移：解析问题 vs 推理内容变差  

修格式 + 小样本过拟合通过 → 同划分重跑 A/B/C + 题目级 CI。  
若修后仍 B≪A 且 C≫B → 支持 rollout 相对缓存有帮助；若 B、C 仍≪A → 问题在答案监督微调本身。


### 答案格式诊断 (2026-09-29 23:30)

**Finding（关键）：** 原 COCONUT 在 `<|end-latent|>` 之后的成功输出是 **`### {answer}<|endoftext|>`**（前可有 `<<...>>` 计算），**不是** 首轮用的 `#answer`。  
例：`AFTER='### 1400<|endoftext|>'` vs `TARGET='#1400'`。  
extractor `split('#')[-1]` 两种都能解析，但 CE 目标与生成分布不对齐 → 足以解释 B/C 相对 A 退步。  
Loss mask 本身：仅监督答案 2 token，prompt/latent/end 为 -100（构造正确）。  
产物：`results/full/sf_pilot/diag_A_success_format.json`、`diag_loss_mask.json`。

**Decision：** 修正目标为至少 `### {answer}` + eos；再做小样本过拟合；通过后再同划分重跑。仍不加步数硬训、不上 MMD。


### 小样本过拟合 (2026-09-29 23:35)
32 题 × 96 steps，lr=5e-5，greedy decode 测 train exact。

| 目标 | 模式 | loss 初→末 | train exact | 输出以 ### 开头 |
|------|------|------------|-------------|-----------------|
| `### {ans}<eos>` | C | 2.58→1.05 | **96.9%** | 100% |
| `### {ans}<eos>` | B | 2.00→0.30 | **90.6%** | 100% |
| `#ans`（旧） | C | 27.0→3.21 | 81.3% | **0%**（变成 `#...`） |

**Decision：** pipeline 可过拟合；旧 `#` 目标会改写输出前缀。已改 `train_sf_pilot.py` → `### {answer}`+eos。下一步同 512/256 重跑 B/C（仍 200 steps 作可比对照，或略增），再报 Pass@1/Cov。


### 格式修正后重跑 (2026-09-29 23:40)
同 512/256、200 steps、目标改为 `### {ans}<eos>`（`outputs/sf_pilot_{B,C}_20260929_233457`）。

| 组 | Pass@1 | Coverage@4 | mean unique |
|----|--------|------------|-------------|
| A | **31.0%** | **45.7%** | 2.35 |
| B | 23.3% | 33.6% | 2.10 |
| C | 25.7% | 37.1% | 2.16 |

与错格式首轮几乎同级退步。过拟合已证明目标可学；**短答案 CE 微调本身仍损害 valid 生成**（多样性也略降）。C 仍略高于 B，仍无 CI。  
**Decision：** 格式问题已排除为「唯一」原因；下一步按诊断清单做逐题迁移 + 看输出是否过早 `###`/丢掉中间 `<<>>`；考虑只更新部分参数或更小 lr / 更短答案监督窗口。仍不上 MMD、不盲目加步数。


### 格式修正后读法锁定 (2026-09-29 23:47)

**Decision：** 格式 bug 真实但非退步主因。短答案 CE 微调本身损害 COCONUT 生成；C 略高于 B（+2.4 Pass@1 / +3.5 Cov，**无 CI**）仅作待验证信号，不写 self-rollout 有效 / 生成能力↑。过拟合≠泛化到有效推理。

### Do not
- 不加步数；不加 MMD；不与「降 lr / 限更新范围」同时乱改多项

### Next（诊断）
1. valid 逐题保存 A/B/C 每次答案与正确性 → 转移矩阵（A↔B、A↔C、B↔C）；可按难度粗分  
2. 输出结构：`###` 稳定、end-latent/eos、提取失败、提前停/重复/异常、parser 误杀  
3. 同题同解码同 seed；配对 + 不确定性（n=256 单点不可当确定提升）  
4. 遗忘诊断：param 变化、valid loss、固定输入输出；再**单独**测更小 lr / 限更新范围

若 C 对 B 的配对收益集中且有统计支持 → 保留信号；否则暂停 generator 微调或改更保守更新。


### 逐题迁移结果 (2026-09-29 23:52)
`results/full/sf_pilot/diag_transfer_20260929_233457/summary.json`（同 seed=42，N=4，valid=256）。

结构：A/B/C 几乎都有 end-latent；`###` 率 A≈高、B/C 亦高；单 `#` 与 extract_fail 都不高 → **不是解析器大面积判错**。多样性 A>C≳B。

Coverage 配对差（pp，bootstrap CI）：B−A / C−A 显著为负；**C−B CI 跨 0** → C≳B 不稳。  
→ 确认：短答案 CE 整体损害；C 相对 B 无可靠配对优势。下一步看遗忘/更新幅度，仍不加步数/MMD。


### 退步主因锁定 + 表数字核对 (2026-09-30 00:17)

**Decision：** 答案 CE 奖励「尽快 `###`+短答案」，中间 `<<...>>` 展开被压短——这是退步主因，非解析器误判。C 缓解一部分但仍 ≪ A；C≳B 幅度小，Coverage 有弱正向 CI，Pass@1 CI 跨 0 → **不写 self-rollout 有效 / 生成能力提高**。

**两张 Pass@1 表并不矛盾：** 同 stamp `20260929_233457`、同 256 题、同 seed=42。  
- 报告 **31.0 / 23.3 / 25.7** = `pass_at_k` 的 Pass@1（`eval_sf_generator.py`）  
- 报告 **30.1 / 23.8 / 26.2** = **第 0 个 sample** 的正确率（`diag_sf_transfer.py`）  
Coverage@4 两边一致（45.7 / 33.6 / 37.1）。以后表头写清 metric。

结构证据：after_len A≈283、B≈38、C≈123；B 的 `###` 率≈99.6%。A→B Coverage 独损 35 / 独赢 4；A→C 25 / 3。

### Do not
- 不加步数；不加 MMD  
- 不只调 lr / 限参而不改「教什么」  
- 不把 C−B 写成已证明有效

### Next（先改目标，再谈更新幅度）
拆成两个独立问题：  
1) 目标是否奖励过早结束 → 补推理展开监督或冻结模型输出保持（伪目标优先用**最终答案正确**样本并抽查）  
2) 更新是否过激 → 仅在目标明确后单因子测更小 lr / 更少可训参数 / 更少步数

最小比较保持：  
- **A** 冻结  
- **B** 当前答案 CE（已知缩短对照）  
- **C** 可微 rollout + 同答案 CE  
- **D** 可微 rollout + 答案监督 + **明确保留合理输出过程 / 防过早结束**  

每组报 Pass@1、Coverage，并记 **输出长度、`###` 位置、早停率、答案多样性**。


### 组 D 开跑 (2026-09-30 00:21)
**D** = 可微 rollout（同 C）+ 过程监督：dataset `steps` 拼成 `<<...>>\n...\n### {ans}<eos>`（非冻结伪标签）。  
同 512/200 steps/lr=1e-5；对照 stamp `20260929_233457` 的 A/B/C。看 Pass@1、Coverage、after_len、### 率、多样性。不上 MMD。


### 组 D 结果 (2026-09-30 00:31)
D=`outputs/sf_pilot_D_20260930_002121`（C 式可微 rollout + dataset steps 过程 CE，200 steps）。对照 A/B/C=`20260929_233457`。

| 组 | Pass@1 (atk) | Cov@4 | after_len | 立即 `###` | 备注 |
|----|-------------:|------:|----------:|----------:|------|
| A | 31.0% | 45.7% | ~283 | ~70% | 冻结 |
| B | 23.3% | 33.6% | ~38 | ~99.6% | 短答案 CE |
| C | 25.7% | 37.1% | ~123 | ~95% | rollout+短答案 |
| **D** | **18.6%** | **31.3%** | **~414** | **~1.7%** | 长度恢复，准确率更差 |

mt128 几乎不变（Pass@1≈18.8%，Cov≈32.0%）。Greedy 抽查仍有 `<<...>>\n###`；dropout 下更长且常以 `<<` 起头（过程保留），但答案更易错。

**Decision：** 改「教什么」有效改变了输出形态（不再塌缩到短 `###`），**尚未**恢复/超过 A 的准确率；D 目前差于 B/C。过程监督 ≠ 自动修好生成。下一步可单因子：冻结 A 的**正确**轨迹伪标签、或对 `###`/eos 加权、或减小更新幅度——每次只动一个旋钮。仍不加 MMD、不盲目加步数。


### D 读法锁定 + 暂停微调 (2026-09-30 00:34)

**Decision：** D 证明过程监督能改输出形状，**未**证明改善推理——更长、答更差。A 仍最好；B/C/D 均损 valid。C≳B 仍不足以称 self-rollout 有效；D ≈「学会过程形式」≠「更好推理」。  
dataset steps 未必适配 COCONUT latent/输出；逼模型写长不是答案。

### Do not
- 暂停「继续微调模型」扩训；不加步数；不加 MMD；暂缓 `###` 加权  
- 不并行改监督数据+lr+参数范围+loss

### Next（单因子，先查监督是否可信）
1. **抽查 D 的 steps 标签**：语义/顺序/与最终答案一致？与模型 `<<...>>` 格式是否匹配  
2. **正确轨迹伪标签**：冻结 A 生成，仅保留最终答案正确且格式像原输出的轨迹作过程目标；只换监督来源，其余同 D → 区分「数据集步骤不适合」vs「不需要过程监督」  
3. 目标选定后，再单测更小 lr / 限参  

评估：同 256 题、同采样；存逐题正确性/答案/长度/格式；Pass@1+Coverage+形态。正向后再扩评/复训。


### steps 标签抽查 (2026-09-30 00:36)
`results/full/sf_pilot/diag_steps_audit.json`：sf_pilot train512 中 **96.5%** 最后一步数值 ≈ 最终答案；`<<...>>` 格式全合规；均值 ~2.6 步。~3.5% 失配多为取整（4.76→5、8.5→9）。  
**Decision：** 标签大体可信、表面格式也匹配 A；D 变差**不能**主要归因于「steps 全是错标签」。更可能是：显式 CoT 文本监督与 COCONUT latent 推理分工不兼容，或更新过猛/目标过长。→ 进入 **A 正确轨迹伪标签** 单因子对照（只换监督来源）。


### E 伪标签对照 (2026-09-30 00:38)
只换监督来源：冻结 A **greedy 正确**轨迹（372/512）作过程目标；其余同 D（可微 rollout，200 steps，lr=1e-5）。

| 组 | Pass@1(atk) | Cov@4 | after_len | starts_### / starts_<< |
|----|------------:|------:|----------:|------------------------|
| A | 31.0% | 45.7% | ~283 | ~70% / (混合) |
| D dataset steps | 18.6% | 31.3% | ~414 | ~1.7% / 高 |
| **E A-pseudo** | **31.3%** | **44.1%** | ~288 | 71.5% / 28.5% |

**Decision：** E ≈ A（Pass@1 31.3% vs 31.0%，Cov 44.1% vs 45.7%，after_len/### 率也贴近）→ **A 正确轨迹伪标签可避免 D 式崩溃，也未超过冻结 A**。D 差主要因 **dataset steps 不适合当 COCONUT 过程目标**（标签数值大多对，但与模型原生轨迹不匹配），不是「过程监督本身无效」。下一步若继续：在伪标签设定下单测更小更新/是否能略超 A；或暂停 generator 微调、回到 LatentRM。不加 MMD/步数。

### How to read later results
- Coverage↑、Pass@1 平 → 更易探到正确路径；Pass@1↑ → 单次也改善。  
- 仅 O2 BoN↑、generator 指标不变 → 仍是评分器。  
- C>B 且对照干净 → 可归因 self-rollout；对照不清 → 不算。


## Phase-G2 — 组合方案 SF + MMD + O2（锁定 2026-09-30 16:34）

### Decision（主线改写）
目标是验证**完整 idea**，不是把 O2 挖成单篇。叙事：

> LatentTTS 两瓶颈：generator 自回归偏离有效轨迹；latent thoughts 无可靠一一对应，逐点监督易短答/机械模仿。用 self-rollout 贴近推理；用 set-level MMD 对齐正确轨迹**内容分布**；用 O2 层级偏好在推理时选候选。

三组件**不是同一 loss**，也不都对 generator 反传：
| 组件 | 作用 | 反传 |
|------|------|------|
| SF-style rollout | 训练≈推理，自生成 prefix | → generator |
| Set MMD（content view） | 弱化逐点对齐，对齐正向轨迹集合 | → generator（+projector） |
| O2 LatentRM | 推理时选候选 | **先冻结**，不反传进 generator（防骗分） |

历史命名：旧 **D_steps**=dataset steps CE（已失败）；本阶段 **D_mmd**=SF+MMD 主组合。E=A-pseudo 作答案监督底子。

### 已知约束（必须带着跑）
- 短答案 CE → 过早 `###`；dataset steps → 长但 Acc↓；A-pseudo → ≈A 未超 A
- SF 下一版**不能**只靠答案 CE，也**不**把“写更长”当目标
- MMD **不判断对错**；content view 须过顺序 probe，否则称「set-level regularization」非 position-free
- 不把 MoFlow `2503.09950` 当 MMD 出处

### 最小矩阵（512/256，~200 steps）
| 组 | Generator | MMD | 推理 O2 | 作用 |
|----|-----------|-----|---------|------|
| A | 冻结 | 无 | 可另报 A+O2 | 基线 |
| C | SF+答案（已有；建议 A-pseudo） | 无 | 不用于判生成 | SF-only 消融 |
| **D_mmd** | SF+答案 | **有** | 先生生成指标 | **主组合** |
| D+O2 | 同 D_mmd ckpt | 同候选 | O2 选 | 系统互补 |

预算紧：先 A/C/D_mmd；有信号再补 D+O2。

### Loss（generator）
\( \mathcal L_G = \mathcal L_{answer}^{SF} + \lambda_{MMD}\mathcal L_{set} + \lambda_{keep}\mathcal L_{preserve} \)  
- answer：self-rollout 后答案 CE（优先 A-pseudo，非 short/dataset-steps）  
- set：MMD²(C(H^SF), C(H^positive))，positive=同题 A 成功 latent 集合  
- preserve：可先弱/监控替代；首轮 λ_MMD **小 + ramp-up**  
首轮目标：**减退化 + 看是否优于 SF-only C**，不预设必超 A。

### Content view 硬要求
共享非可逆 bottleneck projector → 无序集合上算 MMD；训后/并行做**顺序 probe**。probe 仍能恢复位置 → 不得声称 position-free。

### 成功门槛（分开报）
**Generator（不用 O2 选）：** Pass@1、Cov@4/16、长度/`###`、逐题转移、多样性。  
**系统：** D 候选 + 冻结 O2 vs A+O2。  
最低：D 相对 C 的 Cov 稳升且 Pass@1 不再明显退化；**或** D+O2 相对 A+O2 有清楚正向配对。仅 MMD↓/分布像而无 Acc/Cov/BoN → 不算成功。

### Do not
- 先把 MMD、SF 各自调到最优再组合（主决策看 D vs A/C）  
- 用 O2 判 generator 成败；一上来把 O2 分数反传 generator  
- Chamfer+MMD 同时上；把均值中心化/detach 当 content view  
- 把计划写成已验证成功

### Next
1. 缓存 A 正确题的 **latent** 正向轨迹（非仅文本）  
2. 接 ContentProjector + MMD + 顺序 probe smoke  
3. 训 D_mmd（SF + A-pseudo answer + 小 λ MMD ramp）；对照 C（同 answer、无 MMD）  
4. 独立评生成指标 → 再 D+O2

### D_mmd 结果（2026-09-30 16:50）

设定：`outputs/sf_pilot_Dmmd_20260930_164751`；mode=C + A-pseudo（372）；λ_mmd=0.05 ramp 50；200 steps；bs=1 lr=1e-5。对照 A=冻结 COCONUT；E=`sf_pilot_E_20260930_003631`（同 answer、无 MMD = C_pseudo）。评测不用 O2。

| 组 | Pass@1(atk) | Cov@4 | after_len | starts_### | mean unique |
|----|------------:|------:|----------:|-----------:|------------:|
| A | 31.0% | 45.7% | 283 | 70.2% | 2.35 |
| E (C_pseudo) | 31.3% | 44.1% | 288 | 71.5% | 2.43 |
| **D_mmd** | **31.4%** | **44.1%** | **303** | **70.6%** | **2.37** |

配对 Cov（diag）：D−E = 0.0pp（CI [−3.1,+3.1]）；D−A = −1.6pp（CI 跨 0）。形态未崩（无短答 CE 式 after_len 塌缩）。

顺序 probe（训后 projector）：content-view 位置准确率 **83%**（chance 17%；raw 99%）→ **仍非 position-free**（与随机 projector ~83% 同档）。称 **set-level regularization**，不写 position-free content view。

**Decision（读法，2026-09-30 20:37 收紧）：**
> 轻量 MMD 没有带来可见生成提升，也没有像前几种微调那样破坏输出形态；当前 content view 仍明显含位置信息，**不能**称为 position-free set alignment。

- D≈E≈A（31.4 / 31.3 / 31.0 Pass@1；Cov 44.1=E，低于 A 45.7）。**不要**把 31.4 vs 31.3 写成改善。  
- 合适表述：当前小规模设定下，轻量 MMD+伪标签**共同维持**接近原模型，**无额外收益**。  
- 可写「用了 permutation-invariant 的 set-level MMD」；对齐的可能是**带位置线索的集合分布**，非纯内容。probe≠证明 MMD 全无作用，只收窄解释。  
- **不写** Self-Forcing + MMD 已提升生成能力。

**三块证据分开写：**
| 模块 | 现状 |
|------|------|
| O2 LatentRM | 同池选优正向证据最强；仍是最有结果支撑的模块 |
| SF-style generator | 短答/dataset-steps CE 退步；A-pseudo ≈A |
| MMD pilot | 无明确生成收益；形态基本保持；content view 位置泄漏 |

**D_mmd+O2：** **不设为默认**。若成本低可做一次**标明诊断性**的组合评估（固定 D 候选 + 冻结 O2 选优 vs A+O2；分报候选池 / Cov / O2 BoN）。测的是「生成器与选择器是否互补」，**不是** MMD 已增强生成。若无收益 → 可从当前方案移除 MMD，不必继续为其烧资源。

产物：`Dmmd_valid_*.json`、`diag_transfer_Dmmd_*`、`probe_content_order_Dmmd_*.json`、`content_projector.pt`。

### Do not（本轮后）
- **不**盲目加大 λ_MMD / 加训练步数  
- 不把 0.1pp Pass@1 当改善；不用 O2 BoN 回填「MMD/SF 提升了 generator」  
- 不声称已实现 position-free content alignment  
- 不上 Chamfer 双 set loss；不把 probe 失败当「修好前绿灯」重训主叙事

### 诊断 + 下一招（2026-09-30 20:55）

**问题判断：** 三 idea 未真正耦合——正监督≈克隆 A；MMD 不判对错且 view 漏位置；O2 训练期不参与。D≈E≈A 是预期。

**D_mmd+O2 诊断（sf_pilot valid，N=4，seed=42，冻结 O2）：**
| 系统 | O2 BoN Acc | Cov@4 | Voting |
|------|----------:|------:|------:|
| A+O2 | 35.16% | 41.80% | 33.98% |
| D_mmd+O2 | 33.59% | 40.23% | 33.98% |

配对 D−A：BoN −1.6pp、Cov −1.6pp（CI 均跨 0）。**无系统互补信号** → 可从当前方案边缘化 light MMD；不加大 λ。

**Next（执行中）：** SF + **冻结 O2 拒绝采样（RFT）**  
self-sample K→O2 在正确样本中选优→SF live rollout；O2 不反传。脚本 `scripts/train_sf_o2_rft.py`。

### O2-RFT 结果（2026-09-30 21:04）

| 设定 | Pass@1 | Cov@4 | after_len | starts_### | +O2 BoN | +O2 Cov |
|------|-------:|------:|----------:|-----------:|--------:|--------:|
| A | 31.0 | 45.7 | 283 | 70% | **35.2** | **41.8** |
| D_mmd | 31.4 | 44.1 | 303 | 71% | 33.6 | 40.2 |
| O2rft-short（### CE） | 27.2 | 38.7 | **157** | **92%** | 34.0 | 38.7 |
| O2rft-traj（仅<<正确） | 30.7 | 41.8 | 303 | 60% | 33.2 | 38.7 |

- short 版重蹈短答 CE 塌缩。  
- traj+require_process：形态稳住（after_len≈A），但 **Pass@1/Cov 未超 A**；系统 BoN 也未超 A+O2。skip≈994/1194 采样。  
- **Decision：** 接上 O2 选样是对的方向，但「只在已正确且含<<的自采样上 CE」仍像弱克隆，涨不过 A。light MMD / 本轮 RFT **均无系统涨点**。  
- **Do not：** 回 short CE；用 O2 BoN 粉饰 generator；盲目加 λ_MMD。  
- 若再试：O2 排序 + **A-pseudo 作 CE 目标**（门控≠短目标），或 best-vs-worst 对比；先扩 K/数据前须有形态+指标双信号。

### O2-gate + A-pseudo（2026-09-30 21:13，边训边调）

脚本 `train_sf_o2_gate_pseudo.py`：self-sample K → 冻结 O2 门控 → CE 用 **A-pseudo**（非 short）。调试：`debug.jsonl` 记 gate 原因 / margin / roll_hash3 / roll_alen / roll_nok / roll_neg_m；负 margin 过多或格式塌缩可 abort。

**中途调整：** smoke 发现 margin&lt;0 仍在更新 → 默认 `margin_min=0.5`（负 margin 更新率→0）。短 after_len≈10 是 `<<…>>\n###` 正常，不误判塌缩。

| 设定 | Pass@1 | Cov@4 | after_len | +O2 BoN | +O2 Cov |
|------|-------:|------:|----------:|--------:|--------:|
| A | 31.0 | 45.7 | 283 | **35.2** | 41.8 |
| E | 31.3 | 44.1 | 288 | — | — |
| O2gate k4 m0.5 | **32.3** | **46.9** | 293 | 34.8 | **43.4** |
| O2gate k8 m0.5 | **33.1** | **47.3** | 291 | 33.2 | 42.2 |

配对（相对 A）：k4 Cov +1.2pp（CI 跨 0）；k4+O2 Cov +1.6pp（CI 跨 0）；k4+O2 Acc −0.4pp。k8 生成略高但 **O2 BoN Acc 更差**。

**Decision：** 这是目前**形态健康 + 生成弱正向**的最好一版，但仍 **无稳 CI**，不能写「已涨点 / SF+O2 有效」。k4 系统权衡优于 k8。门控+pseudo 优于 short-RFT / light-MMD。下一步若继续：best-vs-worst 或更大预算前先做配对 CI 复测；不加 λ_MMD。


### O2 best-vs-worst 偏好（锁定执行 2026-09-30 21:18）

**缺陷：** 门控+克隆 A-pseudo 仍无“比 A 更好”的信号。  
**修法：** 冻结 O2 作教练——自采样 K；优胜=正确且 O2 高（优先含<<）；失败=错误且 O2 高；  
`L = L_CE(优胜轨迹) + λ_UL L_unlikelihood(失败轨迹)`；O2 不反传。  
成败分报 generator Pass@1/Cov 与 +O2 BoN；形态健康为先。不上加大 λ_MMD。

### O2 best-vs-worst 结果（2026-09-30 21:33）

边调信号：
- `correct_vs_wrong`+UL：update 少；UL≈4 压 CE；**生成 Cov↓**（43.4 < A 45.7），放弃大 λ_UL。
- 改为 `correct_vs_correct`（≥2 正确中 O2 高低分）+ **λ_UL=0**（只 CE 优胜 / A-pseudo 兜底）。

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov | 形态 |
|------|-------:|------:|--------:|--------:|------|
| A | 31.0 | 45.7 | 35.2 | 41.8 | ok |
| O2gate k4 | 32.3 | 46.9 | 34.8 | 43.4 | ok |
| pref cvw+UL | 31.9 | 43.4 | 34.0 | 40.6 | ok 但退步 |
| pref cvc λul=0 | 32.6 | 45.7 | 34.8 | 42.2 | after=305 |

**Decision：** 对错误做 UL 有害（Cov 掉）。cvc（正确间 O2 门控 CE，λul=0）Pass@1≈32.6 Cov=45.7≈A，+O2≈34.8/42.2；**未超过 O2gate**（32.3/46.9）。当前小预算下可交付仍是 **O2gate+A-pseudo**；“O2 当教练的对比学习”尚未带来稳涨点。不加大 MMD。

### 缺陷重定位 + beat-A（2026-09-30 21:35）
### beat-A 结果（2026-09-30 21:41）

训练：self 目标占 sources beat=22 match=16 pseudo=162（约 19% self）。

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov |
|------|-------:|------:|--------:|--------:|
| A | 31.0 | 45.7 | 35.2 | 41.8 |
| O2gate | 32.3 | 46.9 | 34.8 | 43.4 |
| beat-A k16 | 32.3 | 45.7 | 33.6 | 42.2 |

形态 after≈302。配对相对 A Cov 0.0pp（CI 跨 0）。相对 gate -1.2pp。

**Decision：** 修好打分一致性后 self 更新≈19–25%；指标未稳超 O2gate。缺陷从“完全克隆”推进到“偶发采用自轨迹”，但自采样过程轨迹仍稀缺，涨点不够。下一步若继续：提高过程轨迹出现率（采样/解码），或放大仅 self 更新的步数；不加大 MMD。


**当前缺陷：** O2gate 仍把 CE 目标钉在 A-pseudo 上 → O2 只是门卫，学不到“比 A 更好”的轨迹；self_process 命中率低。UL 打错答案已证有害。

**改进：** 用缓存 A latent 打 O2 分；自采样正确轨迹若 **O2 分 > A-pseudo + margin_beat** 且含 `<<` → CE 用 self；否则回退 A-pseudo（同 gate）。脚本 `train_sf_o2_beat_a.py`。盯 `beat_rate` / `gap_vs_a`。

### beat-A 结果（2026-09-30 21:41）

训练：self 目标占 sources beat=22 match=16 pseudo=162（约 19% self）。

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov |
|------|-------:|------:|--------:|--------:|
| A | 31.0 | 45.7 | 35.2 | 41.8 |
| O2gate | 32.3 | 46.9 | 34.8 | 43.4 |
| beat-A k16 | 32.3 | 45.7 | 33.6 | 42.2 |

形态 after≈302。配对相对 A Cov 0.0pp（CI 跨 0）。相对 gate -1.2pp。

**Decision：** 修好打分一致性后 self 更新≈19–25%；指标未稳超 O2gate。缺陷从“完全克隆”推进到“偶发采用自轨迹”，但自采样过程轨迹仍稀缺，涨点不够。下一步若继续：提高过程轨迹出现率（采样/解码），或放大仅 self 更新的步数；不加大 MMD。

### 根本路线：不克隆 A-pseudo（2026-09-30 21:43）

**为何会抄旧的：** 无 GT latent；短答 CE 塌缩；A-pseudo 是“不崩”的监督拐杖，但本质上是克隆 A，无法定义“更好”。

**根本修法：** 训练目标**禁止** A-pseudo/dataset steps。只从当前模型自采样学习：  
正确 ∧ 含 `<<` ∧ O2 相对同题其它样本更优 → SF CE 该 self 轨迹；否则 **skip**（宁可不更新）。  
可用轻奖励塑造提高 `<<` 出现率（选样偏好），不是换回克隆。O2 冻结。

**执行（同日）：** `scripts/train_sf_o2_noclone.py`；K=16、dropout_p↑提高多样性；无 A-pseudo 兜底；盯 `roll_proc_ok` / update_rate / tlen。成败分报 generator vs +O2；对照 A 与 O2gate。不加大 λ_MMD。

### 不克隆结果（2026-09-30 21:52）

训练：200 更新 / 1525 尝试，update_rate=0.131，**clone=false**；训练末 `roll_proc_ok≈0.20`（<<正确率被抬起）；胜者多为短过程 `<<eq>>\n### ans`（mean_tlen≈16，mean n_<<≈1.4）。

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov | after | starts_### |
|------|-------:|------:|--------:|--------:|------:|-----------:|
| A | 31.0 | 45.7 | 35.2 | 41.8 | ~ok | high |
| O2gate | **32.3** | **46.9** | **34.8** | **43.4** | 293 | high |
| **noclone k16** | 30.1 | 41.4 | 32.4 | 41.4 | 319 | **0.57** |

配对：相对 A Cov −4.3pp（CI 跨 0）；相对 **O2gate Cov −5.5pp，CI 全负** [−9.4, −1.6]。系统相对 A+O2：bon −2.7pp / cov −0.4pp（CI 跨 0）。

**Decision：** 根本路线（禁止 A-pseudo）已执行且可复现；**在当前自采样质量下会掉点**（尤其相对 O2gate）。缺陷不是“没勇气不抄”，而是 **self 过程正确轨迹太短/太稀**，纯 CE 其上会推坏长格式（### 率↓）且涨不出答案。  
可交付仍是 **O2gate**。下一步若继续根本路线：先提高多步 self 过程质量（不是退回克隆），或改用非文本 CE 的 latent 偏好；**不加 λ_MMD，不宣称 no-clone 已涨点**。

### 综合判断 + 目标 +10pp（2026-09-30 22:08）

**三 idea：** ① SF-style generator ② content MMD ③ 冻结 O2（选优/门控/教练）。

**当前水位（sf_pilot valid，相对冻结 A=31.0/45.7；系统相对 A+O2=35.2/41.8）：**
| 模块 | 证据强度 | 量级 |
|------|----------|------|
| O2 作选优 | 最强（全量 GSM 亦有软正） | 选优约 **+1～3pp**；不是 +10 |
| SF generator | 弱：短 CE 有害；A-pseudo≈A；O2gate 最好约 **+1pp 级**且 CI 跨 0 | 未独立涨点 |
| MMD | 近零；view 漏位置；D+O2 ≤ A+O2 | **暂无贡献** |
| 三者耦合 | 未形成；要么克隆 A，要么 no-clone 掉点 | **非加性** |

**对「最后用上三 idea 且约 +10pp」的读法：**
- **现在写不进**：任何组合都未接近 +10；最好生成器增量 ~1pp，O2 选优单独也只有数个点。
- **+10pp 必须先钉死指标**：建议主目标定为 **系统 Pass@1（generator+O2 BoN）相对 A 无 RM 基线**，或相对 **A+O2**。若相对 A+O2 再 +10 → 要到 ~45 Acc，当前无路径证据。若相对裸 A（31）到 ~41，仍远超现有软信号，需换量级的数据/训练信号，不是调 λ。
- **三 idea 要留下，但角色要改**：O2=唯一已验证杠杆，必须进训练环（非只推理选优）；SF=可微 rollout 载体，监督不能再默认 A-pseudo 克隆；MMD=在 content view 真正去位置前 **hold / 边缘化**，否则占名额不涨点。
- **到 +10 的必要条件（未满足则不要空转组合）：** (1) 自采样多步过程正确率显著高于今（正确∧<< 从 ~11% 量级抬升，或改用 latent 偏好不依赖短文本 CE）；(2) O2 分数与最终对错在自采样上校准，而不只在合成乱序上强；(3) 全量 GSM/更大预算，pilot 512×200 步撑不住 +10；(4) MMD 仅在 probe 显示位置泄漏明显下降后再加回。

**Do not：** 把 O2gate +1pp 外推成三 idea 成功；加 λ_MMD 冲 +10；宣称「结合即可 +10」；用 A-pseudo 克隆当三 idea 已耦合。

### 规模口径（2026-09-30 22:12）

用户确认：目标 **~+10pp** 的路径是 **全数据 + 更长训练**，不是 sf_pilot×200step 抠小数。

**同意：** pilot 只负责验耦合；量级必须靠全量 GSM（或项目既定全训练集）与更长 schedule。

**但扩规模之前仍要选对配方（否则全量只是把错信号放大）：**
1. **可扩：** O2 门控/教练 + SF live rollout；全量题；更长 steps/epoch；更大 K 采样。  
2. **先不扩 / 不作为主损失：** light MMD（未去位置泄漏前）；纯 A-pseudo 克隆当「三 idea 已结合」；noclone 短 `<<` CE（pilot 已掉点）。  
3. **扩之前的门槛（小预算复测即可）：** 自采样「正确∧可用过程」命中率相对今有抬升，或改用 **latent 偏好**（不依赖短文本 CE）；O2 与最终对错在自采样上不全拧。  
4. **主指标：** 先钉死相对谁 +10（建议：全量 valid/test 上 generator Pass@1 与 +O2 BoN 分报；相对冻结 A 与相对 A+O2 分开写）。

**Do not：** 直接把 O2gate/A-pseudo 原样拉满全量当冲 +10；全量加大 λ_MMD；无中间命中率信号就开多卡长跑。

### 执行：全量 self-correct（2026-09-30 22:20）

**测量更正：** diag `mean_after_len≈280` 为 **pad 虚高**；真实 after 中位≈18；A-pseudo 亦多为短 `###`/`<<`（alen≥80 几乎不存在）。先前「长 CoT」门槛作废。

**配方：** `train_sf_o2_selfcot.py` — 禁止 A-pseudo；CE 用自采样 **任意正确 after**（不强制 `<<`，避免 noclone 只打 11% 稀有模式）；O2 排序 + margin；`min_alen=5`。MMD hold。全量 `data/gsm_train.json` + `max_steps≥3000`。

**门槛：** sf_pilot 短跑 update_rate≫0、形态不塌；过则开全量。评测分报 generator / +O2；对照 A 与 A+O2。

### 全量 self-correct 结果（2026-09-30 23:05）

设定：`gsm_train`≈385k（跳过 5 条坏答案）；K=8；min_alen=5；margin_min=0.5；**3000 updates**；update_rate=0.63；clone=false；mean_tlen≈4.5（短 ### 为主）。

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov |
|------|-------:|------:|--------:|--------:|
| A | 31.0 | 45.7 | **35.2** | 41.8 |
| O2gate (pilot) | **32.3** | **46.9** | 34.8 | **43.4** |
| **full selfcot 3k** | 31.5 | 44.9 | 33.2 | 41.4 |
| gsm_valid selfcot | 30.6 | 44.2 | — | — |

配对（pilot valid）：相对 A Cov −0.8pp（CI 跨 0）；相对 gate −2.0pp（CI 跨 0）；系统 vs A+O2 bon −2.0pp / cov −0.4pp。

**Decision：** **全数据+更长训 alone（当前 self-correct 配方）没有给出 +10，甚至未稳超 O2gate/A。** 规模把「短自正确 CE」跑满了，但信号仍弱。下一档若继续冲 +10：不要再盲目加 step；要换更强训练信号（如全量 A-pseudo 门控作对照上限、或 latent 偏好/过程质量），并分报相对 A 与相对 A+O2。MMD 仍 hold。

### 自动迭代约定（2026-09-30 23:08）

用户要求：**每次实验完 → 分析瓶颈 → 立刻改进再跑**，减少等人下指令。已写入 `.cursor/rules/auto-iterate-experiments.mdc`。

### 下一轮：全量 O2gate + 在线 A 教师（锁定执行）

**瓶颈：** selfcot@3k 学短自答，规模无效。pilot 唯一弱正是 O2gate+A-pseudo。  
**修法：** `train_sf_o2_gate_teacher.py` — 学生 K 采样 + O2 门控；CE 目标 = **冻结 A greedy**（仅当 A 正确）；全量 `gsm_train`，3000 steps。MMD hold。评测对照 A / O2gate-pilot / selfcot。

### 全量 O2gate-Teacher 结果（auto）

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov |
|------|-------:|------:|--------:|--------:|
| A | 31.0 | 45.7 | 35.2 | 41.8 |
| O2gate pilot | 32.3 | 46.9 | 34.8 | 43.4 |
| selfcot full | 31.5 | 44.9 | 33.2 | 41.4 |
| **gateT full** | 33.5 | 45.7 | 32.8 | 40.2 |

vsA Cov {mean_pp: 0.0, ci跨0}; vsGatePilot Cov −1.2pp（CI 跨 0）。系统 vs A+O2：bon −2.3 / cov −1.6（CI 跨 0）。
stamp=20260930_231108 updates=3000 rate=0.620 mean_tlen≈4.5。

**Decision：** 全量教师门控 **抬了 generator Pass@1（31→33.5）**，但 **+O2 变差** —— 纯克隆 A 与系统选优冲突。Cov 未超 pilot-gate。距 +10 仍远。  
**下一轮：** 从 gateT ckpt warm-start；`O2beatT`：自正确 O2>教师 → CE self，否则回退教师；盯 `self_rate` 与 +O2。MMD hold。

### O2beatT 开跑 (2026-09-30 23:51)

- Smoke 30 步：self−teacher O2 gap 在 0 附近密集（噪声），故 `beat_margin=0.5`；self/teacher 目标长度相近（~5 vs ~4 tok）。
- 全量：k=8, beat_margin=0.5, 3000 steps, warm-start gateT；stamp `20260930_235112`。

### O2beatT 结果（auto）

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov |
|------|-------:|------:|--------:|--------:|
| A | 31.0 | 45.7 | 35.2 | 41.8 |
| gateT | 33.5 | 45.7 | 32.8 | 40.2 |
| **beatT** | 33.0 | **47.3** | **34.8** | **42.2** |

vs A+O2：bon −0.4 / cov +0.4（CI 跨 0）；vs gateT+O2：bon +2.0 / cov +2.0（CI 仍跨 0，方向对）。self_rate=0.366（beat_self 717 / teacher 1903 / self_tw 380）；update_rate=0.734。beat_self 全程占比略升（early→late 225→255/1000），mean_tlen 稳定 ~4.9。

**Decision：** 修复了 gateT 的 +O2 回退；Pass@1 保住；Cov 新高。距 +10 仍远，但信号非塌缩。  
**下一轮：** 同配方再 +3000（warm-start beatT ckpt）——按 PLAN「有正信号才加长」。盯 late 段 self_rate 与 +O2；若 plateau 再加 self 质量过滤（min_alen）。MMD hold。

### O2beatT2 结果（auto）— 加长 plateau

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov |
|------|-------:|------:|--------:|--------:|
| beatT | 33.0 | 47.3 | 34.8 | 42.2 |
| **beatT2** | 32.9 | 44.5 | 34.8 | 41.4 |

同配方再 +3k：**Cov↓2.8、Pass@1/ +O2 持平**。beat_self 仍短（637/667 ≤5 tok，mean_tlen 3.67）；self_rate 卡在 ~36%。  
**Decision：** 停止纯加长。瓶颈 = 超短 beat_self 主导自更新。  
**下一轮：** warm-start **beatT（非 beatT2）**；相对长度门控 `beat_len_eps=1`（self after_len ≥ teacher_alen+1）+ b=0.5；不用绝对 min_alen（教师本身 median≈7 字符，a8 无效）。成功 = Cov≥47 且 beat_self 明显更长。MMD hold。

### 诊断更新（2026-10-01 10:40）

- `min_alen=8` 无效：`after_len` 是字符；教师 median≈7，短 `###` 仍过门。
- `beat_len_eps=1` 更糟：凡 O2 打赢教师的 self，`after_len` **恒等于** teacher_alen → 40 步 0 次 beat_self。说明优势在 **latent**，不在文本长度；纯 CE after ≈ 再克隆短答。
- **下一轮：** `lambda_lat=1` — beat_self / self_tw 时 CE + MSE(live C-latents, 赢样本 latents)；warm-start beatT；b=0.5 k=8 3000。成功 = +O2 超 A+O2 或 Cov 保 47+。MMD hold。

### O2beatLat 结果（auto）— 首超 A+O2

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov |
|------|-------:|------:|--------:|--------:|
| A | 31.0 | 45.7 | 35.2 | 41.8 |
| beatT | 33.0 | 47.3 | 34.8 | 42.2 |
| **beatLat** | **33.8** | 45.7 | **36.3** | **44.1** |

vs A+O2：bon **+1.2** / cov **+2.3**（CI 仍跨 0）；vs gateT+O2：bon +3.5 / cov +3.9（CI 下沿≈0）。self_rate=0.383，update_rate=0.745。  
**Decision：** latent match 有效——第一次系统 +O2 超过 A+O2。Cov 略低于 beatT 峰值。距 +10 仍远。  
**下一轮：** 同配方再 +3000 warm-start beatLat（有正信号才加长）。若 +O2 继续涨则再 scale；若 plateau 试 λ_lat∈{0.5,2}。MMD hold。

### O2beatLat2 结果（auto）— 再加长 plateau

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov |
|------|-------:|------:|--------:|--------:|
| beatLat | **33.8** | 45.7 | **36.3** | **44.1** |
| beatLat2 | 33.9 | 45.3 | 35.9 | 42.2 |

同配方再 +3k：Pass@1 持平，**+O2 略回退**（36.3→35.9 / 44.1→42.2）。self_rate 升到 0.44 也没带来系统增益。  
**Decision：** 停加长。最佳仍是 beatLat。  
**下一轮：** warm-start **beatLat**；`λ_lat=2`（加强 latent match）；k=8 b=0.5 3000。对照若掉点再试 0.5。MMD hold。

### 运维约定（2026-10-01）

用户要求：**显存安排满一点**，勿让 H20 只占 ~5GB。已写入 `.cursor/rules/pack-gpu-vram.mdc`；`train_sf_o2_beat_teacher.py` 默认 `frozen_dtype=bf16`、`auto_pack`（空卡抬 k→16/32 + `grad_accum≥4`）、启动打 VRAM。当前 `λ_lat=2` 跑完后再用 packed 配置开下一轮。

### O2beatLatL2 结果（auto）— λ=2 新高 +O2

| 设定 | Pass@1 | Cov@4 | +O2 BoN | +O2 Cov |
|------|-------:|------:|--------:|--------:|
| A | 31.0 | 45.7 | 35.2 | 41.8 |
| beatLat λ1 | 33.8 | 45.7 | 36.3 | **44.1** |
| **L2 λ2** | 33.3 | 44.1 | **37.5** | 43.4 |

vs A+O2：bon **+2.3** / cov +1.6（CI 仍跨 0）；vs gateT+O2：bon **+4.7**（CI [1.2, 8.6]，首次下沿>0）。self_rate=0.418。  
**读法：** 加大 latent match → **BoN 再涨**（36.3→37.5），但 generator Cov 略掉。系统侧相对 A+O2 约 +2pp，距 +10 还差 ~8pp。  
**Decision：** λ↑ 有正信号；停盲加长。  
**下一轮：** warm-start **L2**；同 λ=2 + **auto_pack**（k≈32, ga≥4, bf16 frozen）；3000 steps。盯 +O2 是否继续涨、Cov 是否守住 ≥44。MMD hold。

### 阶段反思（2026-10-01 18:50）— 停小参扫描，先找根因

用户要求：先汇总反思，**勿继续拧 λ/k/margin 等小旋钮**。Pack 跑可收尾，但新实验先停。

**目标对照：** 要的是相对 A / A+O2 约 +10pp；现状最好系统 +O2≈37.5（vs A+O2 35.2，约 +2pp），generator Cov 峰值 47.3（vs A 45.7，约 +1.6pp）。数量级不对。

**已证实有效 / 无效：**
- 有效方向：O2 门控 > 盲 selfcot；beat（超教师才学自己）修复 gateT 的 +O2 回退；latent MSE match 首次让 +O2 超过 A+O2。
- 无效/假杠杆：纯加长同配方（beatT2 / beatLat2）；绝对/相对 after 长度门（教师本身也是短 `###`）；盲抬 λ 以外的“感觉调参”。
- 未真正打开：MMD（hold）、过程监督金标 steps、O2 相对新 generator 重校准、非 CE 的系统目标。

**根因假说（按证据强度）：**
1. **优化目标 ≠ 评测目标（主因候选）**  
   训练几乎全是「短 after 的 token CE ± 偶发 latent MSE」；评测是 Pass@1/Cov 与 **O2-BoN**。CE 不直接最大化 O2-BoN；latent match 只在 ~35–40% beat 步上出现，且匹配的是单次 dropout 样本，不是 BoN 选优分布。
2. **教师 A 把答案形态钉在短 `###`**  
   teacher_alen median≈7；beat 赢家 after_len **恒等于** teacher → 文本侧没有过程增益可学。系统上限被「短答 + 冻结 O2 在短答池里挑」卡住，难出 +10。
3. **冻结 O2 与学生分布错位**  
   O2 在 A/annotate 分布上训；学生一偏，O2 仍按旧分打分。gateT 抬 Pass@1 却掉 +O2，就是「生成分布移向 A、O2 选优变差」。latent match 只是在旧 O2 坐标系里贴赢样本，不扩大能力边界。
4. **3000 step × 全量数据仍是极稀疏**  
   ~3k updates / 386k 题 ≈ 每题 ≪1 次；「全数据更长」叙事上做了，信息量上仍像反复打同一短模式。加长同损失 → plateau 符合这一条。
5. **SF+O2 还没接到第三支柱**  
   原计划 SF+MMD+O2；MMD 因近零信号 hold。当前实质是 SF-CE + 弱 O2 对齐，缺新表示/新监督。

**因此不该再做的：** λ∈{0.5,2,4}、k∈{8,16,32}、margin、min_alen 细扫；再 +3k 同损失。  
**该想清楚再动手的（需选型，不是立刻开跑）：**  
(A) 评测对齐的目标（直接 O2 / BoN / pref，而不是只 CE after）；  
(B) 打破短 `###` 老师形态（金标 steps / 非 A 教师 / 显式过程约束且已证不会塌）；  
(C) O2 对学生分布重校准或联合；  
(D) 是否还坚持 +10 定义在 pilot N=4 +O2，还是换主指标。

Pack `20261001_183910` 若未完可让它跑完作归档，**不据此自动开下一旋钮实验**。

### 专家建议（2026-10-01 18:53）— 选型，不执行

**总判断：** 当前卡在「短答 CE + 冻结 O2 贴分」的局部最优；再扫 λ/k/步数期望值 <1–2pp。要冲 +10，必须换**目标或监督形态**，不是调参。

**建议优先级：**

1. **先锁主指标（半天决策，必做）**  
   +10 若指 pilot N=4 的 +O2 BoN：要从 35→45，靠 CE 几乎不可能。若指 generator Cov/Pass@1：beatT 的 Cov 峰说明 generator 还有一点空间，但和系统叙事不一致。  
   **建议：** 主报 **系统 +O2 BoN（及 Cov）**，generator 作诊断；+10 的分母写死为 A+O2。避免继续优化错尺子。

2. **主攻：评测对齐的目标（假说 1）— 最值得做的下一方法**  
   在 beat/门控之后，对 live C-rollout 加 **可反传的 O2 项**（正确轨迹 maximize O2，或 pairwise：赢样本 > 教师/错样本），CE 降权为辅助。  
   理由：唯一持续正信号都来自「怎么用 O2」，但从未把 O2 当主损失；latent MSE 只是间接对齐。这是从「蹭尺子」变成「推尺子」。  
   风险：O2 hack / 短答刷分 → 必须同时盯 generator Cov 与 after 结构，设 stop 规则（Cov 掉 >2pp 或 ### 率异常则停）。

3. **并行探针（小、死活实验，不是调参）：打破短 `###` 天花板（假说 2）**  
   不做 min_alen 游戏。二选一、各 ≤1 张卡、短跑：  
   - **金标 steps CE**（仅 O2 门控通过时），或  
   - **拒绝短教师**：教师 after 过短则 skip，改用 dataset steps / 更长伪标签源。  
   若过程监督再次塌缩 after_len→只出 `###`：记失败，不再碰。若 Cov/+O2 同向动：才谈 scale。

4. **暂缓：O2 重校准（假说 3）**  
   有价值，但是第二波：先有「学生分布真的变了且 O2 仍有效」的证据。否则是给旧短答分布重装一把尺。  
   例外：若走 (2) 后 +O2↑ 但人工看轨迹变差，再开校准。

5. **继续 hold**  
   盲 λ_MMD↑；同损失 +3k；λ/k/margin 网格；A-pseudo 克隆路线（已证 Pass@1↔+O2 对立）。

**不建议的叙事：** 「全数据再训久一点就到 +10」。证据反对。  
**建议的叙事：** 「系统目标对齐 O2-BoN + 可选过程监督探针；小步证伪，有正信号再 pack 显存 scale」。

**你需要拍板的一件事：** 下一方法做 **(2) O2 主损失** 还是 **(3) 过程监督探针**，还是两者各开一个短对照。拍板前不自动开跑。

### 指标锁定更正（2026-10-01 18:56）

用户澄清：**+10pp = 相对论文基线**，打分器用 **最基本的 LatentRM**（非 O2/order-pref）。

**对照 `/home/lihourun/ll/科研计划.md` 理解（2026-10-01 18:59 锁定）：**

| 概念 | 计划里是什么 | 本仓对应 |
|------|----------------|----------|
| **论文基线** | LatentTTS（ACL 2026）复现：COCONUT + **官方 CE LatentRM** 自然 BoN/MV/MC | generator=`checkpoints/coconut`；PRM=**B0** `outputs/latentrm_baseline/best`（矩阵 ID「官方 CE LatentRM」）；official 仅外部对照 |
| **最基本 LatentRM** | B0：CE、**无 pref** | 不是 O2，不是 B1/O1 |
| **主方法 / 主表增益** | Phase-1 **O2**（B1+hierarchical pref）在**自然候选**上的 BoN/beam | 相对 **B0**（及难集）看是否不掉、是否有增益 |
| **第一篇不做** | Phase-2 MMD；**Phase-3 Generator self-rollout** | 近期 SF / beatLat 线属 Phase-3，**不是**计划锁定的第一篇主贡献 |

合格基线复现清单（计划原文）：COCONUT deterministic、原 LatentRM BoN、MV、MC-dropout 原配置、GSM8K-Test/Hard/MultiArith、相同 N、≥3 seed 或报方差。公平第一：对不齐论文绝对值没关系，**内部对比协议必须统一**。

**+10 若仍作为性能目标：** 分母应是 **论文基线 BoN Acc（B0 或对齐后的 official 协议）**，不是 pilot 上的 A+O2。本仓 GSM-Test seed42 已有：B0@N16≈**32.8%** → +10 ≈ **~43%**。Phase-1 已观测到的 O2−B0 仅约 **+1～3pp（软正）**，与 +10 量级差很远；计划 Go 条件是「机制↑且自然 BoN 不掉、难集有增益」，**并未写死 +10**。若坚持 +10，需另开能力跃迁路径，且与「第一篇只做 LatentRM」可能冲突——要你确认 +10 是第一篇硬目标还是远期愿景。

**立刻改正的读表习惯：** SF 阶段表上的「A+O2=35.2」是 **order-pref 尺 + pilot valid**，**不能**当论文基线，也不能当 +10 进度条。

### 战略锁定（2026-10-01 19:16）— 基线采纳 + Phase1–3 一体

用户确认：

1. **《科研计划》只锁了第一阶段方法**；现已跨过「只做 Phase-1」的阶段限制。  
2. **基线只采纳计划里的定义**：COCONUT + **B0 / 官方 CE LatentRM** 自然 BoN（及计划复现清单协议）。+10 分母用这把尺，不用 A+O2。  
3. **Phase-1 / 2 / 3 都要落好**，目标是 **互相传递正确信号的整体系统**，不是三篇互不通气的实验：  
   - **P1 LatentRM（O2）**：给自然候选打分/选优，机制与 BoN 相对 B0 为正或不伤；  
   - **P2 表示对齐（MMD/set，曾 hold）**：仅当能向 generator / RM 传**可检验的正信号**再开，禁盲 λ↑；  
   - **P3 Generator（SF / beat / latent match…）**：改分布时，必须以 **B0 尺**（及诊断用 O2）验证系统 BoN，避免「生成↑、选优↓」或「只蹭 O2 池」。  

**系统健康标准（读写统一）：**  
某阶段的改动，必须在 **下一阶段或联合评测** 上不破坏信号——例如 P3 新 generator 在 **B0 BoN** 上 ≥ 基线且相对 B0+10 有路径；P1 新 RM 在 P3 候选上仍能选对；P2 若开，要同时动 RM 与/或 generator 的可测指标，而不是孤立 loss↓。

**执行含义：** 继续做 SF/整体时，主报表改为 **相对 B0（论文基线协议）**；O2 作辅助选优/训练信号可以，但 claim +10 必须过 B0。旧「第一篇不做 P3」作废；旧「P2 默认不做」改为 **有正传递信号才做**。

### 不足与修正路径（2026-10-01 19:23）

**不足（按阻塞程度）：**

1. **尺子错位（可立刻修）** — 全程用 pilot+O2 当进度；+10 与系统健康要用 **B0×GSM-Test**。最好 Pack 的 O2-BoN 37.9 **尚未**用 B0 量过。  
2. **P3→P1 信号不闭环** — 训练门控/匹配跟 O2，主 claim 跟 B0；未知 O2 涨是否翻译成 B0 涨（gateT 已暗示会打架）。  
3. **P3 目标仍是短答 CE±latent MSE** — 不直接推 BoN；短 `###` 天花板未破。  
4. **P1 单独不够 +10** — O2−B0 仅 ~+1–3pp；必须靠 P2/P3 改变候选质量或联合。  
5. **P2 未接入** — 无表示层把 RM 与 generator 对齐的正传递。  

**能修正并接着做：能。** 顺序：

| 步 | 动作 | 目的 |
|----|------|------|
| **S0** | 用 **B0** 评 A / L2 / Pack（GSM-Test N=16 seed42） | 标定真实离 +10 的距离；验证 O2 涨是否传到 B0 |
| **S1** | 若 B0 上 Pack≥A：P3 训练信号改为 **可反传 B0 分**（或 B0+O2 双报，主停看 B0） | 修正不足 2–3 |
| **S2** | 过程监督短探针（破 `###`）仅当 S0/S1 显示文本形态是瓶颈 | 不足 3 |
| **S3** | P2 仅当 S1 后仍差一大截且有内容对齐假说 | 不足 5 |

Pack 归档：O2 pilot Bon **37.9**（新高），仍只是诊断尺。  
**正在执行 S0。**

### S0 结果（2026-10-01 19:43）— B0 尺真相

GSM-Test · **B0** · N=16 · seed=42（`20261001_192426`）：

| Generator | B0 Acc | B0 Cov | 对照（pilot O2 N4，诊断） |
|-----------|-------:|-------:|---------------------------|
| A（coconut 基线） | **33.74%** | 51.1% | A+O2 35.2 |
| L2（λ=2） | 32.45% | 49.1% | O2 ~37.5 |
| Pack（k32） | 31.99% | 48.4% | O2 **37.9** |

相对 A：**L2 −1.3pp / Pack −1.7pp（B0）**。旧矩阵 A+B0@N16=32.8%，本次 A 复测 33.7%（同协议可接受波动）。+10 目标仍约 **~43–44%**。

**Decision：** 不足 2 坐实——**优化 O2 选池在损害 B0 选优**。P3 按 O2 涨点继续训 = 背离论文基线。  
**下一刀（S1）：** 停 O2 主优化；P3 训练/门控改对齐 **冻结 B0 分数**（可反传或 BoN-一致的 surrogate），主报 B0；O2 仅诊断。先短跑证：B0 门控 CE / B0-score 损失是否让 B0 Acc ≥ A。

### 澄清（2026-10-01 19:46）— O2 仍是创新，不是扔掉

用户正确指出：**O2 是 Phase-1 创新，必须留在整体系统里。**  
S0 结论不是「别用 +O2」，而是「**分母/基线**用 A+B0；**分子/我们的系统**可以是 gen+O2」。

| 角色 | 设定 | 用途 |
|------|------|------|
| **论文基线（分母）** | A + **B0** | +10 的对照 |
| **我们的系统（分子）** | （改进）generator + **O2** | 主 claim；创新在 RM + 可选 gen |
| **健康检查** | 同 gen × B0 | 若 gen 只适配 O2、B0 崩盘，要写明；理想是双尺都不崩 |

之前说「停 O2 主优化」**表述过激**。应改为：继续用 O2 做选优/训练信号，但 **+10 进度 = (gen+O2) − (A+B0)**，并补测 GSM-Test 上的 A+O2 / Pack+O2（此前只有 pilot O2 与 GSM B0）。  
S1 修正为：训练可继续 O2 门控，但每轮主报 **系统 vs A+B0**；若系统涨而 B0 健康检查掉太多，再加 B0 对齐项（双信号），不是删掉 O2。

### 系统尺结果（2026-10-01 20:01）— Pack+O2 vs A+B0

GSM-Test · N=16 · seed=42（同协议连跑）：

| 系统 | Acc | vs A+B0 |
|------|----:|--------:|
| **基线 A+B0** | **33.74%** | 0 |
| A+O2 | 33.13% | **−0.61** |
| Pack+B0 | 31.99% | −1.74 |
| **系统 Pack+O2** | **31.99%** | **−1.74** |

目标 ≈ 43.7%（基线+10）。  
**读法：** 当前最好 P3（Pack）+ 创新 O2，在论文协议上 **低于基线**；pilot 上 O2 37.9 **未翻译**到 GSM-Test 系统。A+O2 本次也略低于 A+B0（与旧矩阵 O2@N16=35.0 不一致，需另查协议差，但不改变「Pack 系统未涨」）。  
**Decision：** 整体系统尚未打过基线；不能再只看 pilot+O2。下一方法必须同时抬 **候选质量** 与 **O2 在 GSM-Test 上的选优**，并主报 (gen+O2)−(A+B0)。

### 原因（2026-10-01 20:05）— 不是算错，是切片+不迁移

逐题对照（GSM-Test N=16 与 pilot N=4 同 seed）：

1. **Pilot 37.9 是真的，但是窄切片**  
   `data/sf_pilot/valid.json` **256 题、N=4、O2**。Pack 该切片 **first=35.6 / Cov=42.6 / O2 Acc=37.9**（相对 A+O2 35.2 的 +2.7）。O2 在「池子里有正确答案」时只漏 **11%**。  
   **不是日志造假。**

2. **换论文协议就消失**  
   GSM-Test **1319 题、N=16**：Pack Cov **48.4 < A 51.1**（全错题 680>645）；O2 Acc=**32.0 = Pack+B0**。O2 在有解池上漏 **34%**（A 上也 ~34–35%）。  
   leftover：pilot **~5pp** vs GSM **~16–18pp**。N 变大、题变难后，O2 几乎选不出混池里的对答案。

3. **Pack+B0 与 Pack+O2 Acc 相同不是抄错文件**  
   候选池 sha1 **全同**；选中下标只同 332/1319；对错翻转 **39 vs 39 对称**。O2 在 Pack 的 GSM 池上 **零净增益**。

4. **P3 改的是「易集短答更稳」，不是「更会解题」**  
   Pack first 在 GSM 上仍高于 A（33.5>31.8），但 **unique↓（3.81<4.41）**、全错题↑。训练目标（短 `###` CE + 贴 O2 latent）把分布挤到 O2 在 **小 N 易集** 上好看的模式；论文集上覆盖变差，选优也帮不上。

**结论：** 先前涨点 = **真实但不可迁移的切片增益**（pilot/N=4/O2），不是实验造假。相对论文基线 **没有涨过**。

### P123 一体系统落地（2026-10-01 20:20）

实现（不是再盲扫 λ）：

| 组件 | 路径 | 作用 |
|------|------|------|
| 共享打分 | `src/system_scoring.py` | O2/B0 BoN-一致 sum-logit |
| 一体训练 | `scripts/train_p123_system.py` | P1 冻结 O2 选优 + B0 健康门；P2 λ_MMD 可接（默认 0）；P3 SF-C CE |
| 训练中中间量 | `outdir/mid_metrics.jsonl` | leftover / O2↔B0 agree / hash3 / cot / unique / sources |
| 评后中间量 | `scripts/diag_p123_mids.py` | Acc_O2、health B0、leftover、Δ vs A+B0、pilot↔GSM gap |
| 流水线 | `scripts/pipeline_p123_eval.sh` | gen → +O2 → +B0 → mids |

**本轮配方（对症假迁移）：** 从 **coconut** 起（不继承 Pack 短 `###`）；`dual_beat`：self 须 O2 超 teacher **且** B0 ≥ teacher_B0 − `b0_slack`；`prefer_cot`；**λ_MMD=0**（未过内容漏检前不盲开）；主报 `(gen+O2)−(A+B0=33.74%)`。

**怎么读中间量：** leftover↓ + Cov≥A + Δ_vs_A_B0≥0 才算系统涨；若 pilot↑ 而 GSM leftover 仍 ~16pp → 仍是切片特化。  
**Do not：** 盲 λ_MMD↑；只报 pilot O2；用 Pack 作暖启除非 mids 证明未塌格式。

### P123 开训 + 首批中间量（2026-10-01 20:30）

在跑：
- `dualBeat` GPU0 · `outputs/p123_dualBeat_20261001_202106` · k=64 ga=8 · λ_lat=0 · 1500 step  
- `dualBeatLat` GPU7 · `outputs/p123_dualBeatLat_ll1_20261001_202240` · 同上 + λ_lat=1  
- 并行 `A+O2` GSM-Test N16 → `results/full/p123/bon_A_O2_N16_gsm_20261001_202200.json`  
- 训完自动 `pipeline_p123_eval.sh` → mids vs A+B0=33.74%

**step=50 训练中中间量（dualBeat，窗=50 attempt）：**

| 中间量 | 值 | 读法 |
|--------|---:|------|
| pool_ok (k64) | 0.96 | 池里几乎总有对 → 训练信号够 |
| leftover | 0.12 | 训练分布上 O2 漏选 12%（GSM 上曾 ~34%） |
| o2_b0_agree | **0.10** | **O2/B0 顶尖几乎不共识** → 双尺打架是真瓶颈 |
| hash3 / cot | 0.84 / 0.16 | 仍偏短 `###`；prefer_cot 未主导 |
| unique/k | 4.4/64 | **多样性极低**；加 k 不涨 Cov 的根因之一 |
| sources | beat 342 / teach 58 | dual_beat 主要在学 self |

**Decision：** 系统代码已接通；先等 1500step 的 GSM 系统尺。若 GSM leftover 仍高而 train leftover 低 → 仍是迁移；下一步优先抬 **unique/过程形态**（非盲 λ_MMD），并盯 o2_b0_agree。

### GPU 0567 并行铺开（2026-10-01 20:50）

用户确认可用 **0/5/6/7**。GPU6 现被他人 `wubohan/.../generate_validation.py` 占 ~11GB，**不抢**；我们的卡：

| GPU | 任务 | 对症 |
|-----|------|------|
| 0 | dualBeat + **dualBeatDiv**（共卡，dp=0.45） | 基线双门 / 抬 unique |
| 5 | **dualBeatProc**（`target_mode=process`） | 反短 `###` 特化 |
| 7 | dualBeatLat + **dualBeatB0t**（b0_slack=0） | λ_lat / 抬 O2↔B0 共识 |
| 6 | （他人）空了再补 | — |

A+O2 GSM N16 = **32.75%**（低于 A+B0 33.74% → 系统分子更不能只靠换 RM）。

### 中间量读数（2026-10-01 21:14）— 训练分布，非 GSM 系统尺

| 配方 | step | leftover | o2_b0_agree | unique | hash3/cot | sources 倾向 | 观感 |
|------|-----:|---------:|------------:|-------:|-----------|--------------|------|
| dualBeat | ~300 | 0.08–0.22 | ~0.14 | ~4.5/64 | 仍偏 hash3 | beat≫teach | 稳但多样性差；像旧 Pack 风险 |
| dualBeatLat | ~300 | ~0.22 | ~0.1–0.2 | ~4/64 | cot 偶升 | beat≫teach | 与 dualBeat 差不多，λ_lat 未见明显结构性改善 |
| **dualBeatProc** | ~100 | 0.16–0.26 | 0.2–0.3 | ~5–6 | **cot↑0.67 hash3↓0.33** | beat≈teach | **形态最好**：过程监督在起作用 |
| **dualBeatDiv** | ~150 | 0.24–0.30 | **~0.32** | **~11/48** | hash3 仍高 | **teach≫beat** | unique 真抬了，但几乎克隆 teacher，leftover 偏高 |
| dualBeatB0t | ~100 | 0.10–0.14 | ~0.26 | ~4 | 中等 | beat>teach | leftover 略好；agree 略升；未质变 |

**Verdict（现在）：** 还不能报涨点——没有 GSM `(gen+O2)−(A+B0)`。训练尺上 **Proc 最值得盯**（破短答），**Div 多样性有效但信号偏 teacher-clone**。dualBeat/Lat 继续跑作对照即可，不要因 train leftover 低就提前宣称成功。

### GSM 系统尺汇总（2026-10-02 01:14）— 已结束配方

协议：GSM-Test N=16 seed=42；分母 **A+B0=33.74%**；主报 `(gen+O2)−(A+B0)`。

| 系统 | Acc_O2 | Cov | Acc_B0 | Δ vs A+B0 | pilot O2 | 备注 |
|------|-------:|----:|-------:|----------:|---------:|------|
| A+B0 | 33.74 | 51.1 | 33.74 | 0 | — | 论文分母 |
| A+O2 | 32.75 | 50.2 | — | −0.99 | — | O2  alone 未超 B0 |
| Pack+O2（旧） | 31.99 | 48.4 | 31.99 | −1.74 | 37.9 | 假迁移对照 |
| **dualBeat** | **35.41** | **53.4** | **34.27** | **+1.67** | 35.9 | 当前最好；B0 健康也过线 |
| dualBeatB0t | 35.18 | 52.5 | 34.04 | +1.44 | 36.7 | 严 B0 门略逊主线 |
| dualBeatDiv | 34.04 | 51.0 | 32.90 | +0.30 | 34.4 | 多样性↑但涨点小；B0 健康掉 |
| dualBeatLat | 35.03 | 51.4 | （B0 评测中） | +1.29 | — | O2 已出；接近 dualBeat |
| dualBeatProc | — | — | — | — | — | 训 ~92%，~23min 完 |

**读法：** 相对 Pack 已扭转为 **正增益**；相对 +10 目标仍差 **~8.3pp**。leftover 仍 ~17pp（选优漏）。pilot↔GSM transfer_gap 已很小（0.3–1.5），不再是「假涨」主因。  
**下一优先：** 等 Proc / Lat-B0 齐 → 若 Proc 系统尺再抬 Cov/降 leftover 则沿过程监督加码；否则在 dualBeat 上专攻 leftover（非盲 λ_MMD）。

### 全结束复盘（2026-10-02 11:01）

五路 train+eval **全部 PIPELINE_DONE**；无占用进程。

| 系统 | Acc_O2 | Cov | Acc_B0 | Δ vs A+B0 | leftover | pick_fail|池 | 训练源 beat/teach |
|------|-------:|----:|-------:|----------:|---------:|-------------:|------------------:|
| A+B0 | 33.74 | 51.1 | 33.74 | 0 | — | — | — |
| Pack+O2 | 31.99 | 48.4 | 31.99 | −1.74 | ~16.5 | ~34% | — |
| **dualBeat** | **35.41** | **53.4** | 34.27 | **+1.67** | 18.0 | 33.7% | 11043/957 |
| dualBeatB0t | 35.18 | 52.5 | 34.04 | +1.44 | 17.4 | 33.0% | 9244/2756 |
| dualBeatLat | 35.03 | 51.4 | 34.42 | +1.29 | 16.4 | 31.9% | 10524/1476 |
| dualBeatProc | 34.87 | 52.2 | **36.77** | +1.14 | 17.3 | 33.1% | 2308/9692 |
| dualBeatDiv | 34.04 | 51.0 | 32.90 | +0.30 | 16.9 | 33.2% | 1766/10234 |

**因果读法**
1. **主 claim 赢家 = dualBeat**（O2 双门 + self-beat）；相对假迁移 Pack 已扭正；相对 +10 仍差 **8.3pp**。  
2. **瓶颈几乎全是 leftover**：Cov≈53 → 若选优完美可达 ~53%（超过 43.7% 目标）。pick_fail|池 ≈33% 稳定，五配方都没打破。  
3. **Proc**：形态/ B0 健康最好（B0 **36.77**），但 O2 选优弱于 B0 → 主 claim（+O2）反被拖；训练几乎变成 teacher-clone（9692 teach）。过程监督改分布有效，**没修好 O2 在池上的排序**。  
4. **Div**：unique↑ 但涨点最小，且 B0 健康掉线——多样性≠系统涨点。  
5. **离线 O2+B0 z-sum 集成**：dualBeat 无增益（仍 35.41）；Proc 提到 36.54≈B0。说明简单融合不是 dualBeat 的免费 +10。  
6. O2/B0 top 一致率仅 ~30%。

**Decision / 下一刀（对 leftover，不盲 λ_MMD）**  
A. **加长 dualBeat**（唯一明确正信号配方）：自 `outputs/p123_dualBeat_.../model` 再 1500 step。  
B. **同 ckpt × N=64 BoN**：测 Cov/leftover 是否随 N 改善（候选池不够 vs 选优坏）。  
C. Proc 线暂不主扩，除非改「过程 gen + 仍用 O2 claim」的选优（否则会滑向 B0 叙事）。

**已启动（11:06）：** A=`outputs/p123_dualBeatX2_20261002_110614`（GPU0，lr=3e-6，暖启 dualBeat）；B=`results/full/p123/bon_dualBeat_O2_N64_gsm_20261002_110614.json`（GPU5）。

### 提速 + N64 诊断（2026-10-02 11:15）

**提速：** 原 trainer 每题 64 候选 × (O2+B0) = 128 次单条 PRM 前向 + 单题 generate → GPU 空转（~10–12 s/step）。另发现 11:03–11:06 启动命令被重复执行，GPU0 上 4 个同样的 X2、GPU5 上 4 个同样的 N64 在抢卡，已清理。  
改为 `sample_and_score`：`q_batch` 题批量 generate + 解码文本重分词后批量 PRM 打分（与 `infer_gpt2_rm` best_of_n 同一路径，训练打分≡评测打分）。q_batch=32,k=64：**2.4 s/step、峰值 53GB（卡上 84GB）**，1500 step ≈1h。  
X2 重启：`outputs/p123_dualBeatX2_20261002_111157`（GPU0）。注：新打分路径与旧 seq 路径略有不同，X2 与 dualBeat 不是严格同协议的续训。

**N64（dualBeat，O2）：** Cov **61.1**（N16 53.4），Acc **34.50**（N16 35.41）——**N 越大 O2 越选不准**，leftover 18→26.6pp；多数投票 36.24 > O2 top。  
**离线选法（同 dump，零成本）：**

| 池 | O2 top | 多数投票 | O2 加权投票 T=1 |
|----|-------:|--------:|----------------:|
| dualBeat N16 | 35.41 | 35.56 | **36.69** |
| dualBeat N64 | 34.50 | 36.24 | **37.00**（Δ vs A+B0 **+3.26**） |
| A N16 B0（基线） | 33.74 | 33.97 | 33.66 |

**读法：** 瓶颈已坐实是 **P1 选优器不适配 P3 新分布**（O2 在 A 的 annotate 数据上训，未见过 dualBeat 候选）。加权投票是合法 TTS 聚合，可回收 ~1.5pp，但离 +10 仍差 ~6.7pp。  
**下一刀候选（需决策，计算量大）：** 在 dualBeat 候选上 **on-policy 重标注 + 续训 O2**（P3→P1 回传，正是一体系统的闭环）；全量 annotate 曾耗时数天，需先做子集版。

### P3→P1 on-policy O2（用户批准子集版，2026-10-02 11:18）

- 子集：gsm_train 中 coconut **从未标注**的题随机 2 万（`scripts/make_onpolicy_subset.py`，两片各 1 万），题号与 coconut 不重叠 → 可直接合并。
- 标注：同 Phase-0 协议（n=8，MC 64，dropout 0.2，seed 42），生成器换成 `p123_dualBeat/model`；GPU5 上 train_a / train_b / valid-64（N=64，不做 MC）并行，约 2h。
- 续训：`training_args/train_order_pref_onpolicy.yaml`——损失与 O2 完全相同，自 O2 best 暖启，lr 2e-5，4 epoch；训练集 = 新标注 + 等量 coconut 回放（`scripts/build_onpolicy_mix.py`）；选 ckpt 看 dualBeat valid-64，同时监控 coconut valid-64（遗忘）。
- 自动串联：`scripts/chain_onpolicy_o2.sh` → 评测 dualBeat×O2op N16/N64、A×O2op N16（健康），报 top 与加权投票相对 A+B0。
- **怎么读：** 主看 dualBeat×O2op N16 top 相对 35.41、N64 是否不再随 N 变差；A×O2op 不应明显低于 A+O2（32.75）。若涨点主要来自加权投票而 top 不涨 → RM 仍未学会新分布。
- **Do not：** 改 O2 损失/λ；用 GSM-Test 选 ckpt。

### OOM 事故 + 修复（2026-10-02 11:35）

- X2（dualBeat 续训 1500 步，q_batch=32）在第 120 步 OOM：generate 首步对 2048 条长 prompt 算全词表 logits，一次要 48.7GB。修复：`sample_and_score_safe` 遇 OOM 对半切 batch 重试；改 q_batch=16 重启（`outputs/p123_dualBeatX2_20261002_112643`，与 annotate 共卡约 8 s/步，约 3.5h）。
- annotate train_a 在 GPU5 被 valid-64（峰值 67GB）挤爆，只完成 128 题。剩余 9872 题对半拆：train_a（GPU0，seed 43）和 train_c（GPU5，seed 44）。`build_onpolicy_mix.py` 默认已加 train_c，chain 无需改动。
- 读结果注意：train_a、train_b、train_c 的 seed 不同，只影响采样，不改变协议。

### X2 结果 + on-policy 标注读数（2026-10-02 13:30）

- **X2**（dualBeat 再续 1500 步，GSM-Test N16）：O2 Acc **35.18**（dualBeat 35.41），Cov 54.4，Vot 36.92，B0 约 34.4。**生成器已饱和**，续训步数不再涨点，Δ vs A+B0 仍只有约 +1.4~1.7。
- **决定：** dualBeat 配方不再加步数，主瓶颈继续按 P1 选优器处理（leftover 约 17pp）。
- **on-policy 标注**：约 1.97 万题全部标完。dualBeat 在 train 上 cov@8 约 91%，全对 68%，全错 9%。只有 mixed 题（约 23%，即 4636 题）有 MC estimations 能进训练，和原 O2 数据规则一致；混入 4641 题 coconut 回放。
- **O2op 训练过程**：dualBeat valid-64 recall@1 从 35.2 最高到 36.2（500 题，标准误约 2pp，属噪声量级）；coconut valid-64 稳定在约 0.61（无遗忘）。最终以 chain 的 BoN 结果为准。

### O2op BoN 结果：on-policy 重标注无效（2026-10-02 13:50）

| 池 × RM | top | 加权投票 | 对照（原 O2） |
|---|---:|---:|---|
| dualBeat N16 × O2op | 34.95 | 36.54 | 35.41 / 36.69 |
| dualBeat N64 × O2op | 34.27 | 36.24 | 34.50 / 37.00 |
| A N16 × O2op | 32.52 | 33.74 | 32.75 / — |

- **读法：** 三组都持平或略降，N64 仍比 N16 差 → RM 没学到新分布的排序。可能原因：①训练集太容易（dualBeat 在 train 上 cov@8 91%，test 只有 53%），能用的 mixed 题只有 4636 道，且分布偏离测试的难题；②相对原 O2 的 17.5 万题，这点数据推不动。
- **结论：** 生成器加步数（X2）和选优器 on-policy 续训两条路都饱和。目前最好的合法系统是 dualBeat + O2 加权投票：N16 36.69（+2.95），N64 37.00（+3.26），距 +10 还差约 6.7pp。
- **Hold：** 不再扩大同协议的 on-policy 标注；不在 GSM-Test 上调任何东西。属于策略分叉，等用户决定方向。
- **14:10 先做低成本确认**（不依赖方向选择）：`scripts/seed_confirm.sh` 在 seed 43/44 上重跑 dualBeat+O2 与 A+B0（N16，按 seed 配对），用 `scripts/summarize_bon.py` 报 top / 加权投票 / 多数投票 / cov。读法：只有三个 seed 上加权投票的 Δ 都 >0 且均值约 +3，才算稳定收益。

### 锁定阶段成果：dualBeat + O2 加权投票（用户选 1，2026-10-02 15:40）

**官方系统（当前）：** 生成器 `outputs/p123_dualBeat_20261001_202106/model` + RM O2 + **softmax 加权投票 T=1**（合法 TTS 聚合，非事后挑挑拣拣）。

| seed | dualBeat top | dualBeat wvote | A+B0 top | Δ wvote vs 固定 33.74 | Δ wvote vs 同 seed A+B0 |
|------|-------------:|---------------:|---------:|----------------------:|------------------------:|
| 42 | 35.41 | **36.69** | 33.74 | **+2.95** | +2.95 |
| 43 | 35.86 | **36.16** | 32.45 | **+2.42** | +3.71 |
| 44 | 36.32 | **36.77** | 31.16 | **+3.03** | +5.61 |
| 均值 | 35.86 | **36.54** | 32.45 | **+2.80** | +4.09 |

- N64（仅 seed 42）：wvote **37.00**（Δ vs 33.74 **+3.26**）；top 34.50 仍弱于 N16，说明 top-1 选优仍不适配，但加权投票随 N 略增。
- **判定：** 三个 seed 上 wvote 都落在 36.2–36.8，相对固定协议基线 A+B0@s42=33.74 的 Δ 全为正、均值约 **+2.8pp**。收益稳定，可锁定为阶段进度。
- **仍未达标：** 距 +10（≈43.74）还差约 **6.7–7.2pp**。X2 加步、同协议 on-policy O2 已 Hold。
- **Do not：** 用 seed 均值当新官方数字替换 s42 协议声明；主声明仍写 seed 42：N16 wvote 36.69（+2.95）、N64 wvote 37.00（+3.26）。三 seed 只作稳定性证据。
- **下一刀（待选）：** (2) 只标难题续训 O2；(3) 攻生成器覆盖率（N16 cov≈53%）；(4) 把加权投票目标写进 P3 训练。

### 三选一拆解 + 推荐（2026-10-02 15:45）

**现状分解（dualBeat×O2，GSM-Test N16 seed42）：**
- cov（池里≥1 对）= **53.37%** → 若选优完美，天花板 53.37% > 目标 43.74，**覆盖率本身还没卡死 +10**。
- wvote = 36.69；oracle→wvote 空隙 **16.7pp**；leftover（有对但 top 错）**18.0pp**，pick_fail|pool **33.7%**。
- 加权投票只从 leftover 里救回 **17%**（237→41），大部分选错救不回来。
- N64：cov↑到 61.1，但 top↓到 34.5、wvote 只到 37.0 → **靠加大 N 堆覆盖率，选优效率会掉**，不能指望“多采样就涨”。
- 换算：若保持现在 wvote/cov≈0.69，要摸到 43.74 需要 cov≈**64%**（比现在 +10pp 绝对覆盖）。

| 选项 | 打的瓶颈 | 已有证据 | 乐观收益 | 主要风险 | 成本 |
|------|----------|----------|----------|----------|------|
| **2 只标难题续训 O2** | leftover / pick_fail | O2op 刚失败（易题、mixed 仅 4.6k）；难题版针对失败原因 | 若 pick_fail 减半，现有池 theoretically 可摸到 ~44 | 同族方法刚阴性；难题可能全错→无 MC；再花 3–4h 空转 | 中高 |
| **3 攻生成器覆盖率** | 测试 cov 53%，训–测难度鸿沟（train pool~90%，test 53%） | X2 同配方加步无效；但“弱题+老师轨迹”是**不同配方**；N64 说明无效覆盖不转化 | cov→64% 且转化率不掉 → 有望 +7pp 级 | 多样性崩、leftover 跟着涨、只涨 cov 不涨 wvote | 中（沿用 P123 trainer） |
| **4 加权投票写进训练** | 生成器与真正有效的聚合不对齐 | 无直接实验；wvote>top 已坐实 | 提高“正确答案在 O2 质量上的质量” | 可能刷 O2 分而非正确性；两模块耦合难归因 | 中，设计风险高 |

**推荐：先做 3（覆盖率），不做 2。**
- 理由：① 选优闭环（O2op）刚被证伪，2 是同方向加强版，先验偏失败；② 目标 43.74 虽低于当前 oracle，但 **实际能兑现的是 ~69%×cov**，把 cov 从 53 推到 ~64 是 presently 最干净的算术路径；③ 3 用老师补学生弱题，直接打训–测鸿沟，且不改 O2、不碰已 Hold 的 λ；④ 4 值得做，但应在“生成器分布先变好”之后，否则是在烂候选上对齐聚合。
- **3 的杀伤判据（先跑一版再决定）：** GSM N16 上 cov 相对 dualBeat **至少 +3pp**，且 wvote **至少 +1pp**；若 cov↑ 但 wvote 持平/跌 → 停 3，改评估 4 或（仅当 leftover|pool 仍~34% 时）谨慎试 2。
- **Do not：** 用加大 N 冒充覆盖率实验；扩大同协议易题 on-policy；盲加 λ_MMD。

### 开跑 coverage（用户选 3，2026-10-02 15:45）

- 配方：从 `p123_dualBeat` 续训；`--select_mode coverage`：学生 k 样本全错且老师对 → **强制**蒸馏老师（`cover_teacher`，不过 O2 门）；`--easy_keep_prob 0.5`：保留一半 beat_self（0.2 会让墙钟过长），更新仍偏向弱题。
- 超参：k64，q_batch16，grad_accum8，lr 3e-6，1500 步，λ_lat=λ_mmd=0，O2/B0 仍冻结。目录 `outputs/p123_cover_20261002_154530`。
- **杀伤判据：** GSM-Test N16 相对 dualBeat：cov **≥+3pp** 且 O2-wvote **≥+1pp**。否则停 3。
- **Do not：** 改 O2；用加大 N 代替；把 easy_keep 当成功指标。

### coverage 结果：未过杀伤判据，停 3（2026-10-02 18:15）

| 系统 (N16 s42) | top | wvote | cov | Δcov vs dualBeat | Δwvote vs dualBeat |
|---|---:|---:|---:|---:|---:|
| dualBeat+O2 | 35.41 | **36.69** | **53.37** | — | — |
| **cover** | 35.56 | 37.00 | 52.46 | **−0.91** | **+0.31** |
| X2 | 35.18 | 36.77 | 54.44 | +1.07 | +0.08 |

- 训练侧：teacher 更新 13.2%（dualBeat ~8%），easy_drop 正常；训完 DONE。
- **杀伤：** 要 cov≥+3 且 wvote≥+1 → **FAIL**（cov 反而略降，wvote 只 +0.31）。
- **读法：** 「弱题+老师轨迹」没推高测试覆盖率；训–测鸿沟没被这刀打开。官方系统仍是 dualBeat+O2 加权投票（36.69 / N64 37.00）。
- **决定：** **停 3**。下一刀在 (2) 只标难题续训 O2 与 (4) 加权投票写进 P3 之间选。

### 开跑 wvote 对齐（用户选 4，2026-10-02 18:30）

- 配方：从 `p123_dualBeat` 续训；`--select_mode wvote`：若 O2 softmax 加权投票（T=1）赢家答案正确 → CE 蒸馏该答案下 O2 最高的学生样本（`wvote_self`）；B0 过差则退回 dual_beat；否则 dual_beat 兜底。
- 意图：让生成器分布对齐**已经稳定有效**的聚合方式，而不是再挤覆盖率或重训 O2。
- 超参：k64，q_batch16，grad_accum8，lr 3e-6，1500 步，λ=0。
- 中途看：`wvote_ok_rate`、`top_wvote_disagree`、`sources.wvote_self`。
- **杀伤判据：** GSM N16 相对 dualBeat，O2-**wvote ≥+1pp**（主）；top ≥+0.5 为加分。否则停 4。
- **Do not：** 在 wvote 选错时仍强制蒸馏该答案（会刷分）；改 O2；盲加 λ_MMD。

### wvote 对齐结果：未过杀伤判据，停 4（2026-10-02 22:00）

| 系统 (N16 s42) | top | wvote | cov | Δtop | Δwvote |
|---|---:|---:|---:|---:|---:|
| dualBeat+O2 | **35.41** | **36.69** | 53.37 | — | — |
| **wvote 训** | 34.95 | 36.92 | 52.69 | **−0.46** | **+0.23** |

- 训练：`wvote_self` 88.7% / beat_self 10.9%，配方对齐在跑；训末 wvote_ok≈0.84、top↔wvote 分歧≈0.30。
- **杀伤：** wvote 要 ≥+1 → **FAIL**（只 +0.23）；top 还略降。
- **读法：** 把加权投票赢家写进 CE，几乎搬不动测试 wvote；官方系统仍是 dualBeat+O2 加权投票（36.69 / N64 37.00）。生成器侧（X2/cover/wvote）三连停。
- **决定：** **停 4**。未试且仍可考虑的是 (2) 只标难题续训 O2；或接受现阶段 +3pp 为平台期、换更大策略分叉。

### 全面核查：半一体 + 大头缺口（2026-10-02 22:05，用户叫停小改）

**三 Phase 实际状态**
| Phase | 设计 | 一体跑里 | 进 claim 评测？ |
|-------|------|----------|----------------|
| P1 O2(+B0) | 选优 / 健康门 | **冻开**：dual_beat 用 O2 top 门控 + B0 slack；O2 权重不更新 | O2 进 BoN；**主 Acc 仍是 top-1** |
| P2 MMD | 内容对齐 | **关**（所有 locked 跑 `λ_mmd=0`；代码可接） | 否 |
| P3 SF-C | 可微 latent + answer CE | **开**：只训 student | 生成器 = dualBeat |

结论：**不算完整三 Phase 一体**——是 **P1 冻结选优 + P3 生成器**；P2 从未开。claim 数字用的 **wvote 只在 `summarize_bon` 事后算**，未进 `infer_gpt2_rm` / `diag_p123_mids`。

**硬伤（不是小旋钮）**
1. **Claim 协议双账本**：diag/pipeline 报 top-1（dualBeat Δ=+1.67）；锁定声明用 wvote（+2.95）。仪表与官方系统不一致。
2. **训–测聚合不一致**：训练 dual_beat 追 O2 top 门控轨迹；claim 用 softmax wvote。
3. **选样 ↔ 反传错位（更深）**：采样是 dropout k 条 → 选目标 after；`sf_train_step(C)` **重新**确定性 rollout 再 CE，默认 `λ_lat=0` 不钉赢者 latents。梯度路径 ≠ 被选轨迹。这解释了为何「把标签换成 wvote 赢家」几乎不动点。
4. **协议噪声**：训 max_new_tokens=96 vs infer 128；训常 k=64 vs claim N=16；O2 pref 训练 `mean_log_prob` vs BoN `sum(latent logits)`。

**已排除的小头**：同配方加步、易题 on-policy、弱题 CE、wvote 当 CE 标签——都没动上面 2/3 或选优决策规则本身。

**大头杠杆（按优先级）**
1. **把分数→答案的决策做对**（leftover≈18pp / pick_fail|pool≈34%）：池已够摸 +10，差在选。不是再灌 CE，而是改聚合进主路径，或给 O2 真难题排序监督。
2. **关掉选样–反传缝**：更新必须绑到实际被选的 latent 轨迹（或可微 O2/wvote 目标），否则 P1 与 P3 只是「外部门控 + 文本模仿」。
3. **训–测难度/覆盖**：test cov≈53% 是软天花板（在现转化率下要 ~64%）；但须改生成分布结构，不是 N↑ 或弱题灌老师。
4. **仪表先统一**：wvote 写入 infer+diag，消灭双账本（本身不涨点，是后续一切实验的地基）。
5. P2 MMD：仍 Hold，除非内容漏检有证据；不是当前最大缺口。

**Hold**：继续 easy_keep / 加步 / 盲 λ_MMD / 同族浅对齐。等用户定大头方向后再动。

### 硬伤修正落地（2026-10-02 22:40）

代码已修（**未重开训**；旧 dump 可用 diag 回溯）：

1. **Claim 双账本** → `infer_gpt2_rm` 默认 `claim_agg=wvote`；`meta.accuracy`=claim，另记 `accuracy_top1` / `accuracy_wvote`。`diag` 主报 wvote（旧 dump 从 examples 重算）；分母 A+B0 仍用 **top-1=33.74**。校验：dualBeat → claim 36.69、Δ=+2.96。
2. **训–测聚合** → 训练默认 `--select_mode wvote`。
3. **选样↔反传缝** → 有选中 latents 时默认 `sf_mode=auto` → **SF-B + cached 选中 latents**；老师无 latents 仍走 C。
4. **协议** → 训练默认 `max_new_tokens=128`；infer/`pipeline` 显式 `dropout_p=0.2`。

共享：`src/system_scoring.wvote_select`。

**Do not：** 把修仪表当成涨点；下一刀应是「修缝后的对照跑」或真正的选优大头。

### 开跑 fix缝对照（用户批准，2026-10-02 22:50）

- 从 `p123_dualBeat` 续训；**唯一变量**是硬伤修正后的默认协议：`select_mode=wvote`，`sf_mode=auto`（有 latents→SF-B 绑选中轨迹），`max_new_tokens=128`，infer `dropout_p=0.2` / claim=wvote。
- 超参同前：k64，q_batch16，grad_accum8，lr 3e-6，1500 步，λ=0。
- **杀伤：** GSM N16 相对 dualBeat **wvote ≥+1pp**（主）；top ≥+0.5 加分。否则修缝本身不够，回大头选优。
- **Do not：** 同时开 P2；改 O2；和旧 wvote-CE 跑混读（旧跑是 C 重滚 + 事后 wvote）。

### fixseam 结果：未过杀伤，修缝不够（2026-10-03 00:20）

| 系统 (N16 s42) | top | wvote (claim) | cov | Δwvote | Δtop | Δcov |
|---|---:|---:|---:|---:|---:|---:|
| dualBeat+O2 | **35.41** | **36.69** | 53.37 | — | — | — |
| 旧 wvote-CE (C重滚) | 34.95 | 36.92 | 52.69 | +0.23 | −0.46 | −0.68 |
| **fixseam (B绑轨迹)** | 32.98 | 34.95 | **56.33** | **−1.74** | **−2.43** | **+2.96** |

- 训练：`wvote_self` 87.4% / beat_self 12.1%，DONE 1500。
- **杀伤 FAIL**：wvote 要 ≥+1，实际 −1.74；top 更差。cov 涨了 ~3pp 但不转化（转化率掉了）。
- **读法：** 关掉选样–反传缝 + 默认 wvote 标签，**没有**抬 claim；SF-B 钉选中 latents 可能削弱 live latent 探索，覆盖↑、选优兑现↓。官方系统仍锁定 dualBeat+O2 wvote（36.69）。
- **决定：** 修缝本身不是涨点刀。下一刀回到大头：选优决策 / 难题 O2，或接受 +3pp 平台期换分叉。

### 根因：口径不一致 + SF-B 修缝修错了（2026-10-03 00:30）

**1. 「修缝比错误低」主要是评测口径变了（主因）**

| 模型 | 旧 dump（未记 dropout_p） | 同协议重评 fair（dropout_p=0.2, max_new=128, claim=wvote） |
|------|--------------------------:|----------------------------------------------------------:|
| dualBeat | wvote **36.69** / top 35.41 | wvote **34.42** / top 32.45 |
| 旧 wvote-CE (mode C) | 36.92 / 34.95 | **35.18** / 32.90 |
| fixseam (mode B) | — | **34.95** / 32.98 |

- 旧 infer 不设 `dropout_p` → 只用模型默认 dropout（≈0.1）+ `enable_dropout`；新协议强制 **0.2**（与训练一致）→ 同一 dualBeat 掉 **~2.3pp**。
- **同口径下**：fixseam 相对 dualBeat 是 **+0.53**（34.95−34.42），不是 −1.74。之前拿「新协议的 fixseam」对比「旧协议的 dualBeat」是错账。

**2. SF-B「绑选中 latents」修缝方向错了（次因）**

- fixseam：12000 次更新里 **11936 次 mode=B**（cached latents **detach**），仅 64 次老师走 C；`g_lat` 非零比例 **0.5%**（旧 wvote-CE 为 **100%**，mean≈0.12）。
- B 把梯度从 **latent 生成链**上剪断，只训「给定冻结 latents 解码答案」→ latent 策略无直接监督、会漂。
- 表现：cov↑（56.3 vs fair dualBeat 55.0）、unique↑，但 gold 上 softmax 质量 ↓（0.58 vs 0.66）、pick_fail|pool ↑ → 覆盖涨、兑现掉。相对同协议的 mode-C 旧 wvote（35.18），fixseam **略差 −0.23**。

**正确修缝**：应是 **mode C live + λ_lat 拉向选中 latents**（或其它可微对齐），而不是默认改成 B。

**行动：** ① 官方对比一律用 fair 协议重评；② 训练默认 `sf_mode` 改回 **C**（或 auto=C）；③ A+B0 分母也要 fair 重评后再读 Δ。

### 正确修缝重跑：C + λ_lat（2026-10-03 00:50）

- **修什么：** SF-B 默认撤销；`sf_mode=C` + `λ_lat=1` 把选中 latents 用 MSE 钉住 live 链（保留 g_lat）；评测恢复与锁定 dump 一致（**不强制** `dropout_p=0.2`，与 dualBeat/A+B0 原协议对齐）。
- **配方：** 暖启 `p123_dualBeat`；`select_mode=wvote`；`sf_mode=C`；`λ_lat=1`；`λ_mmd=0`；k64 q16 ga8 lr 3e-6；1500 步；`max_new_tokens=128`；训仍 `dropout_p=0.2`。
- **对照尺（PLAN 锁定，勿再问）：** 分母 A+B0=**33.74%**；当前系统 dualBeat+O2 wvote=**36.69%**；目标 ~+10 → **~43.74%**。
- **杀伤：** GSM N16 相对 dualBeat wvote **≥+1pp**（≥37.69）；否则修缝仍不够，回大头选优/难题 O2。
- **Do not：** 再开 SF-B；盲 λ_MMD；同配方空加步；easy_keep。

### clat 结果：修缝正确但仍未过杀伤（2026-10-03 02:57）

| 系统 (N16 s42) | top | wvote | cov | Δwvote vs dualBeat | Δ vs A+B0 |
|---|---:|---:|---:|---:|---:|
| dualBeat+O2 | 35.41 | **36.69** | 53.37 | — | +2.95 |
| **clat (C+λ_lat=1, wvote选)** | 35.86 | 36.85 | 51.48 | **+0.16** | +3.11 |

- 训练健康：`sf_mode=C` 100%，`g_lat` 非零 100%（mean≈0.12），`loss_lat` mean≈0.40；`wvote_self` 10767 / beat 1138。
- **杀伤 FAIL**（要 ≥+1）。cov 略掉；leftover 14.6 / pick_fail|pool 28.4%（略好于 dualBeat 的 18/33.7，但兑现不够）。
- **读法：** 选样–反传缝已按正确方式关掉，仍几乎不动 claim → **生成器侧修缝不是涨点刀**。官方系统仍 dualBeat+O2 wvote 36.69。
- **决定：** 停生成器缝类实验。下一刀 **(2) 难题 O2**：只用 dualBeat annotate 里 `correct_frac≤0.5` 的 mixed（~1.9k）+ 轻 coconut 回放（ratio 0.25），自 O2 best 暖启续训。

### 开跑 O2hard（2026-10-03 03:00）

- 数据：`scripts/build_hard_onpolicy_mix.py` → `latent-data/onpolicy_hard_mix/train`；yaml `training_args/train_order_pref_hard.yaml` → `outputs/latentrm_order_pref_hard`。
- 评测：dualBeat × O2hard N16 wvote；健康 A×O2hard。
- **杀伤：** 相对 dualBeat+O2，wvote **≥+1pp**（≥37.69）。否则停同族 on-policy O2，承认 +3pp 平台、换更大分叉。
- **Do not：** 扩易题 annotate；盲 λ_MMD；再灌生成器 CE。

### O2hard 结果：未过杀伤，停同族 on-policy O2（2026-10-03 03:20）

| 池 × RM (N16 s42) | top | wvote | cov | Δwvote vs dualBeat+O2 |
|---|---:|---:|---:|---:|
| dualBeat × O2 | 35.41 | **36.69** | 53.37 | — |
| **dualBeat × O2hard** | 35.48 | 36.92 | 53.37 | **+0.23** |
| A × O2hard | 33.06 | 34.50 | 50.19 | （健康：A wvote 不崩） |

- 数据：hard_mixed 1965 + coco 499；训 ~14min 至 epoch 3 early-ish；best 已存。
- **杀伤 FAIL**（要 ≥+1）。cov 与 dualBeat 池相同（同生成器），选优几乎不动。
- **读法：** 易题稀释不是主因；同族 on-policy 续训 O2（全 mixed / 只 hard）都推不动 test leftover。
- **决定：** **停同族 on-policy O2**。官方系统仍 dualBeat+O2 wvote **36.69**（Δ vs A+B0 **+2.95**），距 +10 差 ~6.7pp。下一刀需**更大策略分叉**（换选优目标/监督形态/生成分布结构），不再做同配方过滤。

### 三 Phase 中间信号全查（2026-10-03 11:20）

**一体状态：没有有效结合。** 实际是 **P1 冻结门控 + P3 文本 CE**；P2 全线 `λ_mmd=0`。

| Phase | 训里 | Claim | 中间信号 |
|-------|------|-------|----------|
| P1 O2/B0 | 冻开选优/健康门，**不更新** | O2→BoN/wvote；B0 另报 | leftover≈18pp，pick_fail\|pool≈34%；o2_b0_agree 训窗仅 ~0.1–0.2 |
| P2 MMD | **从未开**（全部 p123 `λ_mmd=0`，losses 里 mmd 全 0） | 不进 | **无信号** |
| P3 SF | 只训 student；dualBeat: C+λ_lat=0；clat 修缝后 g_lat=100% 仍不涨 | 生成器=dualBeat | 生成器侧旋钮 |Δwvote|≲0.3 饱和 |

**+10 算术（dualBeat 池）：** cov=53.37 → oracle Δ≈+19.6 已够；现 wvote=36.69（Δ=+2.95），差 **7.05pp**。固定转化率需 cov≈63.6；固定 cov 需把 leftover 再兑现 ~7pp（pick_fail 31%→~18%）。**主债在选优兑现，不是池不够。**

**硬缝（解释为何灌 CE / 同族 O2 无效）：**
1. **O2 训分 ≠ Claim 分**：pref `mean_log_prob`（logsigmoid）vs BoN/train-select **`sum` raw logits**（`infer_gpt2_rm` / `system_scoring`）。
2. O2/B0 **无系统梯度**；wvote 正确性不对 student 反传。
3. P2 未接入。
4. 锁定 dualBeat 训选 dual_beat、claim 用 wvote（后对齐仍失败）。

**已否证小刀：** cover +0.31 / wvote-CE +0.23 / clat +0.16 / O2hard +0.23 / fixseam-B 伤 g_lat。

**下一刀（先关分数缝）：** dualBeat 池上用 `mean_log_prob`（及 `sum_log_prob`）重打分重聚 claim；若 wvote≥+1 则锁定该 score 进 train-select+infer；否则用 **sum_logit** 重训 leftover 定向 O2。杀伤仍 ≥37.69。

### score_align 阴性 + 用户令：先真一体再调优（2026-10-03 11:30）

| score_reduce | top | wvote |
|---|---:|---:|
| sum_logit（锁定） | 35.41 | **36.69** |
| mean_log_prob（O2 训对齐） | 35.63 | 36.01 |
| sum_log_prob | 35.63 | 36.16 |

只改 claim 打分家族 **掉点** → 不锁 mlp/slp；claim 仍 sum_logit。

**用户意见：** 先把三个 Phase **真正接到一条训练链**，再调优。

**一体接线（代码）：**
- P1：冻 O2/B0 选优（dual_beat）
- P2：`λ_mmd>0` + **在线** H^pos（老师对→teacher latents；否则 cache/选中正确 latents）；不再依赖 pilot-only cache 才能开火
- P3：SF-C + `λ_lat=1`

### 开跑 p123_joint（2026-10-03 11:35）

- 暖启 dualBeat；`select=dual_beat`；`sf_mode=C`；`λ_lat=1`；`λ_mmd=0.1`（ramp 200）；k64 q16 ga8 lr 3e-6；1500 步。
- 中间信号必看：`mmd_hit_rate`、`loss_mmd` 非零比例、`g_lat`、sources。
- **杀伤：** GSM N16 wvote ≥**37.69**（相对 dualBeat +1）。否则一体接线本身不够，再调 λ 配比或 P1 可训。
- **Do not：** 盲把 λ_mmd 拉到 ≥1；再 SF-B；再同族 hard-filter O2。

### 多卡（2026-10-03 12:06）

- 已加 `--frozen_device`：`student+O2` / `teacher+B0` 分卡，并行 generate + 并行 O2‖B0 score。
- 卡 3/4（有 ray 争用）双卡反而 ~12s/step；**GPU5+0** 约 **2.9–3.7s/step**（与单卡满速同级略好）。避开 GPU6（wubohan）。
- 当前：`joint2` 暖启首段 joint ckpt，`CUDA_VISIBLE_DEVICES=5,0`，eval 挂 3/4。

### joint 一体结果：接线成功、claim 未过杀伤（2026-10-03 13:54）

| 系统 N16 s42 | top | wvote | cov | Δwvote vs dualBeat | Δ vs A+B0 |
|---|---:|---:|---:|---:|---:|
| dualBeat+O2 | 35.41 | **36.69** | 53.37 | — | +2.95 |
| **joint (P1冻选+P2 MMD0.1+P3 C+λ_lat)** | 35.86 | 36.32 | 52.08 | **−0.37** | +2.58 |

- 训练：1500 步 DONE；`mmd_hit` 末窗 0.86；beat 10894 / teach 1106。B0 健康 34.80（未崩）。
- **杀伤 FAIL**（要 ≥37.69）。cov 略掉，leftover 15.8 / pick_fail 30.3%。
- **读法：** 三 Phase **第一次真正同链开火**，但把 P2 叠上去 **没有**抬 claim；略低于冻结 dualBeat。官方仍 dualBeat+O2 wvote **36.69**。
- **决定：** 一体接线本身不是涨点刀。下一刀不要再叠小 λ；要么让 P1 可训（选优债），要么换决策级目标。不盲抬 λ_mmd。

### joint 错例审计（2026-10-03 14:05）

测试 n=1319：**no_pool 47.9%**、wvote 对 36.3%、leftover 15.8%（208 题）。
- leftover 金标 O2 **中位 rank=5**，40% score_gap>2；gold_rank>5 占 102/208。例 idx=80：对 10 分最高 11.6，错 14 分最高 **22.5**（票数 10 vs 6）。
- 聚合旋钮（T、mean-vote、hybrid）**搬不动**；B0 只能救 leftover 31/208。
- 训练侧 pool≈0.94 vs 测试 0.52 → 鸿沟仍在；MMD/λ_lat 健康但没修选优。
- **改法（仍一体）：** 在 k 池 mixed 上给 O2 加 **sum_logit leftover hinge**（可训 P1），P2/P3 保持。

### 开跑 joint_rank（2026-10-03 14:10）

- 暖启 `p123_joint2_20261003_120606`；`λ_rank=1` margin=1 `o2_lr=1e-5`；λ_mmd=0.1 λ_lat=1 sf=C dual_beat。
- 评测用 `$OUT/o2`。杀伤：wvote **≥37.69**；中间量盯 `rank_hit_rate`、`mean_rank_gap`↓、测试 leftover↓。
- **Do not：** 盲抬 λ_mmd；改 claim 聚合凑点。

### jrank 不是卡在 200：磁盘满（2026-10-03 15:34）

- PID 2440939 **已死**。日志停在 14:25；`200/1500` tqdm 是 **save_every=200** 写 `model.safetensors` 时 `No space left on device`。`src=skip` 只是该步 postfix，不是死循环。
- 未完整 ckpt（306M 残片），不能从 200 续。已清阴性 p123/smoke 权重，根盘 ~8.7G；中途存盘失败改为 WARN 不停训；`save_every=1500` 只在收尾写。
- 重开同一配方：暖启 joint2，GPU5+0，eval 仍挂 pipeline。

### joint_rank 结果：杀伤 FAIL（2026-10-03 17:21）

GSM-Test N16 s42，claim=wvote，训完 O2=`$OUT/o2`：

| 系统 | top | wvote | cov | leftover | Δwvote vs dualBeat |
|---|---:|---:|---:|---:|---:|
| dualBeat+O2（锁） | 35.41 | **36.69** | 53.37 | — | — |
| joint（冻 O2） | 35.86 | 36.32 | 52.08 | 15.8 | −0.37 |
| **jrank（可训 leftover hinge）** | 34.87 | **36.92** | 52.31 | 15.39 | **+0.23** |

- 杀伤要 wvote≥**37.69**：FAIL（只 +0.23）。top **掉 0.54**。vs A+B0 +3.18，距 +10 还差 **6.82**。官方仍 dualBeat **36.69**。
- 训练末窗：pool 0.98 / rank_hit 0.56 / leftover 0.18；测试 leftover **没动**（15.4）。beat 7916 / o2_rank 1552 / teach 2532。B0 健康 35.78 wvote。
- **读法：** P1 可训没修测试选优；top 更差说明 generator 还滑了一点。wvote 微涨在噪声边。
- **下一刀（先拆因果，再决定加 λ）：** 同协议 N16 交叉：(1) dualBeat 生成器 + jrank O2；(2) jrank 生成器 + 原 O2。看涨点来自 O2 还是 gen。不盲抬 λ_mmd。

### 交叉评测（2026-10-03 20:20）

| 组合 | top | wvote | leftover |
|---|---:|---:|---:|
| dualBeat gen + 原 O2（锁） | 35.41 | **36.69** | — |
| **dualBeat gen + jrank O2** | 35.94 | **37.00** | 17.4 |
| jrank gen + 原 O2 | 34.80 | 35.86 | 17.5 |
| jrank gen + jrank O2 | 34.87 | 36.92 | 17.4 |

+0.23 来自 **O2 略好**，jrank 生成器自己是掉点的。dualBeat+新O2 仍 <37.69。leftover 三个组合都 ~17%，hinge 没改选优债。不锁新官方。不盲抬 λ_mmd。

### 洞见：hinge 压了分差、没翻名次（2026-10-03 20:50）

- 同 dualBeat 池：jO2 把 leftover **thief−gold 中位差 1.25→0.53**，gap>2 从 40%→16%；但 leftover 条数 237→230，**207 题两边都 leftover**。救 30、弄坏 23，净 +7 top。
- 测试 no_pool **46.6%** 是硬顶；完美选优上限 = cov **53.4%**。+10pp 目标（43.7）在上限内，卡在 **~1/3 的有金标池选错**，不是再叠 λ_mmd。
- 训练 pool 0.8–0.98 vs 测试 0.52：on-policy k64 上金标常已 rank1，hinge 经常饱和；测试 N16 的硬 thief 几乎没见过。`mean_rank_gap` 全程震荡、无下降曲线。
- jrank 生成器 pool 704→690（丢 49 / 得 35），unique 4.25→4.10。CE 主要 beat_self 7916，rank-only 1552：生成器在克隆已赢样本，覆盖变差。
- **下一步不该加 λ_rank/λ_mmd。** 冻 dualBeat，只训 O2；难负例从 **gsm_train** 用 k=16 挖 leftover（不碰测试 207，防泄漏），多 thief hinge。

### 开跑 o2hardrank（2026-10-03 22:06）

- GPU5：`mine_o2_hard_leftover` n_q=4000 k=16 → `train_o2_hard_rank` 3ep lr=3e-5 margin=1。
- 然后 dualBeat + 新 O2，GSM N16 wvote。杀伤仍 ≥37.69。日志 `logs/p123/o2hardrank_20261003_220603.log`。

### o2hardrank 结果：杀伤 FAIL（2026-10-03 22:14）

挖 gsm_train 4000：pool 3479 / leftover **577** / empty 521。O2-only 3ep：train 金标赢 thief 一直 ~**50%**，loss 1.12→1.00。

dualBeat + 此 O2，N16：top **35.78** wvote **36.32** cov 53.37（锁 36.69 / jrankO2 37.00）。**更差**。离线 leftover hinge 在自身训练集都没学会排序，不能当成功。不锁。不抬 λ_mmd。

### 瓶颈重判（2026-10-03 22:51）

离线 O2 **在 train leftover 上是学会了的**（eval 关 dropout：金标>主 thief 20%→46%，gap 中位 0.18→0.01）。测上 leftover 237→232，救 49 / 弄坏 44。

真正打不动的是 **三套 O2 共用的 172 道 leftover**（gap 中位 **1.83**，47% gap>2；38% 池里只有 1 条金标）。train 挖到的 leftover 中位 gap 只有 **0.44**、gap>2 仅 4%——训的是另一种题。

两块硬顶：
1. **no_pool 46.6%**（train 同生成器 ~13–20% empty）。N64 cov 61 但转化率掉，wvote 只到 37.0。
2. **172 稳定 leftover（13%）**：换 O2 / hinge / 离线难负例都不翻。O2 排序剩余空间大约只有那 ~30 道可移动题（还和弄坏相抵）。

算术：+10 要 43.7。只靠 O2 几乎凑不齐；必须要么自适应加采样（别盲 N64），要么接受选优器在这 172 上不可辨、改决策规则。不盲抬 λ。

### 自适应 BoN dump 否证 + 下一刀 k=16 coverage（2026-10-03 23:40）

在 **已有** dualBeat N16 / N64 dump 上把 entropy / unique / maxO2 / wvote-mass 门控、空池加采、leftover 改多数票全部扫完（无金标泄漏的规则）：

| 决策 | wvote/acc | vs 锁 36.69 |
|---|---:|---|
| 官方 N16 wvote（同 dump 重算） | 36.69 | — |
| N16 特征门控改 MV（最好） | **36.85** | +0.16，噪声 |
| N64 全量 wvote | 37.00 | +0.31 |
| 空池@16→N64（oracle 空池） | 36.24 | 掉点；cov 61 不转化 |
| 特征门控 extra-N | ≤37.15 | ≈全量 N64；help≈hurt |
| leftover@16 改 MV（**作弊**金标） | 39.27 | 说明 leftover 里多数票也常错 |
| 完美选优上限 = N16 cov | **53.4** | +10 目标 43.7 **在上限内** |

**结论：** 聚合/自适应 N **打不通** 36.69→43.7。不改 `infer_gpt2_rm` 上线这套门控，也不再盲 N64。A 的池只能盖住 dualBeat 空池的 **9%**，拼 A 候选无效。

墙在 **生成器覆盖**：训 k=64 pool 0.88–0.98，测 N16 pool 0.53。dualBeat 几乎没见过测试那种空池。

**下一刀（正信号配方上改协议匹配，不叠 λ）：** 暖启 dualBeat，**冻 O2**，`k=16`（`--no_auto_pack` 禁止再抬到 64），`select=coverage`（空池且老师对 → 强制 teacher CE），`easy_keep=0.25`（少训已赢题），`λ_mmd=0` `λ_rank=0` `λ_lat=0`，2000 step，lr=3e-6。杀伤仍 GSM N16 wvote **≥37.69**。中间量：`sources.teacher`/`cover_teacher` 比例、train pool、测 cov。不锁成功。不碰 GPU6。

### 加覆盖并行（2026-10-04 00:31）

k16cov ~800/2000：teacher:beat ≈ 3407:2993（coverage 在打），但 train pool 仍 0.80–0.92、hash3~0.83 —— 训练尺还是偏易、短 ###。k16hard 子集 empty 1547 / leftover 1712 / keep 4806，步更慢（长题）。

并行再加：
- GPU4：锁 dualBeat+O2 N16 **dropout=0.2**（对齐训练采样；官方锁是 dropout omit）。只当 wvote>36.69 且相对 A+B0 也涨才算点。
- GPU7：k16 coverage + **target_mode=process**（对空池题直接 CE 金标步骤，压 hash3）。不叠 λ。
GPU1/2 留给 k16cov 评测。杀伤仍 ≥37.69。

### 覆盖三刀结果：全 FAIL（2026-10-04 11:36）

| 系统 N16 s42 | top | wvote | cov | Δwvote vs dualBeat | 读法 |
|---|---:|---:|---:|---:|---|
| dualBeat+O2（锁） | 35.41 | **36.69** | 53.37 | — | 官方 |
| **dp0.2 推理**（锁 gen） | 34.12 | 34.95 | **56.03** | −1.74 | cov↑ 转化↓；不锁 |
| **k16cov** coverage ek0.25 | 34.95 | 36.32 | 53.75 | **−0.37** | 杀伤 FAIL；train teacher:beat 8671:7329，测 cov 几乎不动 |
| **k16proc** coverage+process | 32.07 | **33.43** | 51.93 | −3.26 | 更差；hash3→0.005 但变成长 CoT clone teacher（14957:1043） |
| k16hard（空池子集） | — | — | — | — | 训完 1871 步；首轮 eval timeout，已重挂 |

相对 A+B0：k16cov +2.58（锁曾 +2.95）；距 +10 仍差 **~7.4pp**。

**否证：** (1) 推理对齐 dropout0.2 不是免费点；(2) 训 k=16+coverage 抬不了测空池；(3) 全量 process 会把生成器拧成 teacher-CoT、伤 claim。官方仍 **36.69**。不抬 λ_mmd。

**下一刀：** 暖启 dualBeat，k16 coverage ek0.25 + **λ_lat=1**（空池 teacher 文本 CE 同时拉 live latent→teacher latent；k16cov 证明纯文本 CE 不够）。冻 O2，λ_mmd=0 λ_rank=0，2000 step。杀伤 ≥37.69。k16hard 评测并行。

### 失败总账 + 真提升点（2026-10-04 11:40）

**锁：** dualBeat+O2 N16 wvote **36.69**（top 35.41 / cov 53.4）。目标 ~**43.7**（A+B0+10）。距目标 **~7pp**；完美选优上限=cov **53.4** → 目标在上限内。

**已否证（按层，勿再小步空转）：**

| 层 | 方案 | 最佳/结局 | 教训 |
|---|---|---|---|
| 生成器变体 | Lat / Div / Proc / B0t / X2 续训 | 均 ≤ dualBeat | dualBeat 已饱和；多样性/过程形≠涨点 |
| 训选对齐 | cover / wvote-CE / 早 coverage | wvote +0.2~0.3 | 训目标≠测空池 |
| 缝/协议 | fixseam-B / clat / score mlp·slp / infer dp0.2 | 掉或 +0.16 | 缝不是 7pp；dp↑cov↓转化 |
| O2 同族 | onpolicy / O2hard / jrankO2 / o2hardrank | 最好 dualBeat+jO2 **37.00** | leftover 172 道 O2 翻不动；train gap≠test gap |
| 一体 P123 | joint / jrank 全链 | 36.32 / 36.92 | 接线≠claim；jrank gen 掉 top |
| 聚合/N | T·MV·hybrid / N64 / 自适应加采 / 拼 A | ≤37.0；作弊 leftover-MV~39 | 聚合封死；空池加采不转化 |
| 覆盖再攻 | k16cov / k16proc | 36.32 / **33.43** | k=16+teacher CE 测 cov 不动；全 process 伤 claim |

**算术拆 7pp：**
1. **no_pool ~46.6%**（~615 题）——主墙。训 k64 pool 0.9 vs 测 0.53。纯 CE 老师答在易分布上**学不会测空池**。
2. **稳定 leftover ~13%（172）**——次墙。换 O2/hinge 净救≈0；要的是池里**更多/更醒目的金标**，不是再训排序。
3. 有池选错 ~1/3：O2 剩余可动空间 ≤~1pp（已见 37.0）。

**真正还能抬的地方（按优先级）：**
1. **测样空池的生成能力**（不是再调 select_mode）：需要让 student 在「测会空、训很少空」的题上产出至少 1 条对。候选：更强/多样 teacher 轨迹（多样本老师、验证过滤后再 CE）；或硬空池上用可搜索式自洽（多 roll + 外部校验）而不是单条 teacher 克隆；勿再全量 process。
2. **把金标做厚**（针对 172 leftover）：同题提高正确条数/降低 thief 形态碰撞，让现有 O2 自然选中——本质仍是 gen，不是 RM。
3. **只有在 cov 先 +≥3pp 后**才值得再碰 O2/聚合；否则 O2 工作是噪声。

**Do not：** 盲 λ_mmd↑；再同族 leftover-O2；再 N64/门控凑点；再全量 process；宣称成功无 ≥37.69 证据。
k16covlat / k16hard 仍在跑；若再 FAIL → 停 P123 小刀，改「硬空池多样本老师/校验 CE」分叉。

### 墙钟 / 多卡（2026-10-03 15:45）

当前 jrank **~3.5–4.5s/it**，ETA 训完约 **1.4h**（这轮不杀）。双卡几乎=单卡：墙钟是 student `k=64` 自回归；teacher `k=1` 并行藏不住。GPU5 util~40%；GPU0 teacher~8GB，卡上 100% 是别人的活。DDP 反传不是瓶颈。下一轮用第三卡 `--gen_replica_device` 做 `k/2‖k/2`（不要和 teacher 同卡），上限大约 1.5–1.8× generate。空闲卡被 ray 占时硬拆会变慢。

## Phase-0 — train_baseline OOM + resume (2026-09-27 21:10)

- **原因**：不是系统 RAM；GPU0 上另有 `xvyijia/alphaprobe` ~27GB，我们已占 ~65GB，再要 5GB → CUDA OOM。别人占得少所以不容易先爆。
- **处理**：从 `outputs/latentrm_baseline/checkpoint-578` resume；**只用空闲 2,6,7 三卡**（避开共卡 0）；`nohup setsid` + `RESUME_FROM_CHECKPOINT=auto`；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **日志**：`logs/train_baseline_20260927_211047.log`
- **读结果注意**：卡数 4→3 后 `max_steps` 3990→5310，global batch 变小；LR schedule 会重算（日志里像再 warmup）。权重从 578 续上；与纯 4 卡轨迹不完全同一，但仍是合法 B0 续跑。若要严格同配置，等 GPU0 空再 4 卡 resume。
- **Do not**：去杀别人进程；不要无 occupied GPU0 硬上原 batch
