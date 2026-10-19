> 论文叙事稿（匿名会议体例）：[`PAPER_NARRATIVE.md`](PAPER_NARRATIVE.md)。本文件为 Experiments 数字骨架。  
> **权威主表 / 消融（mmd_cold 锁）：** [`paper_tables_mmd_cold/MAIN_TABLES.md`](paper_tables_mmd_cold/MAIN_TABLES.md) · [`ABLATION_TABLES.md`](paper_tables_mmd_cold/ABLATION_TABLES.md) · 锁 [`p123/paper_mmd_cold/LOCK.md`](p123/paper_mmd_cold/LOCK.md)。

# Experiments（主文草稿）

本节验证两件事是否构成可测的协同：（i）生成器侧后训练是否提高潜轨迹候选池质量；（ii）顺序偏好打分器是否比逐步对比 LatentRM 更能把同池增益变现，且该优势在（i）之后是否扩大。除非另注，协议统一为：潜步数 $T{=}6$，scorer 推理关闭 dropout，GSM8K-Test 为主表，BoN 主看 $N{\in}\{16,64\}$，主指标 Top-1；不跨生成器底座比较绝对准确率。

**方法接口（实验相关部分）。** 生成器后训练（p2）在线采样 $k$ 条学生轨迹与 1 条 teacher 轨迹，以 O2 为主门控、B0 为健康约束选取 CE 目标，经 SF-C 回传；可选内容视图 MMD。打分器 O2 在逐步 CE 之上增加顺序偏好目标；B0 为同 annotate 下的官方逐步对比 LatentRM。训练门控与测试 BoN 共用同一套 latent 轨迹表示。

---

## 4.1 生成器后训练策略（固定 COCONUT）

**Setup.** 起点固定为官方 COCONUT-GPT2（124M）。对照：冻结（G0）；同起点、同优化步数 3000 的 SLPO/RLOO（G3）与 Dropout-GRPO（G4）；本文 **mmd_cold**（G-Ours：冷启 dual_beat+SF-C+λ_MMD=0.05，1500 step；锁 `outputs/p123_mmd_cold`）。评测打分统一用 **B0**（故意不接 O2，避免把 scorer 增益算进生成器）。指标：Coverage@$N$、B0 Top-1。

**Result.** GSM8K-Test，$N{=}16$，3-seed 均值（dropout=None）：

| Method | Cov@16 | B0 Top-1 | Δ Top-1 vs G0 |
|--------|-------:|---------:|--------------:|
| G0 Frozen | 50.67 | 32.88 | — |
| G3 SLPO (3000) | 50.90 | 32.88 | +0.00 |
| G4 Dropout-GRPO (3000) | 51.23 | 32.85 | −0.03 |
| dualBeat (−MMD） | 53.60 | 34.12±0.55 | +1.24 |
| G-Ours **mmd_cold** | **54.03** | **35.03±0.20** | **+2.15** |

逐 seed B0 Top-1（mmd_cold）：35.18 / 35.10 / 34.80；dualBeat：33.66 / 34.72 / 33.97。旧暖启 p2mmd05 已降级。全矩阵见 `paper_tables_mmd_cold/MAIN_TABLES.md`。G3/G4 训练曲线上 `eval_reward` 停滞在 0.352–0.356。

**Takeaway.** 在相同骨干下，p2（mmd_cold）提高 Cov 与 B0 Top-1；短预算 RL 对照未超过冻结。该结论**不等于**优于官方 SLPO/GRPO 全预算（其步数约 $4{\times}10^4$–$5{\times}10^4$，且每步 `num_generations` 小于本设定的 $k$）。

---

## 4.2 后训练跨底座迁移（COCONUT / CODI / CoLaR）

**Setup.** 各底座内「冻结 → +最终配方」对照；评测接 O2（GPT-2 底座）或 Greedy/Cov/Maj（CoLaR）。禁止用跨底座绝对分差解释策略收益。CODI 上的 B0/O2 为 COCONUT 表征上训练权重的零样本迁移（须标明）。

**Result.** GSM8K-Test，$N{=}16$（除非另注）：

| Base | Policy | 主指标 | Δ (within base) |
|------|--------|-------:|----------------:|
| COCONUT | Frozen | O2 Top1 32.07 | — |
| COCONUT | **mmd_cold** | O2 **36.01**（3-seed **36.32±0.40**） | **+3.94**（s42；3-seed vs 冻≈+2.6） |
| CODI | Frozen | O2 41.14±0.35（N1 42.08） | — |
| CODI | **CODImmd** | O2 **42.13±0.84**（N1 **43.29**） | O2 **+0.99** / N1 **+1.21**† |
| CoLaR fixed_t | Frozen | Greedy 22.44 / Cov@16 35.48 / Maj 23.73 | — |
| CoLaR fixed_t | **colar_mmd_cold**（answer_beat+SF-C+λ_MMD=0.05） | Greedy **36.62** / Cov@16 **48.45** / Maj **37.07** | **+14.2 / +13.0 / +13.3** |

†`codi_samepool_fair/`：两生成器各 generate-once B0+O2（mism=0），seeds 42–44；B0/O2 为零样本 COCONUT RM。池内 O2−B0：冻 +1.31 / CODImmd +1.87。Cov 略降（54.79→53.30），N1/Maj/O2 同向正。旧单点 s42 +0.38 与本表 s42 一致。CoLaR：`colar_mmd_cold/`。

**Takeaway.** 最终配方在 COCONUT 上增益最大；CODI 上 3-seed O2 约 +1pp、N1 +1.2（零样本 RM）；CoLaR fixed_t 底座内大幅正 Δ（不可与官方 `###` 混表）。

---

## 4.3 潜轨迹打分器（固定候选池）

**Setup.** **generate-once** 同池：一次采样后用 B0/O2/SWIFT/HSRM 打分（`scorer_axis_mmd_cold_samepool/`，`answers_sha1` mism=0）。种子 $42$–$46$。主指标 Top-1。

**Result A — 同池 Top-1（GSM，$N{=}16$，均值±std）。**

| Pool | B0 | O2 | SWIFT | HSRM |
|------|---:|---:|------:|-----:|
| Frozen COCONUT | 32.72±0.96 | **33.72±1.03** | 33.69±0.75 | 33.60±0.75 |
| **mmd_cold**（正式同池） | 34.95±0.26 | **36.38±0.60** | 36.16±0.37 | 36.32±0.63 |

配对 O2−B0：冻池 **+1.00pp** $[0.17,1.85]$（历史 5-seed）；**mmd_cold 池 mean Δ ≈ +1.43pp**（5-seed；同池已断言）。O2 vs SWIFT/HSRM Top-1 仍接近（平手区）。

**Result B — 机制与诊断。** Valid 上 order_pair_acc / hierarchy：O2 0.972 / 0.894；B0 0.860 / 0.769；SWIFT 0.449 / 0.242；HSRM 0.475 / 0.441。`auc_cw` 上 O2 相对 SWIFT/HSRM 显著或边缘显著（见 `tables/paired_ranking_*`）。Order-Decision：O2 `strict_win@N16` 在 adjacent/segment/full_reverse 上均高；同骨干下 O2 相对 B0 的 adjacent `pair_acc` 增量约 +0.20（CI 不含 0）。SWIFT/HSRM 的顺序盲与置换不变聚合一致。

**可比性。** 训练划分与候选池对齐，但 O2 为预训练 GPT-2 级骨干，SWIFT/HSRM 为随机初始化小头（约 1.5K / 1.8M）——Top-1 名次不得解读为方法绝对优劣。train/valid/test 问题级重叠为 0。

**Takeaway.** （1）同池 O2−B0 Top1 为正；救援不对称支持相对 B0 的变现补足。（2）相对 SWIFT/HSRM：full-O2 的 `auc_cw` 有优势（~+3pp），但 Top1 接近平手——下一阶段任务是把辨别力转成 Top1，而非否定 O2。（3）`latent_only` 消融显示该 `auc_cw` 优势依赖文本上下文；保留 full-O2，补 `text-only` 与选择端校准以优化输入/打分配方。Order-Decision 仍支持顺序机制可转化为决策。

---

## 4.4 协同读法（对应主 claim）

将 4.1–4.3 合读（而非并列罗列）：

1. **生成器侧**：mmd_cold 在 COCONUT 上抬 Cov/B0 Top1；CODI 上弱正（+0.38）；CoLaR fixed_t 上底座内大幅正 Δ。  
2. **打分器侧**：正式 mmd_cold 同池上 O2−B0 ≈ +1.43pp；O2≈SWIFT/HSRM（Top1）。  
3. **接口**：训练门控使用 O2（+B0 健康约束）；消融显示整门控贡献最大（nogate −4.2pp vs Full）。

据此，主文可陈述：潜 TTS 的后训练与轨迹打分应在同一潜表示上联合设计；仅优化其一会低估可实现增益。**不**把「闭环节拍协同」写成最强 claim（见 PLAN diagnosis-first）。不陈述的内容见下。

---

## 限制（Experiments 内显式写出）

- CoLaR 正结果仅在 **fixed_t 自洽协议**下相对该协议冻结；不可与官方 `###`+max_L 绝对分混表。  
- 4.1 的 RL 对照为同 3000-step 设定，非官方全预算复现。  
- CODI：matched-protocol 3-seed；跨生成器非 hash 同池；RM 为零样本。  
- 不得用跨底座绝对 Acc 差论证策略收益；不得写「O2 Top-1 全面优于 SWIFT/HSRM」。


---

## 实验补全审计（2026-10-19 改锁后）

| 板块 | 主 claim 是否够写 | 状态 |
|------|:--:|------|
| 4.1 生成器策略（COCONUT） | ✅ | mmd_cold 3-seed B0 齐；G1 SFT / G2 RFT **缺** |
| 4.2 跨底座 | ✅ | CODI fair 3-seed O2 **+0.99** / N1 **+1.21**；CoLaR fixed_t 正 Δ |
| 4.3 打分器同池 | ✅ | `scorer_axis_mmd_cold_samepool/` mism=0；5-seed |
| 方法消融 | ✅ | Full=mmd_cold；nogate −4.2；−MMD −0.51 |
| 全预算 SLPO/GRPO、SFT/RFT | ❌ | 非本轮阻塞项 |

**结论：** 最终策略 **mmd_cold** 下的三主实验测量已齐；文档以本文件 + `paper_tables_mmd_cold/` 为准。
