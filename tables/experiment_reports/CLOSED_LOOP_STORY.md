> 论文叙事稿（匿名会议体例）：[`PAPER_NARRATIVE.md`](PAPER_NARRATIVE.md)。本文件为 Experiments 数字骨架。

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
| G-Ours **mmd_cold** | **~54.0** | **35.03** | **+2.15** |

逐 seed B0 Top-1（mmd_cold）：35.18 / 35.10 / 34.80。旧暖启对照 p2mmd05 约 34.3（3-seed）已降级，不作主锁。G3/G4 训练曲线上 `eval_reward` 停滞在 0.352–0.356。

**Takeaway.** 在相同骨干下，p2（mmd_cold）提高 Cov 与 B0 Top-1；短预算 RL 对照未超过冻结。该结论**不等于**优于官方 SLPO/GRPO 全预算（其步数约 $4{\times}10^4$–$5{\times}10^4$，且每步 `num_generations` 小于本设定的 $k$）。

---

## 4.2 后训练跨底座迁移（COCONUT / CODI）

**Setup.** 在 COCONUT 与 CODI 上分别做「冻结 → +p2」底座内对照；评测接 O2；禁止用跨底座绝对分差解释策略收益。CODI 上的 B0/O2 为 COCONUT 表征上训练权重的零样本迁移（须在文中标明）。

**Result.** GSM8K-Test，$N{=}16$，O2 Top-1：

| Base | Policy | O2 Top-1 | Δ (within base) |
|------|--------|---------:|----------------:|
| COCONUT | Frozen | 32.07 | — |
| COCONUT | **mmd_cold** | **36.01**（3-seed **36.32±0.40**） | **+3.94**（vs s42 冻；3-seed vs 冻 O2≈+2.6） |
| CODI | Frozen | 40.94 | — |
| CODI | CODIp2 | 42.00 | **+1.06**（同池） |

锁与全矩阵见 `p123/paper_mmd_cold/LOCK.md`。旧 p2mmd05（暖启）GSM N16 O2≈34.80，已非主系统。CODI 侧数字未因换锁重跑。

**Takeaway.** 在更强的 CODI 冻结点上，p2 仍给出正的同池 Top-1 增量，说明增益不是仅来自弱 COCONUT 的过拟合空间。

---

## 4.3 潜轨迹打分器（固定候选池）

**Setup.** 在固定 dump 上比较 Maj、B0、O2、SWIFT-Latent、HSRM-Latent；禁止不同 scorer 吃不同采样池。种子 $42$–$46$（5-seed）。主指标 Top-1；辅助：对错排序 `auc_cw`、顺序机制 `order_pair_acc` / `hierarchy_consistency`，以及内容固定、仅置换顺序的 Order-Decision 诊断。

**Result A — 同池 Top-1（GSM，$N{=}16$，均值±std）。**

| Pool | B0 | O2 | SWIFT | HSRM |
|------|---:|---:|------:|-----:|
| Frozen COCONUT | 32.72±0.96 | **33.72±1.03** | 33.69±0.75 | 33.60±0.75 |
| p2 池（历史 dump=`p2mmd05`）† | 34.44±0.38 | 35.92±0.28 | 35.88±0.48 | **36.06±0.37** |

†系统主锁已改为 **mmd_cold**（§4.1–4.2）；本表为固定历史候选池上的 scorer 轴，未因换锁重采样。mmd_cold+O2 系统尺见 `paper_mmd_cold/LOCK.md`（GSM N16 3-seed **36.32±0.40**）。

配对 O2−B0：冻池 **+1.00pp** $[0.17,1.85]$；p2 池 **+1.49pp** $[0.59,2.38]$（均显著）。O2 vs SWIFT/HSRM 的 Top-1 配对 CI 跨 0。

**Result B — 机制与诊断。** Valid 上 order_pair_acc / hierarchy：O2 0.972 / 0.894；B0 0.860 / 0.769；SWIFT 0.449 / 0.242；HSRM 0.475 / 0.441。`auc_cw` 上 O2 相对 SWIFT/HSRM 显著或边缘显著（见 `tables/paired_ranking_*`）。Order-Decision：O2 `strict_win@N16` 在 adjacent/segment/full_reverse 上均高；同骨干下 O2 相对 B0 的 adjacent `pair_acc` 增量约 +0.20（CI 不含 0）。SWIFT/HSRM 的顺序盲与置换不变聚合一致。

**可比性。** 训练划分与候选池对齐，但 O2 为预训练 GPT-2 级骨干，SWIFT/HSRM 为随机初始化小头（约 1.5K / 1.8M）——Top-1 名次不得解读为方法绝对优劣。train/valid/test 问题级重叠为 0。

**Takeaway.** （1）同池 O2−B0 Top1 为正；救援不对称支持相对 B0 的变现补足。（2）相对 SWIFT/HSRM：full-O2 的 `auc_cw` 有优势（~+3pp），但 Top1 接近平手——下一阶段任务是把辨别力转成 Top1，而非否定 O2。（3）`latent_only` 消融显示该 `auc_cw` 优势依赖文本上下文；保留 full-O2，补 `text-only` 与选择端校准以优化输入/打分配方。Order-Decision 仍支持顺序机制可转化为决策。

---

## 4.4 协同读法（对应主 claim）

将 4.1–4.3 合读（而非并列罗列）：

1. **生成器侧**：mmd_cold（及历史 p2 配方）在固定骨干、短预算对照下提高 Cov 与（B0 尺）Top-1；在 CODI 上仍有同池正 Δ。  
2. **打分器侧**：同池 O2−B0 为正；续训后 gap 增大。  
3. **接口**：训练门控使用 O2（+B0 健康约束），因此 scorer 进入生成器更新，而不仅用于测试；消融显示整体 RM gate 贡献最大（nogate −3.7pp）。

据此，主文可陈述：潜 TTS 的后训练与轨迹打分应在同一潜表示上联合设计；仅优化其一会低估可实现增益。**不**把「闭环节拍协同」写成最强 claim（见 PLAN diagnosis-first）。不陈述的内容见下。

---

## 限制（Experiments 内显式写出）

- CoLaR（Llama+LatentHead）可作为非 GPT-2 冻结参照；当前 p2 移植未稳定，**不纳入跨架构正结果**。  
- 4.1 的 RL 对照为同 3000-step 设定，非官方全预算复现。  
- COCONUT 冻/p2 的部分 O2 对比非逐条同池，写作须标注。  
- 不得用跨底座绝对 Acc 差论证策略收益；不得写「O2 Top-1 全面优于 SWIFT/HSRM」。


---

## 实验补全审计（2026-10-17）

| 板块 | 主 claim 是否够写 | 状态 |
|------|:--:|------|
| 4.1 生成器策略（COCONUT，短预算 RL 对照） | ✅ | 3-seed B0 矩阵齐；G1 SFT / G2 RFT **缺** |
| 4.2 跨底座（CODI） | ✅ | 同池 +1.06；COCONUT 冻/p2 部分非逐条同池（已标） |
| 4.3 打分器同池 + 机制 + Order-Decision | ✅ | 5-seed 齐；BCE ckpt 缺（数字已有） |
| 协同读法（p2 门控用 O2） | ✅ | 有；组件消融（去 MMD / 去 order-pref 等）**弱** |
| CoLaR 第三架构 p2 | ❌ | 冻结可参；p2 未稳定，不进正结果 |
| 全预算 SLPO/GRPO、冻 CODI×四 scorer | ❌ | 附录/扩展，非主闭环阻塞项 |

**结论：** 生成器×打分器主闭环对主文 **已基本补全**；未补全的是经典 SFT/RFT 对照、部分消融、CoLaR 正结果与全预算 RL——不挡闭环故事，但挡「生成器轴方法大满贯」表述。
