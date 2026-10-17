# 面向潜变量思维链的顺序偏好学习与整体 Rollout 对齐

**匿名作者**

> 中文审阅稿。英文备份：`PAPER_NARRATIVE_en.md`。

---

## 摘要

潜变量思维链（latent CoT）用连续轨迹 $H=(h_1,\ldots,h_T)$ 替代中间文本，只解码最终答案。此时有三类核心失效模式：（i）**顺序歧义**——内容相近的轨迹仍可能因果结构错误；（ii）**token 对应歧义**——潜步之间未必存在一一索引对齐；（iii）**自回归 rollout 漂移**——teacher-forcing 与自生成轨迹分布不一致。我们提出分阶段训练框架，使每类失效对应一类目标：对顺序扰动负样本做**层级偏好学习**（i）；在去时序的 content view 上做**位置无关集合对齐**（Chamfer，可选 MMD）（ii）；对完整轨迹做**整体自 rollout 对齐**——任务损失加集合匹配，而非按位置回归（iii）。第一阶段得到顺序偏好 LatentRM（O2），同时用于 Best-of-$N$ 选解与生成器续训（p2）门控。实验在固定 $T{=}6$、同池评测下表明：p2 相对同预算短训 RL 提升 COCONUT 的覆盖与 BoN，并在 CODI 上有底座内正增益；O2 相对官方逐步 LatentRM（B0）Top-1 为正，且 p2 池上配对差距更大；相对轻量 outcome scorer，有证据支持的是排序（$\mathrm{auc}_{cw}$）与顺序诊断，而非 Top-1 全面胜出。输入消融进一步说明：应保留 O2 的辨别力并改进「分数→选择」，而不是放弃顺序感知打分。

---

## 1. 引言

连续潜变量 CoT 把多步推理存成向量而非 token，既去掉了文本过程奖励，也带来标准 next-token / 扁平 outcome 损失难以覆盖的结构化失效。

**顺序歧义。** 两条轨迹可以局部内容相似，但步骤时间排列不同。打乱或反转推理顺序往往仍保留袋级语义，却破坏逻辑依赖。只预测最终对错的 scorer，对潜过程监督是不充分的。

**Token 对应歧义。** 学生轨迹与 teacher 轨迹未必按索引对齐：长度、切分与表示都可能不同。强制 $\|\hat h_t - h_t^{\mathrm{target}}\|^2$ 预设了未必成立的对应关系；若作用在有序流上，还会与真正的顺序监督互相打架。

**自回归 rollout 漂移。** 训练时 teacher forcing 条件在金标前缀上，推理时条件在模型自身历史上。把自生成状态逐步回归到 teacher 状态，同样假定索引对齐，且未对准「完整轨迹」层面的分布差距。

因此我们用三类互补目标组织学习——**不是简单相加，而是分阶段课程**。主线为
\[
\mathcal L
=
\mathcal L_{\mathrm{task}}
+
\lambda_{\mathrm{pref}}\mathcal L_{\mathrm{pref}},
\]
即任务监督加层级顺序偏好。集合对齐与整体 rollout 对齐仅在顺序偏好被验证有效后再加入，并各自作用在正确的表示上（有序分数 / 去位置 content view / 完整轨迹比较）。

得到的顺序偏好 LatentRM（O2）既用于 BoN 解码，也用于门控生成器续训（p2），使探索与变现共享同一潜接口。实验用固定标尺与锁定候选池隔离坐标轴，既报告相对 B0 的变现增益，也诚实报告相对容量不对齐轻量 scorer 的 Top-1 非显著结果。

**贡献。**
1. 将潜 CoT 失效分解为顺序、对应、rollout 漂移，每类对应一类目标，并给出明确的 staged 开启规则。
2. 带等级顺序扰动与按序列长度校准 margin 的层级偏好 LatentRM，以及顺序 / 层级诊断指标。
3. 位置无关集合对齐与整体自 rollout 对齐：避免把顺序泄漏进置换不变损失，也避免宣称实现了 Self-Forcing 的 DMD/SiD/GAN 目标。
4. 将 O2 同时用作 BoN scorer 与 p2 训练门控，并在 COCONUT/CODI 上给出同池、同预算证据。

---

## 2. 相关工作

**潜 / 连续 CoT。** COCONUT 类生成器及潜长度、蒸馏等变体多在固定解码规则下报准确率。我们强调对潜轨迹的结构化监督，以及训练期与测试期 scorer 的耦合。

**过程奖励与潜奖励。** 逐步 LatentRM、偏好 / outcome PRM，以及 SWIFT/ELHSR、HSRM 等潜适配器提供 BoN 分数。扁平 outcome 损失不编码「错的程度」；我们统一采用**同池**比较。

**曝光偏差与自 rollout。** Scheduled sampling 与 Self-Forcing 式训练通过条件在模型自身输出上缩小训推差距。我们采纳「对齐完整自生成轨迹」这一启发，**不**声称复现 Self-Forcing 的 DMD、SiD 或 GAN 公式。

**集合匹配。** Chamfer 与 MMD 对置换不变。我们只在剥离可恢复时序信息的 content view 上使用，绝不拿来替代有序流上的顺序偏好。

---

## 3. 方法

### 3.1 问题设定

模型生成潜轨迹
\[
H=(h_1,\ldots,h_T),\qquad h_t\in\mathbb{R}^{d},
\]
再解码答案。LatentRM $s_\phi$ 对完整轨迹打标量分（含问题编码）。BoN 取 $\arg\max_i s_\phi(H^{(i)})$。每个训练样本可构造：正确顺序 $H^+$；带等级 $r_k$ 的顺序扰动负样本 $\{H^-_k\}_{k=1}^{K}$；teacher-forced 轨迹 $H^{\mathrm{TF}}$；自生成轨迹 $H^{\mathrm{SF}}$。

完整目标族为
\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{task}}
+
\lambda_{\mathrm{pref}}\mathcal L_{\mathrm{pref}}
+
\lambda_{\mathrm{set}}\mathcal L_{\mathrm{set}}
+
\lambda_{\mathrm{roll}}\mathcal L_{\mathrm{rollout}}.
\]
**分阶段。** 不同时打开全部项。最稳主线是 $\mathcal L_{\mathrm{task}}+\lambda_{\mathrm{pref}}\mathcal L_{\mathrm{pref}}$。集合与 rollout 项在顺序偏好有效后再引入，并分别消融（Chamfer vs MMD；有无 rollout）。

### 3.2 模块一：层级顺序偏好（主创新）

**扰动必须发生在重新编码之前。** 负样本通过对*渲染后的推理流*（或诱导它的离散潜 token 布局）做顺序扰动，再跑 backbone 得到完整轨迹，最后用 $s_\phi$ 打分。我们**不**直接置换已经算好的 hidden states：那会鼓励模型识别「异常向量」，而不是理解逻辑顺序错误。

**带等级的扰动层级。** 每个负样本必须带 `negative_rank`（多负样本层级训练不得静默退化成 flat triplet）：

| 等级 | 扰动 | 难度 |
|-----:|------|------|
| 1 | 相邻交换 | 最轻 |
| 2 | 局部片段交换 | 较轻 |
| 3 | 多片段重排 | 中等 |
| 4 | 完整反转 | 较重 |
| 5 | 重排 + 删除/替换等复合扰动 | 最重 |

同等级负样本之间不强行排序。

**层级偏好损失。** 设 margin 为 $m_{\mathrm{order}}$、$m_{\mathrm{hier}}$，
\begin{align}
\mathcal L_{\mathrm{pos}}
&=
\sum_k
\bigl[
m_{\mathrm{order}}
+
s_\phi(H^-_k)-s_\phi(H^+)
\bigr]_+,
\\
\mathcal L_{\mathrm{hier}}
&=
\sum_{r_i<r_j}
\bigl[
m_{\mathrm{hier}}
+
s_\phi(H^-_j)-s_\phi(H^-_i)
\bigr]_+,
\\
\mathcal L_{\mathrm{pref}}
&=
\mathcal L_{\mathrm{pos}}
+
\lambda_{\mathrm{hier}}\mathcal L_{\mathrm{hier}}.
\end{align}
因此 $s_\phi$ 既要让正确顺序高于一切扰动，也要让较轻扰动高于较重扰动——学习的是「错的程度」，而不只是二元对错。

**Margin 按真实长度校准。** 分数差对长度 $T$ 求平均，局部扰动 $q$ 个位置时约有 $\Delta_{\mathrm{avg}}\approx (q/T)\Delta_{\mathrm{local}}$。故用经验 gap（相邻交换与全反转下 $s_\phi(H^+)-s_\phi(H^-)$ 的中位数与分位数）设定 margin，例如 $m_{\mathrm{order}}=\alpha\cdot\mathrm{median}(\Delta)$，$\alpha\in[0.25,0.75]$，$m_{\mathrm{hier}}=\beta m_{\mathrm{order}}$，$\beta\in[0.1,0.5]$。Gap 尺度估计只是校准工具，**不能**证明某扰动一定可学。

**实例化（O2）。** O2 = $\mathcal L_{\mathrm{task}}$（逐步监督）+ $\mathcal L_{\mathrm{pref}}$ 训练的 LatentRM，用于 BoN，并作为下文生成器续训的主门控。B0 = 官方逐步对比 LatentRM（无层级顺序偏好）。

### 3.3 模块二：位置无关集合对齐

当 token 无固定对应时，按索引回归 $\|\hat h_t-h_t^{\mathrm{target}}\|^2$ 是错误归纳偏置。Chamfer 与 MMD 对置换不变，若直接作用在*有序流*上会与模块一冲突。因此构造 content view
\[
C(H)=\{\tilde h_1,\ldots,\tilde h_T\},
\]
去掉可恢复的时序信息。禁止的「假 content view」包括：恒等、clone/detach、均匀缩放、均值中心化、可逆的逐特征对角仿射——它们仍会把顺序泄漏进集合损失。可用的包括：去掉显式时间位置编码、独立 content 投影 / 瓶颈、无位置重编码、token masking 或随机池化。

**默认：双向 Chamfer**（自生成与 teacher 的 content view）
\[
\mathcal L_{\mathrm{Chamfer}}
=
\frac1T\sum_i\min_j\|\tilde h_i^{\mathrm{SF}}-\tilde h_j^{\mathrm{TF}}\|^2
+
\frac1T\sum_j\min_i\|\tilde h_j^{\mathrm{TF}}-\tilde h_i^{\mathrm{SF}}\|^2.
\]
**可选：多尺度 RBF MMD**，同一 content view。第一轮实验分开比较「无集合损失 / Chamfer / MMD」，不同时打开两者。（不把 MMD 方法学归因于无关的生成流论文；若引用 MoFlow，只借鉴「对候选集合建模」的想法，而非其具体 MMD 实现。）

### 3.4 模块三：整体自 Rollout 对齐

Teacher forcing 与推理的条件前缀不同。我们**不做** $\sum_t w_t\|\hat h_t^{\mathrm{SF}}-h_t^{\mathrm{TF}}\|^2$，以免再次强加虚假一一对应。受 Self-Forcing「用自身历史作后续条件」启发——但**不**宣称 DMD/SiD/GAN——我们对*完整*轨迹对齐：
\[
\mathcal L_{\mathrm{rollout}}
=
\mathcal L_{\mathrm{task}}(H^{\mathrm{SF}})
+
\lambda_{\mathrm{set}}
\mathcal L_{\mathrm{set}}\bigl(C(H^{\mathrm{SF}}),C(H^{\mathrm{TF}})\bigr).
\]
自 rollout 既要完成任务，又要在集合意义下贴近 teacher 内容；顺序仍由 $\mathcal L_{\mathrm{pref}}$ 负责。可选的 TF/SF 前缀课程是后续工程：仅用于推理、带 `@torch.no_grad()` 的 `generate()` 不能直接反传整条 rollout。

### 3.5 复用顺序感知 scorer 的生成器续训（p2）

固定或周期性刷新 $s_\phi$ 后，每步采样 $k$ 条学生轨迹与 1 条 teacher 轨迹，用 O2 打分（B0 作健康约束），选 CE 目标，经 SF-C 更新生成器。可选 content-view MMD 在该循环内实现模块二。发布配方 **p2mmd05** 在已验证系数下打开 MMD。测试时 BoN 使用同一 $s_\phi$——探索与变现共享 scorer。

### 3.6 每项「刻意不做」什么

| 诱惑写法 | 为何避免 |
|----------|----------|
| 无 rank 的 flat triplet | 丢掉「错的程度」；多负样本层级必须要求 `negative_ranks` |
| 事后打乱 hidden states | 学异常检测，不学顺序 |
| 在有序流上做集合损失 | 泄漏 / 抵消顺序监督 |
| 按索引 SF↔TF 回归 | 假定虚假对应；误称 Self-Forcing |
| 一次打开全部 $\lambda$ | 无法分离哪类失效被解决 |

---

## 4. 实验协议

**数据。** 主表 GSM8K-Test；附录 GSM-Hard、MultiArith。  
**潜长度。** 主协议 $T{=}6$；最长新 token $128$；BoN $N\in\{16,64\}$；scorer 推理关 dropout。  
**种子。** Scorer 轴 $\{42,\ldots,46\}$；生成器 RL 矩阵 3 seed。  
**统计。** 配对 bootstrap CI；Top-1 不一致对上 McNemar。  
**坐标轴隔离。** 生成器表固定 B0 标尺；scorer 表锁定候选池（`answers_sha1`）；跨底座只报底座内 $\Delta$。  
**模块一诊断。** Order-pair 准确率、层级一致性、Order-Decision（内容固定、只扰动顺序）、BoN 池上 $\mathrm{auc}_{cw}$。

---

## 5. 实验

遵循 staged 规则：主表以模块一（O2）为主；模块二/三目前以 p2 内可选集合 / rollout 项出现，主表不宣称完整的 Chamfer–MMD 烘培对比。

### 5.1 模块一 → BoN 变现（O2 vs B0 / 轻量 scorer）

**设置。** 锁定 dump；Majority / B0 / O2 / SWIFT / HSRM；5 seed；$N{=}16$。

**Top-1（均值$\pm$std）。**

| 池 | B0 | O2 | SWIFT | HSRM |
|------|---:|---:|------:|-----:|
| 冻结 | $32.72\pm0.96$ | $\mathbf{33.72\pm1.03}$ | $33.69\pm0.75$ | $33.60\pm0.75$ |
| p2 | $34.44\pm0.38$ | $35.92\pm0.28$ | $35.88\pm0.48$ | $\mathbf{36.06\pm0.37}$ |

配对 O2$-$B0：**$+1.00$** [$0.17,1.85$]（冻）；**$+1.49$** [$0.59,2.38$]（p2）。O2 vs SWIFT/HSRM 的 Top-1 CI 跨 0。

**顺序与排序。** Full O2 相对 SWIFT 的 $\mathrm{auc}_{cw}$ 约 +$3.3$ / $+2.8$ pp（冻 / p2）。Order-Decision 与层级指标上 O2 远高于纯 outcome 的 SWIFT/HSRM。救援不对称：p2 池上偏向 O2 相对 **B0**；相对 SWIFT/HSRM 大致对称。O2 与极小 SWIFT/HSRM 头容量不对齐——Top-1 打平不能当架构优劣判决。

**输入探针。** Latent-only O2 的 Top-1 仍接近 full O2 / SWIFT，但 $\mathrm{auc}_{cw}$ 落到 SWIFT 一带。读法：保留层级偏好，改进文本与 latent 如何共同喂给 $s_\phi$，并校准选择——而不是丢掉模块一。

### 5.2 在生成器续训中复用模块一（p2）

**设置。** COCONUT-GPT2；冻结 / SLPO / Dropout-GRPO / p2mmd05；同 **3000** 步；评测只用 **B0**。

| 方法 | Cov@16 | B0 Top-1 | $\Delta$ |
|--------|-------:|---------:|---------:|
| 冻结 | 50.67 | 32.88 | — |
| SLPO (3000) | 50.90 | 32.88 | $+0.00$ |
| Dropout-GRPO (3000) | 51.23 | 32.85 | $-0.03$ |
| **p2mmd05** | **53.88** | **34.32** | **$+1.44$** |

**迁移。** CODI 冻结 $40.94$ $\rightarrow$ CODIp2 $42.00$（**$+1.06$**，同池，O2 Top-1）。

同预算短训 RL 未超过冻结；p2 超过。这不是官方全预算 RL 复现（约 $4$–$5\times 10^4$ 步）。

### 5.3 模块二 / 三现状

p2mmd05 内的 content-view MMD 是当前已跑通的集合对齐路径；干净的 Chamfer vs MMD 对比，以及可反传的 rollout 课程，仍按 staged 后续（附录）。非 GPT-2 移植未稳、缺 SFT/RFT 对照等，一并延期并写明非声明。

---

## 6. 讨论

框架的价值在于**职责分离**：层级偏好管顺序；集合损失在剥离视图上管袋级内容；整体 rollout 管训推轨迹失配。经验上，模块一已得到相对 B0 更好、且具顺序敏感行为（扁平 outcome 头所缺）的 scorer；把它复用进 p2 后，在固定 B0 标尺下池质量上升。后续工作是完成模块二/三的 staged 消融，并把 O2 的 $\mathrm{auc}_{cw}$ 优势更稳地转成 Top-1——而不是把三项损失糊成一团。

**影响。** 方法面向潜 CoT 下的数学 BoN。风险包括过度依赖自动奖励与 BoN 算力。不涉及双用途生物 / 化学应用。

---

## 7. 结论

潜 CoT 以特有方式失败：顺序错、假一一对应、rollout 漂移。我们将层级偏好、位置无关集合对齐与整体自 rollout 对齐分别对应这三类问题，分阶段推进而非盲目加和，并用得到的顺序感知 LatentRM 同时做 BoN 与生成器续训。现有证据支持顺序偏好核心及其在 p2 中的复用；集合与 rollout 模块在上线时遵循同一设计约束。

---

## 致谢

*评审阶段隐去。*

---

## 参考文献

*待补。*

---

## 附录（大纲）

- A. 扰动目录、等级定义、margin 校准图。  
- B. Order-Decision / 层级 / $\mathrm{auc}_{cw}$ 全表。  
- C. Content view 可接受性检查；Chamfer vs MMD 试点。  
- D. Rollout 课程细节；关于 Self-Forcing 目标的非声明。  
- E. 完整 BoN 矩阵；可复现（种子、哈希、配置）。
