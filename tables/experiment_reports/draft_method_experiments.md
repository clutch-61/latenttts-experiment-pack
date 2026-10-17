# Methods and Experiments (draft)

> 初稿：只写方法和实验。术语对齐 You et al., *Parallel Test-Time Scaling for Latent Reasoning Models*（LatentTTS / LatentRM）。数字均为 seed 42（多 seed 处另注明）。主协议：测试时奖励模型**不开** dropout。未写入摘要、引言、相关工作。

> **主线（2026-10 锁定）：生成器 ↔ 打分器协同，非单点涨点。** 贡献 1 = 续训拓宽潜空间探索（Greedy↑、Coverage↑）；贡献 2 = 顺序偏好打分器是变现探索收益的必要补充（O2 相对 B0 的变现优势在续训后 ≈2×）。不再用「B0 变现率下降」作论据——那是一个跨口径混比；严谨同池下 B0 变现率基本持平，真正硬的是 **O2−B0 差距在续训后翻倍**。

---

## 5. Method

### 5.1 Setup and notation.

We start from a frozen latent reasoner in the COCONUT family (Hao et al., 2024): given a question $x$, the model emits a short sequence of continuous latent thoughts $h_{1:L}$ ($L{=}6$ in our runs) and then decodes an answer $a$. Parallel test-time scaling samples $N$ trajectories and aggregates them. Following You et al. (2025), a Latent Reward Model (LatentRM) $g_\phi$ scores a prefix of latents; the trajectory score is the sum of step-wise logits. Best-of-$N$ picks the highest-scoring trajectory. We also report, on the same pool, (i) a score-weighted vote over answers, (ii) unweighted majority vote, and (iii) coverage (whether any of the $N$ answers is correct).

Two LatentRMs appear below, both GPT-2 scoring heads on latent positions:

- **B0** ($g_{\mathrm{B0}}$): the official LatentTTS LatentRM. At each latent step it does a softmax over the $N$ candidate traces (contrastive CE); labels are Monte Carlo estimates of whether that thought leads to a correct answer. This is **not** next-token language-model CE.
- **O2** ($g_{\mathrm{O2}}$): our scorer. Per-trajectory BCE plus a hierarchical preference loss on permutation-corrupted **successful** traces. Do not call this the official You et al.\ LatentRM.

A separate **BCE-only** recipe (no contrastive softmax over $N$, no permutation preference) is a correctness-supervised control; it is not B0.

The **baseline system** is frozen COCONUT plus B0. The **proposed system** updates the generator (Section 5.2–5.3) and, at test time, aggregates with O2.

### 5.2 Dual-gated targets for generator updates.

Let $\pi_\theta$ be the current generator (student) and $\pi_0$ the original COCONUT checkpoint (teacher). For each training question we draw a greedy teacher trajectory $(h^{\mathrm{t}}, a^{\mathrm{t}})$ and $k{=}64$ student trajectories. A student trajectory may become the imitation target only if **all** of the following hold:

1. **Answer correctness.** The decoded answer matches the gold label.
2. **Preference beat.** O2 assigns the student a strictly higher score than the teacher (margin $0$).
3. **Health gate.** B0 does not rate the student worse than the teacher by more than slack $\delta{=}0.5$. This blocks updates that chase O2 while looking unhealthy under B0.

If no student passes the gate, we imitate the teacher when the teacher is correct and O2 prefers it to the best incorrect student by a margin; otherwise the example is skipped. Among passing students we prefer traces that still emit a textual CoT suffix when such a trace exists.

The gate uses frozen O2 and B0; neither is updated in the generator stages below.

### 5.3 Self-forcing through latent rollout.

Imitating a stored latent sequence creates a train–test gap: training conditions on a target latent prefix, while inference rolls latents from the model’s own hidden states. We close this with **self-forcing in mode C**: the student rolls $L$ latent vectors differentiably (each latent is the last-layer hidden state at the previous step, matching COCONUT generation), then we apply cross-entropy on the answer tokens after that live rollout. Gradients therefore flow through the latent steps, not only through a detached prefix.

### 5.4 Light content-view alignment (second stage).

After the gated self-forcing stage, we continue from that checkpoint with the **same gate**, plus a small Maximum Mean Discrepancy (MMD) term on a **content view** of the latents: a bottleneck projector (width $64$) maps rolled student latents and a positive latent set to a representation that is not used as an invertible copy of the ordered trajectory. The positive set is the teacher latent when the teacher is correct, otherwise a cached successful latent or the gated student latent. The MMD weight is $\lambda_{\mathrm{mmd}}{=}0.05$, linearly warmed from $0$ over the first $100$ steps. We do **not** add other latent-alignment losses and do **not** train O2 in this stage.

### 5.5 Two-stage schedule.

Both stages use GSM8K train, learning rate $3{\times}10^{-6}$, dropout $0.2$ on the student, batch of $16$ questions, $k{=}64$ samples, gradient accumulation $8$, at most $128$ new tokens, seed $42$, $1500$ optimizer steps per stage. Stage 1: $\lambda_{\mathrm{mmd}}{=}0$. Stage 2: $\lambda_{\mathrm{mmd}}{=}0.05$ from the Stage-1 checkpoint.

---

## 6. Experiments

### 6.1 Benchmarks and protocol.

We evaluate on GSM8K-Test ($1319$ problems), GSM-Hard (same questions with scaled numbers), and MultiArith, following You et al. (2025) and Hao et al. (2024). Unless noted: seed $42$, latent length $6$, at most $128$ new tokens.

- **$N{=}1$:** deterministic greedy decoding of the generator. No LatentRM.
- **$N\in\{4,16,64\}$:** sample $N$ trajectories; score with the system’s LatentRM; report Best-of-$N$ / weighted vote / majority / coverage on the **same** pool.

**Main protocol:** LatentRM dropout is **off** at test time (no MC-dropout). Table 2 additionally turns MC-dropout $p{=}0.2$ on for the *same* proposed generator and $g_{\mathrm{pref}}$, matching the sampling hyperparameter used to train LatentRM in You et al.

The baseline system uses frozen COCONUT and B0. The proposed system uses the Stage-2 generator and frozen O2. This is a **system** comparison (generator and scorer both change). $N{=}1$ isolates the generator.

### 6.2 Main results: baseline vs.\ proposed (dropout off).

Cells are Best-of-$N$ / weighted vote / majority / coverage (\%). $N{=}1$ is generator accuracy only.

| Dataset | $N$ | Baseline (COCONUT + B0) | Proposed (updated generator + O2) |
|---------|-----|----------------------------------------|-----------------------------------------------------|
| GSM8K-Test | 1 | 34.42 | 35.18 |
| GSM8K-Test | 4 | 32.37 / 32.52 / 32.98 / 42.15 | 35.25 / 36.01 / 35.18 / 44.20 |
| GSM8K-Test | 16 | 33.74 / 33.66 / 33.97 / 51.10 | 34.80 / 36.77 / 35.94 / 52.84 |
| GSM8K-Test | 64 | 31.31 / 33.74 / 33.74 / 59.59 | 34.72 / 36.92 / 36.16 / 60.88 |
| GSM-Hard | 1 | 7.66 | 7.97 |
| GSM-Hard | 4 | 6.68 / 6.98 / 7.51 / 9.71 | 7.66 / 8.35 / 8.19 / 10.62 |
| GSM-Hard | 16 | 7.21 / 7.06 / 7.36 / 12.14 | 8.19 / 8.27 / 8.04 / 12.90 |
| GSM-Hard | 64 | 6.53 / 7.21 / 7.44 / 14.57 | 7.51 / 8.27 / 8.04 / 14.57 |
| MultiArith | 1 | 80.86 | 85.69 |
| MultiArith | 4 | 77.41 / 78.28 / 78.10 / 86.21 | 81.72 / 83.62 / 83.62 / 89.83 |
| MultiArith | 16 | 79.66 / 82.07 / 80.86 / 91.38 | 85.52 / 87.41 / 86.21 / 92.07 |
| MultiArith | 64 | 80.17 / 83.10 / 80.69 / 93.97 | 85.52 / 87.76 / 85.69 / 94.66 |

The system improves on every row. But the two components move different axes: the generator widens **exploration** (Section 6.3), and the scorer is what converts that exploration into accuracy (Section 6.4). We therefore unpack the system gain into its two parts rather than reporting a single blended number.

### 6.3 The generator widens exploration (Greedy and Coverage).

$N{=}1$ greedy and coverage measure the generator **without any scorer** and are thus the clean readout of the generator update. On GSM8K-Test:

| Generator readout | Frozen COCONUT | Stage-2 generator | $\Delta$ |
|---|---|---|---|
| Greedy $N{=}1$ | 34.42 | 35.18 | **+0.76** |
| Coverage@16 | 51.10 | 52.84 | **+1.74** |
| Coverage@64 | 59.59 | 60.88 | **+1.29** |

Greedy is up (+0.76 on GSM, +4.83 on MultiArith), so the generator itself answers more questions correctly even before sampling. Coverage is up at every $N$: the pool of sampled trajectories contains a correct answer more often. In other words, the update genuinely broadens the searchable space of correct reasoning; it is not merely re-ranking what the frozen model already produces.

### 6.4 The scorer is what converts exploration into accuracy (O2 vs B0, same pool).

Coverage gains do not automatically become accuracy gains: a scorer must pick the correct trajectory out of the larger pool. We measure this on **identical candidate pools** (same generator, same seed, same sampling; only the scorer changes, so coverage and majority vote coincide by construction). Three seeds, GSM8K-Test $N{=}16$, dropout off:

| seed | O2 − B0 top1, frozen pool | O2 − B0 top1, Stage-2 pool |
|---|---|---|
| 42 | +0.38 | +1.44 |
| 43 | +0.83 | +1.22 |
| 44 | +1.13 | +2.05 |
| **mean** | **+0.78** | **+1.57** |

On the frozen pool, O2 edges out B0 by a small margin (+0.78 mean). On the Stage-2 pool that margin **doubles** (+1.57), and the direction is stable across all three seeds. Equivalently, the **conversion-rate advantage** of O2 over B0 (gap divided by coverage) goes from +1.5pp to +2.9pp. This is the empirical core of our second contribution: post-training shifts the latent distribution enough that the official contrastive LatentRM B0 cannot fully monetize the new correct trajectories, whereas the order-preference scorer O2 can. (Note: B0’s *absolute* conversion rate is roughly flat, not significantly down; the claim is the **widening O2−B0 gap**, not a B0 collapse.)

### 6.5 The strategy transfers to a stronger backbone (CODI).

To show the update is a general latent-space mechanism rather than a COCONUT-specific overfit, we rerun the same recipe on CODI, a stronger compressed-thought backbone. Within each backbone we compare before/after only; we do not attribute the COCONUT→CODI absolute gap to our method. GSM8K-Test $N{=}16$ top1:

| CODI backbone | top1 |
|---|---|
| Frozen CODI + B0 | 40.03 |
| Frozen CODI + O2 | 40.94 |
| **CODIp2 + O2** | **42.00** |

On an already-strong backbone (frozen CODI+B0 = 40.03), swapping in O2 alone gives +0.91 and the generator update adds another +1.06. On the identical CODIp2 pool, B0 scores 40.49 and O2 scores 42.00 (+1.51, same pool). The exploration–monetization pattern therefore persists across backbones.

### 6.6 Control: MC-dropout on vs.\ off (proposed system only).

Same Stage-2 generator and O2. $N{=}1$ does not use the scorer, so the two columns coincide.

| Dataset | $N$ | Dropout off | MC-dropout $p{=}0.2$ |
|---------|-----|-------------|----------------------|
| GSM8K-Test | 1 | 35.18 | 35.18 |
| GSM8K-Test | 4 | 35.25 / 36.01 / 35.18 / 44.20 | 32.37 / 33.43 / 33.66 / 44.12 |
| GSM8K-Test | 16 | 34.80 / 36.77 / 35.94 / 52.84 | 34.04 / 35.94 / 35.86 / 56.56 |
| GSM8K-Test | 64 | 34.72 / 36.92 / 36.16 / 60.88 | 31.77 / 33.97 / 35.94 / 66.11 |
| GSM-Hard | 1 | 7.97 | 7.97 |
| GSM-Hard | 4 | 7.66 / 8.35 / 8.19 / 10.62 | 7.44 / 7.36 / 7.51 / 9.86 |
| GSM-Hard | 16 | 8.19 / 8.27 / 8.04 / 12.90 | 7.74 / 7.81 / 8.35 / 13.51 |
| GSM-Hard | 64 | 7.51 / 8.27 / 8.04 / 14.57 | 7.21 / 7.59 / 7.89 / 15.71 |
| MultiArith | 1 | 85.69 | 85.69 |
| MultiArith | 4 | 81.72 / 83.62 / 83.62 / 89.83 | 77.59 / 79.14 / 79.48 / 88.62 |
| MultiArith | 16 | 85.52 / 87.41 / 86.21 / 92.07 | 83.10 / 84.66 / 84.14 / 93.97 |
| MultiArith | 64 | 85.52 / 87.76 / 85.69 / 94.66 | 82.24 / 87.59 / 83.62 / 96.03 |

MC-dropout $p{=}0.2$ usually **raises coverage** and **lowers** Best-of-$N$ relative to dropout-off scoring of the same generator. We therefore treat dropout-off as the deployment protocol for the proposed system, and $p{=}0.2$ as a control aligned with LatentTTS training-time sampling, not as the headline number.

### 6.7 What we do not claim.

Gains versus the baseline are consistent across the three sets but modest on GSM8K Best-of-$N$ (about one point at $N{=}16$); weighted vote moves more. We do not claim that content-view MMD is the main driver: Stage 1 already uses the dual gate and self-forcing; Stage 2 only adds $\lambda_{\mathrm{mmd}}{=}0.05$. We also do not claim that Best-of-$N$ saturates with $N$ on GSM-Hard: empty pools dominate. Beam search and additive Gaussian noise sampling from You et al. are not re-run here.

---

## Notes for us (not for the paper)

- 表 6.2 是**系统**对比（生成器 + 打分器都换了）；6.3/6.4 把两个贡献拆开：6.3 生成器探索（Greedy/Cov，无 scorer），6.4 打分器变现（同池 O2−B0，3 seed）。
- **不要用「B0 变现率 66%→64%」**：那是跨口径混比（主表 33.74/51.10 混配新 dump 34.04/52.84）。严格同池 B0 变现率 64.9%→63.7%，噪声内。主论据统一写「O2−B0 gap 续训后 ≈2×」。
- 骨干论文主曲线常用 MC-dropout；我们主表关掉 dropout，应用 6.6 解释，避免和他们的图直接叠坐标。
- 未写 related work / intro。公式编号、图表编号可按投稿模板再套。
