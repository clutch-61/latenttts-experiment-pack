# Order-Aware Preference Learning and Holistic Rollout Alignment for Latent Chain-of-Thought

**Anonymous Author(s)**

---

## Abstract

Latent chain-of-thought (CoT) replaces intermediate text with continuous trajectories $H=(h_1,\ldots,h_T)$ and decodes only the final answer. Three failure modes then become central: (i) *order ambiguity*—content-similar trajectories can encode broken causal structure; (ii) *token correspondence ambiguity*—latent steps need not admit a one-to-one index alignment; (iii) *autoregressive rollout drift*—teacher-forced and self-generated trajectories diverge. We propose a staged training framework that assigns one objective family to each failure: hierarchical preference learning on order-corrupted negatives for (i); position-free set alignment (Chamfer, optionally MMD) on a content-only view for (ii); and holistic self-rollout alignment—task loss plus set matching over full rollouts, not per-index regression—for (iii). The first stage yields an order-preference LatentRM (O2) that is reused both for Best-of-$N$ selection and as a gate in generator post-training (p2). Empirically, under fixed $T{=}6$ and same-pool evaluation, p2 improves coverage and BoN accuracy over matched short-budget RL on COCONUT and transfers within CODI; O2 improves Top-1 over the official step-wise LatentRM (B0), with a larger paired gap after p2; versus lightweight outcome scorers, O2’s supported edge is ranking ($\mathrm{auc}_{cw}$) and order diagnostics rather than Top-1. Input ablations further show that preserving O2’s discrimination—and converting it into selection—is the right optimization path, rather than abandoning order-aware scoring.

---

## 1. Introduction

Continuous latent CoT models store multi-step reasoning as vectors rather than tokens. This removes textual process rewards and introduces structured failure modes that standard next-token or flat outcome losses do not address.

**Order ambiguity.** Two trajectories may share similar local content yet differ in the temporal arrangement of steps. Shuffling or reversing reasoning order can preserve bag-level semantics while destroying logical dependencies. A scorer that only predicts final correctness is therefore under-specified for latent process supervision.

**Token correspondence ambiguity.** Latent steps are not guaranteed to align index-wise with a teacher trajectory: lengths, segmentations, and representations may differ. Enforcing $\|\hat h_t - h_t^{\mathrm{target}}\|^2$ assumes a correspondence that need not exist and can fight genuine order supervision if applied on the ordered stream.

**Autoregressive rollout drift.** Training under teacher forcing conditions on gold prefixes, while inference conditions on the model’s own past. Per-step regression from self-generated states onto teacher states again assumes index alignment and does not target the distributional gap between full rollouts.

We therefore organize learning around three complementary objectives—not as an undifferentiated sum, but as a staged curriculum. The primary line is
\[
\mathcal L
=
\mathcal L_{\mathrm{task}}
+
\lambda_{\mathrm{pref}}\mathcal L_{\mathrm{pref}},
\]
i.e., task supervision plus hierarchical order preference. Set alignment and holistic rollout alignment are added only after order preference is validated, each on its proper representation (ordered scores vs.\ position-stripped content views vs.\ full-rollout comparisons).

The resulting order-preference LatentRM (O2) is used both for BoN decoding and to gate generator post-training (p2), so exploration and realization share one latent interface. Experiments isolate axes with fixed rulers and locked pools, and report both positive realization gains over B0 and honest Top-1 non-results against capacity-mismatched light scorers.

**Contributions.**
1. A failure-mode decomposition for latent CoT—order, correspondence, rollout drift—with one objective family per mode and an explicit staging rule.
2. Hierarchical preference learning for LatentRM with ranked order corruptions, margin calibration by sequence length, and order/hierarchy diagnostics.
3. Position-free set alignment and holistic self-rollout alignment that avoid leaking order into permutation-invariant losses and avoid claiming Self-Forcing DMD/SiD/GAN objectives.
4. Co-use of O2 as BoN scorer and p2 training gate, with same-pool and matched-budget evidence on COCONUT/CODI.

---

## 2. Related Work

**Latent / continuous CoT.** COCONUT-style generators and related latent-length or distillation variants typically report accuracy under a fixed decoding rule. We focus on structured supervision of latent trajectories and on coupling the scorer used in training with the scorer used at test time.

**Process and latent rewards.** Step-wise LatentRM, preference and outcome PRMs, and latent adapters (e.g., SWIFT/ELHSR, HSRM) provide BoN scores. Flat outcome losses do not encode graded order errors; we compare under *same-pool* protocols.

**Exposure bias and self-rollouts.** Scheduled sampling and Self-Forcing-style training reduce train–inference mismatch by conditioning on model outputs. We adopt the *holistic rollout* idea—align full self-generated trajectories—without claiming to reproduce DMD, SiD, or GAN formulations from Self-Forcing.

**Set matching.** Chamfer distances and MMD are permutation-invariant. We apply them only on content views that strip recoverable temporal order, never as a drop-in replacement for order preference on the ordered stream.

---

## 3. Method

### 3.1 Problem setup

Let the model emit a latent trajectory
\[
H=(h_1,\ldots,h_T),\qquad h_t\in\mathbb{R}^{d},
\]
followed by an answer. A LatentRM $s_\phi$ assigns a scalar score to a full trajectory (given the question encoding). BoN selects $\arg\max_i s_\phi(H^{(i)})$. For each training example we may construct: a correctly ordered trajectory $H^+$; order-corrupted negatives $\{H^-_k\}_{k=1}^{K}$ with ranks $r_k$; a teacher-forced rollout $H^{\mathrm{TF}}$; and a self-generated rollout $H^{\mathrm{SF}}$.

The full objective family is
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
**Staging.** We do not open all terms at once. The stable main line is $\mathcal L_{\mathrm{task}}+\lambda_{\mathrm{pref}}\mathcal L_{\mathrm{pref}}$. Set and rollout terms are introduced only after order preference is shown to help, and are ablated separately (Chamfer vs.\ MMD; with/without rollout).

### 3.2 Module I: Hierarchical order preference (primary)

**Corruption must happen before re-encoding.** Negatives are obtained by perturbing the *rendered* reasoning stream (or the discrete latent-token layout that induces it), then re-running the backbone to produce a full trajectory, then scoring that trajectory with $s_\phi$. We do *not* permute already-computed hidden states in place: that shortcut encourages detectors of “broken vectors” rather than sensitivity to logical order.

**Ranked corruption hierarchy.** Each negative carries a required `negative_rank` (multi-negative hierarchical training must not silently fall back to a flat triplet):

| Rank | Corruption | Severity |
|-----:|------------|----------|
| 1 | Adjacent swap | Mild |
| 2 | Local segment swap | Mild–moderate |
| 3 | Multi-segment shuffle | Moderate |
| 4 | Full reverse | Strong |
| 5 | Composite (reorder$+$drop/replace) | Strongest |

Same-rank negatives are not forced to order relative to each other.

**Hierarchical preference loss.** With margins $m_{\mathrm{order}}$ and $m_{\mathrm{hier}}$,
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
Thus $s_\phi$ must prefer the intact order to every corruption *and* prefer milder corruptions to harsher ones—learning graded order errors, not only a binary good/bad bit.

**Margin calibration.** Score gaps average over length-$T$ sequences, so a local corruption of $q$ positions yields roughly $\Delta_{\mathrm{avg}}\approx (q/T)\Delta_{\mathrm{local}}$. Margins are therefore set from empirical gap scales (median and quantiles of $s_\phi(H^+)-s_\phi(H^-)$ under adjacent swap and reverse), e.g.\ $m_{\mathrm{order}}=\alpha\cdot\mathrm{median}(\Delta)$ with $\alpha\in[0.25,0.75]$ and $m_{\mathrm{hier}}=\beta m_{\mathrm{order}}$ with $\beta\in[0.1,0.5]$. Gap-scale estimation is a calibration tool, not proof that a corruption is learnable.

**Instantiation (O2).** O2 is a LatentRM trained with $\mathcal L_{\mathrm{task}}$ (step-wise supervision) plus $\mathcal L_{\mathrm{pref}}$. It is the scorer used in BoN and as the primary gate in generator post-training below. B0 denotes the official step-wise contrastive LatentRM without hierarchical order preference.

### 3.3 Module II: Position-free set alignment

Index-wise regression $\|\hat h_t-h_t^{\mathrm{target}}\|^2$ is the wrong inductive bias when tokens lack a fixed correspondence. Chamfer and MMD are permutation-invariant, so applying them on the *ordered* stream would fight Module I. We therefore build a content view
\[
C(H)=\{\tilde h_1,\ldots,\tilde h_T\}
\]
that removes recoverable temporal information. Disallowed “fake” views include identity, clone/detach, uniform rescaling, mean-centering, and invertible per-feature diagonal maps—transforms that still leak order into a set loss. Admissible views include stripping explicit temporal encodings, a content projector / bottleneck, position-free re-encoding, or stochastic pooling.

**Default: bidirectional Chamfer** between self and teacher content views,
\[
\mathcal L_{\mathrm{Chamfer}}
=
\frac1T\sum_i\min_j\|\tilde h_i^{\mathrm{SF}}-\tilde h_j^{\mathrm{TF}}\|^2
+
\frac1T\sum_j\min_i\|\tilde h_j^{\mathrm{TF}}-\tilde h_i^{\mathrm{SF}}\|^2.
\]
**Optional: multi-scale RBF MMD** on the same content view. We compare *no set loss / Chamfer / MMD* separately; we do not enable both at once in the first studies. (We do not attribute MMD methodology to unrelated generative-flow papers; MoFlow-style ideas, if cited, concern modeling sets of candidates, not our kernel estimator.)

### 3.4 Module III: Holistic self-rollout alignment

Teacher forcing and inference differ in their conditioning prefixes. We do **not** minimize $\sum_t w_t\|\hat h_t^{\mathrm{SF}}-h_t^{\mathrm{TF}}\|^2$, which re-imposes false one-to-one alignment. Inspired by Self-Forcing’s use of the model’s own history as context—but **not** claiming DMD/SiD/GAN objectives—we align *complete* rollouts:
\[
\mathcal L_{\mathrm{rollout}}
=
\mathcal L_{\mathrm{task}}(H^{\mathrm{SF}})
+
\lambda_{\mathrm{set}}
\mathcal L_{\mathrm{set}}\bigl(C(H^{\mathrm{SF}}),C(H^{\mathrm{TF}})\bigr).
\]
Self-rollouts must solve the task and match teacher content *as sets*; order remains the job of $\mathcal L_{\mathrm{pref}}$. Optional curriculum mixing of TF/SF prefixes for backbone training is a later engineering stage: inference-only `generate()` under `@torch.no_grad()` is insufficient for backprop through rollouts.

### 3.5 Generator post-training that reuses the order-aware scorer (p2)

With $s_\phi$ fixed or periodically refreshed, each post-training step samples $k$ student trajectories and one teacher trajectory, scores them with O2 (B0 as a health constraint), selects a CE target, and updates the generator with SF-C. Optional content-view MMD implements Module II inside this loop. The released recipe **mmd_cold** (coconut cold-start + dual_beat + SF-C + $\lambda_{\mathrm{MMD}}{=}0.05$, 1500 steps; lock `outputs/p123_mmd_cold`) enables MMD at a validated coefficient. Warm-start **p2mmd05** is demoted to a control. Crucially, the same $s_\phi$ interface is used for BoN at test time—exploration and realization share one scorer.

### 3.6 What each term is *not*

| Temptation | Why we avoid it |
|------------|-----------------|
| Flat triplet without ranks | Loses graded severity; multi-negative hierarchy must require `negative_ranks`. |
| Permuting post-hoc hidden states | Teaches anomaly detection, not order. |
| Set loss on the ordered stream | Leaks / cancels order supervision. |
| Per-index SF$\leftrightarrow$TF regression | Assumes false correspondence; misnames Self-Forcing. |
| Opening all $\lambda$ at once | Confounds which failure mode was addressed. |

---

## 4. Experimental Protocol

**Data.** Primary: GSM8K-Test; appendix: GSM-Hard, MultiArith.  
**Latent length.** Main protocol $T{=}6$; max new tokens $128$; BoN $N\in\{16,64\}$; scorer dropout off.  
**Seeds.** Scorer axis $\{42,\ldots,46\}$; generator RL matrix: three seeds.  
**Statistics.** Paired bootstrap CIs; McNemar on discordant Top-1.  
**Axis isolation.** Generator tables fix B0 as ruler. Scorer tables lock candidate dumps (`answers_sha1`). Cross-base results report within-base $\Delta$ only.  
**Diagnostics for Module I.** Order-pair accuracy, hierarchy consistency, Order-Decision (content fixed, order corrupted), and $\mathrm{auc}_{cw}$ on BoN pools.

---

## 5. Experiments

Experiments follow the staging rule: Module I (O2) is the main scorer result; Modules II–III appear inside p2 as optional set/rollout terms and are not claimed as fully ablated Chamfer-vs-MMD bake-offs in the main tables.

### 5.1 Module I $\rightarrow$ BoN realization (O2 vs B0 / light scorers)

**Setup.** Locked dumps; Majority / B0 / O2 / SWIFT / HSRM; five seeds; $N{=}16$.

**Top-1 (mean$\pm$std).**

| Pool | B0 | O2 | SWIFT | HSRM |
|------|---:|---:|------:|-----:|
| Frozen | $32.72\pm0.96$ | $\mathbf{33.72\pm1.03}$ | $33.69\pm0.75$ | $33.60\pm0.75$ |
| **mmd_cold** (generate-once same pool) | $34.95\pm0.26$ | $\mathbf{36.38\pm0.60}$ | $36.16\pm0.37$ | $36.32\pm0.63$ |

Paired O2$-$B0: **$+1.00$** [$0.17,1.85$] (frozen); mmd_cold pool mean **$\approx{+}1.43$** (5-seed, sha-aligned). O2 vs SWIFT/HSRM Top-1 remains near-tied.

**Order and ranking.** Full O2 exceeds SWIFT in $\mathrm{auc}_{cw}$ by $\sim$+$3.3$ / $+2.8$ pp (frozen / legacy p2 pool). Order-Decision and hierarchy metrics place O2 far above outcome-only SWIFT/HSRM. Capacity of O2 vs tiny SWIFT/HSRM heads is not matched—Top-1 ties are not architecture verdicts.

**Input probe.** Latent-only O2 keeps Top-1 near full O2/SWIFT but drops $\mathrm{auc}_{cw}$ onto the SWIFT band. We read this as: keep hierarchical preference, improve how text and latents jointly feed $s_\phi$, and calibrate selection—not discard Module I.

### 5.2 Reusing Module I inside generator post-training (p2)

**Setup.** COCONUT-GPT2; frozen / SLPO / Dropout-GRPO / **mmd_cold**; RL controls matched at **3000** steps; evaluate with **B0** only on the generator axis. System lock also reports O2 (see `paper_mmd_cold/LOCK.md`).

| Method | Cov@16 | B0 Top-1 | $\Delta$ |
|--------|-------:|---------:|---------:|
| Frozen | 50.67 | 32.88 | — |
| SLPO (3000) | 50.90 | 32.88 | $+0.00$ |
| Dropout-GRPO (3000) | 51.23 | 32.85 | $-0.03$ |
| dualBeat (−MMD) | 53.60 | $34.12\pm0.55$ | $+1.24$ |
| **mmd_cold** | **54.03** | **$35.03\pm0.20$** | **$+2.15$** |

System lock GSM N16 **O2** 3-seed: **$36.32\pm0.40$** (vs dualBeat $35.81$; legacy p2mmd05 $35.48$). Canonical tables: `paper_tables_mmd_cold/`.

**Transfer.** CODI fair 3-seed (generate-once within-pool mism=0; zero-shot COCONUT RMs): frozen O2 $41.14{\pm}0.35$ $\rightarrow$ **CODImmd** $42.13{\pm}0.84$ (**$+0.99$**); N1 $42.08{\rightarrow}43.29$ (**$+1.21$**). CoLaR fixed_t: frozen $22.44/35.48/23.73$ $\rightarrow$ **colar_mmd_cold** $36.62/48.45/37.07$.

Matched short-budget RL does not beat frozen; mmd_cold does. This is not a full-budget official RL claim ($\sim 4$–$5\times 10^4$ steps).

### 5.3 Modules II–III status

Content-view MMD inside **mmd_cold** is the exercised set-alignment path and is slightly ahead of no-MMD dualBeat; warm-start p2mmd05 / legacy CODIp2 are demoted controls. A clean Chamfer-vs-MMD bake-off and SFT/RFT controls remain staged follow-ups (Appendix).

---

## 6. Discussion

The framework’s value is *responsibility separation*: hierarchical preference owns order; set losses own bag-level content on stripped views; holistic rollout owns train–inference trajectory mismatch. Empirically, Module I already yields a scorer that improves over B0 and exhibits order-sensitive behavior that flat outcome heads lack; reusing that scorer in p2 improves pools under a fixed B0 ruler. Remaining work is to finish staged Module II/III ablations and to turn O2’s $\mathrm{auc}_{cw}$ margin into more reliable Top-1—without collapsing the three losses into one opaque sum.

**Impact.** Methods target mathematical BoN under latent CoT. Risks include over-reliance on automatic rewards and BoN compute. No dual-use biological or chemical applications are addressed.

---

## 7. Conclusion

Latent CoT fails in distinctive ways—wrong order, false token correspondence, and rollout drift. We assign hierarchical preference, position-free set alignment, and holistic self-rollout alignment to these three problems, stage them rather than sum them blindly, and close the loop by using the resulting order-aware LatentRM for both BoN and generator post-training. Current evidence supports the order-preference core and its co-use in p2; set and rollout modules follow the same design constraints as they are brought online.

---

## Acknowledgments

*Blinded for review.*

---

## References

*To be completed.*

---

## Appendix (outline)

- A. Corruption catalog, rank definitions, margin calibration plots.  
- B. Order-Decision / hierarchy / $\mathrm{auc}_{cw}$ full tables.  
- C. Content-view admissibility checks; Chamfer vs.\ MMD pilots.  
- D. Rollout curriculum details; non-claims re: Self-Forcing objectives.  
- E. Full BoN matrices; reproducibility (seeds, hashes, configs).
