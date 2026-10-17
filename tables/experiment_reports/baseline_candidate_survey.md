# Five candidate baselines — survey (2026-10-05)

Work-agent report. Do **not** treat paper-reported accuracies as same-protocol numbers. Our protocol: fixed 6 latent steps, scorer dropout off, $N\in\{1,4,16,64\}$, GSM8K-Test / GSM-Hard / MultiArith, seed 42. Do not expand a full generator×scorer grid.

Local stack: `LatentTTS-main`; generator `checkpoints/coconut`; B0 `outputs/latentrm_baseline/best` (official LatentTTS contrastive LatentRM); O2 `outputs/latentrm_order_pref/best` (ours); BCE recipe `training_args/train_baseline_bce.yaml` (not in the main table).

---

## SLPO — **可复现主实验基线**

**Paper:** [SLPO: Scaling Latent Reasoning via a Surrogate Policy](https://arxiv.org/abs/2607.19691) (You, Liu, Li, Li; 2026-07). Same group as LatentTTS.

1. **Code / weights / license**
   - Code: https://github.com/ModalityDance/SLPO — MIT.
   - Weights: https://huggingface.co/ModalityDance/slpo-coconut-gpt2 — MIT, ungated, ~125M F32 safetensors. `hf download ModalityDance/slpo-coconut-gpt2`.
   - Sibling CODI+SLPO not needed for our stack. Base: `ModalityDance/latent-tts-coconut`.

2. **Architecture / interface / data / hparams**
   - Generator: COCONUT GPT-2 + stop-gate head. Same special tokens `<|start-latent|>`, `<|latent|>`, `<|end-latent|>`.
   - Train: (i) stop-gate cold start on correctness vs latent length; (ii) SLPO RLOO with diagonal-Gaussian surrogate from $K{=}4$ MC-dropout forwards, $G{=}8$ rollouts. Train on GSM8K-Aug (`gsm_train.json`); threshold on `gsm_valid.json`.
   - **Paper inference (not our protocol):** `STOP_POLICY=gate`, threshold 0.6, `MAX_LATENT_LENGTH=12`, dropout off for Acc. Repo also has **fixed-length** RLOO/GRPO configs (`configs/coconut_gpt2_rloo.yaml`) — useful if we force $T{=}6$.
   - Paper-reported Acc (gate, dropout off; **not** our $T{=}6$ BoN): GSM8K 35.63 (mean $T{\approx}5.73$), GSM-Hard 7.66, MultiArith 83.10.

3. **Compatibility**
   - Highest among the five. Same tasks and latent token interface. HF `architectures: LatentCOCONUTGPT2` may include extra stop-gate weights; `from_pretrained` in *our* `COCONUTGPT2` may fail or ignore the gate.
   - Our `infer_gpt2_rm` currently assumes a fixed latent budget like vanilla COCONUT, not adaptive stop.

4. **Min adaptation**
   - Download ckpt. Either: (a) force **6 latent steps, no gate** in generation (match our protocol; may need SLPO `LatentGenerationMixin` with a fixed-length stop policy if the repo exposes it); or (b) keep gate+$T_{\max}{=}12$ only as an *appendix / paper-setting* column, clearly labeled.
   - Then run **our** `infer_gpt2.py` / `infer_gpt2_rm.py` with B0, O2, majority on **the same sampled pools**. If extra modules block load: copy stop-gate class from SLPO or `strict=False` and disable the gate.
   - Cost: download + load debug **hours**; full $N$ matrix on 3 sets **same order as one BoN sweep** (already done for coconut). No retrain required for the generator baseline.

5. **Verdict:** Main-experiment generator baseline. First thing to run.

---

## PMPS — **需适配后比较**（当前无代码/权重）

**Paper:** [Structural Process Supervision…](https://arxiv.org/abs/2609.09928) (Li et al.; 2026-09). Method name PMPS (Prototype-Mediated Process Supervision).

1. **Code / weights:** No GitHub, Hugging Face, or appendix code link found (paper index: no code). License: arXiv preprint only.

2. **Architecture / data / hparams**
   - Teacher–student **self-distillation** in the CODI family: $N_l{=}6$ latent embeddings (GPT-2 / Qwen2.5-0.5B), $N_l{=}8$ on LLaMA-3.2-1B. Learnable $K$ prototypes (GPT-2: $K{=}24$, $d{=}128$), Sinkhorn assignment, bidirectional cross-prediction vs explicit CoT hidden states, Progressive Sequential Alignment (Gaussian positional prior, then anneal). LoRA $r{=}128$, $\alpha{=}32$, batch 256, 8×A100. Train GSM8K-Aug; eval GSM8K-Aug + GSM-Hard + SVAMP + MultiArith. Metric Acc + **token length**, not LatentRM BoN.
   - Coconut appears only as a **baseline they compare against**, not as their student interface.

3. **Compatibility:** Task overlap (Hard / MultiArith) and GPT-2 + 6 latents look close, but latents are **CODI-style continuous thoughts**, not COCONUT special-token recurrence. **B0/O2 cannot score PMPS traces without a new annotate+train.**

4. **Min adaptation:** Reimplement on a CODI or PMPS student; train GPT-2; dump 6-step hidden states; train new B0/O2 (or only majority / $N{=}1$) on those traces. Cost: **full generator retrain on 8×A100-class** plus scorer retrain. Without official code, weeks of reverse-engineering.

5. **Verdict:** Do not put in the main table until code/weights appear. Cite as related generator-training work. If a checkpoint shows up later, treat as adapt-then-compare.

---

## LTF — **只适合相关工作**

**Paper:** [Latent Thought Flow](https://arxiv.org/abs/2606.16222) (Zou, Huang, Li, Zhou; 2026-06). CC BY 4.0 on arXiv.

1. **Code / weights:** None found (reviews list Code/Model N/A).

2. **Architecture:** LoRA ($r{=}128$) + **latent reasoning head**; each step samples $z_t\sim\mathcal{N}(\mu,\sigma^2)$ until an `<eos_r>`-like stop; continuous **GFlowNet** (entropy-weighted subtrajectory balance + reference-prior). Train 100 epochs, batch 256, lr $1{\times}10^{-4}$, RTX PRO 6000. Inference temperature 0.9, top-$p$ 0.95. Backbones: LLaMA-3.2 1B/3B, LLaMA-3.1 8B, DeepSeek-R1-Distill-Qwen-1.5B. **No GPT-2.** Eval includes GSM8K-Aug, GSM-Hard, MultiArith (among others). Variable length (e.g. paper-reported ~3.3 steps on GSM8K-Aug for 1B — not our protocol).

3. **Compatibility:** Wrong backbone, wrong latent interface (Gaussian head vs COCONUT tokens), variable $T$, no artifacts.

4. **Min adaptation:** Port GFlowNet head onto GPT-2 COCONUT and retrain — that is a new method, not a baseline reproduction. Cost: high.

5. **Verdict:** Related work only.

---

## HSRM — **需适配后比较**（仓库几乎空）

**Paper:** [HSRM](https://arxiv.org/abs/2608.30841) (Li & Zhu; 2026-08). Claimed code https://github.com/JXL884/HSRM — **as of this survey the repo contains only an 88-byte README**, no training/eval scripts, no weights.

1. **Code / weights:** GitHub stub. No HF checkpoint. No license file in the stub.

2. **Architecture / data**
   - Frozen **text** generator; extract **last-layer hidden states at textual reasoning-step boundaries**; 2-layer Transformer encoder (~2M, $d{=}256$, 4 heads) → scalar; Best-of-$N$.
   - Train on self-sampled trajectories with **final-answer correctness** (64 candidates per train problem in their setup). No process labels.
   - Their generators: Qwen3 1.7B–14B and Llama-3.2/3.1. Benchmarks: GSM8K, MATH-500, AIME, OlympiadBench. Eval Best-of-8. **Not GPT-2, not COCONUT, not MultiArith/GSM-Hard in the main protocol.**

3. **Compatibility:** Idea matches “frozen gen + trajectory scorer + BoN”. Input is **step-boundary hidden states of token CoT**, not six COCONUT latents. Released weights (none) would not match GPT-2 $d_{\mathrm{model}}$.

4. **Min adaptation**
   - Reimplement the 2M encoder in our stack.
   - Define a step as one COCONUT latent (length 6); cache latents from **one** generator (start with coconut, same dumps we already have for BoN).
   - Train with outcome ranking / BCE on those caches (GSM train annotate or on-policy samples).
   - Score the **same** $N\in\{4,16,64\}$ pools as B0/O2/majority.
   - Cost: **days** if we ignore their empty repo and reimplement; training is cheap vs LatentRM GPT-2. Must not load a Qwen-trained HSRM onto GPT-2 latents.

5. **Verdict:** Worth an adapt-then-compare **scorer** against B0 on coconut (and later SLPO) pools. Not a drop-in baseline until they release code. Do not cite their BoN-8 Qwen/Llama numbers next to our GPT-2 table.

---

## RSP — **只适合相关工作**（除非大改主干）

**Paper:** [Learning Process Rewards via Reasoning State Propagation](https://arxiv.org/abs/2609.39220) (Gan et al.; 2026-09).

1. **Code / weights:** “Code in supplementary material” — **no public GitHub/HF found.**

2. **Architecture / data**
   - **Text PRM** on Qwen3 (default **Qwen3-4B**): special tokens `⟨BREAK⟩` / `⟨REPAIR⟩` + two MLP heads; propagate binary valid/invalid state along **textual** steps.
   - Train: 20% of PRM800K process labels mixed 1:3 with AceMath-RM outcome trajectories; 2 epochs, AdamW $5{\times}10^{-6}$, bf16, **8×H100**. Compare also to Qwen2.5-Math-PRM (that is a *baseline in their paper*, not RSP itself).
   - Eval: beam search on MATH500/Gaokao; Best-of-$N$ with $N\in\{8,16,32,64,128\}$ on MATH500; generators LLaMA / Qwen / DeepSeek. **Not GSM8K-Test / GSM-Hard / MultiArith, not GPT-2.**

3. **Compatibility:** Step = natural-language rationale prefix. Our step = one of six continuous latents. Backbone is a 4B LLM, not a GPT-2 scoring head.

4. **Min adaptation:** Either (a) run RSP as a **text** verifier on decoded CoT after latents (unfair vs B0 which never sees text steps, and coconut often has little CoT); or (b) re-define break/repair on latent prefixes and train a GPT-2-sized head — a new scorer, not RSP. Cost: (a) medium but protocol-incomparable; (b) high, needs process-like labels on latents we do not have (only MC thought-correctness in annotate).

5. **Verdict:** Related work. Do not block the paper on reproducing Qwen3-4B PRM.

---

## What to run (still 2×2, not a grid)

Keep the existing main tables. Add **at most**:

| Axis | Add | Same-pool rule |
|------|-----|----------------|
| Generator | SLPO-COCONUT, **forced $T{=}6$, no dropout** | Sample once; score with majority / B0 / O2 (pairwise, not a 3×3 showcase) |
| Scorer | Optional HSRM-style 2M on coconut latents | Same coconut (or SLPO) dumps as B0 |

PMPS, LTF, RSP: related-work paragraph unless artifacts appear.

**Do not:** paste SLPO 35.63 / HSRM Qwen BoN-8 / PMPS GSM8K-Aug Acc into Table 1. If mentioned, label **paper-reported**, and state gate+$T_{\max}{=}12$ vs our $T{=}6$, or Qwen vs GPT-2, etc.
