# CoLaR 适配说明（第三生成器底座）

源：`ModalityDance/latent-tts-colar`（Llama-3.2-1B-Instruct + LatentHead，hidden=2048）。  
本仓库将其接到与 COCONUT/CODI 同一套 **p2 精神** 续训管线；**不能**接 GPT-2 B0/O2（d=768）。

## 代码入口

| 路径 | 作用 |
|------|------|
| `src/models/colar.py` | CoLaR 模型包装 |
| `src/infer_llama.py` | 评测：`prompt_style=official\|fixed_t`，`model_id` 可指向续训 ckpt |
| `src/generation_mixin.py` | 变长 / 固定 latent 生成 |
| `src/annotate_data.py` | 已支持 `colar` |
| `scripts/train_p123_system.py --generator_type=colar` | p2 训练；强制 `selector_family=none` + `select_mode=answer_beat` |
| `scripts/train_colar_p2.sh` | v1：官方 `###` + max_L 变长 +（曾）MCLinear |
| `scripts/train_colar_p2v2.sh` | v2：固定 T=16、noise、无 MCLinear、短训 |
| `scripts/eval_colar_fixed_t.sh` | 固定 T 协议评测 wrapper |

数字摘要：`tables/experiment_reports/colar_*`。权重与逐题 dump 不入库（自备 `checkpoints/colar`）。

## 协议纪律

1. **不跨底座比绝对 Acc**（CoLaR 官方 Greedy≈40.6 ≠ COCONUT≈34）。
2. **不用 GPT-2 O2/B0 对 CoLaR 做 BoN**。
3. **官方协议**（`###` + `Thinking speed:2` + `max_latent_length=64`）与 **策略自洽协议**（无 `###`、固定 `latent_length=16`）**分表**，禁止混成同一 Δ。
4. Colar-p2 v1 崩塌 → **禁止**宣称 CoLaR 上 p2 已成功；v2 只报 **fixed_t 底座内** 前后对照。

## 结果（GSM8K-Test）

### A. 官方协议（冻结参考行）

`Thinking speed:2` + `max_latent_length=64`，bf16，dropout 采样。

| | Greedy | Cov@16 | Maj@16 | Cov@64 | Maj@64 |
|--|--:|--:|--:|--:|--:|
| 冻结 CoLaR | **40.56** | **64.97** | **45.11** | **77.79** | **47.16** |

### B. Colar-p2 v1（官方同协议续训）→ 崩塌

`answer_beat` + SF-C，曾永久注入 MCLinear；评测用 unwrap 后 ckpt。

| | Greedy | Cov@16 | Maj@16 |
|--|--:|--:|--:|
| 冻结 | 40.56 | 64.97 | 45.11 |
| Colar-p2 v1 | **1.59** | **7.51** | **0.76** |
| Δ | −39 | −57 | −44 |

判定：配方移植失败（非「CoLaR 底座不可用」）。Hold：盲 λ_MMD、把崩塌写成生成器差。

### C. 策略自洽协议 fixed_t（v2 主表）

Prompt：`...(Thinking speed: 2)`（无 `###`），`latent_length=16`，noise 采样；**不**注入 MCLinear。

| | Greedy | Cov@16 | Maj@16 |
|--|--:|--:|--:|
| 冻结 fixed_t | 22.44 | 35.48 | 23.73 |
| Colar-p2v2（400 step, lr=2e-6） | **34.42** | **49.58** | **36.85** |
| Δ（底座内） | **+11.98** | **+14.10** | **+13.12** |

读法：
- 相对 **同协议冻结**，v2 明显正 Δ → p2 精神在 Llama+LatentHead 上可非崩塌地起作用。
- **不可**与官方 40.56 直接比 Δ（协议不同；fixed_t 冻结本身低于官方）。
- 短预算探针；未做全量预算 / MMD / CoLaR-dim LatentRM。写作标「fixed_t 自洽协议，非官方 CoLaR 评测设定」。

## 适配要点（相对 GPT-2 p2）

1. **门控**：无 LatentRM → `answer_beat`（正确且优于 teacher；过滤 `Answer:0`/过短）。
2. **采样**：v2 用 noise，**禁止**训练期永久 MCLinear（v1 根因之一；且曾导致 `down_proj` 存盘键错位）。
3. **长度**：v2 sample ≡ SF，固定 T=16；避免 max_L 变长与 SF-16 错配。
4. **dtype**：CoLaR student 可用 bf16；ckpt 存盘前 unwrap MCLinear。
5. **评测**：`infer_llama --prompt_style=fixed_t --latent_length=16 --model_id=<ckpt>`。

## 复现命令（摘要）

```bash
# 冻结官方
python -m src.infer_llama --model_dtype=bf16 --do_sample=False --n_samples=1

# 冻结 / 续训 fixed_t
bash scripts/eval_colar_fixed_t.sh 0 checkpoints/colar colar_fixed_t 1
bash scripts/eval_colar_fixed_t.sh 0 outputs/colar_p2v2_<stamp>/model colar_p2v2 1

# 训 v2
bash scripts/train_colar_p2v2.sh 0
```

## 对论文 / EXPERIMENTS 的位置

- 主实验 2 的 **C 列**：官方冻结可报探索能力；v1 作阴性；v2 作「自洽协议下的底座内正 Δ」，**不**写成已对齐 COCONUT/CODI 官方 T=6+O2 闭环。
- 需要完整闭环（BoN+O2）时，须另训 CoLaR 维 LatentRM，不在本适配范围内。
