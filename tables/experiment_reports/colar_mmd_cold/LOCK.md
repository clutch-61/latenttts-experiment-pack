# CoLaR fixed_t + mmd_cold lock

- **Train out:** `outputs/colar_mmd_cold_20261019_165530` → lock `outputs/colar_mmd_cold/model`
- **Recipe:** answer_beat + SF-C + λ_MMD=0.05，fixed_t 协议（`scripts/train_colar_mmd_cold.sh`）
- **Eval:** Greedy / Cov@16 / Maj@16（不可接 GPT-2 B0/O2）

## GSM fixed_t

| Policy | Greedy | Cov@16 | Maj@16 |
|--------|-------:|-------:|-------:|
| Frozen fixed_t | 22.44 | 35.48 | 23.73 |
| **colar_mmd_cold** | **36.62** | **48.45** | **37.07** |

相对旧 Colar-p2v2（λ_MMD=0：34.42 / 49.58 / 36.85）：Greedy/Maj 略升，Cov 略降。  
**不可**与官方 `###`+max_L 冻结分（Greedy 40.56）混成同一 Δ。
