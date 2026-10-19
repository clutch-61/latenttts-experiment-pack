# CODI + mmd_cold lock

- **Train out:** `outputs/p123_codi_mmd_cold_20261019_153953` → lock `outputs/p123_codi_mmd_cold/model`
- **Recipe:** coconut-style cold dual_beat + SF-C + λ_MMD=0.05 on CODI generator (`scripts/train_codi_mmd_cold.sh`)
- **Eval protocol:** dropout=None, `claim_agg=top1`, `score_reduce=sum_logit`, T=6, seed 42

## GSM N16 (s42)

| Scorer | Top1 | Cov | Maj |
|--------|-----:|----:|----:|
| O2 | **41.32** | 53.37 | 42.61 |
| B0 | 40.11 | 53.37 | 42.61 |

vs frozen CODI+O2 **40.94** → **+0.38**（同协议；cov 非逐条同池）。旧 CODIp2=42.00 降级。

Hard O2 N16: 10.24；MultiArith O2 N16: 91.03。
