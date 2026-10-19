# Scorer axis — mmd_cold generate-once same pool

- **Dir:** `results/full/scorer_axis_mmd_cold_samepool/`
- **Script:** `scripts/infer_same_pool_multi_scorer.py` + `scripts/run_scorer_axis_mmd_cold_samepool.sh`
- **Generator:** `outputs/p123_mmd_cold/model`；N=16；seeds 42–46
- **Assert:** `same_pool=true`，`same_pool_mism_vs_O2=0`（answers_sha1）

## GSM N16 Top1（mean±sample std）

| B0 | O2 | SWIFT | HSRM |
|---:|---:|------:|-----:|
| 34.95±0.26 | **36.38±0.60** | 36.16±0.37 | 36.32±0.63 |

配对 O2−B0 per seed: +1.36 / +0.38 / +2.35 / +1.21 / +1.82 → mean **+1.43pp**。

**Invalid:** `results/full/scorer_axis_mmd_cold/`（分次 generate，mism≈1049）— 勿进主表。
