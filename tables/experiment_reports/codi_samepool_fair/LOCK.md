# CODI fair same-pool (matched protocol)

- **Script:** `scripts/run_codi_samepool_fair.sh`
- **Pools:** `codi_frozen`=`checkpoints/codi` · `codi_mmd`=`outputs/p123_codi_mmd_cold/model`
- **Protocol:** generate-once B0+O2，N16，seeds 42–44，dnone / top1 / sum_logit / T=6；`mism=0`
- **RM:** B0/O2 = COCONUT 训权重，**零样本**迁到 CODI（须脚注）
- **跨生成器：** 非 hash 同池（by design）

## N1 greedy（无 RM，s42）

| Gen | Pass@1 |
|-----|-------:|
| Frozen | 42.08 |
| CODImmd | **43.29**（+1.21） |

## N16 same-pool Top1（3-seed mean±std）

| Pool | B0 | O2 | Cov | Maj | O2−B0 |
|------|---:|---:|----:|----:|------:|
| Frozen | 39.83±0.23 | 41.14±0.35 | 54.79 | 42.05 | +1.31 |
| CODImmd | 40.26±0.84 | **42.13±0.84** | 53.30 | 42.78 | +1.87 |

CODImmd−Frozen O2 per seed: +0.38 / +0.53 / +2.05 → mean **+0.99pp**。  
B0 尺 mean **+0.43pp**。Cov 略降（探索变窄），N1/Maj/O2 Top1 同向为正。

旧单点 s42 +0.38 与本表 s42 一致；**主文用 3-seed mean +0.99**，并标零样本 RM + 非跨池 hash。
