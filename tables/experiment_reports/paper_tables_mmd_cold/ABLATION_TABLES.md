# 方法消融表（相对官方系统 mmd_cold）

协议：椰子冷启 / 1500 step / 冻结 O2 BoN N16 / 3-seed（42–44）/ `claim_agg=top1`。

**Full = mmd_cold**（dual_beat + SF-C + λ_MMD=0.05）。
各臂为 leave-one-out 或对照；Δ = arm − Full。

| Arm | 相对 Full 去掉/改动 | Top1 mean±std | Δ vs Full | Δ vs dualBeat | 读法 |
|-----|---------------------|--------------:|----------:|--------------:|------|
| **mmd_cold** | （Full） | 36.32±0.40 | 0.00 | 0.51 | 官方系统 |
| dualBeat | −MMD（λ=0） | 35.81±0.46 | -0.51 | 0.00 | Module II leave-one-out；公平冷启 |
| p2mmd05 | 暖启+MMD（非 LOO） | 35.48±0.65 | -0.83 | -0.33 | 降级对照；不作成功故事 |
| sfB | SF-C→SF-B | 34.98±0.48 | -1.34 | -0.83 | 自强制变体 |
| o2only | 关 B0 health | 35.10±1.20 | -1.21 | -0.71 | 门控内拆 |
| b0only | O2→B0 双角色 | 35.61±0.12 | -0.71 | -0.20 | 门控内拆 |
| nogate | 去掉 LatentRM 门控 | 32.12±0.34 | -4.20 | -3.69 | 整门掉点最大 |

## 生成器尺（无 RM；Pass@1 / Cov@16）

| Arm | Pass@1 | Cov@16 |
|-----|-------:|-------:|
| dualBeat | 31.46 | 55.88 |
| **mmd_cold** | **32.22** | **56.03** |
| nogate | 29.04 | 55.42 |
| sfB | 30.33 | 54.81 |
| o2only | 31.92 | 54.59 |
| b0only | 31.61 | 54.74 |
| p2mmd05 | 31.46 | 55.95 |

## Module I（打分器；同池；非生成器消融）

见 `module1_ablation_20261018_063957/`。机制：hierarchy 需多等级；BoN Top1 上重训 O2≈B1（勿主打 order→Top1）。

## 纪律

- 谈 MMD：只用 **mmd_cold vs dualBeat**，不用暖启 p2mmd05。
- 不盲抬 λ_MMD；+0.5pp 为弱正。
- nogate −3.7pp 与短预算 O2-vs-SWIFT gate 消融不同轴，勿混写。

