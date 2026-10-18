# Method ablation（生成器配方 leave-one-out）

> **2026-10-19：** 官方 Full = **mmd_cold**。权威表 → `../paper_tables_mmd_cold/ABLATION_TABLES.md`。

## 设定

- 起点：`checkpoints/coconut`；预算 1500 step；冻结 O2 BoN N16；seeds 42/43/44；`claim_agg=top1`
- Full：`dual_beat` + SF-C + λ_MMD=0.05（mmd_cold）
- dualBeat：同配方 λ_MMD=0（Module II LOO）
- p2mmd05：dualBeat 暖启 + MMD（**非** LOO；降级对照）

## 3-seed O2 Top1

| Arm | mean±std | Δ vs mmd_cold | Δ vs dualBeat |
|-----|----------:|--------------:|--------------:|
| **mmd_cold** | **36.32±0.40** | — | +0.51 |
| dualBeat | 35.81±0.46 | −0.51 | — |
| p2mmd05 | 35.48±0.65 | −0.83 | −0.33 |
| o2only | 35.10±1.20 | −1.21 | −0.71 |
| b0only | 35.61±0.12 | −0.71 | −0.20 |
| sfB | 34.98±0.48 | −1.34 | −0.83 |
| nogate | 32.12±0.34 | −4.20 | −3.69 |

## 结论

1. **整门控**（nogate）掉点最大 → 方法消融尺上门控有贡献（与短预算 SWIFT-gate 轴勿混）。
2. **MMD**：冷启 mmd_cold 相对 dualBeat **+0.51**；暖启 p2mmd05 不涨。
3. SF-C→B、门控内拆为中等负向。

模型：`outputs/methodabl_{nogate,sfB,o2only,b0only,mmd_cold}_*`；锁 symlink `outputs/p123_mmd_cold/model`。
