> **主系统改锁 mmd_cold**（2026-10-19）：官方表见 `paper_tables_mmd_cold/`；锁矩阵见 `p123/paper_mmd_cold/LOCK.md`。p2mmd05 降为暖启对照。

# 论文向消融清单（对齐 Method + staged；以 mmd_cold 为 Full）

> **叙事纠偏：** 主文主角 = exploration–realization gap / B0 异类；O2 = 解药实例；闭环 gate 降级。Module II/III → design space（弱 MMD 可接入）。

## 方法消融主线（拆自家配方）

详表：`paper_tables_mmd_cold/ABLATION_TABLES.md`（相对 **mmd_cold Full**）。旧单 seed 表见 `method_ablation/`。

| 臂 | 相对 Full | GSM N16 O2 3-seed | Δ vs Full |
|----|-----------|------------------:|----------:|
| **mmd_cold** | Full | **36.32±0.40** | — |
| dualBeat | −MMD | 35.81±0.46 | −0.51 |
| p2mmd05 | 暖启对照 | 35.48±0.65 | −0.83 |
| sfB | SF-C→B | 34.98±0.48 | −1.34 |
| o2only | 关 B0 health | 35.10±1.20 | −1.21 |
| b0only | 双 B0 | 35.61±0.12 | −0.71 |
| **nogate** | 无 LatentRM | **32.12±0.34** | **−4.20** |

**读法：** 整门控贡献最大；MMD 冷启弱正；暖启 p2mmd05 不作 MMD 成功故事。

## A. 模块一（打分器）

| ID | 对照 | 状态 |
|----|------|------|
| A1–A5 | B1 / O1 / O2 / adj / noadj | ✅ `module1_ablation_20261018_063957/` |
| A6 | full / latent_only / text_only | ✅ `o2_input_mode/` |

**机制：** hierarchy 需多等级；**BoN Top1 上重训 O2≈B1**（勿主打 order→Top1）。

| 模型 | order_pair | hierarchy | BoN Top1 |
|------|------------:|----------:|---------:|
| B1 | 0.774 | 0.731 | 33.36 |
| O1 | 0.973 | 0.745 | 32.52 |
| O2 | 0.974 | **0.896** | 33.28 |
| adj_only | 0.966 | 0.679 | 32.90 |
| no_adj | 0.944 | 0.881 | 34.12 |

## B. 模块二（MMD）

| 对照 | 状态 | 读法 |
|------|------|------|
| **mmd_cold vs dualBeat** | ✅ 3-seed | +0.51；公平冷启 leave-one-out |
| p2mmd05 vs dualBeat | ✅ 降级 | 暖启 −0.33；不作成功故事 |
| Chamfer vs MMD 烘培 | hold | |

## C. 系统轴

主实验表：`paper_tables_mmd_cold/MAIN_TABLES.md`。同池 scorer 5-seed 仍用历史 p2 dump（未因换锁重采样）。
