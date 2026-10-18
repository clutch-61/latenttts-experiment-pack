# 全论文消融总表（诊断优先；官方系统 = mmd_cold）

目标：**每条消融对应一句能站住的 claim**。闭环 gate 不追求阳性。Module II 仅以冷启 MMD 弱正写入。

## 主结论目标

1. **探索成立：** mmd_cold / dualBeat 相对冻结抬 Cov / B0 Top1；同预算文本 RL ≈0。
2. **变现缺口：** B0 realization 不跟 Cov 涨；traj-level（O2/SWIFT/HSRM）更好 → B0 异类。
3. **解法边界：** traj-level 恢复变现；order-pref 非 BoN Top1 必需。
4. **配方：** 整门控必要（nogate −4.2pp vs Full）；MMD 冷启弱正（+0.5）；暖启 p2mmd05 非故事。

## 消融矩阵

| ID | 对照 | 服务 claim | 状态 |
|----|------|------------|------|
| **D1** | realization frozen vs p2 × scorers | B0 异类 | ✅ `realization_rate_*.tsv` |
| **D2** | B1 5-seed 同池 | order vs BCE traj | 见 scorer_axis |
| **D3** | Module I B1/O1/O2/adj/noadj | 辨序机制 | ✅ |
| **D4** | 输入 full/lat/text | O2 输入 | ✅ |
| **D5** | 短预算 gate（O2 vs SWIFT） | 闭环 | ✅ 阴性（不同轴） |
| **D6** | **mmd_cold vs dualBeat** | Module II | ✅ +0.51；p2mmd05 降级 |
| **D8** | nogate / sfB / o2only / b0only | 配方拆 | ✅ `ABLATION_TABLES.md` |
| **D7** | p2-expanded ID 对齐 | 因果变现 | 代理有；硬写需 ID |
| **D9** | 外集矩阵 | 系统锁 | ✅ `paper_mmd_cold/` + dualBeat dnone 补跑 |

## 权威数字目录

- 主实验表：`../paper_tables_mmd_cold/MAIN_TABLES.md`
- 方法消融：`../paper_tables_mmd_cold/ABLATION_TABLES.md`
- 系统锁：`../p123/paper_mmd_cold/LOCK.md`

## 读数纪律

- 主表 Top1 + `claim_agg=top1`；谈 MMD 只用冷启 LOO。
- 同池必须 `answers_sha1` 对齐。
- 不把 Module II/III 写成已验证大满贯。
