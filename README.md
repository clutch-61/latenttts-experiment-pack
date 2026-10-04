# LatentTTS 后续研究：一体系统（P1 选优 · P2 分布 · P3 Self-Forcing）

> 本仓库是**新工作**的代码与实验打包，不是复现原 LatentTTS 论文主线。  
> 原论文基础设施（COCONUT + LatentRM BoN）仅作底座；目标、配方、结论以本文件与 `results/full/PLAN.md` 为准。

**协议（除非另注）：** GSM8K-Test · BoN N=16 · seed=42 · claim=`wvote`（softmax T=1，`sum_logit`）  
**基线：** A + B0 = **33.74%**  
**目标：** (gen+O2) − (A+B0) ≈ **+10pp** → claim ≈ **43.74%**  
**当前官方锁：** dualBeat + O2 · wvote **36.69%**（top 35.41 · cov 53.37）· vs A+B0 **+2.95pp**

杀伤线（新配方要「过线」）：相对 dualBeat，wvote **≥ +1pp**（≥ **37.69**）。未过线不锁新官方、不宣称成功。

---

## 1. 新 Idea（我们在研究什么）

原系统是「冻结生成器 + 离线 LatentRM 做 BoN」。我们要做的是**生成器与奖励在同一在线环里共训**，并拆成可独立证伪的三层：

| Phase | 名字 | 意图 |
|-------|------|------|
| **P1** | 选优 / leftover | 池里有金标时，O2（及可选 B0 门）要把对的选出来；训选与 claim 对齐 |
| **P2** | 分布对齐（MMD） | student 在线 latent 对齐「正轨迹」内容视图（老师对 → teacher latents；否则 cache / 选中正确 latents） |
| **P3** | Self-Forcing（SF-C） | 可微 latent rollout + 答案 CE；可选 `λ_lat` 拉 live↔选中 latents |

**一体链（代码主入口）：** `scripts/train_p123_system.py`  
冻结/可训组合、`select_mode`（`dual_beat` / `coverage` / `wvote`）、`λ_mmd` / `λ_lat` / `λ_rank`、多卡 packing（`--frozen_device`、`--gen_replica_device`）都在这里。

**评测主入口：** `python -m src.infer_gpt2_rm`（BoN + `claim_agg=wvote`）  
**训练后自动评测：** `scripts/pipeline_p123_eval.sh`  
**中间量诊断：** `scripts/diag_p123_mids.py` · 训练 `mid_metrics.jsonl`

### 关键指标（怎么读数）

- **claim / wvote**：对外主数字（加权投票）
- **top**：O2 argmax；与 wvote 拉开说明聚合在救分
- **cov**：池内是否至少一条金标（完美选优上限）
- **leftover**：有池但 O2 top / claim 仍错
- **分开报：** generator-only vs +O2；健康对照 +B0
- **禁止：** 无证据锁成功；盲抬 `λ_mmd`；把 train pool≈0.9 当成测集已通

---

## 2. 已站住的正信号

| 结果 | 数字 | 含义 |
|------|------|------|
| **dualBeat**（P1 双门 O2+B0 + self-beat，暖启后 SF） | wvote **36.69** | 相对假迁移 / 乱叠 λ 的配方，这是当前**唯一官方锁** |
| 相对 A+B0 | **+2.95pp** | 有真实系统增益，但距 +10 还差 ~7pp |
| N16 完美选优上限 | cov **53.4%** | 43.7 目标**在上限内** → 不是算术不可能 |
| O2 顺序机制（Phase-1） | order_pair / hierarchy 明显高于 B0 | LatentRM 偏好训练有机制信号；**不等于** TTS leftover 已解决 |
| jrank 交叉 | dualBeat gen + jrank O2 → **37.00** | 微涨来自 O2，生成器自己掉 top；仍 < 37.69 |

---

## 3. 实验总表（新配方，按层）

下列均为 GSM-Test N16 s42 · claim=wvote（除非标明）。完整决策流见 `results/full/PLAN.md`。

### 3.1 生成器（P3 侧）

| 配方 | wvote | 结论 |
|------|------:|------|
| dualBeat（锁） | **36.69** | 正信号配方 |
| dualBeatLat / Div / Proc / B0t / X2 续训 | ≤ 锁 | 饱和；多样性↑、过程形↑ ≠ 涨点 |
| joint（P1 冻选 + λ_mmd=0.1 + SF-C + λ_lat） | 36.32 | 一体接线成功，claim 未过杀伤 |
| jrank（上 + 可训 leftover hinge） | 36.92 | +0.23，噪声边；gen top 掉 |
| k16cov（k=16 + coverage + ek0.25） | 36.32 | 训侧 teacher CE 多，**测 cov 几乎不动** |
| k16proc（coverage + process） | **33.43** | 更差：变成 teacher 长 CoT 克隆 |
| infer 对齐 dropout=0.2（锁 gen） | 34.95 | cov↑ 转化↓，不是免费点 |

### 3.2 选优器 O2（P1 侧）

| 配方 | wvote | 结论 |
|------|------:|------|
| 离线 / on-policy / O2hard | ≤ 锁附近 | 同族 on-policy 停 |
| dualBeat + jrank O2 | **37.00** | 最好 O2 侧；仍未过 +1pp 杀伤 |
| o2hardrank（gsm_train k16 leftover hinge） | 36.32 | train 会学、test 172 道稳定 leftover 不翻 |

**稳定 leftover ~172 题：** 换 O2 / hinge 净救≈0；train 挖到的 gap（中位 ~0.44）≠ test gap（中位 ~1.83）。

### 3.3 聚合 / 采样（推理侧）

| 尝试 | 结果 | 结论 |
|------|------|------|
| N64 全量 | wvote 37.00 · cov 61 | cov↑ **不转化** |
| 特征门控 MV / wvote-mass / 自适应 extra-N | ≤ ~37.1 | ≈盲 N64；help≈hurt |
| 空池@16→N64（oracle） | 36.24 | 加采救不了转化 |
| 拼 A 候选盖 dualBeat 空池 | 仅盖住空池的 ~9% | 无效 |
| leftover→MV（**金标作弊**） | ~39.3 | 说明高分题里多数票也常错 |

→ **聚合层封死**；不要再靠 N↑ / 门控凑 +10。

### 3.4 分数缝 / 协议

| 尝试 | 结果 |
|------|------|
| claim 改 `mean_log_prob` / `sum_log_prob` | 掉点 → 仍用 `sum_logit` |
| fixseam-B（SF-B 缓存 latent） | 伤 `g_lat` |
| clat 修缝 | +0.16，未过杀伤 |

---

## 4. 瓶颈（当前真正卡在哪）

距目标约 **7pp**，拆开：

1. **主墙 · 测集空池 ~46.6%**  
   训练 k=64 时 pool 常 0.88–0.98；测试 N16 pool ~0.53。  
   单条 teacher CE（k16cov）**抬不动**测空池 → 不是再调 `select_mode` 能混过去。

2. **次墙 · 稳定 leftover ~13%（172 题）**  
   O2 排序剩余空间大约 ≤1pp（已见 37.0）。要的是池里**更多/更好认的金标**，不是再训 RM。

3. **有池选错（池内 ~1/3）**  
   在 cov 先抬之前，继续拧 O2 / 聚合是噪声。

**下一步优先级（研究分叉）：**

1. 测样空池上的**生成覆盖**：多样本老师 + 校验后再 CE；或硬空池多 roll 自洽——**勿再全量 process**。  
2. 针对 leftover **把金标做厚**（仍是 gen）。  
3. 仅当测 cov **先 +≥3pp** 后再碰 O2/聚合。

**Do not：** 盲 `λ_mmd↑`；再同族 leftover-O2；再 N64/门控凑点；无 ≥37.69 证据宣称成功。

---

## 5. 仓库里有什么

本 pack **刻意不含** 权重与大体量缓存（完整树在本地 `LatentTTS-main/`）。

```
src/                  # 推理 / 生成 mixin / LatentRM / SF rollout / scoring
scripts/              # P123 训练、评测管线、挖难例、诊断
training_args/        # yaml 训练配置
results/full/
  PLAN.md             # 完整决策与数字（权威实验日志）
  p123/               # 评测 meta / mids（大 BoN dump 已抽成 meta）
logs/                 # 训练与 BoN 日志
```

### 关键脚本

| 脚本 | 作用 |
|------|------|
| `scripts/train_p123_system.py` | 一体 P123 训练 |
| `scripts/pipeline_p123_eval.sh` | 训完 → pilot + GSM N16 O2/B0 + mids |
| `scripts/mine_o2_hard_leftover.py` | gsm_train 挖 leftover（**禁止碰 test**） |
| `scripts/mine_k16_hard_questions.py` | gsm_train 标 empty/leftover 子集 |
| `scripts/train_o2_hard_rank.py` | 冻 gen，只训 O2 hinge |
| `src/infer_gpt2_rm.py` | BoN / wvote 评测 |
| `src/sf_rollout.py` | SF-B / SF-C |

### 最小评测示例（需本地权重）

```bash
python -m src.infer_gpt2_rm \
  --generator_type=coconut \
  --generator_id=outputs/p123_dualBeat_20261001_202106/model \
  --prm_id=outputs/latentrm_order_pref/best \
  --prm_mode=best_of_n \
  --data_path=data/gsm_test.json \
  --num_return_sequences=16 --seed=42 \
  --claim_agg=wvote --max_new_tokens=128 \
  --result_json=results/full/p123/bon_smoke.json
```

训练示例见 `scripts/train_p123_system.py --help`；历史配方参数写在各 `outputs/p123_*/train_args.json`（权重未打包）。

---

## 6. 与原 LatentTTS 的关系

| | 原论文仓库 | 本 pack |
|--|-----------|---------|
| 问题 | 并行 TTS + LatentRM | **在线一体**：SF +（可选）MMD + 选优闭环 |
| 主 claim | 论文表 | GSM (gen+O2) vs A+B0，冲 **+10pp** |
| 成功标准 | 论文数字 | 相对 dualBeat **wvote≥+1** 才考虑换锁 |
| 文档 | 原 README / arXiv | **本 README + `PLAN.md`** |

原论文：[Parallel Test-Time Scaling for Latent Reasoning Models](https://arxiv.org/abs/2510.07745) · 仅作底座引用。

---

## 7. 状态（写 README 时）

- 官方锁：**dualBeat + O2 = 36.69 wvote**  
- +10 目标未达成；聚合 / 同族 O2 / 朴素 coverage **已否证**  
- 研究重心转向：**测集难空池上的生成覆盖**（多样本/校验式老师信号）  
- 更细的逐步记录、杀伤判据、Do-not 列表 → [`results/full/PLAN.md`](results/full/PLAN.md)
