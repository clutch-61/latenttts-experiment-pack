# 在线一体 Latent TTS：Self-Forcing × 分布对齐 × 选优闭环

本仓库记录的是我们在 **LatentTTS / COCONUT + LatentRM** 底座上做的**后续研究**，不是原论文的复现说明。

原系统大致是：生成器冻住 → 用 LatentRM 做 Best-of-N。我们关心的是另一件事——

> **能不能把「怎么想」（生成器 latent）和「怎么选」（奖励模型）放进同一条在线训练环，并在 GSM8K 系统评测上相对强基线再涨一大截？**

下面按「问题 → 想法 → 怎么训 → 训出了什么 → 卡在哪 → 代码怎么用」展开。更细的逐步决策与原始数字在 [`results/full/PLAN.md`](results/full/PLAN.md)。

---

## 目录

1. [研究问题与成功标准](#1-研究问题与成功标准)
2. [新 Idea：三层一体（P1 / P2 / P3）](#2-新-idea三层一体p1--p2--p3)
3. [评测协议（读数前必看）](#3-评测协议读数前必看)
4. [已经站住的结果](#4-已经站住的结果)
5. [实验叙事：我们试了什么、为什么失败](#5-实验叙事我们试了什么为什么失败)
6. [当前瓶颈：7 个百分点拆开看](#6-当前瓶颈7-个百分点拆开看)
7. [下一步该攻哪里（以及不该再做什么）](#7-下一步该攻哪里以及不该再做什么)
8. [仓库内容与怎么跑](#8-仓库内容与怎么跑)
9. [和原 LatentTTS 论文的关系](#9-和原-latenttts-论文的关系)

---

## 1. 研究问题与成功标准

### 1.1 问题设定

给定一道数学题，COCONUT 类模型先在连续 latent 空间里「想」，再解码出答案。测试时并行采 N 条轨迹，用 LatentRM 打分，再聚合成最终 claim。

强基线记为 **A + B0**：

- **A**：原始 COCONUT 生成器  
- **B0**：基线 LatentRM（不做顺序偏好）  
- GSM8K-Test · BoN N=16 · seed=42 上，A+B0 ≈ **33.74%**

我们要的不是「再训一个略好的 RM」，而是一个**可一起训练的系统**，在同一套协议上把 claim 往上推。

### 1.2 量化目标

| 项目 | 数值 |
|------|------|
| 基线 A+B0 | 33.74% |
| 目标增益 | 约 **+10pp** |
| 目标 claim | 约 **43.74%** |
| 当前官方锁（dualBeat+O2） | **36.69%**（wvote） |
| 相对 A+B0 已有增益 | **+2.95pp** |
| 距目标还差 | 约 **7pp** |

### 1.3 什么叫「过线」

小改配方很容易在噪声里看起来「涨了一点」。我们约定：

- **主指标：** claim = **加权投票 wvote**（对答案做 softmax 加权，温度 T=1；轨迹分用 `sum_logit`）  
- **杀伤线：** 新系统相对官方锁 dualBeat，wvote 至少 **+1pp**（≥ **37.69%**）  
- 没过线 → **不更换官方锁**，不对外宣称成功  
- 报数时必须分开：**生成器本身** vs **+O2**；并用 **+B0** 做健康对照（避免 RM 崩了却只看 O2）

N=16 池上「完美选优」上限等于覆盖率 cov ≈ **53.4%**。目标 43.7% 落在这个上限里面——所以不是算术上不可能，而是**现有刀法没碰到真正杠杆**。

---

## 2. 新 Idea：三层一体（P1 / P2 / P3）

我们把系统拆成三层，每一层都可以单独打开、单独证伪。代码主入口是 `scripts/train_p123_system.py`。

### 2.1 P3 — Self-Forcing（生成器怎么学「想」）

推理时 latent 是 rollout 出来的；若训练只做普通 teacher-forcing，训练和推理会错位。

**SF-C（默认）：** 训练时做与推理一致的可微 latent rollout，再在答案（及可选过程）上做 CE。梯度可以回到 latent 步骤。

**SF-B：** 用缓存/选中的 latent 当前缀再 CE。实现简单，但 latent 往往被 detach，实验里会把对 latent 的梯度几乎掐死——**不作为默认**。

可选 **`λ_lat`**：让 live rollout 的 latent 靠近「本步选中的正轨迹 latent」（老师或 beat 出来的样本），减轻「答案对了但想法漂了」。

### 2.2 P1 — 选优与 leftover（池里有答案时谁该被强化）

在线采样 k 条学生轨迹，用冻结或可训的 O2（顺序偏好 LatentRM）和 B0 打分，再决定本步 CE 学谁：

| `select_mode` | 行为（直觉） |
|---------------|--------------|
| `dual_beat` | 学生某条又对、O2 明显好于老师、B0 也不比老师差太多 → 学自己（self-beat）；否则在门控下学老师；否则 skip |
| `coverage` | 若学生池**完全没有**正确答案、老师有 → **强制**学老师（先救覆盖）；其余回退 dual_beat |
| `wvote` | 若加权投票选出的答案是对的 → 学该答案下 O2 最高的正确轨迹（训选对齐 claim） |

**leftover：** 池里其实有金标，但 O2 top / claim 仍选错。这是 P1 的核心债。  
可选 **`λ_rank`**：在混合 k 池上对「最好金标 vs 最强盗贼」做 sum_logit hinge，试图直接拧排序。

### 2.3 P2 — 分布对齐（MMD）

即使答案 CE 对了，student 的 latent「内容」仍可能漂到另一区域，让 O2（在旧分布上训的）失效。

**想法：** 把正轨迹 latent 投到内容视图，用多尺度 RBF MMD 拉近 student 与正样本。正样本优先级大致是：老师做对时的 teacher latents → 缓存正样本 → 本步选中的正确轨迹。

`λ_mmd` 控制强度；我们有 ramp，且**禁止**在没有信号时盲把 λ 拉到很大。

### 2.4 三层如何串成一条链

一次更新的典型顺序：

1. 学生与老师采样（学生 k 条，老师 1 条）  
2. O2 / B0 打分  
3. `select_mode` 决定 CE 目标文本（及对应 latents）  
4. SF-C 前向 + 可选 `λ_lat` / `λ_mmd` / `λ_rank`  
5. 反传更新学生（O2 仅在 `λ_rank>0` 时更新）

**一体接线本身已经被跑通**（joint / jrank）：中间量里 MMD 会开火、rank hinge 会触发。但「接线成功」≠「claim 过杀伤」——这是后文实验的主旋律。

---

## 3. 评测协议（读数前必看）

除非某一节显式改协议，否则默认：

| 项 | 值 |
|----|-----|
| 数据 | `data/gsm_test.json`（GSM8K-Test） |
| N | 16（BoN） |
| seed | 42 |
| 生成 | COCONUT · `max_new_tokens=128` |
| RM 分 | `sum_logit`（对 latent 位置 logit 求和） |
| claim | `wvote` |
| 对照 | 同池 +B0；相对 A+B0 的 Δ |

辅助量：

- **cov：** 至少一条轨迹答案正确的题占比（完美选优上限）  
- **leftover：** 有正确轨迹但 claim/top 仍错  
- **pick_fail | pool：** 在有池子集上选错的比例  

训练中间量写在各 run 的 `mid_metrics.jsonl`（pool、leftover、hash3/短答率、sources 计数等）。汇总脚本：`scripts/diag_p123_mids.py`。

---

## 4. 已经站住的结果

这些是我们愿意写进「当前系统」的结论，不是过程中的毛刺。

### 4.1 官方锁：dualBeat + O2

**dualBeat** 的要点：在线用 O2+B0 双门做 self-beat，赢了就学自己，否则在门控下学老师；生成器用 Self-Forcing 更新。O2 在锁上是**冻结**的原顺序偏好模型。

| 指标 | dualBeat+O2 |
|------|------------:|
| top | 35.41% |
| **wvote（锁）** | **36.69%** |
| cov | 53.37% |
| vs A+B0 | **+2.95pp** |

相对更早的「假迁移 / 乱叠损失」配方，dualBeat 把系统尺扭正了，因此被定为**官方锁**。之后所有新刀，都先和 36.69 比。

### 4.2 上限告诉我们：还有空间

N=16 上 cov≈53.4%。若选优完美，claim 理论上可以到 53% 附近。目标 43.7% **低于**这个上限，所以「+10 绝对到不了」不成立；卡点在**覆盖不足 + 难例选不出来**。

### 4.3 O2 不是废的，但也不是 +7pp 的杠杆

Phase-1 机制评测里，O2 在顺序破坏（相邻交换等）上明显强于 B0——说明偏好训练学到了结构信号。  
但在 dualBeat 的测试 leftover 上，换多种 O2 续训，**稳定有约 172 题怎么换都不翻**。O2 侧我们见过的最好组合是：

> dualBeat 生成器 + jrank 训出的 O2 → wvote **37.00%**

只比锁高 0.31pp，**没过 +1pp 杀伤**。所以：O2 有用，但**再拧 O2 不够吃掉剩余 7pp**。

---

## 5. 实验叙事：我们试了什么、为什么失败

这一节按「我们当时以为杠杆在哪」组织，而不是按时间流水账。数字默认 GSM-Test N16 s42 · wvote。

### 5.1 生成器变体：在 dualBeat 附近打转

在 dualBeat 配方上加 λ_lat、抬多样性、过程监督、收紧 B0 门、再续训 1500 步（X2）等：

- **结论：** 没有超过 dualBeat 锁。  
- **读法：** 这条 self-beat 配方上，生成器已经接近饱和；「输出更像过程」「unique 更高」不等于系统 claim 更高（Div 甚至偏 teacher 克隆；Proc 改形状但不涨点）。

### 5.2 训选对齐：cover / wvote-CE

想法：训练时选的目标和测试 claim 不一致，所以要对齐。

- coverage 早期版、wvote 对齐 CE：相对锁大约 **+0.2～0.3pp**  
- **结论：** 方向不荒唐，但不是 7pp 级杠杆；**训目标≠测空池分布**。

### 5.3 修缝与协议：SF-B、clat、换打分家族、推理 dropout

- **SF-B：** 修「采样和反传不是一条路」时若走到缓存 latent，会把 `g_lat` 打没 → 否证为默认。  
- **clat 等修缝：** 最多约 +0.16pp。  
- **claim 改用 mean/sum log_prob：** 掉点 → 仍用 `sum_logit`。  
- **推理强制 dropout=0.2（对齐训练）：** cov 升到 ~56%，wvote 反而到 **34.95%**——覆盖上去了，转化变差。

**读法：** 协议缝存在，但不是主战场。

### 5.4 一体接线 joint / jrank

按用户要求把三 Phase 真正接到一条链：

| 系统 | wvote | 相对锁 | 说明 |
|------|------:|-------:|------|
| dualBeat（锁） | 36.69 | — | O2 冻 |
| joint（冻 O2 + MMD0.1 + SF-C + λ_lat） | 36.32 | −0.37 | 接线成功，claim 未过线 |
| jrank（上 + 可训 leftover hinge） | 36.92 | +0.23 | 未过 +1；top 掉 |

交叉评测表明：jrank 那点微涨主要来自**新 O2**，jrank 生成器自己更差。Hinge 能把 leftover 上的分差压小，但**名次常常翻不过来**；测试 leftover 条数几乎不动。

**读法：** 「三 Phase 都开火」≠「系统涨点」。不要为了一体而盲抬 `λ_mmd`。

### 5.5 同族 O2 攻坚：on-policy / hard / 离线 leftover hinge

思路：O2 没见过 dualBeat 的候选分布，所以在新分布上重标、或专门挖 leftover 做 hinge。

- 最好仍是 dualBeat + jrank O2 → **37.00**  
- 离线 `o2hardrank`：在 train leftover 上可以学到排序，换到 test 的 172 道稳定 leftover 上**救与弄坏大致抵消**  
- train 挖到的 leftover「太容易」（gap 中位 ~0.44），test 稳定 leftover「太难」（gap 中位 ~1.83）——**不是同一种题**

**读法：** 同族 O2 工作基本停；再挖同源 leftover 也是空转。

### 5.6 聚合与自适应 N：dump 上把决策层扫穿

有人会想：锁已经 36.69，也许换投票、难例多采一点就能到 43。

我们在 **已有** dualBeat N16 / N64 dump 上做了不泄漏金标的扫描：

| 策略 | 大约结果 | 含义 |
|------|----------|------|
| 官方 N16 wvote | 36.69 | 锁 |
| 特征门控改多数票 | ≤36.85 | 噪声 |
| 全量 N64 wvote | 37.00 | cov 61 但转化掉 |
| 先知空池再采到 64 | 36.24 | 加采不转化 |
| 特征门控 extra-N | ≤37.15 | 和盲 N64 差不多 |
| leftover 改 MV（**作弊**） | ~39.3 | 连「理想决策」也到不了 43.7 |
| 拼 A 的轨迹填空池 | 只盖住 dualBeat 空池的 ~9% | 两套生成器空池重叠很小 |

**读法：聚合层封死。** 不要再改 `infer_gpt2_rm` 上线 entropy 门控指望 +10。

### 5.7 覆盖再攻：k=16 coverage / process / 难子集

瓶颈重判之后，主攻「训练见过的空池太少」。

| 配方 | wvote | 发生了什么 |
|------|------:|------------|
| k16cov（强制 k=16，coverage，少训已赢题） | 36.32 | 训练里 teacher:beat 接近 1:1，**测试 cov 几乎不动** |
| k16proc（上 + 全量过程监督） | **33.43** | hash3 没了，但变成超长 teacher-CoT 克隆，claim 更差 |
| infer dp0.2 | 34.95 | 见上 |

**读法：** 「协议匹配到 N16 + 空池强制学老师」**不足以**迁移到测试空池；全量 process 有害。单条老师文本 CE，打不穿主墙。

---

## 6. 当前瓶颈：7 个百分点拆开看

把「锁 36.69 → 目标 43.7」还差的约 7pp，按错误类型拆开更清楚：

### 6.1 主墙：测集空池 ≈ 46.6%

大约一半测试题在 N=16 下**一条金标都没有**。  
训练时 k=64、题偏易，pool 经常 0.9+——模型几乎总在「已经有正确轨迹」的世界里做 self-beat。  
于是：再好的 O2 也救不了空池；再对齐 claim 也变不出答案。

k16cov 证明：就算训练时故意 k=16 并强制学老师，**测试覆盖仍几乎不动**。说明缺的不是「再开一个 select 分支」，而是**在测样难题上生成出至少一条对的能力**（或等价的多尝试 + 校验信号）。

### 6.2 次墙：稳定 leftover ≈ 13%（约 172 题）

这些题池里有对的，但多个 O2 都选错，且分差大、金标条数往往很少。  
Hinge / 离线难负例压得了 train 上的小 gap，压不动 test 上的硬 thief。

这里真正需要的是：**同题更多正确轨迹、或正确轨迹在分数上更好认**——本质还是生成，不是再买一个 RM。

### 6.3 有池选错的剩余空间

在「有池」子集里，选错大约占三分之一。O2 侧我们把能拔的毛拔到约 **37.0**，再往上极度困难。  
**只有覆盖先上台阶，再回头打选优才有意义。**

---

## 7. 下一步该攻哪里（以及不该再做什么）

### 7.1 值得做的分叉（按优先级）

1. **测样空池上的生成覆盖**  
   - 多样本老师轨迹，经校验（答案对）再 CE，而不是单条克隆  
   - 或硬空池上多 roll + 外部/自洽校验  
   - **不要**再上全量 `target_mode=process`（已证明有害）

2. **把 leftover 题的金标做厚**  
   让现有 O2 自然选中，而不是指望 O2 学会一种它在 train 里见不到的分差结构。

3. **cov 先 +≥3pp 之后**，才重新考虑 O2 / 聚合小刀。

### 7.2 明确进入「停做」清单

- 盲抬 `λ_mmd`  
- 再做同族 leftover-O2 / 同源 hinge  
- 再靠 N64、entropy 门控、拼 A 候选凑点  
- 再全量 process 监督当主损失  
- 没有 ≥37.69 的证据就锁新官方或写成功叙事  

进行中的收尾（k16hard / k16covlat 等）若再 FAIL，按上一节分叉切，而不是继续拧 P123 旋钮。

---

## 8. 仓库内容与怎么跑

### 8.1 这个 GitHub pack 里有什么

为控制体积，**不包含**模型权重、`latent-data` 标注缓存、完整 BoN 逐题 dump（过大 JSON 只保留 meta）。完整产物在训练机器本地的 `LatentTTS-main/`。

```text
src/                 推理、生成 mixin、LatentRM、SF rollout、打分与 wvote
scripts/             P123 训练、评测管线、挖难例、诊断
training_args/       训练 yaml
results/full/
  PLAN.md            完整实验决策日志（权威）
  p123/              评测 meta / mids 等
logs/                训练与 BoN 日志
```

### 8.2 关键入口

| 路径 | 作用 |
|------|------|
| `scripts/train_p123_system.py` | 一体训练（P1/P2/P3 开关都在参数里） |
| `scripts/pipeline_p123_eval.sh` | 训完自动跑 pilot + GSM N16 O2/B0 + mids |
| `scripts/diag_p123_mids.py` | 汇总系统尺与训练中间量 |
| `scripts/mine_o2_hard_leftover.py` | 仅在 **gsm_train** 挖 leftover（禁止碰 test） |
| `scripts/mine_k16_hard_questions.py` | 标 train 上 empty / leftover 子集 |
| `scripts/train_o2_hard_rank.py` | 冻生成器，只训 O2 hinge |
| `src/infer_gpt2_rm.py` | BoN + wvote 评测 |
| `src/sf_rollout.py` | SF-B / SF-C 实现 |
| `src/system_scoring.py` | `sum_logit`、wvote 等 |

### 8.3 评测示例（需本地权重）

```bash
python -m src.infer_gpt2_rm \
  --generator_type=coconut \
  --generator_id=outputs/p123_dualBeat_20261001_202106/model \
  --prm_id=outputs/latentrm_order_pref/best \
  --prm_model_family=gpt2 \
  --prm_mode=best_of_n \
  --data_path=data/gsm_test.json \
  --num_return_sequences=16 \
  --seed=42 \
  --claim_agg=wvote \
  --max_new_tokens=128 \
  --result_json=results/full/p123/bon_smoke.json
```

### 8.4 训练示例（示意）

```bash
python scripts/train_p123_system.py \
  --ckpt outputs/p123_dualBeat_20261001_202106/model \
  --teacher_ckpt checkpoints/coconut \
  --o2_id outputs/latentrm_order_pref/best \
  --b0_id outputs/latentrm_baseline/best \
  --k 16 --no_auto_pack \
  --select_mode coverage \
  --easy_keep_prob 0.25 \
  --lambda_mmd 0 --lambda_lat 0 --lambda_rank 0 \
  --sf_mode C \
  --max_steps 2000 --lr 3e-6
```

具体历史配方以各次 run 的 `train_args.json` 与 `PLAN.md` 为准。多卡时可用 `--frozen_device`（老师+B0 另一卡）、`--gen_replica_device`（k 对半生成）；请避开占用冲突的 GPU。

---

## 9. 和原 LatentTTS 论文的关系

| | 原 LatentTTS | 本仓库研究 |
|--|-------------|------------|
| 核心问题 | 并行 test-time scaling + LatentRM | **在线一体**：SF + 可选 MMD + 选优闭环 |
| 主结果叙事 | 论文表 / BoN·beam | GSM 上 (gen+O2) 相对 A+B0，冲 **+10pp** |
| 当前锚点 | 论文发布数字 | **dualBeat+O2 = 36.69 wvote** |
| 成功门槛 | 论文设定 | 相对锁 **wvote≥+1pp** 才考虑换锁 |
| 文档 | 原项目 README / arXiv | **本 README + PLAN.md** |

原工作仍是我们的生成与打分基础设施，引用：

[Parallel Test-Time Scaling for Latent Reasoning Models](https://arxiv.org/abs/2510.07745)

我们不在本 README 里重复原论文的安装广告与模型 zoo；需要底座细节时再回看原仓库。

---

## 一句话现状

> **一体链能跑通，系统锁在 dualBeat+O2 的 36.69；再叠 λ、再训 O2、再改聚合，都吃不掉剩下的约 7pp。剩下的主战场是测试集空池上的生成覆盖——先让难题池子里出现金标，再谈选优。**

更细的逐日记录、否证表、Do-not 列表：[`results/full/PLAN.md`](results/full/PLAN.md)。
