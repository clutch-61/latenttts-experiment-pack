# O2：AUC 优势为何未变成 Top-1（同池 dump）

pre_registered_split: val=idx%5==0 n_val=264 n_test=1055 n_all=1319

## 结论（先读）

1. **池覆盖**：约一半题 N=16 池内无正解（frozen cov≈50.2654%，p2≈52.6156%）。这是 Top-1 天花板，与 scorer 无关。
2. **有正解时**：几乎全部失败模式是 `lost_to_strongest_wrong`（among_ach 约 34.0875% / 32.8530%），不是「认出了但打平输」。正确候选在失败题上的平均名次约 5.8035 / 5.4395（N=16）—— **识别到了排序空间里的信号，但没赢过最强错误，且经常不在头部**。
3. **among_ach 最佳正解进 top2/top4**：frozen 68.3258%/79.0347%，p2 69.7406%/80.9798%。AUC 可升、Top-1 不动，符合「整体排序变好但 argmax 仍被最强错误占住」。
4. **救回**：p2 上 O2 救 B0 明显多于反向；frozen 接近对称。融合改次序在锁定 test 上 **≈0**。
5. **下一步做强 O2**：优先 listwise / 对最强错误的 margin（推高 `margin_cw`），并盯 `n_correct∈{1,2,3}`；不要指望全局温度或简单 O2+B0 分数融合。

## 读数纪律

- 全局温度缩放不改变单 scorer 的 argmax，故不测。
- 融合 / 分歧重排在 **val**（idx%5==0）选超参，**test** 只报一次。
- `lost_to_strongest_wrong`：池内有正解，但 max(score_wrong) > max(score_correct)。
- `O2_rescue_B0` / `B0_rescue_O2`：多数 seed 上一方对、另一方错。
- `among_ach_*` 分母 = 多数 seed 池内有 ≥1 正解的题。

## 总表

```
pool	split	n	cov	O2_top1	B0_top1	O2_rescue_B0	B0_rescue_O2	disagree_rate	among_ach_lost_strong	among_ach_win	mean_margin_cw_when_lost	mean_rank_best_correct_when_lost	among_ach_best_correct_top2	among_ach_best_correct_top4
frozen	all	1319	50.2654	33.0553	32.8279	2.5019	2.2745	69.2343	34.0875	65.7617	-2.130782	5.8035	68.3258	79.0347
frozen	val	264	44.3182	34.0909	34.0909	1.5152	1.5152	70.5303	23.0769	76.9231	-2.189958	5.6716	79.4872	88.0342
frozen	test	1055	51.7536	32.7962	32.5118	2.7488	2.4645	68.9100	36.4469	63.3700	-2.122794	5.8213	65.9341	77.1062
p2	all	1319	52.6156	34.9507	33.0553	3.7149	1.8196	71.1145	32.8530	66.4265	-1.711057	5.4395	69.7406	80.9798
p2	val	264	47.7273	36.7424	34.0909	4.1667	1.5152	73.2576	22.2222	76.9841	-1.733104	4.8483	81.7460	92.0635
p2	test	1055	53.8389	34.5024	32.7962	3.6019	1.8957	70.5782	35.2113	64.0845	-1.707923	5.5235	67.0775	78.5211
```

锁定 test：frozen O2 Top1=32.7962 B0=32.5118；p2 O2=34.5024 B0=32.7962。

## 失效模式分布（O2，分母=全题）

```
pool	split	mode	n	frac
frozen	all	lost_to_strongest_wrong	226	17.1342
frozen	all	uncovered	657	49.8105
frozen	all	win	436	33.0553
p2	all	lost_to_strongest_wrong	228	17.2858
p2	all	uncovered	630	47.7635
p2	all	win	461	34.9507
```

## 切片（all + locked test）

```
pool	split	slice	n	O2_top1	B0_top1	O2_rescue	B0_rescue	note
frozen	all	n_correct=1	89	6.7416	11.2360	1.1236	5.6180	
frozen	all	n_correct=2-3	80	26.2500	21.2500	11.2500	6.2500	
frozen	all	n_correct>=4	494	82.7935	82.1862	4.6559	4.0486	
frozen	all	margin12_low	227	78.8546	78.4141	3.9648	3.5242	O2 top1-top2 gap
frozen	all	margin12_mid	219	70.7763	70.3196	5.9361	5.4795	
frozen	all	margin12_high	217	47.0046	46.5438	5.0691	4.6083	
frozen	all	scorer_disagree	966	33.2298	32.8157	3.0021	2.5880	
frozen	all	scorer_agree	353	32.5779	32.8612	1.1331	1.4164	
frozen	all	O2_rescue_B0	33	100.0000	0.0000	100.0000	0.0000	B0 wrong O2 right
frozen	all	B0_rescue_O2	30	0.0000	100.0000	0.0000	100.0000	O2 wrong B0 right
frozen	test	n_correct=1	80	7.5000	10.0000	1.2500	3.7500	
frozen	test	n_correct=2-3	69	24.6377	18.8406	11.5942	5.7971	
frozen	test	n_correct>=4	397	81.3602	81.1083	5.0378	4.7859	
frozen	test	margin12_low	183	77.5956	78.1421	3.2787	3.8251	O2 top1-top2 gap
frozen	test	margin12_mid	181	66.2983	64.6409	7.1823	5.5249	
frozen	test	margin12_high	182	46.1538	45.6044	5.4945	4.9451	
frozen	test	scorer_disagree	766	32.7676	32.1149	3.3943	2.7415	
frozen	test	scorer_agree	289	32.8720	33.5640	1.0381	1.7301	
frozen	test	O2_rescue_B0	29	100.0000	0.0000	100.0000	0.0000	B0 wrong O2 right
frozen	test	B0_rescue_O2	26	0.0000	100.0000	0.0000	100.0000	O2 wrong B0 right
p2	all	n_correct=1	59	6.7797	6.7797	1.6949	1.6949	
p2	all	n_correct=2-3	91	24.1758	13.1868	14.2857	3.2967	
p2	all	n_correct>=4	544	79.9632	77.2059	6.4338	3.6765	
p2	all	margin12_low	231	80.5195	78.3550	4.3290	2.1645	O2 top1-top2 gap
p2	all	margin12_mid	232	68.9655	63.7931	9.4828	4.3103	
p2	all	margin12_high	231	49.7835	46.3203	7.3593	3.8961	
p2	all	scorer_disagree	1030	36.6990	34.5631	4.1748	2.0388	
p2	all	scorer_agree	289	28.7197	27.6817	2.0761	1.0381	
p2	all	O2_rescue_B0	49	100.0000	0.0000	100.0000	0.0000	B0 wrong O2 right
p2	all	B0_rescue_O2	24	0.0000	100.0000	0.0000	100.0000	O2 wrong B0 right
p2	test	n_correct=1	51	7.8431	7.8431	1.9608	1.9608	
p2	test	n_correct=2-3	79	20.2532	11.3924	12.6582	3.7975	
p2	test	n_correct>=4	438	78.5388	76.0274	6.1644	3.6530	
p2	test	margin12_low	194	79.3814	78.8660	3.0928	2.5773	O2 top1-top2 gap
p2	test	margin12_mid	188	64.8936	59.0426	10.1064	4.2553	
p2	test	margin12_high	186	47.3118	44.0860	6.9892	3.7634	
p2	test	scorer_disagree	811	36.6215	34.8952	3.9457	2.2195	
p2	test	scorer_agree	244	27.4590	25.8197	2.4590	0.8197	
p2	test	O2_rescue_B0	38	100.0000	0.0000	100.0000	0.0000	B0 wrong O2 right
p2	test	B0_rescue_O2	20	0.0000	100.0000	0.0000	100.0000	O2 wrong B0 right
```

读切片：`n_correct=1` Top1 极低；`n_correct≥4` ~80%。`margin12_high` Top1 更低 —— 大 top1−top2 分差经常是「自信地错」。分歧题上 O2 相对 B0 的救回更明显（尤其 p2）。

## 候选内融合（改相对次序）

frozen: val_best=rank_avg α=0.60 val=35.15 → test=33.61 (test O2=33.55 B0=32.57 ΔvsO2=+0.06)
p2: val_best=rank_avg α=0.70 val=38.94 → test=35.20 (test O2=35.28 B0=34.03 ΔvsO2=-0.08)

全网格见 `fusion_grid.tsv`。val 选出的融合在 test 上相对纯 O2 ≈0；个别非 VAL_BEST 格子（如 frozen score_z α=0.7）test 略高，但未过 val 选模门。
