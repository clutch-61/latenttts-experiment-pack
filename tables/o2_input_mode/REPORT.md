# O2 输入配方：full / latent-only / text-only（+ SWIFT）

同池 N16、5-seed；`answers_sha1` 全对齐。Δ = seed-avg 后逐题 full−mode，bootstrap 95% CI。

## 结论（先读）

1. **保留 full-O2 输入**（题干+答案文本 + latent）。不要切到 text-only 或 latent-only 当默认。
2. **text-only 失败**：Top1 −3.7～−4.7pp；auc_cw −5.5～−6.8pp；`margin_cw` 严重塌缩（frozen mean≈−20）。文本 alone 不够，且会毁选择。
3. **latent-only**：Top1≈full（Δ CI 含 0）；auc_cw 掉到 SWIFT 带（full−lat Δauc ≈+2.6 / +3.5pp，CI 不含 0）。剥文本后丢失相对 SWIFT 的排序优势。
4. **含义**：full 的 auc 优势来自 **文本×latent 联合**；输入侧优化空间小，主瓶颈仍在选择（`margin_cw` / 最强错误）。

## 结论（分池）

- **frozen**：Top1 full/lat/txt/SWIFT = 33.7225/33.3131/29.0523/33.6922；auc_cw = 60.7911/57.4035/51.5001/57.4691。Δauc full−lat=2.6052 [0.4220,4.7908]；full−txt=6.7907 [3.7361,9.9490]；full−SWIFT=2.3359 [0.1513,4.5434]。
- **p2**：Top1 full/lat/txt/SWIFT = 35.9212/35.8453/32.2214/35.8757；auc_cw = 58.9078/55.4133/53.7659/56.0688。Δauc full−lat=3.5318 [1.4154,5.6242]；full−txt=5.4530 [2.2158,8.6130]；full−SWIFT=3.1874 [0.8612,5.4351]。

读法：text≪full 且 latent 丢 auc → **必须联合输入**；Top1 三模式里 full≈lat≈SWIFT、仅 text 塌 → 输入消融已收敛，下一刀打选择/margin。

## 汇总

```
pool	mode	top1_mean	top1_std	auc_cw_mean	auc_cw_std	margin_cw_mean	cov_mean
frozen	full	33.7225	1.0331	60.7911	2.6939	-0.169157	50.6141
frozen	latent	33.3131	0.2765	57.4035	2.4927	-0.161205	50.6141
frozen	text	29.0523	0.8601	51.5001	1.2295	-19.636647	50.6141
frozen	SWIFT	33.6922	0.7536	57.4691	1.8440	0.006841	50.6141
p2	full	35.9212	0.2754	58.9078	0.9979	-0.227050	53.7528
p2	latent	35.8453	0.5181	55.4133	0.6605	-0.210685	53.7528
p2	text	32.2214	0.7409	53.7659	1.2268	-12.029078	53.7528
p2	SWIFT	35.8757	0.4777	56.0688	0.2494	-0.013165	53.7528
```

## 配对 Δ(full − mode)

```
pool	contrast	metric	delta	lo	hi	n
frozen	full-latent	top1	0.4094	-0.5004	1.3343	1319
frozen	full-latent	auc_cw	2.6052	0.4220	4.7908	611
frozen	full-latent	margin_cw	0.0033	-0.2694	0.2759	611
frozen	full-text	top1	4.6702	3.4723	5.8984	1319
frozen	full-text	auc_cw	6.7907	3.7361	9.9490	611
frozen	full-text	margin_cw	20.6911	16.5158	25.0024	611
frozen	full-SWIFT	top1	0.0303	-0.9553	1.0311	1319
frozen	full-SWIFT	auc_cw	2.3359	0.1513	4.5434	611
frozen	full-SWIFT	margin_cw	-0.4016	-0.6909	-0.1096	611
p2	full-latent	top1	0.0758	-0.9098	1.0614	1319
p2	full-latent	auc_cw	3.5318	1.4154	5.6242	609
p2	full-latent	margin_cw	0.0953	-0.1430	0.3269	609
p2	full-text	top1	3.6998	2.3958	5.0948	1319
p2	full-text	auc_cw	5.4530	2.2158	8.6130	609
p2	full-text	margin_cw	13.2358	10.0303	16.4061	609
p2	full-SWIFT	top1	0.0455	-1.0008	1.0466	1319
p2	full-SWIFT	auc_cw	3.1874	0.8612	5.4351	609
p2	full-SWIFT	margin_cw	-0.3089	-0.5485	-0.0712	609
```
