# O2 top-k → B0 二阶段重排

frozen: val_best=b0_in_topk k=6 α=1.00 val=34.70 → test=33.76 (test O2=33.55 Δ=+0.21)
  oracle top2 on test=37.57 (headroom vs O2=+4.02)
  oracle top3 on test=40.13 (headroom vs O2=+6.58)
  oracle top4 on test=41.91 (headroom vs O2=+8.36)
p2: val_best=b0_in_topk k=2 α=1.00 val=38.64 → test=35.64 (test O2=35.28 Δ=+0.36)
  oracle top2 on test=39.91 (headroom vs O2=+4.63)
  oracle top3 on test=42.52 (headroom vs O2=+7.24)
  oracle top4 on test=44.44 (headroom vs O2=+9.16)

全网格: `topk_rerank_grid.tsv`。
oracle = 短名单内若有正解则选中（上界，非可部署）。
