"""Per-question paired comparison of latent scorers (same candidate pool), 5-seed.

Reads the uniform dir ``results/full/scorer_axis_5seed/{pool}_{scorer}_N{n}_s{seed}.json``
(built by ``assemble_scorer_axis_5seed.sh`` + ``extend_scorer_axis_seed.sh``).

Candidate pools are verified identical across scorers via ``answers_sha1`` before any
comparison. Per question q we compute top1 / top1_ach / auc_cw / mrr / margin_cw / cov.

Paired tests for O2 vs a challenger C:
  - per-seed McNemar exact on discordant top1 pairs
  - per-seed paired bootstrap CI on mean(top1_O2 - top1_C) and mean(auc_cw diff)
  - cross-seed: average each question over seeds, then one paired bootstrap CI

Outputs:
  results/full/scorer_axis_5seed/paired_ranking_stats.tsv
  results/full/scorer_axis_5seed/paired_ranking_pairs.tsv
  results/full/scorer_axis_5seed/paired_ranking_crossseed.tsv
  results/full/scorer_axis_5seed/paired_ranking_crossseed_pairs.tsv
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/full/scorer_axis_5seed"
SEEDS = [42, 43, 44, 45, 46]
POOLS = ["frozen", "p2"]
SCORERS = ["B0", "O2", "SWIFT", "HSRM"]


def path_for(pool: str, scorer: str, n: int, seed: int) -> Path:
    return D / f"{pool}_{scorer}_N{n}_s{seed}.json"


def load(p: Path) -> Dict[int, dict]:
    d = json.load(open(p))
    return {e["idx"]: e for e in d["examples"]}


def per_question_metrics(ex: dict) -> Dict[str, float]:
    scores = np.asarray(ex["scores"], dtype=np.float64)
    corr = np.asarray(ex["corrects"], dtype=bool)
    out = {
        "top1": float(bool(ex["selected_correct"])),
        "top1_ach": float("nan"),
        "mrr": 0.0,
        "auc_cw": float("nan"),
        "margin_cw": float("nan"),
        "cov": float(bool(ex["any_correct"])),
    }
    if not corr.any():
        return out
    out["top1_ach"] = float(bool(ex["selected_correct"]))
    order = np.argsort(-scores, kind="stable")
    rank = int(np.where(corr[order])[0][0]) + 1
    out["mrr"] = 1.0 / rank
    n_pos, n_neg = int(corr.sum()), len(corr) - int(corr.sum())
    if n_pos and n_neg:
        s_c, s_w = scores[corr], scores[~corr]
        diff = s_c[:, None] - s_w[None, :]
        out["auc_cw"] = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
        out["margin_cw"] = float(s_c.max() - s_w.max())
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def boot_ci(a, b, reps, rng):
    m = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[m], b[m]
    d = a - b
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    idx = rng.integers(0, n, size=(reps, n))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    rng = np.random.default_rng(args.boot_seed)

    stats = ["pool\tseed\tscorer\tn_q\ttop1\ttop1_ach\tauc_cw\tmrr\tmargin_cw\tcov"]
    pairs = ["pool\tseed\tpair\tmetric\tdelta_pp\tci_lo\tci_hi\tn\tmcnemar_p\tdisc_ab\tdisc_ba"]
    # pool -> scorer -> metric -> seed -> array
    cross: Dict[str, Dict[str, Dict[str, Dict[int, np.ndarray]]]] = {}

    for pool in POOLS:
        cross[pool] = {}
        for seed in SEEDS:
            paths = {s: path_for(pool, s, args.n, seed) for s in SCORERS}
            missing = [s for s, p in paths.items() if not p.exists()]
            if missing:
                print(f"[{pool} s{seed}] SKIP missing {missing}")
                continue
            data = {s: load(p) for s, p in paths.items()}
            common = set.intersection(*[set(d.keys()) for d in data.values()])
            mismatch = 0
            for idx in common:
                ref = data["O2"][idx]["answers_sha1"]
                for s in data:
                    if data[s][idx]["answers_sha1"] != ref:
                        mismatch += 1
            print(f"[{pool} s{seed}] n_common={len(common)} sha1_mismatch={mismatch}")

            per_q: Dict[str, Dict[str, np.ndarray]] = {}
            idxs = sorted(common)
            for s, d in data.items():
                recs = [per_question_metrics(d[i]) for i in idxs]
                per_q[s] = {k: np.array([r[k] for r in recs], dtype=np.float64)
                            for k in ("top1", "top1_ach", "auc_cw", "mrr", "margin_cw", "cov")}
                cross[pool].setdefault(s, {})
                for k, arr in per_q[s].items():
                    cross[pool][s].setdefault(k, {})[seed] = arr
                stats.append(
                    f"{pool}\t{seed}\t{s}\t{len(idxs)}\t{per_q[s]['top1'].mean()*100:.4f}\t"
                    f"{np.nanmean(per_q[s]['top1_ach'])*100:.4f}\t{np.nanmean(per_q[s]['auc_cw'])*100:.4f}\t"
                    f"{per_q[s]['mrr'].mean()*100:.4f}\t{np.nanmean(per_q[s]['margin_cw']):.6f}\t"
                    f"{per_q[s]['cov'].mean()*100:.4f}"
                )

            for ch in ("SWIFT", "HSRM", "B0"):
                a, b = per_q["O2"]["top1"], per_q[ch]["top1"]
                d, lo, hi, n = boot_ci(a, b, args.reps, rng)
                ab = int(((a == 1) & (b == 0)).sum())
                ba = int(((a == 0) & (b == 1)).sum())
                p = mcnemar_exact(ab, ba)
                pairs.append(f"{pool}\t{seed}\tO2-{ch}\ttop1\t{d*100:.4f}\t{lo*100:.4f}\t{hi*100:.4f}\t"
                             f"{n}\t{p:.6f}\t{ab}\t{ba}")
                da, loa, hia, na = boot_ci(per_q["O2"]["auc_cw"], per_q[ch]["auc_cw"], args.reps, rng)
                pairs.append(f"{pool}\t{seed}\tO2-{ch}\tauc_cw\t{da*100:.4f}\t{loa*100:.4f}\t{hia*100:.4f}\t"
                             f"{na}\t-\t-\t-")

    # ---- cross-seed: average each question over available seeds, one paired CI ----
    cross_rows = ["pool\tscorer\tn_seeds\ttop1\ttop1_ach\tauc_cw\tmrr\tcov"]
    cross_pairs = ["pool\tpair\tmetric\tdelta_pp\tci_lo\tci_hi\tn"]
    for pool in POOLS:
        seed_avg: Dict[str, Dict[str, np.ndarray]] = {}
        for s in cross[pool]:
            have = sorted(cross[pool][s]["top1"].keys())
            seed_avg[s] = {
                k: np.nanmean(np.stack([cross[pool][s][k][sd] for sd in have]), axis=0)
                for k in ("top1", "top1_ach", "auc_cw", "mrr", "cov")
            }
            cross_rows.append(
                f"{pool}\t{s}\t{len(have)}\t{np.nanmean(seed_avg[s]['top1'])*100:.4f}\t"
                f"{np.nanmean(seed_avg[s]['top1_ach'])*100:.4f}\t{np.nanmean(seed_avg[s]['auc_cw'])*100:.4f}\t"
                f"{np.nanmean(seed_avg[s]['mrr'])*100:.4f}\t{np.nanmean(seed_avg[s]['cov'])*100:.4f}"
            )
        for ch in ("SWIFT", "HSRM", "B0"):
            for metric in ("top1", "top1_ach", "auc_cw", "mrr"):
                a, b = seed_avg["O2"][metric], seed_avg[ch][metric]
                d, lo, hi, n = boot_ci(a, b, args.reps, rng)
                cross_pairs.append(f"{pool}\tO2-{ch}\t{metric}\t{d*100:.4f}\t{lo*100:.4f}\t{hi*100:.4f}\t{n}")

    (D / "paired_ranking_stats.tsv").write_text("\n".join(stats) + "\n")
    (D / "paired_ranking_pairs.tsv").write_text("\n".join(pairs) + "\n")
    (D / "paired_ranking_crossseed.tsv").write_text("\n".join(cross_rows) + "\n")
    (D / "paired_ranking_crossseed_pairs.tsv").write_text("\n".join(cross_pairs) + "\n")
    print("wrote 4 TSVs under", D)


if __name__ == "__main__":
    main()
