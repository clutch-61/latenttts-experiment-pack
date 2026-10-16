"""Per-question paired comparison of latent scorers (same candidate pool).

For each (pool, seed) we load the BoN dumps of several scorers. Candidate pools are
verified identical via `answers_sha1` before any comparison.

Per question q we compute:
  - top1            : selected_correct (bool)
  - auc_cw          : P(score of a random correct > score of a random wrong),
                      defined only when the pool has >=1 correct and >=1 wrong
  - mrr             : 1 / rank of the highest-scored correct candidate (0 if none)
  - margin_cw       : max(score[correct]) - max(score[wrong]) (only if both exist)
  - cov             : any_correct (pool property, scorer-independent)

Paired tests for O2 vs a challenger C (same questions, same pool, per seed):
  - McNemar exact (binomial) on discordant top1 pairs
  - paired bootstrap CI (question resampling) on mean(top1_O2 - top1_C)
  - paired bootstrap CI on mean(auc_O2 - auc_C) over questions with a defined auc

Outputs: results/full/lightweight/paired_ranking_stats.tsv
         results/full/lightweight/paired_ranking_perquestion_{pool}_{seed}.tsv
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/full/lightweight"

# pool -> scorer -> seed -> path
POOLS: Dict[str, Dict[str, Dict[int, str]]] = {
    "frozen": {
        "B0": {s: f"results/full/paired_seeds/bon_frozen_B0_gsm_N16_s{s}_20261016_002036.json" for s in (42, 43, 44)},
        "O2": {s: f"results/full/paired_seeds/bon_frozen_O2_gsm_N16_s{s}_20261016_002036.json" for s in (42, 43, 44)},
        "SWIFT": {s: f"results/full/lightweight/bon_swift_gsm_N16_dnone_s{s}_test.json" for s in (42, 43, 44)},
        "HSRM": {s: f"results/full/lightweight/bon_hsrm_gsm_N16_dnone_s{s}_20261016_174900.json" for s in (42, 43, 44)},
    },
    "p2": {
        "B0": {
            42: "results/full/p123/exp1_ours_b0/bon_p2mmd05_B0_gsm_N16_dnone_s42_20261015_230629.json",
            43: "results/full/paired_seeds/bon_p2mmd05_B0_gsm_N16_s43_20261016_011103.json",
            44: "results/full/paired_seeds/bon_p2mmd05_B0_gsm_N16_s44_20261016_011103.json",
        },
        "O2": {
            42: "results/full/p123/exp1_ours_o2/bon_p2mmd05_O2_gsm_N16_dnone_s42_20261016_001127.json",
            43: "results/full/paired_seeds/bon_p2mmd05_O2_gsm_N16_s43_20261016_011103.json",
            44: "results/full/paired_seeds/bon_p2mmd05_O2_gsm_N16_s44_20261016_011103.json",
        },
        "SWIFT": {s: f"results/full/lightweight/bon_p2mmd05_swift_gsm_N16_s{s}.json" for s in (42, 43, 44)},
        "HSRM": {s: f"results/full/lightweight/bon_p2mmd05_hsrm_gsm_N16_s{s}.json" for s in (42, 43, 44)},
    },
}


def load(path: str) -> Dict[int, dict]:
    d = json.load(open(ROOT / path))
    return {e["idx"]: e for e in d["examples"]}


def per_question_metrics(ex: dict) -> Dict[str, float]:
    scores = np.asarray(ex["scores"], dtype=np.float64)
    corr = np.asarray(ex["corrects"], dtype=bool)
    n = len(scores)
    out = {
        "top1": float(bool(ex["selected_correct"])),
        "top1_ach": float("nan"),  # top1 over questions where a correct candidate exists
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
    n_pos, n_neg = int(corr.sum()), n - int(corr.sum())
    if n_pos and n_neg:
        # AUC = mean over correct/wrong pairs of (s_correct > s_wrong)
        s_c, s_w = scores[corr], scores[~corr]
        diff = s_c[:, None] - s_w[None, :]
        out["auc_cw"] = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
        out["margin_cw"] = float(s_c.max() - s_w.max())
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for discordant counts b (a1 only) and c (a2 only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, reps: int, rng: np.random.Generator):
    """CI for mean(a - b) with paired question resampling. NaN-safe (drops NaN pairs)."""
    m = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[m], b[m]
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(reps, n))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.boot_seed)
    OUT.mkdir(parents=True, exist_ok=True)

    summary_rows: List[str] = [
        "pool\tseed\tscorer\tn_q\ttop1\ttop1_ach\tauc_cw\tmrr\tmargin_cw\tcov"
    ]
    pair_rows: List[str] = [
        "pool\tseed\tpair\tmetric\tdelta_pp\tci_lo\tci_hi\tn\tmcnemar_p\tdisc_ab\tdisc_ba"
    ]
    # pool -> scorer -> metric -> {seed: array}
    cross: Dict[str, Dict[str, Dict[str, Dict[int, np.ndarray]]]] = {}

    for pool, scorers in POOLS.items():
        cross[pool] = {}
        for seed in (42, 43, 44):
            data = {name: load(paths[seed]) for name, paths in scorers.items()}
            # ---- same-pool verification (by answers_sha1) ----
            common = set.intersection(*[set(d.keys()) for d in data.values()])
            mismatch = 0
            for idx in common:
                ref = data["O2"][idx]["answers_sha1"]
                for name in data:
                    if data[name][idx]["answers_sha1"] != ref:
                        mismatch += 1
            print(f"[{pool} s{seed}] n_common={len(common)} sha1_mismatch={mismatch}")

            # ---- per-question rows ----
            per_q: Dict[str, Dict[str, np.ndarray]] = {}
            for name, d in data.items():
                idxs = sorted(common)
                recs = [per_question_metrics(d[i]) for i in idxs]
                per_q[name] = {
                    k: np.array([r[k] for r in recs], dtype=np.float64)
                    for k in ("top1", "top1_ach", "auc_cw", "mrr", "margin_cw", "cov")
                }
                cross[pool].setdefault(name, {})
                for k, arr in per_q[name].items():
                    cross[pool][name].setdefault(k, {})[seed] = arr
                top1 = per_q[name]["top1"].mean() * 100
                ach = np.nanmean(per_q[name]["top1_ach"]) * 100
                auc = np.nanmean(per_q[name]["auc_cw"]) * 100
                mrr = per_q[name]["mrr"].mean() * 100
                cov = per_q[name]["cov"].mean() * 100
                summary_rows.append(
                    f"{pool}\t{seed}\t{name}\t{len(idxs)}\t{top1:.4f}\t{ach:.4f}\t{auc:.4f}\t{mrr:.4f}\t"
                    f"{np.nanmean(per_q[name]['margin_cw']):.6f}\t{cov:.4f}"
                )

            # dump per-question answers for re-analysis
            with open(OUT / f"paired_ranking_perquestion_{pool}_s{seed}.tsv", "w") as fh:
                fh.write("idx\tscorer\ttop1\tauc_cw\tmrr\tmargin_cw\n")
                for name in data:
                    for j, idx in enumerate(sorted(common)):
                        fh.write(
                            f"{idx}\t{name}\t{per_q[name]['top1'][j]:.0f}\t"
                            f"{per_q[name]['auc_cw'][j]:.4f}\t{per_q[name]['mrr'][j]:.4f}\t"
                            f"{per_q[name]['margin_cw'][j]:.6f}\n"
                        )

            # ---- paired tests ----
            for ch in ("SWIFT", "HSRM", "B0"):
                a, b = per_q["O2"]["top1"], per_q[ch]["top1"]
                d, lo, hi, n = paired_bootstrap_ci(a, b, args.reps, rng)
                disc_ab = int(((a == 1) & (b == 0)).sum())
                disc_ba = int(((a == 0) & (b == 1)).sum())
                p = mcnemar_exact(disc_ab, disc_ba)
                pair_rows.append(
                    f"{pool}\t{seed}\tO2-{ch}\ttop1\t{d * 100:.4f}\t{lo * 100:.4f}\t{hi * 100:.4f}\t"
                    f"{n}\t{p:.6f}\t{disc_ab}\t{disc_ba}"
                )
                da, loa, hia, na = paired_bootstrap_ci(
                    per_q["O2"]["auc_cw"], per_q[ch]["auc_cw"], args.reps, rng
                )
                pair_rows.append(
                    f"{pool}\t{seed}\tO2-{ch}\tauc_cw\t{da * 100:.4f}\t{loa * 100:.4f}\t{hia * 100:.4f}\t"
                    f"{na}\t-\t-\t-"
                )

    # ---- cross-seed aggregation: average each question over the 3 seeds, then
    #      paired bootstrap over questions (single CI, more stable than per-seed).
    cross_rows: List[str] = [
        "pool\tscorer\ttop1\ttop1_ach\tauc_cw\tmrr\tcov"
    ]
    cross_pairs: List[str] = [
        "pool\tpair\tmetric\tdelta_pp\tci_lo\tci_hi\tn"
    ]
    for pool in POOLS:
        seed_avg: Dict[str, Dict[str, np.ndarray]] = {}
        for name in cross[pool]:
            seed_avg[name] = {
                k: np.nanmean(np.stack([cross[pool][name][k][s] for s in (42, 43, 44)]), axis=0)
                for k in ("top1", "top1_ach", "auc_cw", "mrr", "cov")
            }
            cross_rows.append(
                f"{pool}\t{name}\t{np.nanmean(seed_avg[name]['top1']) * 100:.4f}\t"
                f"{np.nanmean(seed_avg[name]['top1_ach']) * 100:.4f}\t"
                f"{np.nanmean(seed_avg[name]['auc_cw']) * 100:.4f}\t"
                f"{np.nanmean(seed_avg[name]['mrr']) * 100:.4f}\t"
                f"{np.nanmean(seed_avg[name]['cov']) * 100:.4f}"
            )
        for ch in ("SWIFT", "HSRM", "B0"):
            for metric in ("top1", "top1_ach", "auc_cw", "mrr"):
                a, b = seed_avg["O2"][metric], seed_avg[ch][metric]
                d, lo, hi, n = paired_bootstrap_ci(a, b, args.reps, rng)
                cross_pairs.append(
                    f"{pool}\tO2-{ch}\t{metric}\t{d * 100:.4f}\t{lo * 100:.4f}\t{hi * 100:.4f}\t{n}"
                )

    (OUT / "paired_ranking_stats.tsv").write_text("\n".join(summary_rows) + "\n")
    (OUT / "paired_ranking_pairs.tsv").write_text("\n".join(pair_rows) + "\n")
    (OUT / "paired_ranking_crossseed.tsv").write_text("\n".join(cross_rows) + "\n")
    (OUT / "paired_ranking_crossseed_pairs.tsv").write_text("\n".join(cross_pairs) + "\n")
    print("\nwrote", OUT / "paired_ranking_stats.tsv")
    print("wrote", OUT / "paired_ranking_pairs.tsv")
    print("wrote", OUT / "paired_ranking_crossseed.tsv")
    print("wrote", OUT / "paired_ranking_crossseed_pairs.tsv")


if __name__ == "__main__":
    main()
