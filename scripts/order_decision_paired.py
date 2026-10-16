"""Paired cross-model stats for the order-decision test.

Reads the per-pool arrays dumped by ``order_decision_test.py`` (identical pools
across models thanks to deterministic corruption seeds) and reports, for each
(corruption, N):

  - per-model pair_acc / strict_win
  - O2 vs challenger paired bootstrap CI (question-resampling) and McNemar on
    strict_win discordant pools
"""

from __future__ import annotations

import itertools
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/full/order_decision"
MODELS = ["o2", "b0", "swift", "hsrm"]
TP = ["adjacent_swap", "segment_swap", "multi_segment", "full_reverse", "mixed"]


def boot_ci(a, b, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(reps, n))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    rows = ["model\tn\tcorruption\tpair_acc\tstrict_win"]
    pairs = ["pair\tn\tcorruption\tmetric\tdelta\tci_lo\tci_hi\tmcnemar_p\tdisc_ab\tdisc_ba"]
    for n in (4, 16):
        arr = {m: {} for m in MODELS}
        for m in MODELS:
            for tp in TP:
                f = R / f"{m}_N{n}_{tp}.npz"
                if f.exists():
                    z = np.load(f)
                    arr[m][tp] = {"pair": z["pair"], "strict": z["strict"]}
        for tp in TP:
            for m in MODELS:
                if tp in arr[m]:
                    rows.append(f"{m}\t{n}\t{tp}\t{arr[m][tp]['pair'].mean():.4f}\t{arr[m][tp]['strict'].mean():.4f}")
            for ch in ("b0", "swift", "hsrm"):
                if tp not in arr["o2"] or tp not in arr[ch]:
                    continue
                for metric in ("pair", "strict"):
                    a, b = arr["o2"][tp][metric], arr[ch][tp][metric]
                    if len(a) != len(b):
                        continue
                    d, lo, hi = boot_ci(a, b)
                    ab = int(((a == 1) & (b == 0)).sum())
                    ba = int(((a == 0) & (b == 1)).sum())
                    p = mcnemar_exact(ab, ba) if metric == "strict" else float("nan")
                    pairs.append(
                        f"O2-{ch}\t{n}\t{tp}\t{metric}\t{d:.4f}\t{lo:.4f}\t{hi:.4f}\t"
                        f"{p if not np.isnan(p) else '-':}\t{ab}\t{ba}"
                    )
    (R / "order_decision_summary.tsv").write_text("\n".join(rows) + "\n")
    (R / "order_decision_paired.tsv").write_text("\n".join(pairs) + "\n")
    print("wrote", R / "order_decision_summary.tsv")
    print("wrote", R / "order_decision_paired.tsv")


if __name__ == "__main__":
    main()
