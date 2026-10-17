"""Paired cross-model stats for the Order-Decision test, aggregated over seeds.

Reads per-seed per-pool arrays:
  results/full/order_decision/odnpz_{model}_N{n}_s{seed}_{type}.npz
  (arrays: 'pair' per-pool mean pair acc, 'strict' per-pool 0/1 strict win)

For each (corruption, N):
  - per-model pair_acc / strict_win averaged over seeds
  - per-model std over seeds
  - O2 vs challenger: per-seed + cross-seed per-pool mean delta, paired bootstrap
    (pools are identical across models for a given seed), and McNemar on strict_win
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/full/order_decision"
MODELS = ["o2", "b0", "swift", "hsrm"]
TP = ["adjacent_swap", "segment_swap", "multi_segment", "full_reverse", "mixed"]
SEEDS = [42, 43, 44, 45, 46]


def zp(model, n, seed, tp):
    return np.load(R / f"od_{model}_N{n}_s{seed}__{tp}.npz")


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
    if n <= 1000:
        k = min(b, c)
        return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    from math import erf, sqrt
    z = abs(b - c) - 1.0
    if z <= 0:
        return 1.0
    return float(min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0 * n))))))


def main() -> None:
    rows = ["model\tn\tcorruption\tn_seeds\tpair_acc\tpair_std\tstrict_win\tstrict_std"]
    pairs = ["pair\tn\tcorruption\tmetric\tdelta\tci_lo\tci_hi\tn_eff\tmcnemar_p\tdisc_ab\tdisc_ba"]
    for n in (4, 16):
        for tp in TP:
            per_model_seed = {m: {} for m in MODELS}
            for m in MODELS:
                for s in SEEDS:
                    f = R / f"od_{m}_N{n}_s{s}__{tp}.npz"
                    if f.exists():
                        per_model_seed[m][s] = np.load(f)
            for m in MODELS:
                if not per_model_seed[m]:
                    continue
                pa = [v["pair"].mean() for v in per_model_seed[m].values()]
                st = [v["strict"].mean() for v in per_model_seed[m].values()]
                rows.append(
                    f"{m}\t{n}\t{tp}\t{len(pa)}\t{np.mean(pa):.4f}\t{np.std(pa, ddof=1) if len(pa)>1 else 0:.4f}\t"
                    f"{np.mean(st):.4f}\t{np.std(st, ddof=1) if len(st)>1 else 0:.4f}"
                )
            for ch in ("b0", "swift", "hsrm"):
                common_seeds = sorted(set(per_model_seed["o2"]) & set(per_model_seed[ch]))
                if not common_seeds:
                    continue
                # concatenate per-pool arrays across seeds -> one paired bootstrap
                for metric, arr_key in (("pair", "pair"), ("strict", "strict")):
                    a = np.concatenate([per_model_seed["o2"][s][arr_key] for s in common_seeds])
                    b = np.concatenate([per_model_seed[ch][s][arr_key] for s in common_seeds])
                    if len(a) != len(b):
                        continue
                    d, lo, hi = boot_ci(a, b)
                    ab = int(((a == 1) & (b == 0)).sum())
                    ba = int(((a == 0) & (b == 1)).sum())
                    p = mcnemar_exact(ab, ba) if metric == "strict" else float("nan")
                    pairs.append(
                        f"O2-{ch}\t{n}\t{tp}\t{metric}\t{d:.4f}\t{lo:.4f}\t{hi:.4f}\t{len(a)}\t"
                        f"{p if not np.isnan(p) else '-'}\t{ab}\t{ba}"
                    )
    (R / "order_decision_summary_5seed.tsv").write_text("\n".join(rows) + "\n")
    (R / "order_decision_paired_5seed.tsv").write_text("\n".join(pairs) + "\n")
    print("wrote", R / "order_decision_summary_5seed.tsv")
    print("wrote", R / "order_decision_paired_5seed.tsv")


if __name__ == "__main__":
    main()
