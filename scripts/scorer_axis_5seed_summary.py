"""5-seed summary of the scorer-axis BoN table.

Reads results/full/scorer_axis_5seed/{pool}_{scorer}_N{n}_s{seed}.json and writes:
  scorer_axis_5seed_summary.tsv   (per pool x scorer x N x seed + mean/std)
  scorer_axis_5seed_table.tsv     (pool x scorer: top1 mean+-std over seeds, n_seeds)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/full/scorer_axis_5seed"
SEEDS = [42, 43, 44, 45, 46]
POOLS = ["frozen", "p2"]
SCORERS = ["B0", "O2", "SWIFT", "HSRM"]


def acc(p: Path) -> float:
    d = json.load(open(p))
    return float(d["meta"]["accuracy_top1"])


def main() -> None:
    rows = ["pool\tscorer\tn\tseed\ttop1"]
    for pool in POOLS:
        for s in SCORERS:
            for n in (16, 64):
                vals = []
                for seed in SEEDS:
                    f = D / f"{pool}_{s}_N{n}_s{seed}.json"
                    if not f.exists():
                        continue
                    a = acc(f)
                    vals.append(a)
                    rows.append(f"{pool}\t{s}\t{n}\t{seed}\t{a*100:.4f}")

    table = ["pool\tscorer\tn\tn_seeds\ttop1_mean\ttop1_std\ttop1_min\ttop1_max"]
    for pool in POOLS:
        for s in SCORERS:
            for n in (16, 64):
                vals = []
                for seed in SEEDS:
                    f = D / f"{pool}_{s}_N{n}_s{seed}.json"
                    if f.exists():
                        vals.append(acc(f))
                if not vals:
                    continue
                v = np.array(vals) * 100
                table.append(
                    f"{pool}\t{s}\t{n}\t{len(v)}\t{v.mean():.4f}\t"
                    f"{(v.std(ddof=1) if len(v) > 1 else 0.0):.4f}\t{v.min():.4f}\t{v.max():.4f}"
                )

    (D / "scorer_axis_5seed_summary.tsv").write_text("\n".join(rows) + "\n")
    (D / "scorer_axis_5seed_table.tsv").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
