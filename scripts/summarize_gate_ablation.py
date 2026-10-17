"""Regenerate gate_ablation_summary.tsv from all available seed results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results/full/gate_ablation"
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    rows = ["seed\tarm\tgate\tB0_ruler_top1\tO2_ruler_top1\tvoting_quality"]
    by_arm = {"o2": [], "swift": []}
    for seed in SEEDS:
        for arm, gate in (("o2", "O2"), ("swift", "SWIFT")):
            fb = R / f"bon_{arm}_b0_N16_s{seed}.json"
            fo = R / f"bon_{arm}_o2_N16_s{seed}.json"
            if not (fb.exists() and fo.exists()):
                continue
            mb = json.load(open(fb))["meta"]
            mo = json.load(open(fo))["meta"]
            b0 = mb["accuracy_top1"] * 100
            o2 = mo["accuracy_top1"] * 100
            vot = mb["voting_accuracy"] * 100  # same pool
            rows.append(f"{seed}\t{arm}\t{gate}\t{b0:.2f}\t{o2:.2f}\t{vot:.2f}")
            by_arm[arm].append((b0, o2, vot))

    for arm, gate in (("o2", "O2"), ("swift", "SWIFT")):
        xs = by_arm[arm]
        if not xs:
            continue
        a = np.array(xs)
        rows.append(
            f"MEAN\t{arm}\t{gate}\t{a[:,0].mean():.2f}\t{a[:,1].mean():.2f}\t{a[:,2].mean():.2f}"
        )
        if len(xs) > 1:
            rows.append(
                f"STD\t{arm}\t{gate}\t{a[:,0].std(ddof=1):.2f}\t"
                f"{a[:,1].std(ddof=1):.2f}\t{a[:,2].std(ddof=1):.2f}"
            )

    # paired deltas (swift - o2) where both arms have that seed
    rows.append("")
    rows.append("seed\tB0_ruler_swift_minus_o2\tO2_ruler_swift_minus_o2\tvoting_swift_minus_o2")
    paired = []
    for seed in SEEDS:
        fo_b = R / f"bon_o2_b0_N16_s{seed}.json"
        fs_b = R / f"bon_swift_b0_N16_s{seed}.json"
        fo_o = R / f"bon_o2_o2_N16_s{seed}.json"
        fs_o = R / f"bon_swift_o2_N16_s{seed}.json"
        if not all(p.exists() for p in (fo_b, fs_b, fo_o, fs_o)):
            continue
        db = (json.load(open(fs_b))["meta"]["accuracy_top1"] - json.load(open(fo_b))["meta"]["accuracy_top1"]) * 100
        do = (json.load(open(fs_o))["meta"]["accuracy_top1"] - json.load(open(fo_o))["meta"]["accuracy_top1"]) * 100
        dv = (json.load(open(fs_b))["meta"]["voting_accuracy"] - json.load(open(fo_b))["meta"]["voting_accuracy"]) * 100
        rows.append(f"{seed}\t{db:+.2f}\t{do:+.2f}\t{dv:+.2f}")
        paired.append((db, do, dv))
    if paired:
        a = np.array(paired)
        rows.append(f"MEAN\t{a[:,0].mean():+.2f}\t{a[:,1].mean():+.2f}\t{a[:,2].mean():+.2f}")
        if len(paired) > 1:
            rows.append(f"STD\t{a[:,0].std(ddof=1):.2f}\t{a[:,1].std(ddof=1):.2f}\t{a[:,2].std(ddof=1):.2f}")

    out = R / "gate_ablation_summary.tsv"
    out.write_text("\n".join(rows) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
