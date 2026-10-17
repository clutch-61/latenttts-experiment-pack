#!/usr/bin/env python3
"""Summarize one sf_full run: generator vs +O2 metrics, paired bootstrap vs reference BoN dumps."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFS = {
    "A": "results/full/sf_pilot/bon_A_O2_N4_20260930_205539.json",
    "gateT": "results/full/sf_full/bon_O2gateT_O2_N4_pilot_20260930_231108.json",
}


def paired(ref_path: Path, run_path: Path):
    ae = {e["idx"]: e for e in json.load(open(ref_path))["examples"]}
    de = {e["idx"]: e for e in json.load(open(run_path))["examples"]}
    qids = sorted(set(ae) & set(de))
    out = {}
    for key, name in [("selected_correct", "bon"), ("any_correct", "cov")]:
        xa = np.array([ae[i][key] for i in qids], float)
        xd = np.array([de[i][key] for i in qids], float)
        rng = np.random.default_rng(42)
        boots = [(xd[ix] - xa[ix]).mean() * 100 for ix in (rng.integers(0, len(xa), len(xa)) for _ in range(2000))]
        out[name] = (round(float((xd - xa).mean() * 100), 2), [round(float(x), 2) for x in np.percentile(boots, [2.5, 97.5])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    res = ROOT / "results/full/sf_full"
    out = ROOT / args.outdir
    stats = json.load(open(out / "stats.json"))
    print("STATS", stats)
    rows = [json.loads(l) for l in open(out / "debug.jsonl")]
    upd = [r for r in rows if r.get("updated")]
    print("gates", Counter(r.get("gate") for r in rows))
    print("sources", Counter(r.get("src") for r in upd))
    for s in sorted({r.get("src") for r in upd}):
        t = [r["tlen"] for r in upd if r.get("src") == s]
        print(f"  tlen[{s}] mean={sum(t) / len(t):.2f} n={len(t)}")

    g = json.load(open(res / f"{args.tag}_pilotvalid_{args.stamp}.json"))
    b = json.load(open(res / f"bon_{args.tag}_O2_N4_pilot_{args.stamp}.json"))["meta"]
    gv = json.load(open(res / f"{args.tag}_gsmvalid_{args.stamp}.json"))
    print("PILOT GEN", round(g["pass@1"] * 100, 2), round(g["coverage@N"] * 100, 2))
    print("PILOT BON", round(b["accuracy"] * 100, 2), round(b["coverage"] * 100, 2))
    print("GSMVALID GEN", round(gv["pass@1"] * 100, 2), round(gv["coverage@N"] * 100, 2))
    diag = res / f"diag_{args.tag}_{args.stamp}/summary.json"
    if diag.exists():
        s = json.load(open(diag))
        print("STRUCT", {k: s["structure"]["C"][k] for k in ["pass1", "coverage", "mean_after_len", "starts_###"]})
        print("vsA gen-cov", s["transfer_any"]["paired_diff_pp"]["C_minus_A"])
    run_bon = res / f"bon_{args.tag}_O2_N4_pilot_{args.stamp}.json"
    for name, p in REFS.items():
        print(f"{args.tag}-{name}+O2", paired(ROOT / p, run_bon))
    print("REF A 31.0/45.7 A+O2 35.2/41.8 | gatePilot 32.3/46.9 +O2 34.8/43.4 | gateT 33.5/45.7 +O2 32.8/40.2")


if __name__ == "__main__":
    main()
