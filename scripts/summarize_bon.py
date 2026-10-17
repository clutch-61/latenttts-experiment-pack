#!/usr/bin/env python3
"""Print top-1 / O2-softmax-weighted-vote (T=1) / majority vote / coverage for infer_gpt2_rm dumps.

Prefer meta.accuracy_wvote / accuracy_top1 when present (new dumps); else recompute.

Usage: python scripts/summarize_bon.py TAG=path.json [TAG=path.json ...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.system_scoring import wvote_select  # noqa: E402


def summarize(path: str) -> dict:
    d = json.load(open(path))
    ex = d["examples"]
    meta = d.get("meta") or {}
    top = wv = mv = cov = 0
    for e in ex:
        ans = [str(a) for a in e["answers"]]
        s = [float(x) for x in e["scores"]]
        c = [bool(x) for x in e["corrects"]]
        ok = dict(zip(ans, c))
        top += c[max(range(len(s)), key=lambda i: s[i])]
        wv += c[wvote_select(ans, s)]
        mv += ok[Counter(ans).most_common(1)[0][0]]
        cov += any(c)
    n = len(ex)
    out = {
        "n": n,
        "top": 100 * top / n,
        "wvote": 100 * wv / n,
        "mvote": 100 * mv / n,
        "cov": 100 * cov / n,
        "claim_agg": meta.get("claim_agg"),
    }
    if meta.get("accuracy_top1") is not None:
        out["meta_top1"] = 100 * float(meta["accuracy_top1"])
    if meta.get("accuracy_wvote") is not None:
        out["meta_wvote"] = 100 * float(meta["accuracy_wvote"])
    if meta.get("accuracy") is not None:
        out["meta_claim"] = 100 * float(meta["accuracy"])
    return out


def main():
    for arg in sys.argv[1:]:
        tag, path = arg.split("=", 1)
        r = summarize(path)
        extra = ""
        if r.get("meta_claim") is not None:
            extra = f" meta_claim={r['meta_claim']:.2f}"
        print(
            f"{tag:<26} n={r['n']} top={r['top']:.2f} wvote={r['wvote']:.2f} "
            f"mvote={r['mvote']:.2f} cov={r['cov']:.2f}{extra}"
        )


if __name__ == "__main__":
    main()
