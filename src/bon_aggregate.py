"""Official BoN aggregation: O2 softmax-weighted vote (T=1).

Claim metric is this vote, not top-1. Top-1 is kept as a diagnostic.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def softmax_wvote_winner(answers: list, scores: list) -> str:
    """Answer string with the largest sum of exp(score − max)."""
    ans = [str(a) for a in answers]
    sc = [float(x) for x in scores]
    m = max(sc)
    mass: dict[str, float] = defaultdict(float)
    for a, x in zip(ans, sc):
        mass[a] += math.exp(x - m)
    return max(mass, key=mass.get)


def example_claim(ex: dict[str, Any]) -> dict[str, Any]:
    """Per-example top / wvote / majority / coverage from an infer dump row."""
    ans = [str(a) for a in ex["answers"]]
    sc = [float(x) for x in ex["scores"]]
    corr = [bool(x) for x in ex["corrects"]]
    ok = dict(zip(ans, corr))
    top_i = max(range(len(sc)), key=lambda i: sc[i])
    wv = softmax_wvote_winner(ans, sc)
    mv = Counter(ans).most_common(1)[0][0]
    pool = any(corr)
    wv_ok = bool(ok.get(wv, False))
    return {
        "top_correct": bool(corr[top_i]),
        "wvote_correct": wv_ok,
        "mvote_correct": bool(ok.get(mv, False)),
        "pool": pool,
        "leftover_top": pool and (not corr[top_i]),
        "leftover_wvote": pool and (not wv_ok),
        "wvote_answer": wv,
    }


def dump_claim(blob: dict[str, Any] | None) -> dict[str, Any] | None:
    """Claim stats from an infer_gpt2_rm JSON (works on old dumps without wvote fields)."""
    if not blob:
        return None
    exs = blob.get("examples") or []
    if not exs or "answers" not in exs[0] or "scores" not in exs[0]:
        return None
    n = len(exs)
    rows = [example_claim(e) for e in exs]
    pool_n = sum(1 for r in rows if r["pool"])
    leftover_w = sum(1 for r in rows if r["leftover_wvote"])
    leftover_t = sum(1 for r in rows if r["leftover_top"])
    return {
        "n": n,
        "acc_wvote": sum(r["wvote_correct"] for r in rows) / n,
        "acc_top": sum(r["top_correct"] for r in rows) / n,
        "acc_mvote": sum(r["mvote_correct"] for r in rows) / n,
        "cov": pool_n / n,
        "leftover_wvote": leftover_w / n,
        "leftover_top": leftover_t / n,
        "pick_fail_wvote_given_pool": leftover_w / max(pool_n, 1),
        "pick_fail_top_given_pool": leftover_t / max(pool_n, 1),
    }
