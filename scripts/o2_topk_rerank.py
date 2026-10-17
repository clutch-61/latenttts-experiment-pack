"""Stage-2 re-rank: O2 shortlists top-k, B0 (or fused) picks within shortlist.

Addresses lost_to_strongest_wrong when the best correct is near the head
but not argmax. Val = idx%5==0; locked test otherwise. Same-pool dumps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/full/scorer_axis_5seed"
OUT = ROOT / "results/full/scorer_axis_5seed/auc_to_top1"
SEEDS = [42, 43, 44, 45, 46]
POOLS = ["frozen", "p2"]


def load(pool: str, scorer: str, seed: int) -> Dict[int, dict]:
    p = D / f"{pool}_{scorer}_N16_s{seed}.json"
    return {int(e["idx"]): e for e in json.load(open(p))["examples"]}


def is_val(idx: int) -> bool:
    return (idx % 5) == 0


def select(sa: np.ndarray, sb: np.ndarray, method: str, k: int, alpha: float) -> int:
    order = np.argsort(-sa, kind="stable")
    short = order[:k]
    if method == "o2":
        return int(order[0])
    if method == "b0":
        return int(np.argmax(sb))
    if method == "b0_in_topk":
        return int(short[int(np.argmax(sb[short]))])
    if method == "score_z_in_topk":
        a, b = sa[short].astype(np.float64), sb[short].astype(np.float64)

        def z(x):
            s = x.std()
            return np.zeros_like(x) if s < 1e-12 else (x - x.mean()) / s

        return int(short[int(np.argmax(alpha * z(a) + (1.0 - alpha) * z(b)))])
    if method == "oracle_correct_in_topk":
        # upper bound: if any correct in shortlist, pick one (needs corrects)
        raise RuntimeError("oracle needs corrects at call site")
    raise ValueError(method)


def eval_method(
    pool: str,
    idxs: List[int],
    method: str,
    k: int,
    alpha: float,
) -> float:
    accs = []
    for seed in SEEDS:
        o2, b0 = load(pool, "O2", seed), load(pool, "B0", seed)
        ok = []
        for i in idxs:
            sa = np.asarray(o2[i]["scores"], dtype=np.float64)
            sb = np.asarray(b0[i]["scores"], dtype=np.float64)
            corr = np.asarray(o2[i]["corrects"], dtype=bool)
            if method == "oracle_correct_in_topk":
                order = np.argsort(-sa, kind="stable")
                short = order[:k]
                if corr[short].any():
                    sel = int(short[int(np.argmax(corr[short]))])  # any correct
                else:
                    sel = int(order[0])
            else:
                sel = select(sa, sb, method, k, alpha)
            ok.append(bool(corr[sel]))
        accs.append(float(np.mean(ok)))
    return float(np.mean(accs))


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    del args
    OUT.mkdir(parents=True, exist_ok=True)

    ref = load("frozen", "O2", 42)
    idxs = sorted(ref.keys())
    val = [i for i in idxs if is_val(i)]
    test = [i for i in idxs if not is_val(i)]

    rows = ["pool\tmethod\tk\talpha\tval_top1\ttest_top1\tselected"]
    best_lines = []
    grid_spec = [
        ("o2", [16], [1.0]),
        ("b0", [16], [1.0]),
        ("b0_in_topk", [2, 3, 4, 6, 8], [1.0]),
        ("score_z_in_topk", [2, 3, 4], [0.3, 0.5, 0.7]),
        ("oracle_correct_in_topk", [2, 3, 4], [1.0]),
    ]

    for pool in POOLS:
        best = None
        grid = []
        for method, ks, alphas in grid_spec:
            for k in ks:
                for alpha in alphas:
                    v = eval_method(pool, val, method, k, alpha)
                    t = eval_method(pool, test, method, k, alpha)
                    grid.append((method, k, alpha, v, t))
                    if method in ("o2", "b0", "oracle_correct_in_topk"):
                        continue
                    if best is None or v > best[0]:
                        best = (v, t, method, k, alpha)
        assert best is not None
        for method, k, alpha, v, t in grid:
            tag = ""
            if (
                method == best[2]
                and k == best[3]
                and abs(alpha - best[4]) < 1e-12
            ):
                tag = "VAL_BEST"
            rows.append(
                f"{pool}\t{method}\t{k}\t{alpha:.2f}\t{100*v:.4f}\t{100*t:.4f}\t{tag}"
            )
        t_o2 = eval_method(pool, test, "o2", 16, 1.0)
        best_lines.append(
            f"{pool}: val_best={best[2]} k={best[3]} α={best[4]:.2f} "
            f"val={100*best[0]:.2f} → test={100*best[1]:.2f} "
            f"(test O2={100*t_o2:.2f} Δ={100*(best[1]-t_o2):+.2f})"
        )
        for k in (2, 3, 4):
            t_or = eval_method(pool, test, "oracle_correct_in_topk", k, 1.0)
            best_lines.append(
                f"  oracle top{k} on test={100*t_or:.2f} "
                f"(headroom vs O2={100*(t_or-t_o2):+.2f})"
            )

    out_path = OUT / "topk_rerank_grid.tsv"
    out_path.write_text("\n".join(rows) + "\n")
    note = OUT / "topk_rerank_NOTE.md"
    note.write_text(
        "# O2 top-k → B0 二阶段重排\n\n"
        + "\n".join(best_lines)
        + "\n\n全网格: `topk_rerank_grid.tsv`。\n"
        + "oracle = 短名单内若有正解则选中（上界，非可部署）。\n"
    )
    print("\n".join(best_lines))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
