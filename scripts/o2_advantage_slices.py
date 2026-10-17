"""Pre-registered O2 advantage slices vs SWIFT/HSRM (same-pool dumps).

Uses only gold labels and baseline-scorer outcomes to define slices — never O2 scores.

Analyses
--------
1. Pool × scorer interaction (diff-in-diff):
   For each question (seed-averaged), DiD = (O2 - C)_p2 - (O2 - C)_frozen
   on top1 and auc_cw. Bootstrap CI over questions.

2. Baseline ranking-error correction:
   Slice = baseline selected wrong AND pool has ≥1 correct (gold).
   Report O2 Top1, baseline Top1 (=0 by construction), and O2−baseline Δ with CI.
   Also a narrow-margin slice: baseline margin_cw in bottom tertile among
   questions with both correct and incorrect candidates (margin from baseline
   scores only).

3. p2-expanded coverage proxy (question-level, not candidate-id linked):
   Questions where frozen.any_correct=0 and p2.any_correct=1 (seed-avg ≥0.5).
   Compare O2−C gaps on that subset vs complement (exploratory; labeled as such).

Writes under results/full/scorer_axis_5seed/o2_advantage_slices/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/full/scorer_axis_5seed"
OUT = D / "o2_advantage_slices"
SEEDS = [42, 43, 44, 45, 46]
POOLS = ["frozen", "p2"]
SCORERS = ["B0", "O2", "SWIFT", "HSRM"]
CHALLENGERS = ["SWIFT", "HSRM", "B0"]


def path_for(pool: str, scorer: str, n: int, seed: int) -> Path:
    return D / f"{pool}_{scorer}_N{n}_s{seed}.json"


def load(p: Path) -> Dict[int, dict]:
    d = json.load(open(p))
    return {e["idx"]: e for e in d["examples"]}


def metrics(ex: dict) -> Dict[str, float]:
    scores = np.asarray(ex["scores"], dtype=np.float64)
    corr = np.asarray(ex["corrects"], dtype=bool)
    out = {
        "top1": float(bool(ex["selected_correct"])),
        "auc_cw": float("nan"),
        "margin_cw": float("nan"),
        "cov": float(bool(ex["any_correct"])),
        "achievable": float(bool(ex["any_correct"])),
        "baseline_wrong_ach": float(
            bool(ex["any_correct"]) and not bool(ex["selected_correct"])
        ),
    }
    if corr.any() and (~corr).any():
        s_c, s_w = scores[corr], scores[~corr]
        diff = s_c[:, None] - s_w[None, :]
        out["auc_cw"] = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
        out["margin_cw"] = float(s_c.max() - s_w.max())
    return out


def boot_ci(d: np.ndarray, reps: int, rng: np.random.Generator) -> Tuple[float, float, float, int]:
    m = ~np.isnan(d)
    d = d[m]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    idx = rng.integers(0, n, size=(reps, n))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), n


def load_all(n: int) -> Dict[str, Dict[str, Dict[int, Dict[int, dict]]]]:
    """pool -> scorer -> seed -> idx -> example"""
    out: Dict[str, Dict[str, Dict[int, Dict[int, dict]]]] = {}
    for pool in POOLS:
        out[pool] = {}
        for scorer in SCORERS:
            out[pool][scorer] = {}
            for seed in SEEDS:
                p = path_for(pool, scorer, n, seed)
                if not p.exists():
                    raise FileNotFoundError(p)
                out[pool][scorer][seed] = load(p)
    return out


def seed_avg_metric(
    store: Dict[str, Dict[str, Dict[int, Dict[int, dict]]]],
    pool: str,
    scorer: str,
    idxs: List[int],
    key: str,
) -> np.ndarray:
    mats = []
    for seed in SEEDS:
        data = store[pool][scorer][seed]
        mats.append([metrics(data[i])[key] for i in idxs])
    return np.nanmean(np.asarray(mats, dtype=np.float64), axis=0)


def verify_sha1(store, pool: str, seed: int, idxs: List[int]) -> int:
    mismatch = 0
    for i in idxs:
        ref = store[pool]["O2"][seed][i]["answers_sha1"]
        for s in SCORERS:
            if store[pool][s][seed][i]["answers_sha1"] != ref:
                mismatch += 1
    return mismatch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.boot_seed)
    OUT.mkdir(parents=True, exist_ok=True)

    store = load_all(args.n)
    # common question ids across all dumps
    commons = []
    for pool in POOLS:
        for seed in SEEDS:
            keys = [set(store[pool][s][seed].keys()) for s in SCORERS]
            commons.append(set.intersection(*keys))
    idxs = sorted(set.intersection(*commons))
    print(f"n_questions={len(idxs)}")
    for pool in POOLS:
        for seed in SEEDS:
            mm = verify_sha1(store, pool, seed, idxs)
            if mm:
                print(f"WARN sha1 mismatch {pool} s{seed}: {mm}")

    # ---- 1. Diff-in-diff ----
    did_rows = [
        "pair\tmetric\tfrozen_gap_pp\tp2_gap_pp\tdid_pp\tci_lo\tci_hi\tn\tpre_reg"
    ]
    gap_detail = [
        "pair\tmetric\tidx\tfrozen_o2\tfrozen_c\tp2_o2\tp2_c\tfrozen_gap\tp2_gap\tdid"
    ]
    for ch in CHALLENGERS:
        for metric in ("top1", "auc_cw"):
            o2_f = seed_avg_metric(store, "frozen", "O2", idxs, metric)
            c_f = seed_avg_metric(store, "frozen", ch, idxs, metric)
            o2_p = seed_avg_metric(store, "p2", "O2", idxs, metric)
            c_p = seed_avg_metric(store, "p2", ch, idxs, metric)
            gap_f = o2_f - c_f
            gap_p = o2_p - c_p
            did = gap_p - gap_f
            # drop nan (auc_cw undefined when no pos/neg)
            mean, lo, hi, n = boot_ci(did, args.reps, rng)
            did_rows.append(
                f"O2-{ch}\t{metric}\t{np.nanmean(gap_f)*100:.4f}\t"
                f"{np.nanmean(gap_p)*100:.4f}\t{mean*100:.4f}\t{lo*100:.4f}\t{hi*100:.4f}\t"
                f"{n}\tpre_registered"
            )
            for i, qi in enumerate(idxs):
                if np.isnan(did[i]):
                    continue
                gap_detail.append(
                    f"O2-{ch}\t{metric}\t{qi}\t{o2_f[i]:.6f}\t{c_f[i]:.6f}\t"
                    f"{o2_p[i]:.6f}\t{c_p[i]:.6f}\t{gap_f[i]:.6f}\t{gap_p[i]:.6f}\t{did[i]:.6f}"
                )

    (OUT / "diff_in_diff.tsv").write_text("\n".join(did_rows) + "\n")
    (OUT / "diff_in_diff_per_q.tsv").write_text("\n".join(gap_detail) + "\n")

    # ---- 2. Baseline wrong + achievable ----
    corr_rows = [
        "pool\tbaseline\tslice\tn\tbaseline_top1\tO2_top1\tdelta_pp\tci_lo\tci_hi\tpre_reg"
    ]
    for pool in POOLS:
        for bl in CHALLENGERS:
            bl_top1 = seed_avg_metric(store, pool, bl, idxs, "top1")
            o2_top1 = seed_avg_metric(store, pool, "O2", idxs, "top1")
            cov = seed_avg_metric(store, pool, bl, idxs, "cov")
            # seed-avg "wrong & achievable": use mean of per-seed flags
            wrong_ach = seed_avg_metric(store, pool, bl, idxs, "baseline_wrong_ach")
            # primary slice: majority of seeds say wrong&ach (threshold 0.5)
            mask = wrong_ach >= 0.5
            d = (o2_top1 - bl_top1)[mask]
            mean, lo, hi, n = boot_ci(d, args.reps, rng)
            corr_rows.append(
                f"{pool}\t{bl}\twrong_achievable\t{n}\t"
                f"{bl_top1[mask].mean()*100:.4f}\t{o2_top1[mask].mean()*100:.4f}\t"
                f"{mean*100:.4f}\t{lo*100:.4f}\t{hi*100:.4f}\tpre_registered"
            )

            # narrow-margin slice from baseline scores only (among auc-definable)
            margin = seed_avg_metric(store, pool, bl, idxs, "margin_cw")
            definable = ~np.isnan(margin)
            if definable.sum() >= 30:
                thr = np.nanpercentile(margin[definable], 33.333)
                narrow = definable & (margin <= thr)
                d2 = (o2_top1 - bl_top1)[narrow]
                mean2, lo2, hi2, n2 = boot_ci(d2, args.reps, rng)
                corr_rows.append(
                    f"{pool}\t{bl}\tnarrow_margin_p33\t{n2}\t"
                    f"{bl_top1[narrow].mean()*100:.4f}\t{o2_top1[narrow].mean()*100:.4f}\t"
                    f"{mean2*100:.4f}\t{lo2*100:.4f}\t{hi2*100:.4f}\tpre_registered"
                )

            # complement of wrong_ach (sanity): overall for reference
            d3 = o2_top1 - bl_top1
            mean3, lo3, hi3, n3 = boot_ci(d3, args.reps, rng)
            corr_rows.append(
                f"{pool}\t{bl}\tall_questions\t{n3}\t"
                f"{bl_top1.mean()*100:.4f}\t{o2_top1.mean()*100:.4f}\t"
                f"{mean3*100:.4f}\t{lo3*100:.4f}\t{hi3*100:.4f}\treference"
            )

    (OUT / "baseline_error_correction.tsv").write_text("\n".join(corr_rows) + "\n")

    # ---- 3. p2-expanded coverage proxy (exploratory) ----
    cov_f = seed_avg_metric(store, "frozen", "O2", idxs, "cov")
    cov_p = seed_avg_metric(store, "p2", "O2", idxs, "cov")
    expanded = (cov_f < 0.5) & (cov_p >= 0.5)
    complement = ~expanded
    exp_rows = [
        "pair\tmetric\tslice\tn\tgap_pp\tci_lo\tci_hi\tnote"
    ]
    for ch in CHALLENGERS:
        for metric in ("top1", "auc_cw"):
            o2_p = seed_avg_metric(store, "p2", "O2", idxs, metric)
            c_p = seed_avg_metric(store, "p2", ch, idxs, metric)
            gap = o2_p - c_p
            for name, mask in (("p2_new_cov", expanded), ("not_p2_new_cov", complement)):
                mean, lo, hi, n = boot_ci(gap[mask], args.reps, rng)
                exp_rows.append(
                    f"O2-{ch}\t{metric}\t{name}\t{n}\t{mean*100:.4f}\t"
                    f"{lo*100:.4f}\t{hi*100:.4f}\texploratory_proxy"
                )
    (OUT / "p2_expanded_proxy.tsv").write_text("\n".join(exp_rows) + "\n")

    # ---- summary markdown ----
    lines = [
        "# O2 advantage slices（先验检验，5-seed 同池）",
        "",
        "切片定义只用 gold / 基线 scorer；不用 O2 分数。",
        "主比较：GSM8K-Test N=16，种子 42–46 题级平均后配对。",
        "",
        "## 1. 差中之差（p2 gap − frozen gap）",
        "",
        "```",
        *did_rows,
        "```",
        "",
        "## 2. 基线排错且池内可达（wrong_achievable）+ 窄间隔",
        "",
        "```",
        *corr_rows,
        "```",
        "",
        "## 3. p2 新覆盖代理（探索性）",
        "",
        f"n_p2_new_cov = {int(expanded.sum())} / {len(idxs)}",
        "",
        "```",
        *exp_rows,
        "```",
        "",
        "## 读数纪律",
        "",
        "- Top1 与 auc_cw 分开写；auc_cw 赢 ≠ Top1 赢。",
        "- DiD CI 跨 0 → 不支持「p2 池上 O2 优势扩大」。",
        "- wrong_achievable 上若 O2 Top1 无稳定正 Δ → 不包装成「特别会纠错」。",
        "- §3 为无候选 ID 的题级代理，不作主 claim。",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print("wrote", OUT)
    print("\n".join(did_rows))
    print("---")
    print("\n".join([r for r in corr_rows if "wrong_achievable" in r or r.startswith("pool")]))


if __name__ == "__main__":
    main()
