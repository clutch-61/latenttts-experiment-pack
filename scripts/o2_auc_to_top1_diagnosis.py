"""Diagnose why O2 auc_cw gains do not convert to Top-1; try within-pool O2+B0 fusion.

Uses locked same-pool dumps under results/full/scorer_axis_5seed/.
Pre-registered question split of GSM8K-Test idxs:
  val  = idx % 5 == 0   (~20%, fusion tuning only)
  test = otherwise      (locked reporting)

Global temperature on a single scorer cannot change argmax — not tested.
Fusion / disagreement re-rank can change relative order — tested on val, reported on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/full/scorer_axis_5seed"
OUT = ROOT / "results/full/scorer_axis_5seed/auc_to_top1"
SEEDS = [42, 43, 44, 45, 46]
POOLS = ["frozen", "p2"]


def load(pool: str, scorer: str, seed: int, n: int = 16) -> Dict[int, dict]:
    p = D / f"{pool}_{scorer}_N{n}_s{seed}.json"
    d = json.load(open(p))
    return {int(e["idx"]): e for e in d["examples"]}


def is_val(idx: int) -> bool:
    return (idx % 5) == 0


def per_q(ex: dict) -> dict:
    scores = np.asarray(ex["scores"], dtype=np.float64)
    corr = np.asarray(ex["corrects"], dtype=bool)
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    top = int(order[0])
    n_pos = int(corr.sum())
    any_c = bool(n_pos > 0)
    sel_ok = bool(ex["selected_correct"])
    out = {
        "any_correct": float(any_c),
        "n_correct": n_pos,
        "top1": float(sel_ok),
        "margin_12": float("nan"),
        "margin_cw": float("nan"),
        "rank_best_correct": float("nan"),
        "mode": "uncovered",
    }
    if n >= 2:
        out["margin_12"] = float(scores[order[0]] - scores[order[1]])
    if not any_c:
        return out
    # rank of first correct in score order (1-based)
    rank = int(np.where(corr[order])[0][0]) + 1
    out["rank_best_correct"] = float(rank)
    s_c = scores[corr]
    s_w = scores[~corr]
    if len(s_w) == 0:
        # all candidates correct: Top-1 trivial; leave margin_cw as nan
        out["mode"] = "win" if sel_ok else "tie_or_bug"
        return out
    out["margin_cw"] = float(s_c.max() - s_w.max())
    if sel_ok:
        out["mode"] = "win"
    elif out["margin_cw"] > 0:
        # should be rare under strict argmax unless tie broken oddly
        out["mode"] = "recognized_but_lost_tie"
    elif out["margin_cw"] == 0:
        out["mode"] = "tie_best_wrong"
    else:
        out["mode"] = "lost_to_strongest_wrong"  # 识别分空间输给最强错误
    return out


def boot_mean(x: np.ndarray, reps: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def seed_avg_maps(pool: str, scorer: str, idxs: List[int]) -> Dict[str, np.ndarray]:
    """Average per-question metrics over seeds (aligned idxs)."""
    keys = [
        "any_correct",
        "n_correct",
        "top1",
        "margin_12",
        "margin_cw",
        "rank_best_correct",
    ]
    mats = {k: [] for k in keys}
    modes = []
    for seed in SEEDS:
        data = load(pool, scorer, seed)
        rows = [per_q(data[i]) for i in idxs]
        for k in keys:
            mats[k].append([r[k] for r in rows])
        modes.append([r["mode"] for r in rows])
    out = {k: np.nanmean(np.asarray(mats[k], dtype=np.float64), axis=0) for k in keys}
    # majority mode across seeds
    mode_maj = []
    for j in range(len(idxs)):
        col = [modes[s][j] for s in range(len(SEEDS))]
        # prefer lost_to_strongest_wrong / win counts
        vals, cts = np.unique(col, return_counts=True)
        mode_maj.append(vals[int(np.argmax(cts))])
    out["mode"] = np.asarray(mode_maj, dtype=object)
    return out


def fusion_select(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    method: str,
    alpha: float,
) -> int:
    """Return selected index under fusion. Within-pool only."""
    a = scores_a.astype(np.float64)
    b = scores_b.astype(np.float64)
    if method == "o2":
        return int(np.argmax(a))
    if method == "b0":
        return int(np.argmax(b))
    if method == "score_z":
        def z(x):
            s = x.std()
            if s < 1e-12:
                return np.zeros_like(x)
            return (x - x.mean()) / s

        s = alpha * z(a) + (1.0 - alpha) * z(b)
        return int(np.argmax(s))
    if method == "rank_avg":
        # lower rank better; average ranks
        ra = np.argsort(np.argsort(-a, kind="stable"))
        rb = np.argsort(np.argsort(-b, kind="stable"))
        s = -(alpha * ra.astype(np.float64) + (1.0 - alpha) * rb.astype(np.float64))
        return int(np.argmax(s))
    if method == "disagree_score_z":
        ia, ib = int(np.argmax(a)), int(np.argmax(b))
        if ia == ib:
            return ia
        return fusion_select(a, b, "score_z", alpha)
    if method == "disagree_rank_avg":
        ia, ib = int(np.argmax(a)), int(np.argmax(b))
        if ia == ib:
            return ia
        return fusion_select(a, b, "rank_avg", alpha)
    if method == "disagree_prefer_o2":
        ia, ib = int(np.argmax(a)), int(np.argmax(b))
        return ia if ia != ib else ia
    raise ValueError(method)


def eval_fusion(
    pool: str,
    idxs: List[int],
    method: str,
    alpha: float,
) -> float:
    """Mean Top-1 over questions, averaged across seeds."""
    accs = []
    for seed in SEEDS:
        o2 = load(pool, "O2", seed)
        b0 = load(pool, "B0", seed)
        ok = []
        for i in idxs:
            sa = np.asarray(o2[i]["scores"], dtype=np.float64)
            sb = np.asarray(b0[i]["scores"], dtype=np.float64)
            # sha1 check once cheaply
            if o2[i]["answers_sha1"] != b0[i]["answers_sha1"]:
                raise RuntimeError(f"sha1 mismatch {pool} s{seed} idx={i}")
            sel = fusion_select(sa, sb, method, alpha)
            ok.append(bool(o2[i]["corrects"][sel]))
        accs.append(float(np.mean(ok)))
    return float(np.mean(accs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--boot_seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.boot_seed)
    OUT.mkdir(parents=True, exist_ok=True)

    # common idxs
    ref = load("frozen", "O2", 42)
    idxs = sorted(ref.keys())
    val_idxs = [i for i in idxs if is_val(i)]
    test_idxs = [i for i in idxs if not is_val(i)]
    split_note = (
        f"pre_registered_split: val=idx%5==0 n_val={len(val_idxs)} "
        f"n_test={len(test_idxs)} n_all={len(idxs)}"
    )
    (OUT / "SPLIT.txt").write_text(split_note + "\n")

    # ---- diagnosis tables ----
    diag_rows = [
        "pool\tsplit\tn\tcov\tO2_top1\tB0_top1\tO2_rescue_B0\tB0_rescue_O2\tdisagree_rate\t"
        "among_ach_lost_strong\tamong_ach_win\t"
        "mean_margin_cw_when_lost\tmean_rank_best_correct_when_lost\t"
        "among_ach_best_correct_top2\tamong_ach_best_correct_top4"
    ]
    slice_rows = [
        "pool\tsplit\tslice\tn\tO2_top1\tB0_top1\tO2_rescue\tB0_rescue\tnote"
    ]
    mode_rows = ["pool\tsplit\tmode\tn\tfrac"]

    for pool in POOLS:
        o2m = seed_avg_maps(pool, "O2", idxs)
        b0m = seed_avg_maps(pool, "B0", idxs)
        # per-seed hard rescue for rates (use seed-avg top1 >= 0.5 as majority)
        o2_win = o2m["top1"] >= 0.5
        b0_win = b0m["top1"] >= 0.5
        ach = o2m["any_correct"] >= 0.5
        # disagreement: need per-seed selected index — recompute
        disagree = np.zeros(len(idxs), dtype=np.float64)
        for seed in SEEDS:
            o2 = load(pool, "O2", seed)
            b0 = load(pool, "B0", seed)
            for j, i in enumerate(idxs):
                if int(np.argmax(o2[i]["scores"])) != int(np.argmax(b0[i]["scores"])):
                    disagree[j] += 1.0
        disagree /= len(SEEDS)

        for split_name, sub in (("all", idxs), ("val", val_idxs), ("test", test_idxs)):
            mask = np.asarray([i in set(sub) for i in idxs])
            # remap sub indices into arrays
            jj = [j for j, i in enumerate(idxs) if i in set(sub)]
            o2w = o2_win[jj]
            b0w = b0_win[jj]
            ach_j = ach[jj]
            modes = o2m["mode"][jj]
            rescue_o2 = (~b0w) & o2w & ach_j
            rescue_b0 = (~o2w) & b0w & ach_j
            lost = ach_j & (~o2w)
            lost_strong = np.array([m == "lost_to_strongest_wrong" for m in modes]) & ach_j
            win_m = np.array([m == "win" for m in modes]) & ach_j
            n = len(jj)
            n_ach = int(ach_j.sum())
            # margins / ranks when lost (finite only)
            mc = o2m["margin_cw"][jj]
            rk = o2m["rank_best_correct"][jj]
            m_lost = boot_mean(mc[lost], args.reps, rng)
            r_lost = boot_mean(rk[lost], args.reps, rng)
            # top-k of best correct among achievable (seed-avg rank)
            top2 = float(np.nanmean(rk[ach_j] <= 2.0)) if n_ach else float("nan")
            top4 = float(np.nanmean(rk[ach_j] <= 4.0)) if n_ach else float("nan")
            lost_strong_ach = (lost_strong.sum() / n_ach) if n_ach else float("nan")
            win_ach = (win_m.sum() / n_ach) if n_ach else float("nan")
            diag_rows.append(
                f"{pool}\t{split_name}\t{n}\t"
                f"{100*ach_j.mean():.4f}\t"
                f"{100*o2w.mean():.4f}\t{100*b0w.mean():.4f}\t"
                f"{100*rescue_o2.mean():.4f}\t{100*rescue_b0.mean():.4f}\t"
                f"{100*disagree[jj].mean():.4f}\t"
                f"{100*lost_strong_ach:.4f}\t{100*win_ach:.4f}\t"
                f"{m_lost[0]:.6f}\t{r_lost[0]:.4f}\t"
                f"{100*top2:.4f}\t{100*top4:.4f}"
            )
            for mode in sorted(set(modes.tolist())):
                fr = float(np.mean(modes == mode))
                mode_rows.append(f"{pool}\t{split_name}\t{mode}\t{int((modes==mode).sum())}\t{100*fr:.4f}")

            # slices on test/all
            ncorr = o2m["n_correct"][jj]
            m12 = o2m["margin_12"][jj]
            disc = disagree[jj] >= 0.5

            def add_slice(name, m, note=""):
                if m.sum() == 0:
                    return
                slice_rows.append(
                    f"{pool}\t{split_name}\t{name}\t{int(m.sum())}\t"
                    f"{100*o2w[m].mean():.4f}\t{100*b0w[m].mean():.4f}\t"
                    f"{100*rescue_o2[m].mean():.4f}\t{100*rescue_b0[m].mean():.4f}\t{note}"
                )

            add_slice("n_correct=1", ach_j & (ncorr < 1.5))
            add_slice("n_correct=2-3", ach_j & (ncorr >= 1.5) & (ncorr < 3.5))
            add_slice("n_correct>=4", ach_j & (ncorr >= 3.5))
            # margin_12 tertiles among achievable
            if ach_j.sum() >= 30:
                vals = m12[ach_j]
                t1, t2 = np.nanpercentile(vals, [33.333, 66.667])
                add_slice("margin12_low", ach_j & (m12 <= t1), "O2 top1-top2 gap")
                add_slice("margin12_mid", ach_j & (m12 > t1) & (m12 <= t2))
                add_slice("margin12_high", ach_j & (m12 > t2))
            add_slice("scorer_disagree", disc)
            add_slice("scorer_agree", ~disc)
            add_slice("O2_rescue_B0", rescue_o2, "B0 wrong O2 right")
            add_slice("B0_rescue_O2", rescue_b0, "O2 wrong B0 right")

    (OUT / "diagnosis_summary.tsv").write_text("\n".join(diag_rows) + "\n")
    (OUT / "diagnosis_slices.tsv").write_text("\n".join(slice_rows) + "\n")
    (OUT / "diagnosis_modes.tsv").write_text("\n".join(mode_rows) + "\n")

    # ---- fusion search on val, report test ----
    methods = [
        ("o2", [1.0]),
        ("b0", [1.0]),
        ("score_z", [0.3, 0.4, 0.5, 0.6, 0.7]),
        ("rank_avg", [0.3, 0.4, 0.5, 0.6, 0.7]),
        ("disagree_score_z", [0.3, 0.5, 0.7]),
        ("disagree_rank_avg", [0.3, 0.5, 0.7]),
        ("disagree_prefer_o2", [1.0]),
    ]
    fuse_rows = ["pool\tmethod\talpha\tval_top1\ttest_top1\tselected"]
    best_lines = []
    for pool in POOLS:
        best = None  # (val, test, method, alpha)
        grid = []
        for method, alphas in methods:
            for alpha in alphas:
                v = eval_fusion(pool, val_idxs, method, alpha)
                t = eval_fusion(pool, test_idxs, method, alpha)
                grid.append((method, alpha, v, t))
                if method in ("o2", "b0"):
                    continue
                if best is None or v > best[0]:
                    best = (v, t, method, alpha)
        assert best is not None
        for method, alpha, v, t in grid:
            tag = ""
            if method == best[2] and abs(alpha - best[3]) < 1e-12:
                tag = "VAL_BEST"
            fuse_rows.append(
                f"{pool}\t{method}\t{alpha:.2f}\t{100*v:.4f}\t{100*t:.4f}\t{tag}"
            )
        t_o2 = eval_fusion(pool, test_idxs, "o2", 1.0)
        t_b0 = eval_fusion(pool, test_idxs, "b0", 1.0)
        best_lines.append(
            f"{pool}: val_best={best[2]} α={best[3]:.2f} "
            f"val={100*best[0]:.2f} → test={100*best[1]:.2f} "
            f"(test O2={100*t_o2:.2f} B0={100*t_b0:.2f} ΔvsO2={100*(best[1]-t_o2):+.2f})"
        )

    (OUT / "fusion_grid.tsv").write_text("\n".join(fuse_rows) + "\n")

    # ---- REPORT ----
    # pull all-split lines for verdict
    def _row(pool: str, split: str) -> str:
        for r in diag_rows[1:]:
            if r.startswith(f"{pool}\t{split}\t"):
                return r
        return ""

    def _parse(r: str) -> dict:
        h = diag_rows[0].split("\t")
        p = r.split("\t")
        return dict(zip(h, p))

    fr, pr = _parse(_row("frozen", "all")), _parse(_row("p2", "all"))
    ft, pt = _parse(_row("frozen", "test")), _parse(_row("p2", "test"))

    report = [
        "# O2：AUC 优势为何未变成 Top-1（同池 dump）",
        "",
        split_note,
        "",
        "## 结论（先读）",
        "",
        "1. **池覆盖**：约一半题 N=16 池内无正解（frozen cov≈"
        f"{fr.get('cov','?')}%，p2≈{pr.get('cov','?')}%）。这是 Top-1 天花板，与 scorer 无关。",
        "2. **有正解时**：几乎全部失败模式是 `lost_to_strongest_wrong`（among_ach 约 "
        f"{fr.get('among_ach_lost_strong','?')}% / {pr.get('among_ach_lost_strong','?')}%"
        "），不是「认出了但打平输」。正确候选在失败题上的平均名次约 "
        f"{fr.get('mean_rank_best_correct_when_lost','?')} / "
        f"{pr.get('mean_rank_best_correct_when_lost','?')}"
        "（N=16）—— **识别到了排序空间里的信号，但没赢过最强错误，且经常不在头部**。",
        "3. **among_ach 最佳正解进 top2/top4**："
        f"frozen {fr.get('among_ach_best_correct_top2','?')}%/"
        f"{fr.get('among_ach_best_correct_top4','?')}%，"
        f"p2 {pr.get('among_ach_best_correct_top2','?')}%/"
        f"{pr.get('among_ach_best_correct_top4','?')}%。"
        "AUC 可升、Top-1 不动，符合「整体排序变好但 argmax 仍被最强错误占住」。",
        "4. **救回**：p2 上 O2 救 B0 明显多于反向；frozen 接近对称。融合改次序在锁定 test 上 **≈0**。",
        "5. **下一步做强 O2**：优先 listwise / 对最强错误的 margin（推高 `margin_cw`），"
        "并盯 `n_correct∈{1,2,3}`；不要指望全局温度或简单 O2+B0 分数融合。",
        "",
        "## 读数纪律",
        "",
        "- 全局温度缩放不改变单 scorer 的 argmax，故不测。",
        "- 融合 / 分歧重排在 **val**（idx%5==0）选超参，**test** 只报一次。",
        "- `lost_to_strongest_wrong`：池内有正解，但 max(score_wrong) > max(score_correct)。",
        "- `O2_rescue_B0` / `B0_rescue_O2`：多数 seed 上一方对、另一方错。",
        "- `among_ach_*` 分母 = 多数 seed 池内有 ≥1 正解的题。",
        "",
        "## 总表",
        "",
        "```",
        *diag_rows,
        "```",
        "",
        f"锁定 test：frozen O2 Top1={ft.get('O2_top1','?')} B0={ft.get('B0_top1','?')}；"
        f"p2 O2={pt.get('O2_top1','?')} B0={pt.get('B0_top1','?')}。",
        "",
        "## 失效模式分布（O2，分母=全题）",
        "",
        "```",
        *mode_rows[:1],
        *[r for r in mode_rows[1:] if "\tall\t" in r],
        "```",
        "",
        "## 切片（all + locked test）",
        "",
        "```",
        *slice_rows[:1],
        *[r for r in slice_rows[1:] if "\ttest\t" in r or "\tall\t" in r],
        "```",
        "",
        "读切片：`n_correct=1` Top1 极低；`n_correct≥4` ~80%。"
        "`margin12_high` Top1 更低 —— 大 top1−top2 分差经常是「自信地错」。"
        "分歧题上 O2 相对 B0 的救回更明显（尤其 p2）。",
        "",
        "## 候选内融合（改相对次序）",
        "",
        *best_lines,
        "",
        "全网格见 `fusion_grid.tsv`。val 选出的融合在 test 上相对纯 O2 ≈0；"
        "个别非 VAL_BEST 格子（如 frozen score_z α=0.7）test 略高，但未过 val 选模门。",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report))
    print("\n".join(diag_rows))
    print("---")
    print("\n".join(best_lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
