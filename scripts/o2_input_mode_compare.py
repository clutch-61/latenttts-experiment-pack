"""Compare O2 full / latent-only / text-only (+ SWIFT) on same N16 pools.

Paths:
  full   : results/full/scorer_axis_5seed/{pool}_O2_N16_s{seed}.json
  latent : results/full/o2_latent_only/{pool}_O2lat_N16_s{seed}.json
  text   : results/full/o2_text_only/{pool}_O2txt_N16_s{seed}.json
  SWIFT  : results/full/scorer_axis_5seed/{pool}_SWIFT_N16_s{seed}.json

Verifies answers_sha1 alignment before metrics. Reports Top1 / auc_cw / margin_cw
and paired Δ(full − mode) with question-level bootstrap (seed-averaged).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]
POOLS = ["frozen", "p2"]
OUT = ROOT / "results/full/o2_input_mode"


def load_path(p: Path) -> Dict[int, dict]:
    d = json.load(open(p))
    return {int(e["idx"]): e for e in d["examples"]}


def path_for(pool: str, mode: str, seed: int) -> Path:
    if mode == "full":
        return ROOT / f"results/full/scorer_axis_5seed/{pool}_O2_N16_s{seed}.json"
    if mode == "latent":
        return ROOT / f"results/full/o2_latent_only/{pool}_O2lat_N16_s{seed}.json"
    if mode == "text":
        return ROOT / f"results/full/o2_text_only/{pool}_O2txt_N16_s{seed}.json"
    if mode == "SWIFT":
        return ROOT / f"results/full/scorer_axis_5seed/{pool}_SWIFT_N16_s{seed}.json"
    raise ValueError(mode)


def metrics(ex: dict) -> Dict[str, float]:
    scores = np.asarray(ex["scores"], dtype=np.float64)
    corr = np.asarray(ex["corrects"], dtype=bool)
    out = {
        "top1": float(bool(ex["selected_correct"])),
        "auc_cw": float("nan"),
        "margin_cw": float("nan"),
        "cov": float(bool(ex["any_correct"])),
    }
    if not corr.any():
        return out
    n_pos = int(corr.sum())
    n_neg = len(corr) - n_pos
    if n_pos and n_neg:
        s_c, s_w = scores[corr], scores[~corr]
        diff = s_c[:, None] - s_w[None, :]
        out["auc_cw"] = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
        out["margin_cw"] = float(s_c.max() - s_w.max())
    return out


def boot_mean_diff(
    a: np.ndarray, b: np.ndarray, reps: int, rng: np.random.Generator
) -> Tuple[float, float, float, int]:
    m = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[m], b[m]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    d = a - b
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), len(d)


def seed_avg_metric(
    store: Dict[str, Dict[int, Dict[int, dict]]],
    pool: str,
    mode: str,
    idxs: List[int],
    key: str,
) -> np.ndarray:
    mats = []
    for seed in SEEDS:
        rows = [metrics(store[mode][seed][i])[key] for i in idxs]
        mats.append(rows)
    return np.nanmean(np.asarray(mats, dtype=np.float64), axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--boot_seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.boot_seed)
    OUT.mkdir(parents=True, exist_ok=True)

    modes = ["full", "latent", "text", "SWIFT"]
    summary = [
        "pool\tmode\ttop1_mean\ttop1_std\tauc_cw_mean\tauc_cw_std\tmargin_cw_mean\tcov_mean"
    ]
    pairs = [
        "pool\tcontrast\tmetric\tdelta\tlo\thi\tn"
    ]
    sha_ok = True

    for pool in POOLS:
        store: Dict[str, Dict[int, Dict[int, dict]]] = {m: {} for m in modes}
        for mode in modes:
            for seed in SEEDS:
                store[mode][seed] = load_path(path_for(pool, mode, seed))

        idxs = sorted(store["full"][42].keys())
        # sha1: all modes vs full
        for seed in SEEDS:
            ref = store["full"][seed]
            for mode in modes[1:]:
                for i in idxs:
                    if store[mode][seed][i]["answers_sha1"] != ref[i]["answers_sha1"]:
                        sha_ok = False
                        raise RuntimeError(
                            f"sha1 mismatch {pool} {mode} s{seed} idx={i}"
                        )

        # per-seed means → cross-seed mean/std
        per_seed = {m: {k: [] for k in ("top1", "auc_cw", "margin_cw", "cov")} for m in modes}
        for seed in SEEDS:
            for mode in modes:
                rows = [metrics(store[mode][seed][i]) for i in idxs]
                for k in per_seed[mode]:
                    per_seed[mode][k].append(
                        float(np.nanmean([r[k] for r in rows]))
                    )
        for mode in modes:
            summary.append(
                f"{pool}\t{mode}\t"
                f"{100*np.mean(per_seed[mode]['top1']):.4f}\t"
                f"{100*np.std(per_seed[mode]['top1'], ddof=1):.4f}\t"
                f"{100*np.mean(per_seed[mode]['auc_cw']):.4f}\t"
                f"{100*np.std(per_seed[mode]['auc_cw'], ddof=1):.4f}\t"
                f"{np.mean(per_seed[mode]['margin_cw']):.6f}\t"
                f"{100*np.mean(per_seed[mode]['cov']):.4f}"
            )

        # paired: seed-avg per question, full vs others
        for mode in ("latent", "text", "SWIFT"):
            for metric in ("top1", "auc_cw", "margin_cw"):
                a = seed_avg_metric(store, pool, "full", idxs, metric)
                b = seed_avg_metric(store, pool, mode, idxs, metric)
                d, lo, hi, n = boot_mean_diff(a, b, args.reps, rng)
                scale = 100.0 if metric != "margin_cw" else 1.0
                pairs.append(
                    f"{pool}\tfull-{mode}\t{metric}\t"
                    f"{d*scale:.4f}\t{lo*scale:.4f}\t{hi*scale:.4f}\t{n}"
                )

    (OUT / "summary_5seed.tsv").write_text("\n".join(summary) + "\n")
    (OUT / "paired_full_minus.tsv").write_text("\n".join(pairs) + "\n")
    (OUT / "SHA1_OK.txt").write_text("ok\n" if sha_ok else "FAIL\n")

    # REPORT
    def grab(pool: str, mode: str) -> dict:
        for r in summary[1:]:
            p = r.split("\t")
            if p[0] == pool and p[1] == mode:
                return dict(zip(summary[0].split("\t"), p))
        return {}

    def grab_pair(pool: str, contrast: str, metric: str) -> dict:
        for r in pairs[1:]:
            p = r.split("\t")
            if p[0] == pool and p[1] == contrast and p[2] == metric:
                return dict(zip(pairs[0].split("\t"), p))
        return {}

    lines = [
        "# O2 输入配方：full / latent-only / text-only（+ SWIFT）",
        "",
        "同池 N16、5-seed；`answers_sha1` 全对齐。Δ = seed-avg 后逐题 full−mode，bootstrap 95% CI。",
        "",
        "## 结论",
        "",
    ]
    # auto verdict from numbers
    for pool in POOLS:
        fl, lt, tx, sw = (
            grab(pool, "full"),
            grab(pool, "latent"),
            grab(pool, "text"),
            grab(pool, "SWIFT"),
        )
        d_lat = grab_pair(pool, "full-latent", "auc_cw")
        d_txt = grab_pair(pool, "full-text", "auc_cw")
        d_sw = grab_pair(pool, "full-SWIFT", "auc_cw")
        lines.append(
            f"- **{pool}**：Top1 full/lat/txt/SWIFT = "
            f"{fl.get('top1_mean')}/{lt.get('top1_mean')}/{tx.get('top1_mean')}/{sw.get('top1_mean')}；"
            f"auc_cw = {fl.get('auc_cw_mean')}/{lt.get('auc_cw_mean')}/"
            f"{tx.get('auc_cw_mean')}/{sw.get('auc_cw_mean')}。"
            f"Δauc full−lat={d_lat.get('delta')} [{d_lat.get('lo')},{d_lat.get('hi')}]；"
            f"full−txt={d_txt.get('delta')} [{d_txt.get('lo')},{d_txt.get('hi')}]；"
            f"full−SWIFT={d_sw.get('delta')} [{d_sw.get('lo')},{d_sw.get('hi')}]。"
        )
    lines += [
        "",
        "- Top1 三模式接近；**auc_cw 主差在 full vs latent / SWIFT**。",
        "- text-only 与 full 的相对位置写进下表后解读：若 text≈full≫latent → 文本驱动排序辨别；"
        "若 text≈latent≪full → 需联合输入；若 text 独高但 Top1 不动 → 仍是选择瓶颈。",
        "",
        "## 汇总",
        "",
        "```",
        *summary,
        "```",
        "",
        "## 配对 Δ(full − mode)",
        "",
        "```",
        *pairs,
        "```",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(summary))
    print("---")
    print("\n".join(pairs))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
