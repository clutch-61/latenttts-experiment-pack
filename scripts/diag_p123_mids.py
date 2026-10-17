#!/usr/bin/env python3
"""Intermediate quantities for P1–P3 system root-cause analysis.

Reads BoN / generator dump JSONs and reports:
  Cov, Acc (claim=wvote by default), Acc_top1, leftover vs claim selection,
  systemΔ vs A+B0, pilot↔GSM transfer gap.

Old dumps (no claim_agg): meta.accuracy was top-1; wvote is recomputed from examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.system_scoring import wvote_select  # noqa: E402


def _load(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"MISSING {p}")
        return None
    return json.load(open(p))


def _wvote_acc_from_examples(exs) -> float | None:
    if not exs or "scores" not in exs[0]:
        return None
    n_ok = 0
    for e in exs:
        ans = [str(a) for a in e["answers"]]
        sc = [float(x) for x in e["scores"]]
        c = [bool(x) for x in e["corrects"]]
        n_ok += int(c[wvote_select(ans, sc)])
    return n_ok / max(len(exs), 1)


def _leftover_vs_wvote(exs) -> tuple[float, float] | tuple[None, None]:
    if not exs or "any_correct" not in exs[0]:
        return None, None
    n = len(exs)
    leftover_n = 0
    pool = 0
    for e in exs:
        if not e.get("any_correct"):
            continue
        pool += 1
        if "claim_correct" in e:
            ok = bool(e["claim_correct"])
        elif "selected_wvote_correct" in e:
            ok = bool(e["selected_wvote_correct"])
        elif "scores" in e:
            ans = [str(a) for a in e["answers"]]
            sc = [float(x) for x in e["scores"]]
            c = [bool(x) for x in e["corrects"]]
            ok = bool(c[wvote_select(ans, sc)])
        else:
            ok = bool(e.get("selected_correct"))
        if not ok:
            leftover_n += 1
    return leftover_n / max(n, 1), leftover_n / max(pool, 1)


def _acc_cov(blob):
    if blob is None:
        return None
    if isinstance(blob, dict) and "meta" in blob and isinstance(blob["meta"], dict):
        m = blob["meta"]
        if "accuracy" in m:
            exs = blob.get("examples") or []
            top1 = float(m["accuracy_top1"]) if m.get("accuracy_top1") is not None else None
            wvote = float(m["accuracy_wvote"]) if m.get("accuracy_wvote") is not None else None
            claim_agg = m.get("claim_agg")
            if wvote is None:
                wvote = _wvote_acc_from_examples(exs)
            if top1 is None and exs and "selected_correct" in exs[0]:
                top1 = sum(1 for e in exs if e.get("selected_correct")) / max(len(exs), 1)
            if claim_agg is None:
                # Historical dumps stored top-1 in accuracy
                if top1 is None:
                    top1 = float(m["accuracy"])
                claim_agg = "wvote"
                claim = wvote if wvote is not None else top1
            else:
                claim = float(m["accuracy"])
            out = {
                "acc": claim,
                "acc_top1": top1,
                "acc_wvote": wvote,
                "cov": float(m["coverage"]) if m.get("coverage") is not None else None,
                "claim_agg": claim_agg,
            }
            lef, fail = _leftover_vs_wvote(exs)
            if lef is not None:
                out["leftover_rate"] = lef
                out["o2_pick_fail_given_pool"] = fail
            return out
    if "Accuracy" in blob or "accuracy" in blob:
        acc = blob.get("Accuracy", blob.get("accuracy"))
        cov = blob.get("Coverage", blob.get("coverage"))
        if isinstance(acc, str) and acc.endswith("%"):
            acc = float(acc[:-1]) / 100.0
        if isinstance(cov, str) and cov.endswith("%"):
            cov = float(cov[:-1]) / 100.0
        return {"acc": float(acc), "cov": float(cov) if cov is not None else None}
    if "pass@1_first_sample" in blob or "pass_at_1" in blob or "first_acc" in blob:
        first = blob.get("pass@1_first_sample", blob.get("pass_at_1", blob.get("first_acc")))
        cov = blob.get("coverage", blob.get("cov_at_n", blob.get("cov")))
        return {"acc": float(first), "cov": float(cov) if cov is not None else None}
    if isinstance(blob, list):
        n = len(blob)
        ok = sum(1 for x in blob if x.get("correct") or x.get("is_correct"))
        return {"acc": ok / max(n, 1), "cov": None}
    for key in ("summary", "metrics", "stats"):
        if key in blob and isinstance(blob[key], dict):
            return _acc_cov(blob[key])
    return None


def _pct(x):
    if x is None:
        return None
    return round(100.0 * x, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system_o2", required=True, help="gen+O2 BoN json")
    ap.add_argument("--system_b0", default=None, help="same gen + B0 BoN json (health)")
    ap.add_argument("--baseline_ab0", default=None, help="A+B0 BoN json (claim denominator)")
    ap.add_argument("--gen_only", default=None, help="eval_sf_generator dump")
    ap.add_argument("--pilot_o2", default=None, help="optional pilot gen+O2 for transfer gap")
    ap.add_argument("--mid_jsonl", default=None, help="train mid_metrics.jsonl")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--a_b0_acc", type=float, default=0.3374, help="fallback A+B0 Acc if no dump")
    args = ap.parse_args()

    sys_o2 = _acc_cov(_load(args.system_o2))
    sys_b0 = _acc_cov(_load(args.system_b0)) if args.system_b0 else None
    base = _acc_cov(_load(args.baseline_ab0)) if args.baseline_ab0 else None
    gen = _acc_cov(_load(args.gen_only)) if args.gen_only else None
    pilot = _acc_cov(_load(args.pilot_o2)) if args.pilot_o2 else None

    # Paper protocol denominator is A+B0 top-1 (33.74), not wvote
    if base and base.get("acc_top1") is not None:
        base_acc = base["acc_top1"]
    elif base:
        # old A+B0 dump: meta.accuracy is top-1
        base_acc = base["acc_top1"] if base.get("acc_top1") is not None else (
            float(_load(args.baseline_ab0)["meta"]["accuracy"]) if args.baseline_ab0 else args.a_b0_acc
        )
        # For old dumps _acc_cov may have overwritten acc with wvote; prefer raw meta
        raw = _load(args.baseline_ab0)
        if raw and raw.get("meta", {}).get("claim_agg") is None:
            base_acc = float(raw["meta"]["accuracy"])
    else:
        base_acc = args.a_b0_acc

    report = {
        "system_o2": sys_o2,
        "system_b0": sys_b0,
        "baseline_a_b0_acc": base_acc,
        "gen_only": gen,
        "pilot_o2": pilot,
    }
    if sys_o2:
        claim_acc = sys_o2.get("acc_wvote") if sys_o2.get("acc_wvote") is not None else sys_o2["acc"]
        leftover = sys_o2.get("leftover_rate")
        if leftover is None and sys_o2.get("cov") is not None:
            leftover = sys_o2["cov"] - claim_acc
        delta = claim_acc - base_acc
        report["mids"] = {
            "system_acc_o2_pct": _pct(claim_acc),
            "system_acc_o2_top1_pct": _pct(sys_o2.get("acc_top1")),
            "system_acc_o2_wvote_pct": _pct(sys_o2.get("acc_wvote")),
            "claim_agg": sys_o2.get("claim_agg"),
            "system_cov_pct": _pct(sys_o2.get("cov")),
            "leftover_pp": _pct(leftover) if leftover is not None else None,
            "pick_fail_given_pool_pct": _pct(sys_o2.get("o2_pick_fail_given_pool")),
            "health_acc_b0_pct": _pct(
                (sys_b0.get("acc_top1") if sys_b0 and sys_b0.get("acc_top1") is not None else sys_b0["acc"])
                if sys_b0
                else None
            ),
            "delta_vs_A_B0_pp": round(100.0 * delta, 2),
            "target_plus10_acc_pct": _pct(base_acc + 0.10),
            "gap_to_plus10_pp": round(100.0 * ((base_acc + 0.10) - claim_acc), 2),
        }
        if pilot and pilot.get("acc") is not None:
            pilot_claim = pilot.get("acc_wvote") if pilot.get("acc_wvote") is not None else pilot["acc"]
            report["mids"]["pilot_o2_acc_pct"] = _pct(pilot_claim)
            report["mids"]["transfer_gap_pp"] = round(100.0 * (pilot_claim - claim_acc), 2)
        if gen:
            report["mids"]["gen_first_pct"] = _pct(gen.get("acc"))
            report["mids"]["gen_cov_pct"] = _pct(gen.get("cov"))

    if args.mid_jsonl and Path(args.mid_jsonl).exists():
        lines = [json.loads(l) for l in open(args.mid_jsonl) if l.strip()]
        summaries = [x for x in lines if "leftover_rate" in x]
        if summaries:
            report["train_mid_last"] = summaries[-1]

    print(json.dumps(report, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out_json, "w"), indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
