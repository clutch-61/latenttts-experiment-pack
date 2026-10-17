"""Order-decision test: turn "order sensitivity" into the ONLY usable signal.

Motivation
----------
Earlier audits showed O2 (order-preference trained) wins on the *mechanism* axis
(order/hierarchy pair accuracy) but this never converted into a BoN top-1 gain,
because in a real generator pool the correctness of a candidate is decided by
answer extraction -- the scorer only has to rank *already-correct* trajectories,
so order perception is not needed.

This script builds the condition under which order perception IS necessary, and
measures whether each scorer's order sensitivity becomes a usable decision:

  Pool construction (per positive trajectory)
    - candidate 0 : the intact `corrects=1` trajectory (ground-truth order)
    - candidates 1..N-1 : permutation-corrupted variants of that SAME trajectory
  All candidates share identical content (same multiset of latent steps) and the
  same answer label, so answer-based / content-based signals CANNOT discriminate.
  The only usable cue is temporal order.

  Metric: top1 = fraction of pools where the intact candidate is selected.
  A score-blind-to-order scorer is at chance (1/N) because permutation-invariant
  aggregations (mean/sum pooling) map the whole pool to identical scores.

This is a synthetic diagnostic, NOT GSM8K accuracy; it isolates the mechanism.

Usage:
    python scripts/order_decision_test.py --family light --scorer swift \
        --prm_id outputs/swift_latent/best.pt
    python scripts/order_decision_test.py --family gpt2 \
        --prm_id outputs/latentrm_order_pref/best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import CachedPickleDatasetV2, DataCollatorForLatentRM  # noqa: E402
from src.lightweight_scorers import build_lightweight_scorer  # noqa: E402
from src.order_pref.corruption import CORRUPTION_SPECS, apply_corruption  # noqa: E402
from src.order_pref.preference_loss import sequence_score_from_logits  # noqa: E402

CORRUPTIONS = ["adjacent_swap", "segment_swap", "multi_segment", "full_reverse"]


def gather_positives(data_dir: str, max_pos: int) -> List[dict]:
    ds = CachedPickleDatasetV2(data_dir=data_dir, verbose=False, include_gt=True, get_single_sample=True)
    feats = []
    for i in tqdm(range(len(ds)), desc="gather-correct"):
        f = ds[i]
        if "corrects" not in f:
            continue
        if float(f["corrects"].reshape(-1)[0]) > 0.5:
            feats.append(f)
        if max_pos and len(feats) >= max_pos:
            break
    return feats


def build_pool(pos_h: torch.Tensor, n: int, corruption: str, gen: torch.Generator) -> torch.Tensor:
    """(T, D) -> (N, T, D): row 0 intact, rows 1..N-1 corrupted."""
    cands = [pos_h]
    for _ in range(n - 1):
        cands.append(apply_corruption(pos_h, corruption, gen))
    return torch.stack(cands, dim=0)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["gpt2", "light"], required=True)
    ap.add_argument("--scorer", default="swift", help="light family only: swift|hsrm")
    ap.add_argument("--prm_id", required=True)
    ap.add_argument("--data_dir", default="latent-data/coconut/valid-64")
    ap.add_argument("--n", type=int, default=4, help="pool size (1 intact + N-1 corrupted)")
    ap.add_argument("--max_pos", type=int, default=1500, help="0 = all correct trajectories")
    ap.add_argument("--chunk", type=int, default=256, help="rows per forward")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_npz_stem = (args.out_json[:-5] if args.out_json.endswith(".json") else args.out_json) or "order_decision"
    feats = gather_positives(args.data_dir, args.max_pos)
    print(f"[data] n_pos={len(feats)} n={args.n}")
    if not feats:
        raise SystemExit("no correct trajectories")

    # ---- model wrapper: (list_of_latent (B,T,D), input_ids (B,S)|None) -> scores (B,) ----
    if args.family == "light":
        ckpt = torch.load(args.prm_id, map_location="cpu", weights_only=False)
        model = build_lightweight_scorer(
            args.scorer, hidden_size=ckpt.get("hidden_size", 768), model_dim=ckpt.get("model_dim", 256)
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()

        def score(embeds: torch.Tensor, input_ids, labels, attn) -> torch.Tensor:
            return model(latent_embeds=embeds)

    else:
        from transformers import AutoTokenizer
        from src.models.gpt2 import COCONUTGPT2ForTokenClassification

        tok = AutoTokenizer.from_pretrained(args.prm_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        latent_id = tok.convert_tokens_to_ids("<|latent|>")
        model = COCONUTGPT2ForTokenClassification.from_pretrained(
            args.prm_id,
            latent_id=latent_id,
            latent_start_id=tok.convert_tokens_to_ids("<|start-latent|>"),
            latent_end_id=tok.convert_tokens_to_ids("<|end-latent|>"),
        ).to(device).eval()
        collator = DataCollatorForLatentRM(tok, latent_token_id=latent_id, remove_pad_token=True)
        pad_id = tok.pad_token_id
        # pre-collate one row per positive to reuse input_ids/labels/attn across candidates
        rows = []
        for f in feats:
            b = collator([{k: v for k, v in f.items()}])
            rows.append((b["input_ids"][0], b["labels"][0], b.get("attention_mask")))

        def _pad_stack(idxs):
            """Right-pad input_ids (pad_id), labels (-100), attn (0) to the chunk max length."""
            maxlen = max(int(rows[j][0].numel()) for j in idxs)
            ii = torch.full((len(idxs), maxlen), pad_id, dtype=torch.long)
            lb = torch.full((len(idxs), maxlen), -100, dtype=torch.float32)
            at = torch.zeros((len(idxs), maxlen), dtype=torch.long)
            for r, j in enumerate(idxs):
                ids_j, lab_j, att_j = rows[j]
                n = int(ids_j.numel())
                ii[r, :n] = ids_j
                lb[r, :n] = lab_j
                at[r, :n] = att_j if att_j is not None else 1
            return ii, lb, at

        def score(embeds: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor, attn) -> torch.Tensor:
            out = model(input_ids=input_ids, attention_mask=attn, latent_embeds=list(embeds))
            return sequence_score_from_logits(out.logits, labels, reduce="mean_log_prob")

    # ---- run: per corruption type (plus a mixed type) ----
    results: Dict[str, dict] = {}
    rng = torch.Generator().manual_seed(args.seed)
    types = CORRUPTIONS + ["mixed"]
    for ti, ctype in enumerate(types):
        # deterministic per-type seed so every model sees IDENTICAL pools (paired stats)
        gen = torch.Generator().manual_seed(args.seed + 1000 * (ti + 1))
        pair_acc_sum = 0.0  # mean over (intact, corrupt_j) of (s_i > s_j) + 0.5*ties
        strict_win = 0      # intact is strictly greater than every corrupted candidate
        tie_frac = 0.0      # fraction of pools where all candidates score exactly equal
        std_sum = 0.0       # mean within-pool score std (0 => order-blind aggregation)
        total = 0
        per_pool_pair = []  # per-pool mean pair acc (for paired cross-model stats)
        per_pool_strict = []  # per-pool 0/1 strict win
        for start in tqdm(range(0, len(feats), args.chunk), desc=f"pools[{ctype}]"):
            chunk = feats[start : start + args.chunk]
            cands = []
            for f in chunk:
                if ctype == "mixed":
                    h = f["latent_embeds"].float()
                    row = [h]
                    for j in range(args.n - 1):
                        row.append(apply_corruption(h, CORRUPTIONS[j % len(CORRUPTIONS)], gen))
                    cands.append(torch.stack(row, dim=0))
                else:
                    cands.append(build_pool(f["latent_embeds"].float(), args.n, ctype, gen))
            cand = torch.stack(cands, dim=0)  # (B, N, T, D)
            B = cand.shape[0]
            flat = cand.reshape(B * args.n, *cand.shape[2:]).to(device)
            if args.family == "gpt2":
                ii, lb, at = _pad_stack([start + i for i in range(B)])
                ii = ii.repeat_interleave(args.n, dim=0).to(device)
                lb = lb.repeat_interleave(args.n, dim=0).to(device)
                at = at.repeat_interleave(args.n, dim=0).to(device)
            else:
                ii = lb = at = None
            s = score(flat, ii, lb, at).reshape(B, args.n)
            sv = s[:, :1]                       # intact (B,1)
            sc = s[:, 1:]                       # corrupted (B, N-1)
            pp = ((sv > sc).float() + 0.5 * (sv == sc).float()).mean(dim=1)  # (B,)
            pw = (sv > sc.max(dim=1, keepdim=True).values).all(dim=1).float()  # (B,)
            pair_acc_sum += float(pp.sum().item())
            strict_win += int(pw.sum().item())
            per_pool_pair.append(pp.cpu())
            per_pool_strict.append(pw.cpu())
            tie_frac += float((s.std(dim=1) == 0).float().sum().item())
            std_sum += float(s.std(dim=1).mean().item())
            total += B
        results[ctype] = {
            "pair_acc": pair_acc_sum / total,          # 0.5 = order-blind
            "strict_win": strict_win / total,
            "tie_frac": tie_frac / total,
            "mean_score_std": std_sum / total,
            "chance_pair_acc": 0.5,
            "n_pools": total,
        }
        per_pool = {
            "pair": torch.cat(per_pool_pair).numpy(),
            "strict": torch.cat(per_pool_strict).numpy(),
        }
        np.savez(f"{out_npz_stem}__{ctype}.npz", **per_pool)
        print(f"[{ctype}] pair_acc={results[ctype]['pair_acc']:.4f} "
              f"strict_win={results[ctype]['strict_win']*100:.2f}% "
              f"tie_frac={results[ctype]['tie_frac']*100:.1f}% "
              f"score_std={results[ctype]['mean_score_std']:.3e}")

    blob = {
        "family": args.family,
        "scorer": args.scorer if args.family == "light" else "gpt2",
        "prm_id": args.prm_id,
        "data_dir": args.data_dir,
        "n_pool": args.n,
        "n_pos": len(feats),
        "measure": "pair_acc=P(s_intact>s_corrupt)+0.5*ties; 0.5=order-blind",
        "results": results,
    }
    print(json.dumps(blob, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
