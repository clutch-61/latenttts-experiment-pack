"""Order discrimination / hierarchy consistency for lightweight latent scorers.

Same mechanism diagnostic as ``src/eval_order_pref.py`` (which covers
COCONUTGPT2ForTokenClassification RMs: O2/B0/B1), but for the tiny
SWIFT-Latent / HSRM-Latent scorers that consume latent hidden states directly.

For each *correct* trajectory we build the same permutation-corrupted negatives
(``build_negative_bundle``) and report:
  - ``order_pair_acc``: fraction of (pos, neg_k) where s_pos > s_neg
  - ``hierarchy_consistency``: among negatives, milder rank scores higher
  - per-corruption-type win rate

This tests whether an outcome-only scorer (SWIFT/HSRM) has any latent order
sensitivity at all, i.e. whether O2 (order-preference trained) holds an
intrinsic advantage on this axis.

Example:
    python scripts/eval_order_lightweight.py \
        --scorer swift --prm_id outputs/swift_latent/best.pt \
        --data_dir latent-data/coconut/valid-64 --batch_size 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import CachedPickleDatasetV2  # noqa: E402
from src.lightweight_scorers import build_lightweight_scorer  # noqa: E402
from src.order_pref.corruption import (  # noqa: E402
    CORRUPTION_SPECS,
    DEFAULT_PHASE1_CORRUPTIONS,
    build_negative_bundle,
)
from src.order_pref.metrics import (  # noqa: E402
    hierarchy_consistency,
    order_discrimination_accuracy,
)


@torch.no_grad()
def main(
    scorer: str = "swift",
    prm_id: str = "outputs/swift_latent/best.pt",
    data_dir: str = "latent-data/coconut/valid-64",
    batch_size: int = 128,
    max_examples: int = 0,
    device: str = "cuda",
):
    """Evaluate order metrics for a lightweight scorer.

    Args:
        max_examples: If <=0 scan the full dataset; else cap the number of
            positive (correct) trajectories scored.
    """
    ckpt = torch.load(prm_id, map_location="cpu", weights_only=False)
    hidden_size = ckpt.get("hidden_size", 768)
    model_dim = ckpt.get("model_dim", 256)
    model = build_lightweight_scorer(scorer, hidden_size=hidden_size, model_dim=model_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    ds = CachedPickleDatasetV2(data_dir=data_dir, include_gt=True, get_single_sample=True)
    names = list(DEFAULT_PHASE1_CORRUPTIONS)
    ranks = [CORRUPTION_SPECS[n].rank for n in names]

    # Gather correct trajectories only (decision-locked analysis set, matches
    # eval_order_pref.py which keeps corrects=1 rows).
    pos_list: List[torch.Tensor] = []
    n = len(ds)
    for i in tqdm(range(n), desc="gather-correct"):
        feat = ds[i]
        if "corrects" not in feat:
            continue
        if float(feat["corrects"].reshape(-1)[0]) > 0.5:
            pos_list.append(feat["latent_embeds"].float())
        if max_examples and len(pos_list) >= max_examples:
            break

    print(
        {
            "scorer": scorer,
            "prm_id": prm_id,
            "data_dir": data_dir,
            "dataset_len_traj": n,
            "n_pos_correct": len(pos_list),
        }
    )
    if not pos_list:
        print("no correct trajectories found; abort")
        return

    pos = torch.stack(pos_list, dim=0).to(device)  # (P, T, D)
    P = pos.shape[0]

    # Build negatives for every positive trajectory.
    neg_batches = []  # each (P, T, D)
    for t in pos_list:
        negs, _, _ = build_negative_bundle(t)
        neg_batches.append(torch.stack(negs, dim=0))  # (K, T, D)
    negs = torch.stack(neg_batches, dim=0).to(device)  # (P, K, T, D)
    K = negs.shape[1]

    pair_accs: List[float] = []
    hier_accs: List[float] = []
    per_rank = {nm: [] for nm in names}

    for start in tqdm(range(0, P, batch_size), desc="score"):
        end = min(start + batch_size, P)
        pos_b = pos[start:end]  # (B, T, D)
        neg_b = negs[start:end]  # (B, K, T, D)
        B = pos_b.shape[0]

        s_pos = model(latent_embeds=pos_b)  # (B,)
        flat_neg = neg_b.reshape(B * K, *neg_b.shape[2:])  # (B*K, T, D)
        s_neg = model(latent_embeds=flat_neg).reshape(B, K)  # (B, K)

        disc = order_discrimination_accuracy(s_pos, s_neg)
        hier = hierarchy_consistency(s_neg, ranks)
        pair_accs.append(disc["order_pair_acc"])
        hier_accs.append(hier["hierarchy_consistency"])
        for j, nm in enumerate(names):
            per_rank[nm].append(float((s_pos > s_neg[:, j]).float().mean().item()))

    def avg(xs):
        return sum(xs) / max(1, len(xs))

    print({"n_scored": P, "n_metric_batches": len(pair_accs)})
    print({"order_pair_acc": avg(pair_accs), "hierarchy_consistency": avg(hier_accs)})
    print({k: avg(v) for k, v in per_rank.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", type=str, default="swift", choices=["swift", "hsrm"])
    ap.add_argument("--prm_id", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default="latent-data/coconut/valid-64")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    a = ap.parse_args()
    main(
        scorer=a.scorer,
        prm_id=a.prm_id,
        data_dir=a.data_dir,
        batch_size=a.batch_size,
        max_examples=a.max_examples,
        device=a.device,
    )
