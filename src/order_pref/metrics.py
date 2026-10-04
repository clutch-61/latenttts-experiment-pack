"""Diagnostic metrics for order preference (not task accuracy)."""

from __future__ import annotations

from typing import Dict, Sequence

import torch


def order_discrimination_accuracy(
    score_pos: torch.Tensor,
    score_negs: torch.Tensor,
) -> Dict[str, float]:
    """Pairwise: fraction of (pos, neg_k) where s_pos > s_neg."""
    # score_pos (B,), score_negs (B, K)
    wins = (score_pos.unsqueeze(1) > score_negs).float()
    return {
        "order_pair_acc": float(wins.mean().item()),
        "order_pair_acc_per_rank_mean": float(wins.mean(dim=0).mean().item()),
    }


def hierarchy_consistency(
    score_negs: torch.Tensor,
    negative_ranks: Sequence[int],
) -> Dict[str, float]:
    """Among negatives, milder rank should score higher than more severe."""
    b, k = score_negs.shape
    if k < 2:
        return {"hierarchy_consistency": 1.0, "n_pairs": 0.0}
    ranks = list(negative_ranks)
    total = 0
    ok = 0
    for i in range(k):
        for j in range(k):
            if ranks[i] < ranks[j]:
                total += b
                ok += int((score_negs[:, i] > score_negs[:, j]).sum().item())
    return {
        "hierarchy_consistency": (ok / total) if total else 1.0,
        "n_pairs": float(total),
    }
