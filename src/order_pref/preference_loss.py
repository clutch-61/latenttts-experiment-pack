"""Hierarchical pairwise preference loss (VideoComp-style), LatentRM sequence scores."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


def sequence_score_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduce: str = "mean_log_prob",
) -> torch.Tensor:
    """Aggregate LatentRM per-token logits into one score per sequence.

    Must stay consistent between train preference and eval_order_pref.
    LatentTTS BCE eval uses sum of log(sigmoid) over latent steps; with fixed T,
    mean vs sum only rescales margins by T.

    Args:
        logits: (B, S, 1) or (B, S)
        labels: (B, S) with -100 on ignored positions
        reduce:
            - mean_log_prob / sum_log_prob: match BCE probability scoring (preferred)
            - mean_logit / sum_logit: raw logits (legacy / ablation)
    """
    if logits.dim() == 3:
        logits = logits.squeeze(-1)
    mask = labels != -100
    denom = mask.float().sum(dim=-1).clamp(min=1.0)

    if reduce in ("mean_log_prob", "sum_log_prob"):
        log_p = F.logsigmoid(logits)
        # ignored positions -> 0 contribution
        log_p = log_p.masked_fill(~mask, 0.0)
        scored = log_p.sum(dim=-1)
        if reduce == "mean_log_prob":
            return scored / denom
        return scored

    if reduce in ("mean_logit", "sum_logit", "mean", "sum"):
        # mean/sum kept as aliases of *logit* for older call sites
        masked = logits.masked_fill(~mask, 0.0)
        scored = masked.sum(dim=-1)
        if reduce in ("mean_logit", "mean"):
            return scored / denom
        return scored

    raise ValueError(f"unknown reduce={reduce!r}")


def hierarchical_preference_loss(
    score_pos: torch.Tensor,
    score_negs: torch.Tensor,
    negative_ranks: Optional[Sequence[int]],
    order_margin: float = 0.1,
    hierarchy_margin: float = 0.05,
    lambda_hier: float = 1.0,
) -> torch.Tensor:
    """Preference hinge over one positive and K negatives.

    Same-rank negatives are never compared. Hierarchy only when r_i < r_j.
    """
    if score_pos.dim() != 1:
        raise ValueError(f"score_pos must be (B,), got {tuple(score_pos.shape)}")
    if score_negs.dim() != 2 or score_negs.shape[0] != score_pos.shape[0]:
        raise ValueError(f"score_negs must be (B, K) matching B, got {tuple(score_negs.shape)}")

    b, k = score_negs.shape
    del b
    if k == 0:
        return score_pos.new_zeros(())

    if k > 1:
        if negative_ranks is None:
            raise ValueError(
                "negative_ranks is required for hierarchical multi-negative training; "
                "flat-triplet fallback is disabled."
            )
        if len(negative_ranks) != k:
            raise ValueError("len(negative_ranks) must equal K")
        if len(set(int(r) for r in negative_ranks)) <= 1:
            raise ValueError("hierarchy requires at least two distinct negative_ranks")

    order = F.relu(order_margin + score_negs - score_pos.unsqueeze(1)).mean()

    hier = score_pos.new_zeros(())
    if k > 1 and lambda_hier != 0.0:
        ranks = [int(r) for r in negative_ranks]
        pair_losses: List[torch.Tensor] = []
        for i in range(k):
            for j in range(k):
                if ranks[i] < ranks[j]:
                    pair_losses.append(F.relu(hierarchy_margin + score_negs[:, j] - score_negs[:, i]))
                # ranks[i] == ranks[j]: intentionally no order constraint
        if pair_losses:
            hier = torch.stack(pair_losses, dim=0).mean()

    return order + lambda_hier * hier
