"""Calibrate order/hierarchy margins from observed pos-neg score gaps."""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from .corruption import apply_corruption
from .preference_loss import sequence_score_from_logits


@torch.no_grad()
def estimate_order_gap_scale(
    model,
    latent_embeds: Sequence[torch.Tensor],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    corruption: str = "adjacent_swap",
    score_reduce: str = "mean",
    alpha: float = 0.5,
    beta: float = 0.25,
) -> Dict[str, float]:
    """Forward LatentRM on H+ and corrupted H-, report gap stats + suggested margins.

    This is a scale check only; it does not prove learnability.
    """
    device = next(model.parameters()).device
    gaps = []
    for h in latent_embeds:
        h = h.to(device)
        h_neg = apply_corruption(h, corruption)
        # one-item batch reuse of same ids/labels
        out_pos = model(input_ids=input_ids[:1], latent_embeds=[h])
        out_neg = model(input_ids=input_ids[:1], latent_embeds=[h_neg])
        s_pos = sequence_score_from_logits(out_pos.logits, labels[:1], reduce=score_reduce)
        s_neg = sequence_score_from_logits(out_neg.logits, labels[:1], reduce=score_reduce)
        gaps.append((s_pos - s_neg).item())

    if not gaps:
        raise ValueError("empty latent_embeds")
    t = torch.tensor(gaps, dtype=torch.float32)
    median = float(t.median().item())
    q10 = float(t.quantile(0.1).item())
    q90 = float(t.quantile(0.9).item())
    return {
        "n": float(len(gaps)),
        "median_gap": median,
        "q10_gap": q10,
        "q90_gap": q90,
        "recommended_order_margin": max(1e-4, abs(alpha * median)),
        "recommended_hierarchy_margin": max(1e-4, abs(beta * alpha * median)),
        "corruption": corruption,
    }
