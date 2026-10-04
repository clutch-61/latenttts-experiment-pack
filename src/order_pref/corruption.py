"""Temporal order corruptions on continuous latent trajectories H in R^{T x d}.

Unlike visual RoT, LatentTTS has no rendered-frame re-encode path. The honest
intervention is to permute latent thoughts along time before LatentRM scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    rank: int  # larger = more severe logical damage
    fn: Callable[[torch.Tensor, torch.Generator | None], torch.Tensor]


def _adjacent_swap(h: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    t = h.shape[0]
    if t < 2:
        return h.clone()
    if gen is None:
        i = int(torch.randint(0, t - 1, (1,)).item())
    else:
        i = int(torch.randint(0, t - 1, (1,), generator=gen).item())
    out = h.clone()
    out[i], out[i + 1] = out[i + 1].clone(), out[i].clone()
    return out


def _segment_swap(h: torch.Tensor, gen: torch.Generator | None = None, seg_len: int = 2) -> torch.Tensor:
    t = h.shape[0]
    if t < seg_len * 2:
        return _adjacent_swap(h, gen)
    max_a = t - 2 * seg_len
    if gen is None:
        a = int(torch.randint(0, max_a + 1, (1,)).item())
    else:
        a = int(torch.randint(0, max_a + 1, (1,), generator=gen).item())
    b = a + seg_len
    # second segment starts after first; pick a non-overlapping start
    candidates = list(range(0, a)) + list(range(b, t - seg_len + 1))
    if not candidates:
        return _adjacent_swap(h, gen)
    if gen is None:
        c = candidates[int(torch.randint(0, len(candidates), (1,)).item())]
    else:
        c = candidates[int(torch.randint(0, len(candidates), (1,), generator=gen).item())]
    out = h.clone()
    sa, sb = out[a : a + seg_len].clone(), out[c : c + seg_len].clone()
    out[a : a + seg_len] = sb
    out[c : c + seg_len] = sa
    return out


def _multi_segment_shuffle(h: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    t = h.shape[0]
    if t < 3:
        return _adjacent_swap(h, gen)
    # split into ~3 chunks and shuffle chunk order
    n_parts = 3 if t >= 3 else t
    cuts = torch.linspace(0, t, n_parts + 1).long()
    parts = [h[cuts[i] : cuts[i + 1]] for i in range(n_parts)]
    order = torch.randperm(n_parts, generator=gen)
    return torch.cat([parts[i] for i in order.tolist()], dim=0)


def _full_reverse(h: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    del gen
    return torch.flip(h, dims=[0]).contiguous()


def _reverse_plus_adjacent(h: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    return _adjacent_swap(_full_reverse(h, gen), gen)


CORRUPTION_SPECS: Dict[str, CorruptionSpec] = {
    "adjacent_swap": CorruptionSpec("adjacent_swap", 1, _adjacent_swap),
    "segment_swap": CorruptionSpec("segment_swap", 2, _segment_swap),
    "multi_segment": CorruptionSpec("multi_segment", 3, _multi_segment_shuffle),
    "full_reverse": CorruptionSpec("full_reverse", 4, _full_reverse),
    "composite": CorruptionSpec("composite", 5, _reverse_plus_adjacent),
}


def apply_corruption(
    h: torch.Tensor,
    name: str,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    if name not in CORRUPTION_SPECS:
        raise KeyError(f"unknown corruption {name!r}; choose from {list(CORRUPTION_SPECS)}")
    if h.dim() != 2:
        raise ValueError(f"expected H shape (T, d), got {tuple(h.shape)}")
    return CORRUPTION_SPECS[name].fn(h, gen)


DEFAULT_PHASE1_CORRUPTIONS = (
    "adjacent_swap",
    "segment_swap",
    "multi_segment",
    "full_reverse",
)


def build_negative_bundle(
    h_pos: torch.Tensor,
    names: Sequence[str] | None = None,
    gen: torch.Generator | None = None,
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """Return (negatives, negative_ranks, names) for hierarchical preference.

    Default names omit `composite` (decision-locked Phase-1).
    """
    names = list(names) if names is not None else list(DEFAULT_PHASE1_CORRUPTIONS)
    negs: List[torch.Tensor] = []
    ranks: List[int] = []
    used: List[str] = []
    for name in names:
        spec = CORRUPTION_SPECS[name]
        negs.append(spec.fn(h_pos, gen))
        ranks.append(spec.rank)
        used.append(name)
    if len(negs) > 1 and len(set(ranks)) <= 1:
        raise ValueError(
            "Multi-negative preference training requires distinct negative_ranks; "
            "flat-triplet fallback is disabled."
        )
    return negs, ranks, used
