"""Order-aware preference extras for LatentTTS LatentRM (Phase-1).

Adapted for continuous latent thoughts (COCONUT/CODI/CoLaR), not visual RoT frames.
"""

from .corruption import (
    CORRUPTION_SPECS,
    DEFAULT_PHASE1_CORRUPTIONS,
    apply_corruption,
    build_negative_bundle,
)
from .preference_loss import hierarchical_preference_loss, sequence_score_from_logits
from .gap_scale import estimate_order_gap_scale
from .set_align import bidirectional_chamfer, mmd_rbf_multiscale
from .content_view import content_view, reject_fake_content_view

__all__ = [
    "CORRUPTION_SPECS",
    "DEFAULT_PHASE1_CORRUPTIONS",
    "apply_corruption",
    "build_negative_bundle",
    "hierarchical_preference_loss",
    "sequence_score_from_logits",
    "estimate_order_gap_scale",
    "bidirectional_chamfer",
    "mmd_rbf_multiscale",
    "content_view",
    "reject_fake_content_view",
]
