"""Qwen3.5-MoE adapter: registers the MoE slicer specs + per-track MoE model.

The MoE model reuses the dense Qwen3.5 attention slicing verbatim (imported inside
`slicer.qwen3_5_moe`) and only swaps the dense MLP for the sparse MoE block. This
is the single source of truth for "is Qwen3.5-MoE PT-supported."
"""
from __future__ import annotations

from typing import Any

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

from parallm.adapters import AttnOps, ModelAdapter, register_model_adapter
from parallm.model.tracks.qwen3_5_moe import (
    PTTrackTextModel,
    build_per_track_text_config,
)
from parallm.slicer.qwen3_5_moe import (
    VALID_LAYER_TYPES,
    build_masks,
    moe_decoder_layer_specs,
    top_level_specs,
)
from parallm.utils.max_tracks import ConstraintSet


def _qwen3_5_moe_constraints(cfg: Any) -> ConstraintSet:
    # Same attention constraints as dense; MLP is MoE so N must divide the expert
    # width and the shared-expert width. num_experts is replicated (the router runs
    # in full on every track) so it imposes no divisibility constraint.
    return ConstraintSet(
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        divides=(
            int(cfg.linear_num_key_heads),
            int(cfg.linear_num_value_heads),
            int(cfg.moe_intermediate_size),
            int(cfg.shared_expert_intermediate_size),
        ),
    )


QWEN3_5_MOE_ADAPTER = ModelAdapter(
    model_type="qwen3_5_moe_text",
    layer_specs=moe_decoder_layer_specs,
    top_level_specs=top_level_specs,
    valid_layer_types=VALID_LAYER_TYPES,
    track_text_model_cls=PTTrackTextModel,
    build_per_track_text_config=build_per_track_text_config,
    build_masks=build_masks,
    full_text_model_cls=Qwen3_5MoeTextModel,
    constraints=_qwen3_5_moe_constraints,
    # Merging (concatenated slabs) IS supported — see `tracks/qwen3_5_moe.py` for why
    # widening experts beats stacking tracks on an expert-group-bound MLP.
    supports_merged_tracks=True,
    # The batched fold is not: `engine._batched_mlp` is a dense SwiGLU, and under
    # G > 1 each stream carries its own residual, so the shared router picks a
    # different top-k per stream and there is no single `grouped_mm` to issue.
    supports_batched_exec=False,
    # Inert while the fold is off, but stated rather than defaulted: this family
    # reuses the dense Qwen3.5 attention verbatim, so the plain default would be
    # silently wrong numbers if the flag above ever flipped.
    attn_ops=AttnOps(gated_q=True, centered_norm=True),
)

register_model_adapter(QWEN3_5_MOE_ADAPTER)
