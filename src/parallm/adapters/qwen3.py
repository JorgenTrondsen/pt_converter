"""Qwen3 adapter: registers the dense Qwen3 slicer specs + per-track model.

Covers Qwen/Qwen3-8B / -14B / -32B and anything else shipping `model_type: qwen3` —
they differ only in layer/head counts and widths, so one adapter serves all. This is
the single source of truth for "is Qwen3 PT-supported."

Note Qwen3 ships a FLAT config (no `text_config`) and its dense model class is
`Qwen3Model`, not `...TextModel`.
"""
from __future__ import annotations

from typing import Any

from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from parallm.adapters import AttnOps, ModelAdapter, register_model_adapter
from parallm.model.tracks.qwen3 import (
    PTTrackTextModel,
    build_per_track_text_config,
)
from parallm.slicer.qwen3 import (
    VALID_LAYER_TYPES,
    build_masks,
    decoder_layer_specs,
    top_level_specs,
)
from parallm.utils.max_tracks import ConstraintSet


def _qwen3_constraints(cfg: Any) -> ConstraintSet:
    # `intermediate_size` is deliberately absent from `divides`:
    # `Colwise(pad_full_size=...)` zero-pads the MLP exactly (silu(0)*up = 0), so it
    # imposes no constraint. For the 32B (64 q heads, 8 kv) this yields N in
    # {64, 32, 16, 8} whatever the MLP width.
    return ConstraintSet(
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        divides=(),
    )


QWEN3_ADAPTER = ModelAdapter(
    model_type="qwen3",
    layer_specs=decoder_layer_specs,
    top_level_specs=top_level_specs,
    valid_layer_types=VALID_LAYER_TYPES,
    track_text_model_cls=PTTrackTextModel,
    build_per_track_text_config=build_per_track_text_config,
    build_masks=build_masks,
    full_text_model_cls=Qwen3Model,
    constraints=_qwen3_constraints,
    # Dense SwiGLU MLP + ordinary self-attention: both the concatenated merge and
    # the batched fold apply, so F is expressed as `exec_groups` over one merged
    # track — merged step time at any fusion width, including F=1.
    supports_merged_tracks=True,
    supports_batched_exec=True,
    # Plain q_proj (no gate) and a plain RMSNorm weight — the two places Qwen3
    # differs from Qwen3.5 inside `engine._batched_attn`.
    attn_ops=AttnOps(gated_q=False, centered_norm=False),
)

register_model_adapter(QWEN3_ADAPTER)
