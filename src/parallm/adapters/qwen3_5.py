"""Qwen3.5 adapter: glues the existing qwen3_5 slicer specs and per-track text model
into the model-agnostic `ModelAdapter` shape and registers it.

This is the single source of truth for "is Qwen3.5 PT-supported." Nothing outside
this file should import from `parallm.slicer.qwen3_5` or
`parallm.model.tracks.qwen3_5`.
"""
from __future__ import annotations

from typing import Any

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.adapters import AttnOps, ModelAdapter, register_model_adapter
from parallm.model.tracks.qwen3_5 import (
    PTTrackTextModel,
    build_per_track_text_config,
)
from parallm.slicer.qwen3_5 import (
    VALID_LAYER_TYPES,
    build_masks,
    decoder_layer_specs,
    top_level_specs,
)
from parallm.utils.max_tracks import ConstraintSet


def _qwen3_5_constraints(cfg: Any) -> ConstraintSet:
    # `linear_num_key_heads` and `intermediate_size` are deliberately absent:
    # `GDNFusedQKV` replicates k-heads and `Colwise(pad_full_size=...)` zero-pads
    # the MLP, both exactly, so neither has to divide N. Only the GDN value heads
    # (the parallel unit) do. This is what lets the 27B reach its N=24 ceiling.
    return ConstraintSet(
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        divides=(int(cfg.linear_num_value_heads),),
    )


QWEN3_5_ADAPTER = ModelAdapter(
    model_type="qwen3_5_text",
    layer_specs=decoder_layer_specs,
    top_level_specs=top_level_specs,
    valid_layer_types=VALID_LAYER_TYPES,
    track_text_model_cls=PTTrackTextModel,
    build_per_track_text_config=build_per_track_text_config,
    build_masks=build_masks,
    full_text_model_cls=Qwen3_5TextModel,
    constraints=_qwen3_5_constraints,
    # `q_proj` carries [q | gate] (`GatedQColwise`) and the RMSNorm weight is
    # zero-centered — the two places `engine._batched_attn` has to branch.
    attn_ops=AttnOps(gated_q=True, centered_norm=True),
)

register_model_adapter(QWEN3_5_ADAPTER)
