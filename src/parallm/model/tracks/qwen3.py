"""Per-track Qwen3 (dense) text model — the shared body lives in `tracks.base`.

This family divides `intermediate_size` on top of the common attention sizing, and
overrides `_resolve_position_ids`: the base implementation expands to the 4-way
(text/temporal/height/width) form Qwen3.5's mrope wants, while `Qwen3RotaryEmbedding`
takes plain 2-D position ids.

SDPA (the base default) is correct here — `Qwen3PreTrainedModel._supports_sdpa` is
True and there is nothing like gpt-oss's attention sinks for it to drop.
"""
from __future__ import annotations

import torch

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from parallm.model.tracks.base import PTTrackTextModelBase, apply_common_per_track_sizing
from parallm.slicer.base import align_chunk


class PTTrackTextModel(PTTrackTextModelBase):
    DECODER_LAYER_CLS = Qwen3DecoderLayer
    RMSNORM_CLS = Qwen3RMSNorm
    ROTARY_CLS = Qwen3RotaryEmbedding

    def _resolve_position_ids(self, inputs_embeds, position_ids):
        """Plain 2-D position ids, used by both the rotary and the masks."""
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)
        return position_ids, position_ids


def build_per_track_text_config(text_config, n_tracks: int, fuse_size: int = 1):
    cfg = apply_common_per_track_sizing(text_config, n_tracks, fuse_size)
    # Ceil, then round to a GEMM-friendly width: a width that doesn't divide is
    # zero-padded by the slicer's `Colwise(pad_full_size=...)` (exact for SwiGLU),
    # and `align_chunk` makes that padding pay for itself. MUST match the slicer's
    # `_even_chunk` or the shards won't load — including under `fuse_size`, where
    # the merged width is F *aligned* per-track slabs, not one aligned F-wide slab.
    inter = int(cfg.intermediate_size)
    cfg.intermediate_size = align_chunk(
        -(-inter // n_tracks), full_size=inter, n_tracks=n_tracks
    ) * fuse_size
    return cfg
