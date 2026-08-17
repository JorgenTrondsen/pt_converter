"""Per-track Qwen3.5 (dense) text model — the shared body lives in `tracks.base`.

This family divides `intermediate_size` on top of the common attention sizing.
"""
from __future__ import annotations

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

from parallm.model.tracks.base import PTTrackTextModelBase, apply_common_per_track_sizing
from parallm.slicer.base import align_chunk


class PTTrackTextModel(PTTrackTextModelBase):
    DECODER_LAYER_CLS = Qwen3_5DecoderLayer
    RMSNORM_CLS = Qwen3_5RMSNorm
    ROTARY_CLS = Qwen3_5TextRotaryEmbedding


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
