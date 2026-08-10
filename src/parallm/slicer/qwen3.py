"""Per-parameter SlicerSpecs for Qwen3 (dense: Qwen3-8B/14B/32B, ...).

Source of truth for how every parameter inside a `Qwen3DecoderLayer` is
partitioned across N tracks. Verified against the installed
`transformers/models/qwen3/modeling_qwen3.py` (5.14.1).

The simplest family so far — textbook pre-norm GQA self-attention plus a dense
SwiGLU MLP, with no biases, no sinks, no MoE and no linear-attention layers. Two
things worth naming:

1. **`q_proj` is a plain colwise split.** Unlike Qwen3.5, whose `q_proj` carries
   `[q | gate]` doubled per head (`GatedQColwise`), Qwen3's out_features are just
   `num_heads * head_dim` head-major, so a `Colwise()` hands each track whole heads.

2. **`q_norm` / `k_norm` are per-head RMSNorms on `head_dim`**, so they are
   `Replicated` — and `sync=False` by default, the same divergence Qwen3.5 and
   gpt-oss use (each track gets its own KV head instead of a bit-identical copy of
   the kv-group's, turning GQA into per-track MHA). `--sync-attention-heads`
   (force_sync) restores the bit-identical behaviour.
"""
from __future__ import annotations

from typing import Any

from parallm.slicer.base import (
    Colwise,
    KVReplicatedColwise,
    LayerSpec,
    Replicated,
    Rowwise,
    build_decoder_layer_specs,
    standard_top_level_specs,
)


def attention_specs(text_cfg: Any) -> LayerSpec:
    """`Qwen3Attention` slicing. Identical for full and sliding layers — they
    differ only in the mask they are handed (see `build_masks`)."""
    if getattr(text_cfg, "attention_bias", False):
        # No released Qwen3 sets this. Refuse rather than silently dropping the
        # biases: q/k/v would take their weight's spec, but `o_proj.bias` sits on a
        # row-parallel output and needs `SummedBias` or it is added N times at the
        # sync (see `slicer.gpt_oss.attention_specs`).
        raise ValueError(
            "qwen3: attention_bias=True is not sliced yet — add the q/k/v biases "
            "alongside their weights and SummedBias() for o_proj.bias"
        )
    num_kv = int(text_cfg.num_key_value_heads)
    return {
        # out_features = num_heads * head_dim, head-major: a plain colwise split
        # hands each track whole heads.
        "q_proj.weight": Colwise(),
        # rows are per-kv-head; every track in a kv-group starts from the same copy.
        "k_proj.weight": KVReplicatedColwise(num_kv_heads=num_kv, sync=False),
        "v_proj.weight": KVReplicatedColwise(num_kv_heads=num_kv, sync=False),
        "o_proj.weight": Rowwise(),  # cols = num_heads * head_dim -> partial sum
        "q_norm.weight": Replicated(sync=False),  # RMSNorm on head_dim, per-track
        "k_norm.weight": Replicated(sync=False),
    }


def mlp_specs(text_cfg: Any) -> LayerSpec:
    # `pad_full_size` lets intermediate_size not divide n_tracks. Zero-padding a
    # SwiGLU lane is exact: silu(0)*up = 0.
    inter = int(text_cfg.intermediate_size)
    return {
        "gate_proj.weight": Colwise(pad_full_size=inter),
        "up_proj.weight": Colwise(pad_full_size=inter),
        "down_proj.weight": Rowwise(pad_full_size=inter),
    }


# Both layer types are ordinary self-attention under the same `self_attn` module;
# `sliding_attention` (emitted by Qwen3Config when `use_sliding_window=True`)
# differs only by mask. `valid_layer_types` derives from the keys.
ATTENTION_SPECS = {
    "full_attention": ("self_attn", attention_specs),
    "sliding_attention": ("self_attn", attention_specs),
}
VALID_LAYER_TYPES = tuple(ATTENTION_SPECS)


def decoder_layer_specs(text_cfg: Any, layer_type: str) -> LayerSpec:
    """All sliceable params under one Qwen3 decoder layer (with prefixes)."""
    return build_decoder_layer_specs(
        text_cfg,
        layer_type,
        attention_specs=ATTENTION_SPECS,
        mlp_specs=mlp_specs,
    )


def build_masks(per_track_cfg, inputs_embeds, attention_mask, position_ids) -> dict:
    """``{layer_type: mask}`` — mirrors ``Qwen3Model.forward``'s mask mapping.

    The sliding entry is built only when the stack actually HAS sliding layers
    (HF's own ``has_sliding_layers`` guard): with `use_sliding_window=False`
    `Qwen3Config.__post_init__` sets `sliding_window = None`, and asking for a
    sliding-window mask then has nothing to window.
    """
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    kwargs = dict(
        config=per_track_cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    masks = {"full_attention": create_causal_mask(**kwargs)}
    if "sliding_attention" in (getattr(per_track_cfg, "layer_types", None) or ()):
        masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
    return masks


# Embeddings / final norm / lm_head slice identically for every family so far.
top_level_specs = standard_top_level_specs
