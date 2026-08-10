"""Model-adapter registry.

A `ModelAdapter` packages everything model-specific that the PT engine needs:
slicer specs, per-track text-model class, per-track config builder, and the
per-layer-type attention masks. The engine (`slicer/convert.py` and
`model/pt_model.py`) holds no model knowledge and looks up the right adapter
via `get_adapter_for_config(config)`.

To add a new model, write a new module under `src/parallm/adapters/`
that builds a `ModelAdapter` and calls `register_model_adapter(adapter)`.
Add a one-line import to this file so the registration runs at package
import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from parallm.slicer.base import LayerSpec
from parallm.utils.max_tracks import ConstraintSet


@dataclass(frozen=True)
class AttnOps:
    """The two full-attention facts the BATCHED fold cannot read off a module.

    `engine._batched_attn` reimplements a family's attention with the track dim
    folded into the batch, so it cannot call the family's own modules and has to be
    told where they differ. Everything else it needs (head_dim, scaling, kv groups,
    eps) it reads off the per-stream skeleton layer.

    Defaults are the PLAIN forms — a q_proj that is just `num_heads * head_dim`, and
    an RMSNorm whose weight multiplies directly. Qwen3.5 (and its MoE sibling) set
    both flags; see `parallm.model.batched`.
    """

    #: `q_proj` carries `[q | gate]` doubled per head, and the attention output is
    #: multiplied by `sigmoid(gate)` (`GatedQColwise` on the slicer side).
    gated_q: bool = False
    #: The per-head RMSNorm weight is zero-centered, i.e. applied as `(1 + w)`.
    centered_norm: bool = False


@dataclass(frozen=True)
class ModelAdapter:
    """All callbacks the PT engine needs to convert and run one model family.

    All callbacks operate on a *text config* (the sub-config containing
    `num_hidden_layers`, `num_attention_heads`, etc.). The registry resolves
    the right adapter from a top-level config by checking `config.model_type`
    first and falling back to `config.text_config.model_type`.
    """

    model_type: str
    # ----- Slicer side -----
    layer_specs: Callable[[Any, str], LayerSpec]
    top_level_specs: Callable[[Any], LayerSpec]
    valid_layer_types: tuple[str, ...]
    # ----- Per-track model side -----
    track_text_model_cls: type
    build_per_track_text_config: Callable[[Any, int], Any]
    # ----- Attention masks -----
    # `(per_track_cfg, inputs_embeds, attention_mask, position_ids) -> {layer_type: mask}`.
    # This is the `causal_mask_mapping` every HF text model builds internally; hoisting
    # it onto the adapter is what keeps the engine's forward from importing one family's
    # modeling module and branching on that family's layer-type names.
    build_masks: Callable[..., dict[str, Any]]
    # ----- Layer types -----
    # `(text_cfg) -> [layer_type per layer]`. Defaults to the config's own `layer_types`
    # (every hybrid family ships one), falling back to all-`full_attention` for a
    # homogeneous stack — so most families need not supply it.
    get_layer_types: "Callable[[Any], list[str]] | None" = None
    # ----- Full (unsliced) text model -----
    # The HF text-model class used to instantiate the *dense* model for replica
    # calibration (its state_dict keys match the slicer's canonical `layers.*`
    # names). Optional so lightweight/test adapters can omit it.
    full_text_model_cls: type | None = None
    # ----- Track-count constraints -----
    # Divisibility rules for the KV-replicated max-tracks scan (see
    # `parallm.utils.max_tracks`). Takes a *text config*, returns a `ConstraintSet`.
    # Optional so lightweight/test adapters can omit it.
    constraints: "Callable[[Any], ConstraintSet] | None" = None
    # ----- Capabilities -----
    # Whether a rank's tracks may be MERGED into one wide track (`parallm.model.merge`).
    # False for families whose per-track slabs have no verified concatenation rule — the
    # trainer then keeps the looped path instead of failing deep inside
    # `build_per_track_text_config`.
    supports_merged_tracks: bool = True
    # Whether a MERGED track may additionally be RUN as G independent streams
    # (`exec_groups`, `parallm.model.batched`). This is what decouples step time from
    # the fusion width F on the dense path; without it, F is expressed as the merge
    # width itself, so a rank holds K/F merged tracks and loops over them.
    # False for MoE families: `engine._batched_mlp` is a dense SwiGLU, and under G > 1
    # each stream carries its own residual, so the shared router picks a different
    # top-k per stream and there is no single `grouped_mm` to issue.
    supports_batched_exec: bool = True
    # Where this family's attention differs from the plain form the batched fold
    # assumes. Only read when `supports_batched_exec` is True.
    attn_ops: AttnOps = AttnOps()

    def layer_types_for(self, text_cfg) -> list[str]:
        """The per-layer type list, via `get_layer_types` or the config default."""
        if self.get_layer_types is not None:
            return list(self.get_layer_types(text_cfg))
        types = getattr(text_cfg, "layer_types", None)
        if types is not None:
            return list(types)
        return ["full_attention"] * int(text_cfg.num_hidden_layers)


_REGISTRY: dict[str, ModelAdapter] = {}


def register_model_adapter(adapter: ModelAdapter) -> ModelAdapter:
    """Idempotent registration: re-registering the same model_type overwrites
    the previous adapter (useful for tests and for swapping experimental
    adapters in a notebook)."""
    _REGISTRY[adapter.model_type] = adapter
    return adapter


def get_adapter_for_config(config) -> ModelAdapter:
    """Resolve an adapter by config.model_type, falling back to text_config.model_type."""
    model_type = getattr(config, "model_type", None)
    if model_type in _REGISTRY:
        return _REGISTRY[model_type]
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        text_model_type = getattr(text_cfg, "model_type", None)
        if text_model_type in _REGISTRY:
            return _REGISTRY[text_model_type]
    registered = sorted(_REGISTRY.keys())
    raise KeyError(
        f"No ModelAdapter registered for model_type={model_type!r} "
        f"(text_config.model_type={getattr(text_cfg, 'model_type', None)!r}). "
        f"Registered adapters: {registered}"
    )


def list_registered() -> list[str]:
    """Inspect the registry; useful for CLI help and error messages."""
    return sorted(_REGISTRY.keys())


# Import-time adapter registrations. Add one line per shipped adapter.
from parallm.adapters import gpt_oss as _gpt_oss  # noqa: E402,F401
from parallm.adapters import qwen3 as _qwen3  # noqa: E402,F401
from parallm.adapters import qwen3_5 as _qwen3_5  # noqa: E402,F401
from parallm.adapters import qwen3_5_moe as _qwen3_5_moe  # noqa: E402,F401
