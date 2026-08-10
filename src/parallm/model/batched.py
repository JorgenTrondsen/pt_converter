"""Run a MERGED track as G independent members — fusion's step time without fusion.

`parallm.model.merge` concatenates F shards' slabs into one wide track. That is
what makes training fast (1/F the kernels, one full-width residual instead of F),
but it is also exactly what fusion DOES: the concatenated ``o_proj``/``down_proj``
reduce over the member-concatenated input dim, so the wide GEMM sums the members'
deltas. No flag can suppress that — the merge IS the fusion.

The way out is that members mix in only TWO places: those contracting projections,
and the shared input feeding the norms and the expanding projections. Everything
else is already member-independent — attention and GDN are per-head, ``conv1d`` is
depthwise, ``A_log``/``dt_bias`` are ``PerHead``, and ``q_norm``/``k_norm`` are
already ``[F, head_dim]`` (`PTWrappedModel._widen_diverged_norms`). So VIEW each
merged weight as ``[G, ...]`` and drive G residual streams through batched matmuls,
never summing G. Same storage, same checkpoint, same optimizer state, unfused
semantics.

G streams of width ``F = merge_group // G``: ``G == merge_group`` is fully unfused
(every shard its own stream, F=1), ``G == 1`` is the merged track itself, and an
intermediate G is `--fuse-tracks F` — F members pooling inside each stream, G
streams staying independent. Any G dividing ``merge_group`` works; the only place
width shows up is the DIVERGED norms (`q_norm`/`k_norm`), which are stored ``[K, d]``
and must be handed over as ``[G, F, d]``.

⚠ **Precondition for F > 1: exactly ONE attention head and one kv head per member.**
Those norms are per-head RMSNorm weights broadcast against ``(..., heads, d)``, so
``[G, F, d]`` only lines up when a group's head count IS F. This is not new — the
merged track's own forward has always relied on it — but it was implicit, and it is
checked here rather than silently producing a wrong norm. 27B at N=24 has 24 q heads
over 24 tracks, so it holds.

The fold itself is `engine._batched_{attn,gdn,mlp}`, which the decode path already
proves bit-exact; this module only re-sources their weights.
"""
from __future__ import annotations

import torch

from parallm.engine import (
    _batched_attn,
    _batched_gdn,
    _batched_mlp,
    _hf,
    _submodule,
)
from parallm.slicer.base import GDNFusedQKV, Replicated
from parallm.slicer.convert import resolve_param_specs


class MergedShadow:
    """Serves the batched fold from ONE merged track's parameters.

    Implements the provider interface `engine._batched_*` consumes — ``n_tracks``,
    ``grid``, ``blinear``, ``stacked`` — but where `DenseShadow` stacks N separate
    modules, this returns per-member VIEWS of the single merged parameter. Grads
    therefore land in the right slabs of the one ``nn.Parameter`` with no gather.

    ``grid[0][li]`` is a meta-device layer skeleton built from the PER-STREAM
    config, so the fold reads a stream's head counts and dims off it rather than
    the merged layer's K-multiplied ones (same trick as `PackedShadow`). Its
    Linears are never called — every projection goes through ``blinear`` — but its
    two residual norms are re-pointed at the merged layer's real ones, which are
    ``Replicated(sync=True)`` and so genuinely shared by all members.
    """

    def __init__(self, text_model, adapter, text_config, n_groups: int,
                 group_width: int = 1):
        self.n_tracks = n_groups          # the fold's "N" is the number of STREAMS
        self.group_width = group_width    # members pooled inside each stream
        self._fold_paths: dict[str, tuple[str, ...]] = {}
        # Where THIS family's attention differs from the plain form the fold assumes
        # (gated q_proj, zero-centered norm). `engine._batched_attn` reads it.
        self.attn_ops = adapter.attn_ops
        self._layers = text_model.layers
        self._specs = resolve_param_specs(adapter, text_config)
        n_shards = text_model.pt_cfg.n_tracks * n_groups * group_width
        per_group_cfg = adapter.build_per_track_text_config(
            text_config, n_shards, fuse_size=group_width
        )
        if group_width > 1:
            per_member = adapter.build_per_track_text_config(
                text_config, n_shards, fuse_size=1
            )
            heads = (per_member.num_attention_heads, per_member.num_key_value_heads)
            if heads != (1, 1):
                raise ValueError(
                    f"--fuse-tracks {group_width} on the batched path needs exactly one "
                    f"q head and one kv head per member (this convert has {heads}). "
                    f"q_norm/k_norm diverge per member and are broadcast per HEAD, so "
                    f"[G, F, d] only aligns when a group's head count is F. Use "
                    f"--no-merge-fused to take the looped path instead."
                )
        layer_cls = type(text_model.layers[0])
        skeletons = []
        for li, merged in enumerate(text_model.layers):
            with torch.device("meta"):
                skel = layer_cls(per_group_cfg, li)
            skel.input_layernorm = merged.input_layernorm
            skel.post_attention_layernorm = merged.post_attention_layernorm
            skeletons.append(skel)
        self.grid = [skeletons]

    def blinear(self, li: int, rel: str, x: torch.Tensor,
                per_track: bool = False) -> torch.Tensor:
        """All G members' Linear at ``rel``: ``[M, in]`` shared x (or ``[G, M,
        in]`` per-member x) → ``[G, M, out]``. Same body as
        `engine.DenseShadow.blinear`; the difference is entirely in ``stacked``."""
        return torch.matmul(x, self.stacked(li, f"{rel}.weight").transpose(-2, -1))

    def stacked(self, li: int, path: str) -> torch.Tensor:
        """The merged param at ``path`` viewed per member, ``[G, ...]``.

        NOT cached, deliberately — `DenseShadow` can cache because its providers
        are frozen, and this one's parameters are being trained. Most paths return
        views that would track an in-place update anyway, but `_regroup_qkv` is a
        real copy: caching it would pin the GDN qkv and conv weights to their
        step-0 values and silently train around them. Rebuilding is ~2 ms/step at
        27B, so there is nothing to buy back.
        """
        w = _submodule(self._layers[li], path)
        spec = self._specs[f"layers.{li}.{path}"]
        if isinstance(spec, GDNFusedQKV):
            return self._regroup_qkv(w, li)
        if isinstance(spec, Replicated):
            if spec.sync:
                # Collapsed to one shared copy at merge — every stream reads it.
                return w.unsqueeze(0).expand(self.n_tracks, *w.shape)
            # sync=False (q_norm/k_norm) was stacked [K, d] by _widen_diverged_norms.
            # Split K into streams x members; at F=1 the member axis is squeezed so
            # the result stays [G, d] and `_rms_tracks` sees its original form.
            return (w if self.group_width == 1
                    else w.unflatten(0, (self.n_tracks, self.group_width)))
        dim = getattr(spec, "dim", None)
        if dim is None:
            raise TypeError(
                f"{type(spec).__name__} has no concat dim, so {path!r} cannot be "
                f"viewed per member — add the case rather than guessing a dim"
            )
        # The merge was cat(dim); the unmerge is the matching unflatten, with the
        # member axis brought to the front for the batched matmul.
        return w.unflatten(dim, (self.n_tracks, -1)).movedim(dim, 0)

    def layer_fold(self, li: int, prefix: str) -> "_LayerFold":
        """ONE layer's ``prefix`` weights, resolved for the compiled region.

        Path lists are cached per prefix, not per layer: every full-attention layer
        carries the same parameter NAMES, and so does every GDN one, so the first
        layer to ask for a prefix answers for all of them. The TENSORS are resolved
        fresh on every call — `stacked` must not cache (see its docstring).
        """
        paths = self._fold_paths.get(prefix)
        if paths is None:
            head = f"layers.{li}.{prefix}"
            paths = tuple(k[len(f"layers.{li}."):] for k in self._specs
                          if k.startswith(head))
            self._fold_paths[prefix] = paths
        return _LayerFold(self, li, paths)

    def _regroup_qkv(self, w: torch.Tensor, li: int) -> torch.Tensor:
        """``GDNFusedQKV`` merges SEGMENT-major — ``[Q0..Q_{K-1} | K0.. | V0..]`` —
        because the merged track's own forward splits its slab as ``[Q|K|V]``. The
        fold wants STREAM-major, so regroup: three views, one cat. Segment sizes come
        from the per-stream skeleton, so a stream of F members takes F consecutive
        members' worth of each segment — which is contiguous, because each segment is
        member-major. Covers ``in_proj_qkv.weight`` and the depthwise
        ``conv1d.weight``, whose channels follow the same layout."""
        gdn = self.grid[0][li].linear_attn
        segs = [gdn.num_k_heads * gdn.head_k_dim] * 2 + [gdn.num_v_heads * gdn.head_v_dim]
        parts = torch.split(w, [s * self.n_tracks for s in segs], dim=0)
        return torch.cat(
            [p.unflatten(0, (self.n_tracks, s)) for p, s in zip(parts, segs)], dim=1
        )


class _LayerFold:
    """ONE layer's stacked weights, resolved BEFORE the compiled region.

    `torch.compile` SPECIALIZES ON PYTHON INTS, and `engine._batched_*` indexes its
    provider by the layer index. Handing a compiled callable that int recompiles the
    whole graph per layer: dynamo reported ``li == 7``, the walk hit
    ``recompile_limit`` (8) after 8 layers and ran the remaining 56 EAGER — slower
    than not compiling at all. Resolving the weights here makes the compiled function
    take TENSORS and one nn.Module, both of which dynamo treats as graph inputs, so
    the graph count is set by the two SHAPES a residual arrives in (shared ``[B,T,H]``
    out of a sync, per-member ``[G,B,T,H]`` mid-window) and is independent of DEPTH —
    which is what `tests/test_qwen3_slice.py` asserts.

    Serves the same provider interface the fold reads off a shadow — ``n_tracks``,
    ``attn_ops``, ``grid``, ``blinear``, ``stacked`` — with ``li`` accepted and
    IGNORED, so `engine._batched_*` and the decode providers stay untouched.
    """

    __slots__ = ("n_tracks", "attn_ops", "grid", "_w")

    def __init__(self, shadow, li: int, paths):
        self.n_tracks = shadow.n_tracks
        self.attn_ops = shadow.attn_ops
        self.grid = [[shadow.grid[0][li]]]
        self._w = {p: shadow.stacked(li, p) for p in paths}

    def stacked(self, li: int, path: str) -> torch.Tensor:
        return self._w[path]

    def blinear(self, li: int, rel: str, x: torch.Tensor,
                per_track: bool = False) -> torch.Tensor:
        return torch.matmul(x, self._w[f"{rel}.weight"].transpose(-2, -1))


def _fold_attn(fold, x, position_embeddings, attention_mask):
    """The compiled unit: one layer's attention half over a resolved fold."""
    return x + _batched_attn(fold, 0, x, position_embeddings, attention_mask, None)


def _fold_mlp(fold, x):
    """The compiled unit: one layer's MLP half over a resolved fold."""
    return x + _batched_mlp(fold, 0, x)


_COMPILED: dict[str, object] = {}


def enable_batched_compile(
    mode: str = "both", dynamic: bool | None = False, inductor_mode: str = "default"
) -> None:
    """`seam.enable_seam_compile` for the batched fold — the path a family with
    ``supports_batched_exec`` takes at every F, INCLUDING the frozen teacher, whose
    own walk is a `PTWrappedModel` at the exact schedule.

    Called by `seam.enable_seam_compile` so one ``--compile`` flag covers both track
    representations. A caller never picks between them: which one runs is decided by
    ``exec_groups``, not by the user, and before this the flag silently compiled
    nothing whenever the batched path was live.

    The GDN half is deliberately absent — `engine._batched_gdn` calls FLA /
    ``causal_conv1d`` kernels that would break the graph anyway, so
    `batched_token_mixer` keeps it on the eager branch.
    """
    if mode not in ("mixer", "mlp", "both"):
        raise ValueError(f"compile mode must be mixer|mlp|both, got {mode!r}")
    if inductor_mode == "reduce-overhead":
        raise ValueError(
            "reduce-overhead (CUDA graphs) is not supported for the track walk: a "
            "graph's output pool is reused by the next per-layer invocation while the "
            "backward still needs those tensors."
        )
    _hf()  # resolve the modeling module before the first trace: its import is a break
    kw = {"dynamic": dynamic}
    if inductor_mode != "default":
        kw["mode"] = inductor_mode
    if mode in ("mixer", "both"):
        _COMPILED["mixer"] = torch.compile(_fold_attn, **kw)
    if mode in ("mlp", "both"):
        _COMPILED["mlp"] = torch.compile(_fold_mlp, **kw)


def _attn_fn():
    return _COMPILED.get("mixer", _fold_attn)


def _mlp_fn():
    return _COMPILED.get("mlp", _fold_mlp)


def batched_token_mixer(shadow, li, x, position_embeddings, attention_mask, position_ids):
    """`seam.seam_token_mixer` for a merged track run as G members.

    ``x`` is ``[G, B, T, H]`` when the members carry their own residual, or
    ``[B, T, H]`` right after a sync when they all read the same one — the delta
    is ``[G, B, T, H]`` either way, so the add broadcasts a shared residual into
    per-member streams exactly as the boundary semantics require.

    ``position_ids`` is accepted for signature parity with `seam.seam_token_mixer`
    and unused: the fold consumes the precomputed rotary embeddings directly.

    ``block_type`` is Qwen3.5's hybrid marker; a family with only self-attention has
    no such attribute, hence the ``getattr`` — the same guard `seam.seam_token_mixer`
    and `train.teacher.HookedTeacher._attn_submodule` use.
    """
    if getattr(shadow.grid[0][li], "block_type", None) == "linear_attention":
        return x + _batched_gdn(shadow, li, x, attention_mask, None)
    return _attn_fn()(shadow.layer_fold(li, "self_attn."), x,
                      position_embeddings, attention_mask)


def batched_seam_mlp(shadow, li, x):
    """`seam.seam_mlp` for a merged track run as G members."""
    return _mlp_fn()(shadow.layer_fold(li, "mlp."), x)


def batched_halves(shadow, use_ckpt: bool, position_embeddings, position_ids):
    """``(run_mixer, run_mlp)`` for the batched path, mirroring
    `seam.checkpointed_halves` so the walks can swap one for the other.

    The callables take ``(layer_idx, x, mask)`` / ``(layer_idx, x)`` rather than a
    layer module: there is only one merged module per rank and the fold indexes it
    by depth.
    """
    def _mixer(li, x, mask):
        return batched_token_mixer(
            shadow, li, x, position_embeddings, mask, position_ids
        )

    if not use_ckpt:
        return _mixer, lambda li, x: batched_seam_mlp(shadow, li, x)

    from torch.utils.checkpoint import checkpoint

    return (
        lambda li, x, mask: checkpoint(_mixer, li, x, mask, use_reentrant=False),
        lambda li, x: checkpoint(batched_seam_mlp, shadow, li, x, use_reentrant=False),
    )
