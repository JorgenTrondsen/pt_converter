"""Declarative slicing specs for converting dense weights into per-track slices.

Each parameter in a decoder layer is tagged with a `SlicerSpec` that says how
to split it across N tracks. The slicer engine applies the spec to produce N
per-track tensors and (for verification) can reassemble them.

Specs operate on a single tensor at a time. Track index is supplied at slice
time. Specs are deliberately tiny and side-effect free so they can be unit
tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class SlicerSpec(Protocol):
    """Protocol for a slicing rule. Each spec knows how to slice and reassemble.

    Specs may optionally implement ``replication_groups(n_tracks, force_sync=False)``
    to declare that some tracks hold *identical* slices and therefore must share a
    gradient during training. Specs that don't implement it are treated as fully
    unique per track (one singleton group per track). Replicated-style specs carry
    a ``sync`` flag: when ``sync`` is False they return singleton groups (the copies
    are free to *diverge* under training), unless ``force_sync=True`` overrides it
    back to a single shared group (the legacy bit-identical behaviour).
    """

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor: ...

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor: ...

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]: ...

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor: ...

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]: ...


def align_chunk(
    chunk: int,
    align: int | None = None,
    *,
    full_size: int | None = None,
    n_tracks: int | None = None,
) -> int:
    """Round a zero-padded per-track slab up to a friendlier multiple.

    ``full_size`` / ``n_tracks``: never round so far that trailing tracks fall
    entirely past the split dim and hold nothing but zeros. **Every track gets a
    slab** — the model is split as evenly as it can be across n_tracks, because that
    is what max-tracks means; a track holding only an attention head is not a whole
    track. Ignored under an explicit ``align``, where a kernel requirement wins.

    Qwen3-32B at N=64 is the case: 25600 divides by 64 EXACTLY (400 each), yet the
    rounding widens the slab to 448 and 25600/448 = 57.14 — 57 tracks get full slabs,
    track 57 gets the last 64 lanes, and tracks 58-63 begin past the end and are pure
    padding. The tax grows with N (1 dead track at N=32, 6 at N=64), i.e. it bites
    hardest exactly where max-tracks wants to go.

    ⚠ The even slab is measurably WORSE downstream — trained n=2/arm at 32B/N=64:
    −0.031 macro and −0.098 math vs the 448 slab, plus ~5.5% on the MLP bmm. That is
    an accepted cost, not an oversight: an even split is the architecture we want. Do
    not "fix" it back by reading those numbers; see the ledger.

    ``align=None`` (the default) is the THROUGHPUT alignment: round to 64. Measured
    on the 27B (bf16, M=2048, K=5120): a **726**-wide GEMM — `ceil(17408/24)` — runs
    at 101 TFLOP/s, while the **768**-wide one that does 5.8% MORE work runs at 160.
    Since the extra lanes are zeros (exact for SwiGLU), rounding up is free quality
    and ~1.5x on the dense model's dominant GEMM. It is SKIPPED when the rounding
    would inflate the slab by more than 1/8 — that only happens at tiny (test-sized)
    widths, where there is no tensor-core win anyway.

    An explicit ``align`` is a KERNEL REQUIREMENT and is applied unconditionally: an
    unaligned width there is a crash, not a slowdown, so the 1/8 escape hatch would
    be actively wrong. `torch._grouped_mm` (the MoE `grouped_mm` experts path)
    rejects a contracted dim whose row stride is not a multiple of 16 bytes —
    "strides should be multiple of 16 bytes" — so a bf16 per-track expert width must
    be a multiple of 8 whatever the padding costs. Falling back instead is 4.7x
    slower (measured, 32 experts at hidden 2880: 3.6 vs 16.9 ms/MLP call).
    """
    if align is not None:
        return -(-chunk // align) * align
    up = -(-chunk // 64) * 64
    if up * 8 > chunk * 9:
        return chunk
    if (full_size is not None and n_tracks is not None
            and -(-full_size // up) < n_tracks):
        # Rounding up would need fewer slabs than there are tracks, so the
        # trailing ones would sit entirely past `full_size` and hold only zeros.
        return chunk
    return up


def _even_chunk(
    size: int, n_tracks: int, pad_full_size: int | None, label: str,
    align: int | None = None,
) -> int:
    """Per-track chunk along a split dim. Without `pad_full_size` the dim must
    divide evenly; with it the chunk rounds *up* (and to a GEMM-friendly multiple)
    and short slices are zero-filled."""
    if pad_full_size is None:
        if size % n_tracks != 0:
            raise ValueError(f"{label}: dim size {size} not divisible by n_tracks {n_tracks}")
        return size // n_tracks
    if size != pad_full_size:
        raise ValueError(f"{label}: expected dim size {pad_full_size}, got {size}")
    return align_chunk(-(-size // n_tracks), align, full_size=size, n_tracks=n_tracks)


def _cat_merge(slices: list[torch.Tensor], dim: int, chunk: int | None = None) -> torch.Tensor:
    """Concatenate F consecutive tracks' slabs into one F-wide track's slab.

    The inverse of slicing across a *group* rather than across all N tracks: with
    `fuse_size == tracks_per_rank` the group's members share their sublayer input,
    so one wide track computing the concatenated weights is the same function as F
    narrow tracks summing their deltas — and it issues 1/F the kernels.

    ``chunk`` (the padded per-track width) is passed by the specs that zero-pad, so
    a slab converted under an older/narrower rule is widened BEFORE the cat. That
    matters: padding the merged slab at the end instead would shift members 1..F-1
    off their own lane boundaries, and the split back to F shards would corrupt them.
    """
    if chunk is not None:
        slices = [_chunk_slice(s, dim, 0, chunk) for s in slices]
    return torch.cat(slices, dim=dim).contiguous()


def _chunk_split(merged: torch.Tensor, dim: int, fuse: int) -> list[torch.Tensor]:
    """Inverse of `_cat_merge`: the F equal slabs a merged track was built from."""
    if merged.shape[dim] % fuse:
        raise ValueError(
            f"split: dim {dim} size {merged.shape[dim]} not divisible by fuse {fuse}"
        )
    return [c.contiguous() for c in torch.chunk(merged, fuse, dim=dim)]


def _chunk_slice(weight: torch.Tensor, dim: int, track_idx: int, chunk: int) -> torch.Tensor:
    """Track `track_idx`'s `chunk`-wide slab along `dim`, zero-padded if it runs past the end."""
    size = weight.shape[dim]
    start = min(track_idx * chunk, size)
    got = weight.narrow(dim, start, min(chunk, size - start))
    if got.shape[dim] == chunk:
        return got.contiguous()
    tail = list(got.shape)
    tail[dim] = chunk - got.shape[dim]
    return torch.cat([got, got.new_zeros(tail)], dim=dim).contiguous()


@dataclass(frozen=True)
class Colwise:
    """Split the output dimension (dim 0 of nn.Linear.weight) evenly across tracks.

    Sums back via `torch.cat` along dim 0. The output activations of each
    track represent a *disjoint subset* of output features; the next op
    (typically a row-parallel op) must consume them appropriately.

    ``pad_full_size`` (the dense size along ``dim``) lifts the divisibility
    requirement: slices round up and the overhang is zero. For a SwiGLU MLP that
    is exact — a zero row in gate_proj gives silu(0)*up = 0, so the padded lane
    contributes exactly 0.0 to the down_proj sum.
    """

    dim: int = 0  # output dim of nn.Linear weight is 0
    pad_full_size: int | None = None
    align: int | None = None  # see `align_chunk`: None = throughput, int = hard requirement

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        chunk = _even_chunk(
            weight.shape[self.dim], n_tracks, self.pad_full_size, "Colwise", self.align
        )
        return _chunk_slice(weight, self.dim, track_idx, chunk)

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        out = torch.cat(slices, dim=self.dim)
        if self.pad_full_size is not None:
            out = out.narrow(self.dim, 0, self.pad_full_size)
        return out.contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = _even_chunk(
            out[self.dim], n_tracks, self.pad_full_size, "Colwise", self.align
        )
        return tuple(out)

    def _merge_chunk(self, n_tracks: int) -> int | None:
        if self.pad_full_size is None:
            return None
        return _even_chunk(
            self.pad_full_size, n_tracks, self.pad_full_size, "Colwise", self.align
        )

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        return _cat_merge(slices, self.dim, self._merge_chunk(n_tracks))

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        return _chunk_split(merged, self.dim, fuse)


@dataclass(frozen=True)
class Rowwise:
    """Split the input dimension (dim 1 of nn.Linear.weight) evenly across tracks.

    Each track's slice produces a *partial sum* of the output; the all-reduce
    at the next sync point completes the addition. ``pad_full_size`` works as in
    `Colwise` (zero input columns contribute nothing to the partial sum).
    """

    dim: int = 1  # input dim of nn.Linear weight is 1
    pad_full_size: int | None = None
    align: int | None = None  # see `align_chunk`: None = throughput, int = hard requirement

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        chunk = _even_chunk(
            weight.shape[self.dim], n_tracks, self.pad_full_size, "Rowwise", self.align
        )
        return _chunk_slice(weight, self.dim, track_idx, chunk)

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        out = torch.cat(slices, dim=self.dim)
        if self.pad_full_size is not None:
            out = out.narrow(self.dim, 0, self.pad_full_size)
        return out.contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = _even_chunk(
            out[self.dim], n_tracks, self.pad_full_size, "Rowwise", self.align
        )
        return tuple(out)

    def _merge_chunk(self, n_tracks: int) -> int | None:
        if self.pad_full_size is None:
            return None
        return _even_chunk(
            self.pad_full_size, n_tracks, self.pad_full_size, "Rowwise", self.align
        )

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        return _cat_merge(slices, self.dim, self._merge_chunk(n_tracks))

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        return _chunk_split(merged, self.dim, fuse)


@dataclass(frozen=True)
class PerHead:
    """Slice a 1-D per-head parameter (A_log, dt_bias) across tracks."""

    dim: int = 0
    num_heads: int | None = None  # asserted equal to size along dim if provided

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        size = weight.shape[self.dim]
        if self.num_heads is not None and self.num_heads != size:
            raise ValueError(f"PerHead expected dim {self.dim} == {self.num_heads}, got {size}")
        if size % n_tracks != 0:
            raise ValueError(f"PerHead size {size} not divisible by n_tracks {n_tracks}")
        chunk = size // n_tracks
        return weight.narrow(self.dim, track_idx * chunk, chunk).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(slices, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = out[self.dim] // n_tracks
        return tuple(out)

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        return _cat_merge(slices, self.dim)

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        return _chunk_split(merged, self.dim, fuse)


@dataclass(frozen=True)
class Replicated:
    """Replicated parameter: every track holds the same full tensor at conversion.

    Used for norms whose statistic is per-head-dim (RMSNorm on head_dim) and
    for final norms that operate on the synced hidden state.

    ``sync`` controls whether the copies are kept bit-identical during training.
    When True (default) they form one replication group and their gradients are
    averaged each step; when False each copy is its own singleton group and is
    free to diverge (e.g. per-track attention head norms).
    """

    sync: bool = True

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        return weight.detach().clone()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        return slices[0].detach().clone()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        return tuple(full_shape)

    def replication_groups(self, n_tracks: int, force_sync: bool = False) -> list[list[int]]:
        if self.sync or force_sync:
            # Every track holds the same tensor; one group spanning all tracks.
            return [list(range(n_tracks))]
        # Diverge: each copy trains independently.
        return [[t] for t in range(n_tracks)]

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        if self.sync:
            # `sync_replicated_grads` keeps these bit-identical, so one copy IS the
            # group. Assert rather than trust: a drifted copy here would be
            # silently averaged away by the merge.
            for s in slices[1:]:
                if not torch.equal(s, slices[0]):
                    raise ValueError(
                        "Replicated(sync=True).merge: copies differ, so they cannot "
                        "be collapsed to one. Was sync_replicated_grads skipped?"
                    )
            return slices[0].detach().clone()
        # Diverged copies (q_norm/k_norm): stack into a leading per-member dim.
        # Exact only because these are per-head RMSNorm weights applied to a
        # (..., heads, head_dim) tensor, where [F, head_dim] broadcasts per head —
        # verified against a per-head loop. A `sync=False` param that is NOT
        # applied per-head cannot be merged this way.
        return torch.stack([s.detach() for s in slices], dim=0).contiguous()

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        if self.sync:
            return [merged.detach().clone() for _ in range(fuse)]
        if merged.shape[0] != fuse:
            raise ValueError(
                f"Replicated(sync=False).split: leading dim {merged.shape[0]} != fuse {fuse}"
            )
        return [merged[i].contiguous() for i in range(fuse)]


@dataclass(frozen=True)
class OwnerOnly:
    """Parameter that lives on exactly one track. Other tracks omit it entirely.

    Used for `embed_tokens.weight` and `lm_head.weight`: the owner track does
    the embedding lookup and the final logits projection; the embedding output
    is broadcast to peer tracks via the existing all-reduce sync primitive
    (zero-padded delta), and the synced post-final-norm hidden state is local
    to every track so the owner can apply lm_head without further communication.

    `slice` returns `None` on non-owner tracks; the slicer engine interprets
    that as "skip this key" so the per-track state_dict simply omits it.
    """

    owner_track: int = 0

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor | None:
        if track_idx != self.owner_track:
            return None
        return weight.detach().clone()

    def reassemble(self, slices: list[torch.Tensor | None]) -> torch.Tensor:
        for s in slices:
            if s is not None:
                return s.detach().clone()
        raise ValueError("OwnerOnly.reassemble got no non-None slices")

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        return tuple(full_shape)

    def replication_groups(self, n_tracks: int, force_sync: bool = False) -> list[list[int]]:
        # The param lives on exactly one track. No gradient sync needed.
        # `force_sync` is accepted for signature uniformity and ignored.
        return [[self.owner_track]]

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        # Only the group containing the owner holds this at all; the caller skips
        # keys absent from a group's shards, so a present slice is the owner's.
        for s in slices:
            if s is not None:
                return s.detach().clone()
        raise ValueError("OwnerOnly.merge got no non-None slices")

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        # Back to the owner's shard only; peers omit the key. Groups are aligned
        # runs of `fuse` starting at a multiple of `fuse` (PTWrappedModel enforces
        # it), so the owner's position within its group is owner_track % fuse.
        own = self.owner_track % fuse
        return [merged.detach().clone() if i == own else None for i in range(fuse)]


@dataclass(frozen=True)
class SummedBias:
    """Additive bias on a ROW-PARALLEL output (``o_proj.bias``, ``experts.down_proj_bias``).

    A row-parallel projection makes every track produce a *partial sum* that the next
    `SyncBoundary` all-reduce completes. A `Replicated` bias would therefore be added
    N times. Instead the owner track keeps the full bias and its peers hold zeros, so
    the summed partials carry it exactly once — exact in bf16, unlike a ``w / N`` split.

    The copies are deliberately free to DIVERGE under training (singleton replication
    groups): only their SUM is the effective bias, so there is nothing to keep
    bit-identical and each track's copy is real extra capacity.
    """

    owner_track: int = 0

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        if track_idx == self.owner_track:
            return weight.detach().clone()
        return torch.zeros_like(weight)

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        out = slices[0].detach().clone()
        for s in slices[1:]:
            out = out + s
        return out.contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        return tuple(full_shape)

    def replication_groups(self, n_tracks: int, force_sync: bool = False) -> list[list[int]]:
        # `force_sync` is accepted for signature uniformity and ignored: averaging these
        # copies would not preserve their sum, which is the only thing that is meaningful.
        return [[t] for t in range(n_tracks)]

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        # The merged track's row-parallel GEMM reduces over the members' concatenated
        # input dim, i.e. it sums their outputs — so one bias equal to the members' sum.
        return self.reassemble(slices)

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        # Any split whose sum is `merged` is exact; put it all on the group's owner.
        own = self.owner_track % fuse
        return [
            merged.detach().clone() if i == own else torch.zeros_like(merged)
            for i in range(fuse)
        ]


@dataclass(frozen=True)
class FusedSegmentColwise:
    """Slice an nn.Linear whose output is a concatenation of N logical segments.

    For Qwen3.5 `in_proj_qkv`: out_features = [Q (key_dim) | K (key_dim) | V (value_dim)].
    Each segment must be sliced colwise *independently* then re-concatenated in
    segment order. Used wherever the dense layer fuses multiple per-head
    projections into a single weight matrix.

    `segments` is a list of segment sizes along `dim`. They must sum to
    `weight.shape[dim]`. Each segment size must be divisible by n_tracks.
    """

    segments: tuple[int, ...]
    dim: int = 0

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        total = sum(self.segments)
        if total != weight.shape[self.dim]:
            raise ValueError(
                f"FusedSegmentColwise: segments sum {total} != weight dim {weight.shape[self.dim]}"
            )
        per_track_segs: list[torch.Tensor] = []
        offset = 0
        for seg in self.segments:
            if seg % n_tracks != 0:
                raise ValueError(f"FusedSegmentColwise: segment {seg} not divisible by n_tracks {n_tracks}")
            chunk = seg // n_tracks
            per_track_segs.append(
                weight.narrow(self.dim, offset + track_idx * chunk, chunk).contiguous()
            )
            offset += seg
        return torch.cat(per_track_segs, dim=self.dim).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        # slices is one tensor per track, each already fused [Qslice|Kslice|Vslice].
        # To reassemble we split each track's slice back into segments, then
        # concatenate per-segment across tracks, then re-fuse.
        n_tracks = len(slices)
        track0 = slices[0]
        per_track_chunks = [s // n_tracks for s in self.segments]  # each track holds these per-segment sizes
        # split each track-tensor back into its (per-segment) chunks
        split_per_track = [
            torch.split(s, per_track_chunks, dim=self.dim) for s in slices
        ]
        # gather per segment
        segments_concat = []
        for seg_idx in range(len(self.segments)):
            segments_concat.append(
                torch.cat([split_per_track[t][seg_idx] for t in range(n_tracks)], dim=self.dim)
            )
        return torch.cat(segments_concat, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = out[self.dim] // n_tracks
        return tuple(out)

    def _per_track_segments(self, n_tracks: int) -> list[int]:
        return [s // n_tracks for s in self.segments]

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        # A plain cat would give [QKV_0 | QKV_1 | ...]; the wide track's forward
        # wants [Q_all | K_all | V_all], so merge segment-wise.
        segs = self._per_track_segments(n_tracks)
        parts = [torch.split(s, segs, dim=self.dim) for s in slices]
        return torch.cat(
            [torch.cat([p[i] for p in parts], dim=self.dim) for i in range(len(segs))],
            dim=self.dim,
        ).contiguous()

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        segs = self._per_track_segments(n_tracks)
        wide = torch.split(merged, [s * fuse for s in segs], dim=self.dim)
        per_seg = [torch.chunk(w, fuse, dim=self.dim) for w in wide]
        return [
            torch.cat([per_seg[i][t] for i in range(len(segs))], dim=self.dim).contiguous()
            for t in range(fuse)
        ]


@dataclass(frozen=True)
class GDNFusedQKV:
    """Slice a GatedDeltaNet `[Q (key_dim) | K (key_dim) | V (value_dim)]` slab by *value* head.

    GDN is GQA-shaped over its value heads: `ratio = num_v_heads // num_k_heads`
    v-heads share one k-head, and the forward repeat-interleaves q/k by that ratio.
    The parallel unit is therefore the v-head — track `t` owns v-heads
    `[t*vpt, (t+1)*vpt)` and needs whichever k-heads those v-heads read
    (`k = v // ratio`).

    Two regimes, both exact:

      - `num_k_heads % n_tracks == 0`: each track's k-heads are one contiguous
        block, emitted **compactly** (`num_k_heads // n_tracks` of them, dense
        ratio preserved). Byte-identical to a plain per-segment colwise split.
      - otherwise: the block straddles k-heads unevenly (e.g. 16 k / 48 v over 24
        tracks gives 1 or 2 k-heads depending on the track), and per-track shapes
        must stay uniform for the batched forward. So every track gets ONE k-head
        copy per v-head — per-track ratio 1, which the forward handles natively
        (it only repeat-interleaves when the ratio exceeds 1). Costs a `ratio`x
        replication of Q/K.

    Deliberately has no ``replication_groups``: a track's slab always carries its
    own unique V segment, so no two tracks hold an identical tensor. Duplicated
    k-heads are free to diverge under training, like full-attention `k_proj`.
    """

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    dim: int = 0

    def _k_head_indices(self, track_idx: int, n_tracks: int) -> list[int]:
        if self.num_v_heads % n_tracks != 0:
            raise ValueError(
                f"GDNFusedQKV: num_v_heads {self.num_v_heads} not divisible by n_tracks {n_tracks}"
            )
        if self.num_k_heads % n_tracks == 0:
            per_track = self.num_k_heads // n_tracks
            return list(range(track_idx * per_track, (track_idx + 1) * per_track))
        vpt = self.num_v_heads // n_tracks
        ratio = self.num_v_heads // self.num_k_heads
        return [(track_idx * vpt + j) // ratio for j in range(vpt)]

    def _dims(self) -> tuple[int, int]:
        return self.num_k_heads * self.head_k_dim, self.num_v_heads * self.head_v_dim

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        key_dim, value_dim = self._dims()
        if weight.shape[self.dim] != 2 * key_dim + value_dim:
            raise ValueError(
                f"GDNFusedQKV: expected dim {self.dim} = {2 * key_dim + value_dim}, "
                f"got {weight.shape[self.dim]}"
            )
        k_idx = self._k_head_indices(track_idx, n_tracks)
        v_chunk = (self.num_v_heads // n_tracks) * self.head_v_dim
        parts = [
            weight.narrow(self.dim, base + i * self.head_k_dim, self.head_k_dim)
            for base in (0, key_dim)
            for i in k_idx
        ]
        parts.append(weight.narrow(self.dim, 2 * key_dim + track_idx * v_chunk, v_chunk))
        return torch.cat(parts, dim=self.dim).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        # Q/K are de-duplicated (a k-head may be held by several tracks); V concatenates.
        n_tracks = len(slices)
        v_chunk = (self.num_v_heads // n_tracks) * self.head_v_dim
        q: dict[int, torch.Tensor] = {}
        k: dict[int, torch.Tensor] = {}
        v: list[torch.Tensor] = []
        for t, s in enumerate(slices):
            k_idx = self._k_head_indices(t, n_tracks)
            n_k = len(k_idx)
            for j, i in enumerate(k_idx):
                q[i] = s.narrow(self.dim, j * self.head_k_dim, self.head_k_dim)
                k[i] = s.narrow(self.dim, (n_k + j) * self.head_k_dim, self.head_k_dim)
            v.append(s.narrow(self.dim, 2 * n_k * self.head_k_dim, v_chunk))
        order = sorted(q)
        return torch.cat([q[i] for i in order] + [k[i] for i in order] + v, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        n_k = len(self._k_head_indices(0, n_tracks))
        out[self.dim] = (
            2 * n_k * self.head_k_dim + (self.num_v_heads // n_tracks) * self.head_v_dim
        )
        return tuple(out)

    def _per_track_segments(self, n_tracks: int) -> list[int]:
        n_k = len(self._k_head_indices(0, n_tracks))
        k = n_k * self.head_k_dim
        return [k, k, (self.num_v_heads // n_tracks) * self.head_v_dim]

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        # Segment-wise, like FusedSegmentColwise: the merged track's forward splits
        # its slab as [Q | K | V], so the F members' Q's must be adjacent. The
        # k-head COPIES ride along unchanged — the merged track has one k-head per
        # v-head (ratio 1), which is what each member already had.
        segs = self._per_track_segments(n_tracks)
        parts = [torch.split(s, segs, dim=self.dim) for s in slices]
        return torch.cat(
            [torch.cat([p[i] for p in parts], dim=self.dim) for i in range(3)],
            dim=self.dim,
        ).contiguous()

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        segs = self._per_track_segments(n_tracks)
        wide = torch.split(merged, [s * fuse for s in segs], dim=self.dim)
        per_seg = [torch.chunk(w, fuse, dim=self.dim) for w in wide]
        return [
            torch.cat([per_seg[i][t] for i in range(3)], dim=self.dim).contiguous()
            for t in range(fuse)
        ]


@dataclass(frozen=True)
class KVReplicatedColwise:
    """Slice an out-dim that is per-kv-head, with replication across tracks in a kv-group.

    Used for `k_proj.weight` and `v_proj.weight` in GQA full-attention when we
    push N above num_kv_heads. Track `t` belongs to kv-group
    `g = t // (n_tracks // num_kv_heads)`; it stores rows
    `[g*chunk : (g+1)*chunk]` where `chunk = weight.shape[dim] // num_kv_heads`.
    Every track in the same kv-group therefore holds an *identical* slice.

    Constraints:
      - n_tracks must be a multiple of num_kv_heads (so the kv-group factor
        `tracks_per_kv_group = n_tracks // num_kv_heads` is integer).
      - weight.shape[dim] must be divisible by num_kv_heads.

    `reassemble` returns the concatenation of the *unique* per-group slices
    (one per kv-group), reconstructing the dense tensor. Caller is expected
    to either (a) drop duplicate tracks within each group before calling, or
    (b) pass exactly one slice per kv-group.

    ``sync`` controls whether the within-group copies are kept bit-identical
    during training. When True (legacy) the copies form one replication group
    per kv-group and their gradients are averaged each step, exactly reproducing
    dense GQA. When False the copies are free to diverge — each track gets its
    own KV head (GQA → per-track MHA), a capacity superset that distillation can
    exploit. Slicing/reassembly are unaffected by this flag; only the
    training-time replication grouping changes.
    """

    num_kv_heads: int
    dim: int = 0
    sync: bool = True

    def _chunk(self, weight_shape: tuple[int, ...]) -> int:
        size = weight_shape[self.dim]
        if size % self.num_kv_heads != 0:
            raise ValueError(
                f"KVReplicatedColwise: dim {self.dim} size {size} not divisible by "
                f"num_kv_heads {self.num_kv_heads}"
            )
        return size // self.num_kv_heads

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        if n_tracks % self.num_kv_heads != 0:
            raise ValueError(
                f"KVReplicatedColwise: n_tracks {n_tracks} must be a multiple of "
                f"num_kv_heads {self.num_kv_heads}"
            )
        chunk = self._chunk(tuple(weight.shape))
        tracks_per_group = n_tracks // self.num_kv_heads
        g = track_idx // tracks_per_group
        return weight.narrow(self.dim, g * chunk, chunk).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        # `slices` is one unique slice per kv-group (length == num_kv_heads).
        if len(slices) != self.num_kv_heads:
            raise ValueError(
                f"KVReplicatedColwise.reassemble expects exactly {self.num_kv_heads} "
                f"unique slices (one per kv-group), got {len(slices)}"
            )
        return torch.cat(slices, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        # Every track in the same kv-group sees the same shape; n_tracks is
        # accepted for interface uniformity but not used.
        out = list(full_shape)
        out[self.dim] = full_shape[self.dim] // self.num_kv_heads
        return tuple(out)

    def replication_groups(self, n_tracks: int, force_sync: bool = False) -> list[list[int]]:
        if not (self.sync or force_sync):
            # Diverge: every track's KV head trains independently, so the answer is
            # singletons and the kv-group factor is never formed. Divisibility is
            # deliberately NOT required here — under merged tracks this is called
            # with the LOGICAL track count (24 shards / F per rank), which need not
            # be a multiple of num_kv_heads: F=2 at N=24 gives 6 logical tracks
            # against 4 kv heads. Slicing still happens at the shard count, where
            # divisibility does hold.
            return [[t] for t in range(n_tracks)]
        # Keep each kv-group's copies bit-identical (dense GQA). Only this branch
        # partitions BY kv-group, so only this branch needs the factor to exist.
        if n_tracks % self.num_kv_heads != 0:
            raise ValueError(
                f"KVReplicatedColwise.replication_groups: n_tracks {n_tracks} "
                f"must be a multiple of num_kv_heads {self.num_kv_heads}"
            )
        tpg = n_tracks // self.num_kv_heads
        return [[g * tpg + i for i in range(tpg)] for g in range(self.num_kv_heads)]

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        # The merged track gets one kv-head per q-head (num_key_value_heads = F,
        # ratio 1), each member keeping its own — possibly already diverged — copy.
        return _cat_merge(slices, self.dim)

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        return _chunk_split(merged, self.dim, fuse)


@dataclass(frozen=True)
class GatedQColwise:
    """Slice Qwen3.5's doubled `q_proj` whose output carries [q_h0, gate_h0, q_h1, gate_h1, ...]
    interleaved per head (size: num_heads * 2 * head_dim along dim 0).

    For each head the q-half and gate-half are interleaved (the view+chunk in
    forward() reshapes to (-1, head_dim*2) then chunks 2 along the last dim).
    So per head the layout is contiguous: head_i occupies a block of
    `2*head_dim` rows starting at `i * 2 * head_dim`.

    To slice per-track we keep complete heads. With n_tracks dividing num_heads,
    track t gets heads [t*heads_per_track : (t+1)*heads_per_track], which is a
    contiguous slab of rows of size `heads_per_track * 2 * head_dim`.
    """

    num_heads: int
    head_dim: int
    dim: int = 0

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        expected = self.num_heads * 2 * self.head_dim
        if weight.shape[self.dim] != expected:
            raise ValueError(
                f"GatedQColwise: expected dim {self.dim} = {expected}, got {weight.shape[self.dim]}"
            )
        if self.num_heads % n_tracks != 0:
            raise ValueError(f"GatedQColwise: num_heads {self.num_heads} not divisible by n_tracks {n_tracks}")
        heads_per_track = self.num_heads // n_tracks
        block = heads_per_track * 2 * self.head_dim
        return weight.narrow(self.dim, track_idx * block, block).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(slices, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = (self.num_heads // n_tracks) * 2 * self.head_dim
        return tuple(out)

    def merge(self, slices: list[torch.Tensor], n_tracks: int) -> torch.Tensor:
        # Head-major ([q|gate] per head), so a plain cat already lands the merged
        # track's heads in order — the view+chunk in forward() sees F*2*head_dim.
        return _cat_merge(slices, self.dim)

    def split(self, merged: torch.Tensor, fuse: int, n_tracks: int) -> list[torch.Tensor]:
        return _chunk_split(merged, self.dim, fuse)


# Mapping from a fully-qualified parameter sub-path inside a decoder layer to
# its slicer spec. A `LayerSpec` is a dict that callers use to slice every
# parameter in one layer.
LayerSpec = dict[str, "SlicerSpec"]


# A family's token-mixer spec table: layer type -> (module prefix under the decoder
# layer, spec builder). Deriving `valid_layer_types` from its keys is what stops the
# adapter's declared type list drifting from the types it can actually slice.
AttentionSpecMap = dict[str, "tuple[str, object]"]


def build_decoder_layer_specs(
    text_cfg,
    layer_type: str,
    *,
    attention_specs: AttentionSpecMap,
    mlp_specs,
) -> LayerSpec:
    """Assemble one decoder layer's specs: replicated norms + the token-mixer specs
    for `layer_type` (under its own module prefix) + the MLP specs (prefixed `mlp.`).

    `attention_specs` maps every layer type the family supports to
    ``(module_prefix, spec_fn)`` — Qwen3.5 has ``full_attention -> self_attn`` and
    ``linear_attention -> linear_attn``; gpt-oss has ``full_attention`` and
    ``sliding_attention`` both under ``self_attn`` (they differ only by mask). This
    shape is identical across dense and MoE.
    """
    out: LayerSpec = {
        "input_layernorm.weight": Replicated(),
        "post_attention_layernorm.weight": Replicated(),
    }
    entry = attention_specs.get(layer_type)
    if entry is None:
        raise ValueError(
            f"Unknown layer_type {layer_type!r}; this family slices {sorted(attention_specs)}"
        )
    prefix, spec_fn = entry
    for k, v in spec_fn(text_cfg).items():
        out[f"{prefix}.{k}"] = v
    for k, v in mlp_specs(text_cfg).items():
        out[f"mlp.{k}"] = v
    return out


def standard_top_level_specs(_text_cfg) -> LayerSpec:
    """Params outside any decoder layer: embeddings, final norm, lm head.

    Identical for every family so far, so it is the default rather than per-family
    boilerplate. `embed_tokens` and `lm_head` live on track 0 only; the input embedding
    is broadcast to peer tracks via the start-of-forward sync (zero-padded all-reduce),
    and lm_head is applied locally on track 0 to the synced post-final-norm hidden
    state. `norm.weight` stays replicated because every track applies the final RMSNorm
    to its (synced) hidden state.
    """
    return {
        "embed_tokens.weight": OwnerOnly(owner_track=0),
        "norm.weight": Replicated(),
        "lm_head.weight": OwnerOnly(owner_track=0),
    }
