"""The parallm inference engine: lockstep decode with sparse-replica replay.

Deployed semantics (fixed by the probe that measured the frontier — git tag
``pre-parallm-pivot``, ``eval/intervention.py::phased_intervention_forward``,
``replica:*`` channel, whole-network copies):

- Between sync boundaries every track carries the same **shadow estimate** of the
  full residual, advanced comm-free by replaying ALL ``N`` tracks' sublayers with
  degraded weight copies (including the rank's own track — the carry must be
  bit-identical on every rank, so the shadow chain never mixes in exact weights).
- Alongside, each track runs its OWN sublayers exactly on that carry and
  accumulates the exact deltas. A boundary is ONE all-reduce of those deltas
  (`SyncBoundary`), which recombines the true residual and resets the carry.
- Sync placement is phased/post-attn (LEVER B): boundaries in ``sync_indices``
  sync after the token mixer (the boundary MLP then reads the TRUE residual);
  the final layer's boundary is post-MLP and produces the fully-synced hidden
  for the lm_head. Total sync events = len(boundaries < last) + 1.

The chunk forward serves both prefill/eval (whole sequence, ``caches=None`` or
fresh) and decode (1-token chunks with caches); incremental ≡ full-sequence by
causality, which is the engine's second correctness rail (the first: undegraded
copies ⇒ bit-equal to the dense model, up to fp summation order).

Comm rounds per decode token = 1 embed broadcast + boundary all-reduces. The
sampled token id itself is never broadcast: peers hold no ``embed_tokens`` and
contribute zeros to the embed collective, so only the head needs real ids.
``latency_s`` sleeps after every round — the single-node simulation of the
multi-node link (a real deployment needs the one-round hierarchical exchange
noted in the plan; stock ring all-reduce would pay (n_nodes-1) rounds).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from parallm.adapters import AttnOps
from parallm.model.pt_model import PTWrappedModel


# --------------------------------------------------------------------------- #
# Shadow providers. The batched replay path never calls the grid modules'
# Linears: all N tracks' projections go through ``blinear`` (one batched op per
# rel), and the diverged small params (q/k norms, conv, dt, A, gated norm) are
# read per-track via ``_stacked``. The grid remains the storage for those
# params and the per-layer module attrs (head dims, eps, ...).
# --------------------------------------------------------------------------- #
_GEMV_MAX_M = 64


def _submodule(mod, dotted: str):
    for p in dotted.split("."):
        mod = getattr(mod, p)
    return mod


class GridStacked:
    """``stacked`` for providers whose storage is a ``grid[track][layer]``.

    Split out from the providers so the batched forward can source its ``[N, ...]``
    weights from something else — `parallm.model.batched.MergedShadow` serves them
    as views of ONE merged parameter, which is how a merged track runs unfused.
    """

    #: Where the served family's attention differs from the plain form (see
    #: `parallm.adapters.AttnOps`). The decode providers below are Qwen3.5-only by
    #: construction — the GDN fold and the replica pool are written to its shapes —
    #: so they pin its flavour here. `MergedShadow` (the training-side fold, which IS
    #: family-agnostic) overrides this from its adapter.
    attn_ops = AttnOps(gated_q=True, centered_norm=True)

    def stacked(self, li: int, path: str) -> torch.Tensor:
        """The N tracks' param at ``path`` stacked ``[N, ...]`` (cached). A
        single-track grid gets a zero-copy view — the own-track provider must not
        duplicate the rank's full weights."""
        key = (li, path)
        t = self._stacked.get(key)
        if t is None:
            mods = [_submodule(tr[li], path) for tr in self.grid]
            t = mods[0].unsqueeze(0) if len(mods) == 1 else torch.stack(mods)
            self._stacked[key] = t
        return t


def _norm_flat(norm, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """RMSNorm the residual and flatten it to ``blinear``'s input layout.

    ``x [B, T, H]`` — every track reads the same synced residual (decode, and any
    boundary sublayer) — flattens to ``[M, H]`` and ``blinear`` broadcasts it over
    the N tracks. ``x [N, B, T, H]`` — each track carries its own residual (the
    unfused own-carry walk) — flattens to ``[N, M, H]``. The norm weight is
    track-synced in both cases, so one module applies to every member.
    """
    h = norm(x)
    return h.reshape(-1, B * T, h.shape[-1]) if h.ndim == 4 else h.reshape(B * T, -1)


class DenseShadow(GridStacked):
    """Materialized shadow modules, ``grid[track][layer]``. Exact copies can
    share the student's own modules (weights are read-only here); degraded
    copies come from ``model.replica.degrade_track_layers``."""

    def __init__(self, grid: "Sequence[Sequence[nn.Module]]"):
        self.grid = grid
        self.n_tracks = len(grid)
        self._stacked: dict = {}
        # Interface parity with PackedShadow, whose callers read both.
        self.meta: dict = {}
        self.ring = None

    def layers(self, i: int) -> "list[nn.Module]":
        return [track[i] for track in self.grid]

    def blinear(self, li: int, rel: str, x: torch.Tensor,
                per_track: bool = False) -> torch.Tensor:
        """All N tracks' Linear at ``rel``: ``[M, in]`` shared x (or ``[N, M,
        in]`` per-track x) → ``[N, M, out]``. matmul broadcasts the shared x."""
        w = self.stacked(li, f"{rel}.weight")
        return torch.matmul(x, w.transpose(-2, -1))


class _RelEntry:
    """One (layer, rel) of the pool, batched across all N tracks for a single
    unpack kernel chain per rel: packed survivor bitmaps ``[N, bytes]``, the
    tracks' survivor values concatenated flat (codes re-nibble-packed without
    the per-track pad, so residency stays at the packed bits/weight), and
    per-row scales ``[N, out]``. ``masked_scatter_`` consumes the flat values
    in row-major mask order across the stacked ``[N, out, in]`` tensor, which
    is exactly the track-then-row pack order.

    ``pin=False``: tensors live on ``device`` (resident mode). ``pin=True``:
    tensors stay in pinned host DRAM and stream through a `_Ring` slot.
    ``ent``: the entry's entropy-coded plane (blob/offsets/raw_len from the
    repack artifact) replaces the pinned raw codes — the ring's pump copies
    the blob and GPU-decodes it into the slot, ~21-26% fewer PCIe/DRAM bytes,
    bit-identical decoded plane. Streamed-only."""

    def __init__(self, packs: "list[dict]", device, pin: bool = False,
                 ent: "dict | None" = None):
        import numpy as np

        shapes = {tuple(p["shape"].tolist()) for p in packs}
        assert len(shapes) == 1, f"rel shape mismatch across tracks: {shapes}"
        self.shape = (len(packs), *shapes.pop())  # (N, out, in)
        n, o, i = self.shape
        place = (lambda t: t.pin_memory()) if pin else (lambda t: t.to(device))
        self.mask = place(torch.stack([p["mask"] for p in packs]))
        self.scale = None

        # Per-(row, bitmap-block) survivor offsets into the flat value stream
        # (absolute across the track-major concat) — the fused kernel's v2
        # addressing, which frees each block iteration from the previous one's
        # popcount. Derived from the bitmap at load (~2% of pool bytes); small,
        # so always device-resident even when the payload is pinned host-side.
        from parallm.packed_gemv import GEMV_BLOCK_I

        nb = -(-i // GEMV_BLOCK_I)
        bits = np.stack([
            np.unpackbits(p["mask"].numpy())[: o * i].reshape(o, i)
            for p in packs
        ])                                        # [N, o, i]
        pad = np.zeros((n, o, nb * GEMV_BLOCK_I - i), dtype=bits.dtype)
        bpc = np.concatenate([bits, pad], -1).reshape(n, o, nb, -1).sum(-1)
        flat = np.cumsum(bpc.reshape(-1).astype(np.int64))  # inclusive, track-major
        starts = np.concatenate([[0], flat[:-1]]).reshape(n, o, nb)
        self.block_start = torch.from_numpy(starts.astype(np.int32)).to(device)
        track_nnz = bits.sum((1, 2))

        self.blob = None
        if "vals" in packs[0]:
            assert ent is None, "bf16 vals pools have no entropy-coded plane"
            self.bits = None
            self.vals = place(torch.cat([p["vals"] for p in packs]))
            self.nnz = self.values_numel = self.vals.numel()
            self.values_dtype = self.vals.dtype
        else:
            self.bits = int(packs[0]["bits"].item())
            self.scale = place(torch.stack([p["scale"] for p in packs]).float())
            self.nnz = int(track_nnz.sum())
            self.values_dtype = torch.uint8
            if ent is not None:
                # Codes never materialize raw: the pinned blob is the DRAM
                # copy, the decoded bytes exist only in the ring slot.
                self.codes = None
                self.values_numel = int(ent["raw_len"])
                self.blob = ent["blob"].pin_memory()
                self.blob_offsets = ent["offsets"]  # host-side; plans bake it
            else:
                plane, nnz = assemble_codes(packs, track_nnz)
                assert nnz == self.nnz, (nnz, self.nnz)
                self.codes = place(plane)
                self.values_numel = plane.numel()

    @property
    def values(self) -> torch.Tensor:
        return self.vals if self.bits is None else self.codes

    def unpack_from(self, mask: torch.Tensor, values: torch.Tensor,
                    out: torch.Tensor, f32: "torch.Tensor | None",
                    scale: "torch.Tensor | None" = None) -> None:
        """Materialize the degraded dense weights into ``out`` ``[N, out, in]``
        from the given (device) packed buffers — bit-exact to per-track
        ``unpack_sparse_weight``."""
        n, o, i = self.shape
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=mask.device)
        mask = (((mask.unsqueeze(-1) >> shifts) & 1)
                .reshape(n, -1)[:, : o * i].to(torch.bool).reshape(n, o, i))
        if self.bits is None:
            out.zero_()
            out.masked_scatter_(mask, values)
            return
        if self.bits == 4:
            values = torch.stack(((values >> 4) & 0xF, values & 0xF), dim=-1)
            values = values.reshape(-1)[: self.nnz]
        signed = values.to(torch.float32) - float(2 ** (self.bits - 1))
        f32.zero_()
        f32.masked_scatter_(mask, signed)
        f32 *= scale[:, :, None]
        out.copy_(f32)


def _ring_geometry(num_layers: int, n_slots: int, window_layers: int) -> int:
    """Validate window-granular ring geometry; returns ``ring_windows``.

    The constraints are exactly what makes streamed slots CUDA-graph-safe:
    slot ``li % n_slots`` must be the same device address every token
    (``num_layers % n_slots == 0`` — otherwise the layer→slot map drifts
    across passes), slots must partition into whole windows, and at least two
    ring windows must exist: window w+1's leading shadow-MLP re-reads boundary
    layer w, so one window's slot set frees a window LATE — at
    ``ring_windows == 1`` a single captured window would need one slot to hold
    two layers' data."""
    if window_layers < 1 or n_slots % window_layers:
        raise ValueError(
            f"ring layers ({n_slots}) must be whole sync windows of {window_layers} layers")
    rw = n_slots // window_layers
    if rw < 2:
        raise ValueError(
            "the stream ring needs --ring-windows >= 2 (window w+1 re-reads boundary layer w)")
    if num_layers % n_slots:
        raise ValueError(
            f"num_layers ({num_layers}) % ring layers ({n_slots}) != 0 — "
            "the layer->slot map would drift across tokens (breaks CUDA-graphed slots)")
    return rw


class _Ring:
    """Device ring for the streamed pool, window-granular so it composes with
    CUDA-graphed windows: ``n_slots`` per-layer slots at FIXED device
    addresses (slot = ``li % n_slots``, token-stable by ``_ring_geometry``),
    refilled one whole sync window at a time from pinned host DRAM on a
    dedicated copy stream. All host/event work happens in ``begin_window`` /
    ``end_window`` — the eager boundary region of the replay, never inside a
    capture — so a captured window bakes its layers' slot addresses and the
    copies swap the contents between replays. The sync stalls are the hiding
    budget, exactly the bench_stream_overlap law.

    Slot-reuse hazard: window w+1's leading shadow-MLP re-reads boundary layer
    w, so the refill for occurrence ``W`` must wait for window ``W-rw+1`` to
    finish (its ``done`` event), and ``end_window(occ)`` may host-enqueue
    pumps only up to occurrence ``occ+rw-1`` — the newest whose gating done
    event exists. A ``wait_event`` on a never-recorded event is a silent
    no-op, not an error: enqueuing one occurrence further is a data race."""

    def __init__(self, shadow: "PackedShadow", n_slots: int, device,
                 window_layers: int):
        self.shadow = shadow
        self.device = device
        self.n_slots = n_slots
        self.rw = _ring_geometry(shadow.num_layers, n_slots, window_layers)
        self.D = window_layers
        self.n_windows = shadow.num_layers // window_layers
        self.copy_stream = torch.cuda.Stream(device)
        self.ent = shadow.ent
        # Preallocate every slot's buffers for the layers it will host
        # (li ≡ slot mod n_slots), values sized to the max across those layers —
        # no mid-flight growth, so no cross-stream reallocation hazard. With the
        # entropy-coded pool, values live in ONE uint8 arena so a single batched
        # decode launch can write any mix of (slot, rel) planes by offset.
        sizing: "list[dict]" = []  # slot -> rel -> [max values numel, dtype, e0]
        for s in range(n_slots):
            szs: dict = {}
            for li in range(s, shadow.num_layers, n_slots):
                for rel in shadow.rels_at[li]:
                    e = shadow.entries[(li, rel)]
                    cur = szs.get(rel)
                    if cur is None:
                        szs[rel] = [e.values_numel, e.values_dtype, e]
                    elif cur[0] < e.values_numel:
                        cur[0] = e.values_numel
            sizing.append(szs)
        if self.ent is not None:
            total = sum(sz for szs in sizing for (sz, _d, _e) in szs.values())
            self.arena = torch.empty(total, dtype=torch.uint8, device=device)
        self.arena_off: "dict[tuple, int]" = {}  # (slot, rel) -> arena byte offset
        self.slots: "list[dict]" = []
        off = 0
        for s in range(n_slots):
            slot: dict = {}
            for rel, (numel, dt, e0) in sizing[s].items():
                slot[rel] = {"mask": torch.empty_like(e0.mask, device=device)}
                if e0.scale is not None:
                    slot[rel]["scale"] = torch.empty_like(e0.scale, device=device)
                if self.ent is not None:
                    self.arena_off[(s, rel)] = off
                    slot[rel]["values"] = self.arena[off: off + numel]
                    off += numel
                else:
                    slot[rel]["values"] = torch.empty(numel, dtype=dt, device=device)
            self.slots.append(slot)
        if self.ent is not None:
            self._build_ent_plans()
        self.ready: "dict[int, torch.cuda.Event]" = {}  # occ -> refill complete
        self.done: "dict[int, torch.cuda.Event]" = {}   # occ -> window compute complete
        self.copy_log: "list[tuple]" = []  # (bytes, start_ev, end_ev) per occurrence
        self.next_pump = 0
        self.next_use = 0
        self._pump_to(self.rw)  # initial fill: rw fresh slot sets, nothing to wait on

    def _build_ent_plans(self) -> None:
        """Static per-(window, chunk) decode plans: staging layout for the
        chunk's blobs plus per-block indirection tables (staging word offset,
        arena byte offset, symbol count) for ONE batched decode launch. Chunks
        pipeline within a window — chunk c decodes while chunk c+1's blobs
        copy — through two ping-pong staging buffers."""
        import numpy as np

        from parallm.entropy_codec import build_decode_lut

        bb = self.ent["block_bytes"]
        self.lut = torch.from_numpy(build_decode_lut(self.ent["lengths"])).to(self.device)
        n_chunks = max(1, int(self.ent["dec_chunks"]))
        self.ent_plans: "list[list[dict]]" = []
        max_stage = 0
        for w in range(self.n_windows):
            entries = [(li, rel, self.shadow.entries[(li, rel)])
                       for li in range(w * self.D, (w + 1) * self.D)
                       for rel in self.shadow.rels_at[li]]
            total = sum(e.blob.numel() for _, _, e in entries)
            groups: "list[list]" = [[]]
            acc = 0
            for item in entries:  # contiguous split, balanced by blob bytes
                if len(groups) < n_chunks and acc >= total * len(groups) / n_chunks:
                    groups.append([])
                groups[-1].append(item)
                acc += item[2].blob.numel()
            chunks = []
            for group in groups:
                woff, ooff, nsym, copies, base = [], [], [], [], 0
                for li, rel, e in group:
                    offs = e.blob_offsets.numpy().astype(np.int64)
                    nb = len(offs) - 1
                    woff.append(base // 4 + offs[:-1] // 4)  # blocks are 4B-aligned
                    ooff.append(self.arena_off[(li % self.n_slots, rel)]
                                + np.arange(nb, dtype=np.int64) * bb)
                    nsym.append(np.minimum(
                        bb, e.values_numel - np.arange(nb, dtype=np.int64) * bb
                    ).astype(np.int32))
                    copies.append((base, e.blob))
                    base += e.blob.numel()  # blob keeps its own tail pad
                chunks.append({
                    "woff": torch.from_numpy(np.concatenate(woff)).to(self.device),
                    "ooff": torch.from_numpy(np.concatenate(ooff)).to(self.device),
                    "nsym": torch.from_numpy(np.concatenate(nsym)).to(self.device),
                    "copies": copies, "bytes": base,
                })
                max_stage = max(max_stage, base)
            self.ent_plans.append(chunks)
        self.staging = [torch.empty(max_stage, dtype=torch.uint8, device=self.device)
                        for _ in range(2)]
        self.staging_u32 = [s.view(torch.uint32) for s in self.staging]
        self.decode_stream = torch.cuda.Stream(self.device)
        self.staging_free: "list[torch.cuda.Event | None]" = [None, None]
        self.chunk_seq = 0

    def slot_bufs(self, li: int, rel: str) -> dict:
        """The fixed slot buffers for (li, rel) — a pure address function, safe
        to call (and bake) inside a CUDA-graph capture."""
        return self.slots[li % self.n_slots][rel]

    def _pump_to(self, limit: int) -> None:
        """Host-enqueue whole-window refills up to occurrence ``limit``
        (exclusive): raw H2D copies on the copy stream, plus — entropy-coded
        pool — chunked blob copies pipelined with batched decode launches on
        the decode stream (chunk c decodes while chunk c+1 copies)."""
        from parallm.entropy_codec import decode_batched_gpu

        while self.next_pump < limit:
            occ = self.next_pump
            lo = (occ % self.n_windows) * self.D
            # Slot set previously held occurrence occ-rw; its last reader is
            # window occ-rw+1 (the boundary-layer re-read).
            ev_done = self.done.pop(occ - self.rw + 1) if occ >= self.rw else None
            nbytes = 0
            with torch.cuda.stream(self.copy_stream):
                if ev_done is not None:
                    self.copy_stream.wait_event(ev_done)
                t0 = torch.cuda.Event(enable_timing=True)
                t0.record(self.copy_stream)
            if self.ent is not None:
                # Blobs FIRST: their decodes then hide under the mask/scale
                # copies that follow on the copy stream (compute needs the
                # mask/scale only at the window, not for decoding).
                if ev_done is not None:  # decodes write the same slot arena
                    self.decode_stream.wait_event(ev_done)
                for chunk in self.ent_plans[occ % self.n_windows]:
                    par = self.chunk_seq % 2
                    self.chunk_seq += 1
                    with torch.cuda.stream(self.copy_stream):
                        if self.staging_free[par] is not None:
                            self.copy_stream.wait_event(self.staging_free[par])
                        stg = self.staging[par]
                        for base, blob in chunk["copies"]:
                            stg[base: base + blob.numel()].copy_(blob, non_blocking=True)
                        ev_c = torch.cuda.Event()
                        ev_c.record(self.copy_stream)
                        nbytes += chunk["bytes"]
                    with torch.cuda.stream(self.decode_stream):
                        self.decode_stream.wait_event(ev_c)
                        decode_batched_gpu(self.staging_u32[par], chunk["woff"],
                                           chunk["ooff"], chunk["nsym"],
                                           self.lut, self.arena,
                                           self.ent["block_bytes"])
                        ev_d = torch.cuda.Event()
                        ev_d.record(self.decode_stream)
                        self.staging_free[par] = ev_d
            with torch.cuda.stream(self.copy_stream):
                for li in range(lo, lo + self.D):
                    slot = self.slots[li % self.n_slots]
                    for rel in self.shadow.rels_at[li]:
                        e = self.shadow.entries[(li, rel)]
                        bufs = slot[rel]
                        bufs["mask"].copy_(e.mask, non_blocking=True)
                        nbytes += e.mask.nbytes
                        if e.blob is None:  # raw plane; ent decodes instead
                            bufs["values"][: e.values_numel].copy_(
                                e.values, non_blocking=True)
                            nbytes += e.values.nbytes
                        if e.scale is not None:
                            bufs["scale"].copy_(e.scale, non_blocking=True)
                            nbytes += e.scale.nbytes
            if self.ent is None:
                with torch.cuda.stream(self.copy_stream):
                    t1 = torch.cuda.Event(enable_timing=True)
                    t1.record(self.copy_stream)
            else:
                with torch.cuda.stream(self.copy_stream):
                    ev_ce = torch.cuda.Event()  # mask/scale copies complete
                    ev_ce.record(self.copy_stream)
                with torch.cuda.stream(self.decode_stream):
                    self.decode_stream.wait_event(ev_ce)
                    t1 = torch.cuda.Event(enable_timing=True)
                    t1.record(self.decode_stream)  # implies copies AND decodes
            self.ready[occ] = t1
            if len(self.copy_log) < 4096:
                self.copy_log.append((nbytes, t0, t1))
            self.next_pump += 1

    def begin_window(self) -> None:
        """Eager, on the compute stream, just before the window runs/replays:
        gate the window's kernels on its slot refill."""
        torch.cuda.current_stream().wait_event(self.ready.pop(self.next_use))
        self.next_use += 1

    def end_window(self) -> None:
        """Eager, right after the window's kernels are enqueued: record the
        compute-done event a later refill of these slots waits on, then extend
        the pump horizon to occurrence ``occ+rw-1``."""
        occ = self.next_use - 1
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        self.done[occ] = ev
        self._pump_to(occ + self.rw)

    def copy_stats(self) -> "list[tuple[int, float]]":
        """(bytes, ms) per pumped window occurrence. Synchronizes the device —
        call after generation."""
        torch.cuda.synchronize(self.device)
        return [(b, t0.elapsed_time(t1)) for b, t0, t1 in self.copy_log]

    def device_bytes(self) -> int:
        """Device-resident ring footprint: slot buffers, plus the arena /
        staging / LUT / decode-plan tables in ent mode (slot "values" alias
        the arena and are not double counted)."""
        total = sum(
            t.nbytes for slot in self.slots for bufs in slot.values()
            for k, t in bufs.items() if not (self.ent is not None and k == "values"))
        if self.ent is not None:
            total += self.arena.nbytes + sum(s.nbytes for s in self.staging) + self.lut.nbytes
            total += sum(c[k].nbytes for w in self.ent_plans for c in w
                         for k in ("woff", "ooff", "nsym"))
        return total


def _load_ent(path: str, dec_chunks: int):
    """Load a repack_replicas_entropy.py artifact: the shared table + per
    (layer, rel) blob/offsets/raw_len planes."""
    from safetensors import safe_open

    planes: dict = {}
    lengths = None
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        assert meta.get("codec") == "huff12", f"unknown pool codec: {meta}"
        for key in f.keys():
            parts = key.split("|")
            if parts[0] == "__table__":
                lengths = f.get_tensor(key).numpy()
                continue
            li, rel, field = int(parts[0]), parts[1], parts[2]
            d = planes.setdefault((li, rel), {})
            t = f.get_tensor(key)
            d[field] = int(t.item()) if field == "raw_len" else t
    ent = {"lengths": lengths, "block_bytes": int(meta["block_bytes"]),
           "dec_chunks": dec_chunks}
    return ent, planes


def _unpack_nibbles_torch(packed: torch.Tensor, n: int) -> torch.Tensor:
    u = torch.stack(((packed >> 4) & 0xF, packed & 0xF), dim=-1).reshape(-1)
    return u[:n]


def assemble_codes(packs: "list[dict]", track_nnz=None) -> "tuple[torch.Tensor, int]":
    """The engine-order codes byte plane for one (layer, rel) across all N
    tracks: per-track nibbles re-packed without the per-track pad (bits==4) or
    the tracks' int8 codes concatenated. Returns ``(uint8 plane, nnz)``.

    Single source of truth shared by ``_RelEntry`` and the entropy repacker —
    the compressed artifact must encode EXACTLY the bytes the ring streams."""
    import numpy as np

    bits = int(packs[0]["bits"].item())
    if bits == 4:
        if track_nnz is None:
            track_nnz = [
                int(np.unpackbits(p["mask"].numpy()).sum()) for p in packs
            ]
        per_track = [
            _unpack_nibbles_torch(p["codes"], int(track_nnz[k]))
            for k, p in enumerate(packs)
        ]
        flat = torch.cat(per_track)
        a = flat.numpy()
        if a.size % 2:
            a = np.append(a, np.uint8(0))
        return torch.from_numpy((a[0::2] << 4) | (a[1::2])), flat.numel()
    codes = torch.cat([p["codes"] for p in packs])
    return codes, codes.numel()


class PackedShadow(GridStacked):
    """Shadow provider backed by a packed replica pool file: HF layer skeletons
    (meta-built, exact non-Linear params from the track shards) hold the small
    diverged params; every Linear is computed by ``blinear`` straight from the
    packed pool (fused kernel at decode M, dense-unpack scratch fallback at
    prefill/eval M). Residency = packed pool bytes + one dense layer of
    scratch — the dense pool (~N× layer bytes × layers) never exists. The
    skeletons' Linear weights stay on meta: calling one fails loudly.
    """

    def __init__(self, pool_path: str, tracks_dir: str, student: PTWrappedModel,
                 device, dtype: torch.dtype = torch.bfloat16,
                 stream_ring_layers: "int | None" = None, fused: bool = False,
                 stream_window_layers: "int | None" = None,
                 ent_path: "str | None" = None, dec_chunks: int = 4):
        import os

        from safetensors import safe_open

        from parallm.model.replica_pack import load_replica_pool

        self.ent, ent_planes = None, {}
        if ent_path is not None:
            assert stream_ring_layers is not None, \
                "the entropy-coded pool decodes into ring slots — streamed only"
            self.ent, ent_planes = _load_ent(ent_path, dec_chunks)
        pool, self.meta = load_replica_pool(pool_path)
        self.n_tracks = max(k[0] for k in pool) + 1
        num_layers = max(k[1] for k in pool) + 1
        self.num_layers = num_layers
        self.device, self.dtype, self.fused = device, dtype, fused
        rels_at = {}  # layer_idx -> sorted rels
        for (_t, li, rel) in pool:
            rels_at.setdefault(li, set()).add(rel)
        self.rels_at = {li: sorted(rs) for li, rs in rels_at.items()}

        # Batched pool entries + shared scratch (one dense layer, all tracks).
        # Streamed mode keeps the packed entries in pinned host DRAM and pulls
        # them through a device ring; resident mode keeps them on device.
        # Fused mode computes straight from the packed bits (the GEMV kernel),
        # so scratch only materializes lazily for large-M calls (eval/prefill).
        pin = stream_ring_layers is not None
        self.entries = {}  # (li, rel) -> _RelEntry
        self.scratch = {}  # rel -> dense [N, out, in], lazily allocated
        self._f32 = {}     # rel -> fp32 tmp (qwanda rels only)
        self._scratch_mark = {}  # rel -> layer whose weights scratch holds
        self._stacked = {}       # (li, path) -> stacked small params [N, ...]
        for li, rels in self.rels_at.items():
            for rel in rels:
                e = _RelEntry([pool[(t, li, rel)] for t in range(self.n_tracks)],
                              device, pin=pin,
                              ent=ent_planes[(li, rel)] if self.ent else None)
                if fused:
                    assert e.shape[2] % 8 == 0, (rel, e.shape)  # byte-aligned rows
                self.entries[(li, rel)] = e

        # Exact non-Linear params per (track, layer) from the shards.
        smalls: "dict[tuple[int, str], torch.Tensor]" = {}
        for t in range(self.n_tracks):
            with safe_open(os.path.join(tracks_dir, f"track_{t}.safetensors"),
                           framework="pt", device="cpu") as f:
                for key in f.keys():
                    if not key.startswith("layers."):
                        continue
                    parts = key.split(".")
                    li, sub = int(parts[1]), ".".join(parts[2:])
                    rel = sub[: -len(".weight")] if sub.endswith(".weight") else sub
                    if (t, li, rel) in pool:
                        continue  # a degraded Linear — lives in the pool
                    smalls[(t, ".".join(parts[1:]))] = f.get_tensor(key).to(device=device, dtype=dtype)

        # Meta-built skeletons; every non-Linear param must resolve to a shard
        # small (pool Linears are computed via blinear and stay on meta —
        # anything else left on meta fails loudly at first use).
        layer_cls = type(student.text_models[0].layers[0])
        cfg = student.per_track_text_config
        self.grid = []
        for t in range(self.n_tracks):
            track_layers = []
            for li in range(num_layers):
                with torch.device("meta"):
                    layer = layer_cls(cfg, li)
                for name, _p in list(layer.named_parameters()):
                    rel = name[: -len(".weight")] if name.endswith(".weight") else name
                    if (li, rel) not in self.entries:
                        _assign_param(layer, name, smalls[(t, f"{li}.{name}")])
                layer.eval()
                track_layers.append(layer)
            self.grid.append(track_layers)
        if stream_ring_layers is not None:
            assert stream_window_layers, "streamed pool needs the sync-window size in layers"
            self.ring = _Ring(self, stream_ring_layers, device, stream_window_layers)
        else:
            self.ring = None

    def layers(self, i: int) -> "list":
        return [track[i] for track in self.grid]

    def blinear(self, li: int, rel: str, x: torch.Tensor,
                per_track: bool = False) -> torch.Tensor:
        """All N tracks' packed Linear at ``rel`` in one op: ``[M, in]`` shared
        x (``x_tstride=0`` — every track reads the same carry-derived input) or
        ``[N, M, in]`` per-track x → ``[N, M, out]``. Decode-M uses the fused
        kernel; larger M (prefill/eval) uses the dense-unpack scratch."""
        e = self.entries[(li, rel)]
        n, o, i = e.shape
        M = x.shape[-2]
        if self.fused and x.is_cuda and M <= _GEMV_MAX_M:
            from parallm.packed_gemv import _launch

            bufs = self.active_bufs(li, rel)
            return _launch(x.contiguous(), bufs["mask"], bufs["values"],
                           bufs.get("scale"), e.block_start, n, M, o, i, e.bits,
                           M * i if per_track else 0)
        return torch.matmul(x, self.dense_weights(li, rel).transpose(-2, -1))

    # ---- packed-buffer plumbing (blinear reads through these) ----
    def active_bufs(self, li: int, rel: str) -> dict:
        """The device-resident packed buffers for (li, rel): the layer's fixed
        ring slot when streamed, else the entry's own tensors."""
        if self.ring is not None:
            return self.ring.slot_bufs(li, rel)
        e = self.entries[(li, rel)]
        bufs = {"mask": e.mask, "values": e.values}
        if e.scale is not None:
            bufs["scale"] = e.scale
        return bufs

    def _alloc_scratch(self, rel: str, e: _RelEntry) -> None:
        self.scratch[rel] = torch.empty(e.shape, dtype=self.dtype, device=self.device)
        if e.bits is not None:
            self._f32[rel] = torch.empty(e.shape, dtype=torch.float32, device=self.device)

    def dense_weights(self, li: int, rel: str) -> torch.Tensor:
        """The dense degraded weights ``[N, out, in]`` from a lazily allocated
        shared scratch, unpacked once per (layer, rel) visit."""
        e = self.entries[(li, rel)]
        if rel not in self.scratch:
            self._alloc_scratch(rel, e)
        assert e.shape == self.scratch[rel].shape, (li, rel)
        if self._scratch_mark.get(rel) != li:
            bufs = self.active_bufs(li, rel)
            e.unpack_from(bufs["mask"], bufs["values"][: e.values_numel],
                          self.scratch[rel], self._f32.get(rel), bufs.get("scale"))
            self._scratch_mark[rel] = li
        return self.scratch[rel]


def _assign_param(module, dotted: str, tensor: torch.Tensor) -> None:
    *path, leaf = dotted.split(".")
    for p in path:
        module = getattr(module, p)
    setattr(module, leaf, nn.Parameter(tensor, requires_grad=False))


# --------------------------------------------------------------------------- #
# The track-batched forward: N tracks' sublayers as ONE batched op chain
# (track dim = batch dim) instead of N per-track HF module calls — the
# module-internal tiny-kernel tail was the measured decode floor. It serves
# BOTH streams of the replay: the N shadow tracks (via PackedShadow, summed
# delta advances the carry) and the K own tracks (via a DenseShadow over the
# rank's exact modules, per-track deltas). All inputs are track-shared by
# construction (the carry / the synced base), so only per-track weights ride
# the batch dim. Every rank runs the same shadow chain on the same inputs,
# preserving the rank-bit-identical carry.
# --------------------------------------------------------------------------- #
_HF_MOD = None


def _hf():
    """The Qwen3.5 modeling module, resolved at first call so its FLA/CUDA
    kernel bindings (monkeypatched to None on CPU test runs) stay authoritative.

    Memoized in a module global rather than imported per call because `_batched_attn`
    runs INSIDE a compiled region: an ``import`` statement is a hard graph break, and
    dynamo split every attention half into two fragments reached through a resume
    trampoline. Past the first call this is a global read of a module object, which
    dynamo constant-folds. `enable_batched_compile` warms it so the very first trace
    is already past the import.

    The GDN fold genuinely needs THIS family's module. `_batched_attn` takes only
    ``apply_rotary_pos_emb`` and ``repeat_kv`` from it, which are the standard
    rotate-half rope and GQA expand — byte-identical across every Llama-derived
    family, Qwen3 included. That reuse is pinned by the batched-equivalence rails,
    which compare the fold against the family's OWN modules."""
    global _HF_MOD
    if _HF_MOD is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m

        _HF_MOD = m
    return _HF_MOD


def _norm_eps(norm) -> float:
    """An RMSNorm module's epsilon. HF is not consistent about the attribute name —
    Qwen3.5 stores ``eps``, Qwen3 stores ``variance_epsilon``."""
    eps = getattr(norm, "eps", None)
    return float(eps if eps is not None else norm.variance_epsilon)


def _rms_tracks(
    x: torch.Tensor, w: torch.Tensor, eps: float, centered: bool
) -> torch.Tensor:
    """Per-track RMSNorm on ``x [N, ..., d]``.

    ``w`` is ``[N, d]`` when each of the N streams owns one copy, or ``[N, F, d]``
    when a stream is a GROUP of F members that each kept their own — the middle
    dims are right-aligned against x's, so the broadcast lands on the head axis
    either way. The ``[N, d]`` form reproduces the previous view exactly.

    ``centered`` is `AttnOps.centered_norm`: Qwen3.5 applies ``(1 + w)``, Qwen3 (and
    the plain Llama form) applies ``w`` directly.
    """
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    wf = 1.0 + w.float() if centered else w.float()
    wv = wf.view(w.shape[0], *([1] * (x.ndim - w.ndim)), *w.shape[1:])
    return (out * wv).type_as(x)


def _batched_attn(shadow, li: int, x, position_embeddings, attention_mask, cache):
    """All N tracks' full-attention mixers → per-track deltas ``[N, B, T, h]``.
    The attention forward with the track dim folded into batch. ``x`` is the shared
    residual ``[B, T, h]`` or a per-track one ``[N, B, T, h]`` (see `_norm_flat`);
    everything downstream of the projections is per-track either way, because
    attention never mixes heads.

    Family-independent except for `shadow.attn_ops` (see `parallm.adapters.AttnOps`):
    Qwen3.5 doubles ``q_proj`` into ``[q | gate]`` and gates the output, Qwen3 does
    not. Head dims, scaling, kv groups and eps all come off the skeleton layer."""
    m = _hf()
    ops = shadow.attn_ops
    layer0 = shadow.grid[0][li]
    attn = layer0.self_attn
    N, B, T = shadow.n_tracks, x.shape[-3], x.shape[-2]
    d = attn.head_dim
    h2 = _norm_flat(layer0.input_layernorm, x, B, T)
    qw = shadow.blinear(li, "self_attn.q_proj", h2)
    if ops.gated_q:
        q, gate = qw.view(N, B, T, -1, 2 * d).chunk(2, dim=-1)
    else:
        q, gate = qw.view(N, B, T, -1, d), None
    q = _rms_tracks(q, shadow.stacked(li, "self_attn.q_norm.weight"),
                    _norm_eps(attn.q_norm), ops.centered_norm)
    k = shadow.blinear(li, "self_attn.k_proj", h2).view(N, B, T, -1, d)
    k = _rms_tracks(k, shadow.stacked(li, "self_attn.k_norm.weight"),
                    _norm_eps(attn.k_norm), ops.centered_norm)
    v = shadow.blinear(li, "self_attn.v_proj", h2)
    q = q.reshape(N * B, T, -1, d).transpose(1, 2)
    k = k.reshape(N * B, T, -1, d).transpose(1, 2)
    v = v.reshape(N * B, T, -1, d).transpose(1, 2)
    cos, sin = position_embeddings
    # Condition on the ROTARY's own batch dim, not on B: a family whose
    # `_resolve_position_ids` returns one shared row (Qwen3, matching
    # `Qwen3Model.forward`) hands back `[1, T, d]`, which already broadcasts over the
    # track-major batch. Only a genuinely per-row cos has to be tiled, and then row
    # n*B+b must read position row b — which is what `repeat` gives.
    if cos.shape[0] > 1:
        cos, sin = cos.repeat(N, 1, 1), sin.repeat(N, 1, 1)
    q, k = m.apply_rotary_pos_emb(q, k, cos, sin)
    if cache is not None:
        # Static KV: attention spans the whole s_max buffer; a boolean mask
        # from the device position keeps it causal and hides unwritten slots
        # (shape-static, so CUDA graphs can capture the step).
        k, v = cache.update_kv(li, k, v)
        qpos = cache.pos_t + torch.arange(T, device=k.device)
        mask = (torch.arange(k.shape[2], device=k.device)[None, :]
                <= qpos[:, None]).view(1, 1, T, -1)
    else:
        mask = attention_mask
        if mask is not None and mask.shape[0] > 1:
            mask = mask.repeat(N, 1, 1, 1)
    o = nn.functional.scaled_dot_product_attention(
        q, m.repeat_kv(k, attn.num_key_value_groups),
        m.repeat_kv(v, attn.num_key_value_groups),
        attn_mask=mask, scale=attn.scaling,
        is_causal=cache is None and mask is None and T > 1,
    )
    o = o.transpose(1, 2).reshape(N, B * T, -1)
    if gate is not None:
        o = o * torch.sigmoid(gate.reshape(N, B * T, -1))
    y = shadow.blinear(li, "self_attn.o_proj", o, per_track=True)
    return y.view(N, B, T, -1)


def _batched_gdn(shadow, li: int, x, attention_mask, cache):
    """All N tracks' gated-delta-net mixers → per-track deltas. Mirrors
    ``Qwen3_5GatedDeltaNet.forward``; tracks fold into the conv CHANNEL dim
    (depthwise conv keys weights by channel, not batch) and into the BATCH dim of
    the delta-rule kernels. ``x`` is shared ``[B, T, h]`` or per-track
    ``[N, B, T, h]`` (see `_norm_flat`)."""
    m = _hf()
    layer0 = shadow.grid[0][li]
    gdn = layer0.linear_attn
    N, B, T = shadow.n_tracks, x.shape[-3], x.shape[-2]
    Hv, Hk = gdn.num_v_heads, gdn.num_k_heads
    dk, dv = gdn.head_k_dim, gdn.head_v_dim
    C, K = gdn.conv_dim, gdn.conv_kernel_size
    x = m.apply_mask_to_padding_states(x, attention_mask)
    h2 = _norm_flat(layer0.input_layernorm, x, B, T)
    qkv = shadow.blinear(li, "linear_attn.in_proj_qkv", h2)
    z = shadow.blinear(li, "linear_attn.in_proj_z", h2)
    b = shadow.blinear(li, "linear_attn.in_proj_b", h2)
    a = shadow.blinear(li, "linear_attn.in_proj_a", h2)

    conv_w = shadow.stacked(li, "linear_attn.conv1d.weight").reshape(N * C, K)
    xc = qkv.view(N, B, T, C).permute(1, 0, 3, 2).reshape(B, N * C, T)
    state = cache.conv.get(li) if cache is not None else None
    logged = cache is not None and cache.log is not None
    if state is not None and T == 1 and not logged:
        conv_update = m.causal_conv1d_update or m.torch_causal_conv1d_update
        xc = conv_update(xc, state, conv_w, None, gdn.activation)
    else:
        if state is not None:  # chunked continuation: prepend the conv context
            xc = torch.cat([state, xc], dim=-1)
        if logged:
            # Draft-verify: defer the commit — the rollback recuts the conv
            # window from this raw pre-conv context at the accepted length.
            cache.log.setdefault(li, {}).update(conv_ctx=xc, kc=K)
        elif cache is not None:
            ctx = nn.functional.pad(xc, (K - xc.shape[-1], 0))
            if li in cache.conv:
                cache.conv[li].copy_(ctx)  # keep the address graphs captured
            else:
                cache.conv[li] = ctx
        L = xc.shape[-1]
        if m.causal_conv1d_fn is not None:
            xc = m.causal_conv1d_fn(x=xc, weight=conv_w, bias=None,
                                    activation=gdn.activation)
        else:
            xc = nn.functional.silu(nn.functional.conv1d(
                xc, conv_w.unsqueeze(1), padding=K - 1, groups=N * C)[..., :L])
        if state is not None:
            xc = xc[..., -T:]
    qkv = xc.view(B, N, C, T).permute(1, 0, 3, 2)  # [N, B, T, C]
    q, k, v = torch.split(qkv, [Hk * dk, Hk * dk, Hv * dv], dim=-1)
    q = q.reshape(N * B, T, Hk, dk)
    k = k.reshape(N * B, T, Hk, dk)
    v = v.reshape(N * B, T, Hv, dv)
    beta = b.view(N * B, T, Hv).sigmoid()
    A_log = shadow.stacked(li, "linear_attn.A_log")
    dt_bias = shadow.stacked(li, "linear_attn.dt_bias")
    g = -A_log.float().exp().view(N, 1, 1, Hv) * nn.functional.softplus(
        a.view(N, B, T, Hv).float() + dt_bias.view(N, 1, 1, Hv))
    g = g.reshape(N * B, T, Hv)
    if Hv // Hk > 1:
        q = q.repeat_interleave(Hv // Hk, dim=2)
        k = k.repeat_interleave(Hv // Hk, dim=2)
    rec = cache.rec.get(li) if cache is not None else None
    if rec is not None and T == 1 and not logged:
        fn = m.fused_recurrent_gated_delta_rule or m.torch_recurrent_gated_delta_rule
    else:
        fn = m.chunk_gated_delta_rule or m.torch_chunk_gated_delta_rule
    core, rec_out = fn(q, k, v, g=g, beta=beta, initial_state=rec,
                       output_final_state=cache is not None and not logged,
                       use_qk_l2norm_in_kernel=True)
    if logged:
        # Draft-verify: ``rec`` keeps the pre-chunk state; the rollback
        # replays these saved recurrence inputs over the accepted prefix.
        cache.log.setdefault(li, {})["qkvgb"] = (q, k, v, g, beta)
    elif cache is not None:
        cache.update_rec(li, rec_out)
    # Gated RMSNorm (per-track weight, NOT zero-centered), norm before gate —
    # the torch reference math of Qwen3_5RMSNormGated.
    normw = shadow.stacked(li, "linear_attn.norm.weight")
    cf = core.view(N, B, T, Hv, dv).float()
    cn = cf * torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + gdn.layer_norm_epsilon)
    cn = normw.view(N, 1, 1, 1, dv) * cn.to(core.dtype)
    out = (cn * nn.functional.silu(z.view(N, B, T, Hv, dv).float())).to(core.dtype)
    y = shadow.blinear(li, "linear_attn.out_proj", out.reshape(N, B * T, Hv * dv),
                       per_track=True)
    return y.view(N, B, T, -1)


def _batched_mlp(shadow, li: int, x):
    """All N tracks' MLPs → per-track deltas. ``x`` is the shared residual
    ``[B, T, h]`` or a per-track one ``[N, B, T, h]`` (see `_norm_flat`)."""
    layer0 = shadow.grid[0][li]
    N, B, T = shadow.n_tracks, x.shape[-3], x.shape[-2]
    h2 = _norm_flat(layer0.post_attention_layernorm, x, B, T)
    g = shadow.blinear(li, "mlp.gate_proj", h2)
    u = shadow.blinear(li, "mlp.up_proj", h2)
    y = shadow.blinear(li, "mlp.down_proj", layer0.mlp.act_fn(g) * u, per_track=True)
    return y.view(N, B, T, -1)


# --------------------------------------------------------------------------- #
# Caches: plain batched tensors (track dim folded into batch/channels,
# matching the batched forward) — one instance for the K own tracks, one for
# the N shadow tracks. No HF cache classes anywhere in decode. Every buffer is
# STATIC (preallocated KV written by position, in-place conv/recurrent states)
# and the write position is a device tensor, so a captured CUDA graph replays
# correctly as ``pos_t`` advances.
# --------------------------------------------------------------------------- #
@dataclass
class BatchedShadowCache:
    pos_t: torch.Tensor          # device scalar: tokens already cached (shared)
    s_max: int                   # KV capacity in tokens
    kv: dict = None              # li -> (k, v) [N*B, kv_heads, s_max, d]
    conv: dict = None            # li -> [B, N*conv_dim, K], updated in place
    rec: dict = None             # li -> [N*B, Hv, dk, dv], updated in place
    log: dict = None             # draft-verify verify chunk: li -> rollback
                                 # record (raw conv context + recurrence inputs);
                                 # while set, GDN state commits are DEFERRED to
                                 # _rollback_gdn at the accepted prefix length

    def __post_init__(self):
        self.kv, self.conv, self.rec = {}, {}, {}

    def update_kv(self, li: int, k: torch.Tensor, v: torch.Tensor):
        buf = self.kv.get(li)
        if buf is None:
            shape = (k.shape[0], k.shape[1], self.s_max, k.shape[3])
            buf = (k.new_zeros(shape), k.new_zeros(shape))
            self.kv[li] = buf
        idx = self.pos_t + torch.arange(k.shape[2], device=k.device)
        buf[0].index_copy_(2, idx, k)
        buf[1].index_copy_(2, idx, v)
        return buf

    def update_rec(self, li: int, rec_out: torch.Tensor) -> None:
        if li in self.rec:
            self.rec[li].copy_(rec_out)  # keep the static buffer graphs captured
        else:
            self.rec[li] = rec_out


@dataclass
class EngineCaches:
    own: BatchedShadowCache      # the K local tracks, batched
    shadow: BatchedShadowCache   # all N shadow tracks, batched
    pos_t: torch.Tensor          # device mirror of ``pos`` (graph-readable)
    pos: int = 0                 # tokens already cached (authoritative for positions)

    @classmethod
    def build(cls, student: PTWrappedModel, s_max: int = 512) -> "EngineCaches":
        dev = next(student.parameters()).device
        pos_t = torch.zeros((), dtype=torch.long, device=dev)
        return cls(own=BatchedShadowCache(pos_t, s_max),
                   shadow=BatchedShadowCache(pos_t, s_max), pos_t=pos_t)


def _stall(latency_s: float, timing: "dict | None") -> None:
    """One cross-node comm round: simulated wire latency + accounting."""
    if timing is not None:
        timing["rounds"] = timing.get("rounds", 0) + 1
    if latency_s > 0:
        time.sleep(latency_s)


# --------------------------------------------------------------------------- #
# The replay chunk forward.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def replay_chunk(
    student: PTWrappedModel,
    shadow,
    input_ids: torch.LongTensor,
    sync_indices: Sequence[int],
    caches: "EngineCaches | None" = None,
    attention_mask: torch.Tensor | None = None,
    latency_s: float = 0.0,
    timing: "dict | None" = None,
    graphs: "dict | None" = None,
):
    """Advance every track through one chunk of tokens (whole prompt at prefill,
    1 token per decode step). Returns full-chunk logits ``(B, T, V)`` on the
    head rank, ``None`` on peers. All ranks must call in lockstep.

    ``graphs`` (a dict owned by the caller, one per generation) turns on
    CUDA-graph windows for cached 1-token decode: each inter-sync segment is
    captured once (after one eager warmup step) and replayed thereafter, so the
    whole window costs one launch. Comms (embed collective, boundary
    all-reduces), the lm_head, and sampling stay eager between replays; every
    tensor a graph touches is static by construction (see BatchedShadowCache)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

    tm0 = student.text_models[0]
    num_layers = len(tm0.layers)
    last = num_layers - 1
    sync_set = set(int(i) for i in sync_indices) - {last}

    # The K local tracks run through the same batched forward as the shadow,
    # via a provider over their exact modules (stacked-weight views).
    own = getattr(student, "_engine_own_provider", None)
    if own is None:
        own = DenseShadow([list(tm.layers) for tm in student.text_models])
        student._engine_own_provider = own

    # Embed on the owner track; the collective broadcasts h to every rank.
    h = student.embed(input_ids)
    _stall(latency_s, timing)

    # Scaffolding, offset by the cached position (mirrors Qwen3_5TextModel.forward).
    B, T = input_ids.shape
    past = caches.pos if caches is not None else 0
    pos = torch.arange(past, past + T, device=h.device)
    position_ids = pos.view(1, 1, -1).expand(4, B, -1)
    text_position_ids = position_ids[0]
    if past > 0:
        # Cached continuation: no mask is built here — _batched_attn derives
        # the offset-causal mask from the cache position (1-token decode and
        # k+1-token draft-verify chunks alike), and the GDN layers prepend
        # their cached conv/recurrent context.
        causal_mask = None
        linear_attn_mask = None
    else:
        causal_mask = create_causal_mask(
            config=tm0.config, inputs_embeds=h, attention_mask=attention_mask,
            past_key_values=None, position_ids=text_position_ids,
        )
        linear_attn_mask = (None if attention_mask is None
                            or bool(torch.all(attention_mask == 1))
                            else attention_mask)
    position_embeddings = tm0.rotary_emb(h, position_ids[1:])

    own_cache = caches.own if caches is not None else None
    shadow_cache = caches.shadow if caches is not None else None
    if caches is not None:
        caches.pos_t.fill_(past)  # the device mirror graphs read

    # Optional per-window breakdown: device-synced host clocks at each segment
    # boundary. The synchronize serializes the overlap the normal path enjoys
    # (kernels running through the sleep), so the profiled total reads HIGHER
    # than the unprofiled wall — it is the serial cost decomposition.
    prof = timing is not None and timing.get("profile") and h.is_cuda

    def _mark():
        torch.cuda.synchronize()
        return time.perf_counter()

    if prof:
        win_ms: "list[float]" = []
        sync_ms: "list[float]" = []
        seg_t = _mark()

    def mixer(provider, i, x, mask, cache):
        if tm0.config.layer_types[i] == "linear_attention":
            return _batched_gdn(provider, i, x, mask, cache)
        return _batched_attn(provider, i, x, position_embeddings, mask, cache)

    # The layer walk, segmented at the sync events: segment w runs the
    # PREVIOUS boundary's post-sync MLPs, the full layers up to boundary w, and
    # boundary w's own mixer (post-attn placement; the boundary-layer shadow
    # mixer is skipped — its delta is always discarded by the carry reset).
    # The last boundary is the final layer, synced post-MLP for the head.
    boundaries = sorted(sync_set) + [last]
    K = len(student.text_models)
    ring = getattr(shadow, "ring", None)
    if ring is not None:  # the ring pumps whole windows — schedules must agree
        assert boundaries == list(range(ring.D - 1, num_layers, ring.D)), \
            (boundaries, ring.D)

    # Single-track dense replay (the drafter path): own provider IS the shadow
    # and there is exactly one exact track, so the carry chain alone computes
    # the dense forward — skip the duplicate own-chain kernels (halves the
    # per-step graph).
    n1 = K == 1 and shadow.n_tracks == 1 and shadow is own

    def seg_fn(w, carry, base, track_h):
        b = boundaries[w]
        if w > 0:
            pb = boundaries[w - 1]
            if n1:
                carry = base + _batched_mlp(shadow, pb, base).sum(0)
            else:
                # Exact MLP reads the TRUE residual; shadow MLP advances the
                # carry from it with degraded weights.
                y_own = _batched_mlp(own, pb, base)
                track_h = [base + y_own[k] for k in range(K)]
                carry = base + _batched_mlp(shadow, pb, base).sum(0)
        for i in range(boundaries[w - 1] + 1 if w > 0 else 0, b + 1):
            mask = (linear_attn_mask
                    if tm0.config.layer_types[i] == "linear_attention" else causal_mask)
            if n1:
                carry = carry + mixer(shadow, i, carry, mask, shadow_cache).sum(0)
                if i < b or b == last:
                    carry = carry + _batched_mlp(shadow, i, carry).sum(0)
                continue
            # Own exact mixers on the carry (write own KV/GDN state), K-batched.
            y_own = mixer(own, i, carry, mask, own_cache)
            track_h = [th + y_own[k] for k, th in enumerate(track_h)]
            if i < b or b == last:  # full layer (the last layer syncs post-MLP)
                # Shadow mixer: ALL N degraded copies advance the carry comm-free.
                carry_attn = carry + mixer(shadow, i, carry, mask, shadow_cache).sum(0)
                y_own = _batched_mlp(own, i, carry_attn)
                track_h = [th + y_own[k] for k, th in enumerate(track_h)]
                carry = carry_attn + _batched_mlp(shadow, i, carry_attn).sum(0)
        return (carry, [carry]) if n1 else (carry, track_h)

    # CUDA-graph windows: cached decode at a FIXED chunk width — 1-token steps
    # and, for draft-verify, the k+1-token verify chunk (``graphs["chunk_T"]``).
    # One eager warmup step per width (allocations, JIT/autotune, cuBLAS
    # workspaces, lazy KV/conv/rec buffers) before capture; statics per width.
    use_g = (graphs is not None and caches is not None and past > 0
             and (T == 1 or T == graphs.get("chunk_T")) and h.is_cuda)
    if use_g and not graphs.get(("warm", T)):
        graphs[("warm", T)] = True
        use_g = False
    if use_g:
        st = graphs.get(("static", T))
        if st is None:
            st = graphs[("static", T)] = {
                "carry": torch.empty_like(h), "base": torch.empty_like(h),
                "track_h": [torch.empty_like(h) for _ in range(K)],
                "cos": position_embeddings[0].clone(),
                "sin": position_embeddings[1].clone(),
                "graphs": [None] * len(boundaries),
            }
        else:
            st["cos"].copy_(position_embeddings[0])
            st["sin"].copy_(position_embeddings[1])
        position_embeddings = (st["cos"], st["sin"])
        st["carry"].copy_(h)
        st["base"].copy_(h)
        for tk in st["track_h"]:
            tk.copy_(h)
        carry, base, track_h = st["carry"], st["base"], st["track_h"]
    else:
        carry = h                # the shared shadow estimate
        base = h                 # true residual, valid at boundaries
        track_h = [h] * K        # base + own exact deltas since

    for w in range(len(boundaries)):
        if ring is not None:
            ring.begin_window()  # eager: gate this window's kernels on its refill
        if use_g:
            g = st["graphs"][w]
            if g is None:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):  # capture records, replay executes
                    c2, th2 = seg_fn(w, carry, base, track_h)
                    carry.copy_(c2)
                    for tk, t2 in zip(track_h, th2):
                        tk.copy_(t2)
                st["graphs"][w] = g
            g.replay()
        else:
            carry, track_h = seg_fn(w, carry, base, track_h)
        if ring is not None:
            ring.end_window()  # eager: record done, extend the pump horizon
        if prof:
            t = _mark()
            win_ms.append((t - seg_t) * 1e3)
        nb = student.sync_module(track_h, base)  # ONE all-reduce of exact deltas
        if use_g:
            base.copy_(nb)  # keep the static pointer the graphs captured
        else:
            base = nb
        _stall(latency_s, timing)
        if prof:
            seg_t = _mark()
            sync_ms.append((seg_t - t) * 1e3)

    if prof:
        timing.setdefault("window_ms", []).append(win_ms)
        timing.setdefault("sync_ms", []).append(sync_ms)

    if caches is not None:
        caches.pos += T
    if student.lm_head is None:
        return None
    return student.lm_head(tm0.norm(base))


def make_engine_forward_fn(student: PTWrappedModel, shadow, sync_indices):
    """``ForwardFn`` for ``eval/lm_eval_adapter.PTLM``: whole-sequence engine
    replay, so lm-eval scores the exact deployed forward."""
    def _fn(input_ids, attention_mask):
        return replay_chunk(student, shadow, input_ids, sync_indices,
                            attention_mask=attention_mask)
    return _fn


# --------------------------------------------------------------------------- #
# Single-stream generation.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(
    student: PTWrappedModel,
    shadow,
    input_ids: torch.LongTensor,
    sync_indices: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    latency_s: float = 0.0,
    timing: "dict | None" = None,
    use_cuda_graphs: bool = False,
) -> torch.LongTensor:
    """Prefill + token-by-token decode. Returns the generated ids ``(B, new)``
    on the head rank (peers return the placeholder ids they decoded with).

    Peers never see the sampled ids: they hold no ``embed_tokens``, contribute
    zeros to the embed collective, and only need chunk *shapes*, so every rank
    decodes exactly ``max_new_tokens``.
    """
    # ponytail: no EOS early-stop — peers can't observe it without an extra
    # broadcast round; add a stop-flag round if interactive use ever needs it.
    caches = EngineCaches.build(student,
                                s_max=input_ids.shape[1] + max_new_tokens)
    # Graphs compose with the stream ring: slots sit at fixed addresses the
    # captures bake, and all ring host/event work runs in the eager boundary
    # region (geometry enforced at _Ring construction).
    graphs: "dict | None" = {} if use_cuda_graphs else None
    per_token_ms: "list[float]" = []

    t0 = time.perf_counter()
    logits = replay_chunk(student, shadow, input_ids, sync_indices, caches,
                          latency_s=latency_s, timing=timing)
    prefill_ms = (time.perf_counter() - t0) * 1e3

    out = []
    tok = torch.zeros((input_ids.shape[0], 1), dtype=torch.long,
                      device=input_ids.device)
    for step_i in range(max_new_tokens):
        if logits is not None:
            step = logits[:, -1, :].float()
            if temperature > 0:
                probs = torch.softmax(step / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)
            else:
                tok = step.argmax(dim=-1, keepdim=True)
        out.append(tok)
        if step_i == max_new_tokens - 1:
            break  # the last token needs no further forward
        t0 = time.perf_counter()
        logits = replay_chunk(student, shadow, tok, sync_indices, caches,
                              latency_s=latency_s, timing=timing, graphs=graphs)
        per_token_ms.append((time.perf_counter() - t0) * 1e3)

    if timing is not None:
        timing["prefill_ms"] = prefill_ms
        timing["per_token_ms"] = per_token_ms
    return torch.cat(out, dim=1)


# --------------------------------------------------------------------------- #
# Block draft-verify decode (see eval/draft_verify.py for the schedule math).
# --------------------------------------------------------------------------- #
def _rollback_gdn(cache: BatchedShadowCache, upto: int) -> None:
    """Commit a logged verify chunk's GDN states at ``upto`` accepted chunk
    tokens: the conv state is the raw pre-conv context window ending there;
    the recurrent state is the chunk recurrence replayed over the accepted
    prefix from the saved inputs (local, comm-free — the inputs came from
    synced residuals). KV needs no data rollback: the write position rolls
    back and the offset mask hides the stale tail until it is overwritten."""
    m = _hf()
    fn = m.chunk_gated_delta_rule or m.torch_chunk_gated_delta_rule
    if not cache.log:
        return
    lis = sorted(cache.log)
    for li in lis:
        d = cache.log[li]
        # copy_ (never reassign): the verify-window graphs captured this
        # buffer's address; the log entries likewise persist across blocks —
        # each replay refreshes the same graph-pool tensors in place.
        cache.conv[li].copy_(d["conv_ctx"][..., upto: d["kc"] + upto])
    # ONE kernel replay for all logged layers: every GDN layer shares head
    # geometry, so layers stack on the batch dim (per-layer eager calls were
    # ~launch-bound: 2 caches x ~L layers per block).
    parts = [cache.log[li]["qkvgb"] for li in lis]
    q, k, v, g, beta = (torch.cat([p[j][:, :upto] for p in parts], dim=0)
                        for j in range(5))
    init = torch.cat([cache.rec[li] for li in lis], dim=0)
    _core, rec_out = fn(q, k, v, g=g, beta=beta, initial_state=init,
                        output_final_state=True, use_qk_l2norm_in_kernel=True)
    nb = parts[0][0].shape[0]
    for i, li in enumerate(lis):
        cache.update_rec(li, rec_out[i * nb: (i + 1) * nb])


@torch.no_grad()
def generate_draft_verify(
    student: PTWrappedModel,
    shadow,
    input_ids: torch.LongTensor,
    sync_indices: Sequence[int],
    max_new_tokens: int,
    draft_propose,
    k: int,
    latency_s: float = 0.0,
    timing: "dict | None" = None,
    use_cuda_graphs: bool = False,
) -> torch.LongTensor:
    """Prefill + block draft-verify greedy decode: the head drafts ``k``
    tokens locally (``draft_propose(seq (1, T), k) -> (k,)``; never called on
    peers), then ONE lockstep verify chunk over the k+1 pending positions
    stands in for k+1 per-token passes. Every emitted token is the verifier's
    greedy choice given its prefix — emitted ≡ ``generate``'s plain decode by
    construction; only τ = tokens/verify-event varies with the drafter.

    Comm per block = the chunk's usual rounds (1 embed + boundary
    all-reduces) + ONE tiny accept-count broadcast, which every rank needs to
    roll its caches back to the accepted prefix. Peers decode zero-id chunks
    exactly as in ``generate``. Returns emitted ids (1, >= max_new_tokens; the
    final block may overshoot by < k — bought with the same verify event)."""
    import torch.distributed as dist

    assert k >= 1 and input_ids.shape[0] == 1, "draft-verify decode is B=1, k>=1"
    caches = EngineCaches.build(
        student, s_max=input_ids.shape[1] + max_new_tokens + k + 1)
    head = student.lm_head is not None
    device = input_ids.device
    distributed = dist.is_available() and dist.is_initialized() \
        and dist.get_world_size() > 1
    graphs: "dict | None" = {"chunk_T": k + 1} if use_cuda_graphs else None

    t0 = time.perf_counter()
    logits = replay_chunk(student, shadow, input_ids, sync_indices, caches,
                          latency_s=latency_s, timing=timing)
    prefill_ms = (time.perf_counter() - t0) * 1e3
    # Deferred-commit (draft-verify) mode from here on: entries persist across
    # blocks — under graphs each replay refreshes the same logged tensors.
    caches.own.log, caches.shadow.log = {}, {}

    tok = torch.zeros((1, 1), dtype=torch.long, device=device)
    if head:
        tok = logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
    seq = torch.cat([input_ids, tok], dim=1) if head else None
    out = [tok]
    tokens, blocks = 1, 0
    per_block_ms: "list[float]" = []
    accept_t = torch.zeros((), dtype=torch.long, device=device)

    prof = timing is not None and timing.get("profile") and input_ids.is_cuda

    def _mark():
        torch.cuda.synchronize()
        return time.perf_counter()

    while tokens < max_new_tokens:
        t0 = time.perf_counter()
        if head:
            proposals = draft_propose(seq, k).view(-1).to(device)
            chunk = torch.cat([seq[:, -1:], proposals.view(1, -1)], dim=1)
        else:
            chunk = torch.zeros((1, k + 1), dtype=torch.long, device=device)
        if prof:
            timing.setdefault("draft_ms", []).append((_mark() - t0) * 1e3)
        start_pos = caches.pos
        logits = replay_chunk(student, shadow, chunk, sync_indices, caches,
                              latency_s=latency_s, timing=timing, graphs=graphs)
        if head:
            # Chunk position j predicts sequence slot start_pos+j+1, so the
            # argmaxes are the greedy targets for the k proposal slots plus
            # the bonus/correction slot (the tag-era verify-step math).
            targets = logits[0].float().argmax(-1)  # (k+1,)
            accepted = int((targets[:k] == proposals).cumprod(0).sum())
            accept_t.fill_(accepted)
        if distributed:
            dist.broadcast(accept_t, src=0)
        # No stall: the accept count flies while the head drafts the next
        # block (~k drafter steps > one wire latency), so peers have rolled
        # back before the next chunk's embed round lands — the embed stall
        # is the only wire wait either side still pays between chunks.
        a = int(accept_t)
        if prof:
            tr = time.perf_counter()
        _rollback_gdn(caches.own, a + 1)
        _rollback_gdn(caches.shadow, a + 1)
        if prof:
            timing.setdefault("rollback_ms", []).append((_mark() - tr) * 1e3)
        caches.pos = start_pos + a + 1
        if head:
            emitted = targets[: a + 1].view(1, -1)
            seq = torch.cat([seq, emitted], dim=1)
        else:
            emitted = torch.zeros((1, a + 1), dtype=torch.long, device=device)
        out.append(emitted)
        tokens += a + 1
        blocks += 1
        per_block_ms.append((time.perf_counter() - t0) * 1e3)
        if timing is not None:
            timing.setdefault("accepts", []).append(a)

    if timing is not None:
        timing["prefill_ms"] = prefill_ms
        timing["per_block_ms"] = per_block_ms
        timing["blocks"] = blocks
        timing["tokens"] = tokens
    return torch.cat(out, dim=1)
