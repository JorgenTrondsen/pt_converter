"""Per-layer / per-track activation probe: what each track's state looks like
against the dense teacher's, at BOTH sublayer phases of every layer.

The trainer's only activation signal is one scalar per supervised tap
(`distill_step`'s ``layer_relmse``), and `eval/fidelity.py` only ever sees the
MERGED state at boundary depths. Neither can answer the question the whole PT
premise rests on: what does an individual track's contribution look like, and
where along the stack does it stop resembling the teacher's?

Everything here rides the teacher-forced block loop, so each row is **that
layer's own error**, not accumulated drift: the loop resets the carry to the
teacher's tensor at every boundary, and the teacher is the frozen slices on the
`exact` schedule — which already computes post-attn AND post-mlp at every depth.
Almost nothing new is computed; it is recorded.

Three quantity families per (layer, phase, track), each scored cos / norm-ratio
/ relMSE against the teacher:

1. **residual** — the running ``h`` this track carries, vs the teacher's residual
   at the same point.
2. **delta** — this track's own SUBLAYER output (``attn_k(...)`` / ``mlp_k(...)``)
   vs the teacher's. At N tracks one track carries ~1/N of the teacher's update,
   so ``d_nr ≈ 1/N`` is EXPECTED and carries no information; ``d_cos`` is the
   signal — whether the track's contribution points along the true update.
3. **nin** — the NORMED tensor the sublayer actually reads.

Plus a merged row (``track = -1``) holding the all-reduced state, the summed
delta, and ``coh = ‖Σ_k δ_k‖ / Σ_k ‖δ_k‖`` — near 1 if the tracks are redundant,
near 1/√N if their deltas are mutually orthogonal. Never measured before.

Two rows are self-validating rails, and they are the reason this probe can be
trusted where an older per-layer panel could not (it compared post-attn against
post-mlp for months without anyone noticing):

* **post-mlp residual at step 0 sits at the bf16 floor.** The student's MLP reads
  the teacher's exact ``R_i`` through weights that ARE the teacher's, so the
  merged post-mlp state must reproduce ``Hm_i``. A large value here means the
  probe is misaligned, not that the model is bad.
* **post-mlp ``nin_*`` is a pure norm-weight-drift meter.** Student and teacher
  feed the second norm the IDENTICAL tensor ``R_i``, so any difference is
  ``post_attention_layernorm.weight`` moving under training. Exactly zero at step
  0. It also sizes the confound in the post-attn ``nin_*`` row, which mixes
  residual error with ``input_layernorm.weight`` drift.

Cost/safety notes that are load-bearing:

* Every capture is detached and reduced to scalars IMMEDIATELY. The TF loop
  backwards and frees its graph per boundary segment; a probe holding live
  references would pin the whole stack's graph.
* Metrics land in a GPU buffer and cross to the host ONCE, at flush. A ``.item()``
  per metric would be thousands of device syncs on the hot path.
* Normed inputs are RECOMPUTED outside the compiled region, never tapped inside
  it — ``seam.seam_token_mixer`` / ``batched._fold_attn`` and friends are what
  ``--compile both`` compiles, and a tap in there is a graph break away from
  blowing ``recompile_limit``.
* The teacher's norm modules are CLONED at construction, reading ``p._host``
  when present: under ``--teacher-stream``, `HostResidentLayers.release` sets
  ``p.data`` to an empty tensor after each layer, so the live modules are unusable
  once the forward has moved on.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist

# Column order of the per-row metric buffer. Kept as one tuple so the buffer, the
# JSONL keys and the reporter cannot drift apart.
METRICS = (
    "res_cos", "res_nr", "res_relmse",
    "d_cos", "d_nr", "d_relmse",
    "nin_cos", "nin_nr", "nin_relmse",
    "dnorm",
    # Merged-row only (NaN → null on per-track rows): the GLOBAL N×N track-vs-track
    # structure. `coh` says HOW redundant the tracks are; these say whether that
    # redundancy is spread evenly or concentrated in a few near-duplicate pairs.
    "gram_mean", "gram_max", "gram_pr",
)
_M = len(METRICS)
PHASES = ("attn", "mlp")


def _as_list(x) -> list[torch.Tensor]:
    """Both track representations as a plain list: a ``[K, ...]`` stacked tensor
    (batched merged path) or an already-per-track list (looped path)."""
    if isinstance(x, torch.Tensor):
        return list(x.unbind(0)) if x.ndim == 4 else [x]
    return list(x)


def _msum(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Sum of a (B, T) per-token quantity over non-pad positions."""
    if mask is None:
        return x.sum()
    return (x * mask.to(x.dtype)).sum()


def _mmean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    return _msum(x, mask) / mask.sum().clamp(min=1)


def pair_terms(
    s: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four PER-TOKEN (B, T) tensors every pair metric is built from.

    Split out from `pair_stats` because the (B, T) axis is where the interesting
    structure lives and the aggregate throws it away. ⚠ ``err`` and ``den`` are
    returned SEPARATELY, not as a ratio: the aggregate relMSE is ``Σerr / Σden``
    — a norm-WEIGHTED quantity — while ``cos`` and ``ratio`` are reduced by an
    UNWEIGHTED mean. That asymmetry is exactly why a handful of high-‖t‖
    positions can move relMSE 13x while cos and ratio, diluted by ~2000 ordinary
    tokens, do not move at all. Keeping both terms lets a per-position dump
    reproduce the aggregate exactly, which is the rail on the dump.
    """
    s = s.float()
    t = t.float()
    s_n = s.pow(2).sum(-1).sqrt()
    t_n = t.pow(2).sum(-1).sqrt()
    cos = (s * t).sum(-1) / (s_n * t_n).clamp(min=eps)
    ratio = s_n / t_n.clamp(min=eps)
    err = (s - t).pow(2).sum(-1)
    den = t.pow(2).sum(-1)
    return cos, ratio, err, den


def pair_stats(
    s: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor | None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(cos, norm_ratio, relmse)`` for one (B, T, H) student/teacher pair.

    Same three definitions the objective and `eval/fidelity.py` already use
    (`losses.block_direction_magnitude`, `losses.block_mse(normalize=True)`) —
    written once more here only because those take a masked mean over a single
    pair and this runs a few thousand pairs per probe step, so it keeps
    everything on device and returns 0-dim tensors rather than floats.
    """
    cos, ratio, err, den = pair_terms(s, t, eps)
    return _mmean(cos, mask), _mmean(ratio, mask), _msum(err, mask) / _msum(den, mask).clamp(min=eps)


def track_gram(deltas: list[torch.Tensor], mask: torch.Tensor | None,
               eps: float = 1e-12, group=None) -> tuple[float, float, float]:
    """``(mean_cos, max|cos|, participation_ratio)`` over ALL N tracks.

    ``coh`` is an aggregate: it says the tracks are, say, 2.7x redundant, but not
    whether all of them lean the same way or two are near-duplicates and the rest
    are fine. Those imply different things — the first is a property of the layer,
    the second is wasted width.

    The participation ratio ``(Σλ)²/Σλ²`` of the Gram's eigenvalues is the
    **effective number of independent directions** these K tracks span: K when they
    are mutually orthogonal, 1 when they are all parallel. That is the max-tracks
    question asked directly.

    ``gram_mean`` is the SIGNED mean off-diagonal cosine, deliberately: it is what
    has to be commensurable with ``coh``/``redun`` (``redun² ≈ 1 + (K−1)·mean_cos``),
    and an absolute value would report alignment MAGNITUDE, which cannot be compared
    to a net sum. ``gram_max`` keeps the absolute value — for spotting a
    near-duplicate pair, direction does not matter.

    **GLOBAL, and it has to be.** A first version scored only this rank's K tracks
    and was measured useless at 32B/N=64: a rank holds CONTIGUOUS shards (rank r has
    8r..8r+7, i.e. adjacent heads), local structure came out flat across depth
    (``gram_pr`` 4.2-5.6 of 8 at every band) and **uncorrelated with the global
    ``redun`` (Pearson r = −0.03)**, which itself spans 1.04-1.79. The redundancy
    that varies with depth is a LONG-RANGE property across all N tracks and is
    invisible in any 8 adjacent ones.

    So the deltas are all-gathered and the full N×N Gram is formed — EXACT, no random
    projection: at the 32B/N=64 shape the gathered tensor is 1.34 GB (64 × B×T×H
    bf16) against a ~27 GB peak on 40 GB cards, transient per (layer, phase). A
    projection would have bought nothing but error.

    ⚠ Collective: every rank must call this, with equal K. Zero-delta tracks are
    dropped first, or they would dilute the mean and bound ``gram_pr`` below its true
    value — shards cut before `slicer.base.align_chunk` learned to keep every track
    covered can strand a few with no MLP slab at all.
    """
    x = torch.stack(deltas, 0) if not isinstance(deltas, torch.Tensor) else deltas
    if mask is not None and not bool(mask.all()):
        # (B, T) onto x's MIDDLE axes. Right-aligned padding lands the mask's T on
        # x's B; unreachable so far only because the training mask is all ones.
        m = mask.to(x.dtype).reshape(
            (1,) * (x.ndim - mask.ndim - 1) + tuple(mask.shape) + (1,))
        x = x * m
    x = x.reshape(x.shape[0], -1).contiguous()
    if dist.is_available() and dist.is_initialized():
        out = x.new_empty((x.shape[0] * dist.get_world_size(group), x.shape[1]))
        dist.all_gather_into_tensor(out, x, group=group)
        x = out
    g = (x @ x.T).float()
    live = g.diagonal() > eps
    if int(live.sum()) < 2:
        return float("nan"), float("nan"), float("nan")
    g = g[live][:, live]
    d = g.diagonal().clamp(min=eps).sqrt()
    c = g / d[:, None] / d[None, :]
    k = c.shape[0]
    off = c[~torch.eye(k, dtype=torch.bool, device=c.device)]
    lam = torch.linalg.eigvalsh(g.double()).clamp(min=0)
    pr = (lam.sum() ** 2 / lam.pow(2).sum().clamp(min=eps)).item()
    return off.mean().item(), off.abs().max().item(), pr


def _clone_norm(mod: torch.nn.Module, device) -> torch.nn.Module:
    """A private copy of a norm module that survives teacher-layer streaming.

    Deep-copied rather than reconstructed so no family's norm class or
    constructor signature is named here (Qwen3 applies ``w``, Qwen3.5 applies
    ``1 + w``, and only the module knows which). Parameter storage comes from
    ``p._host`` when `HostResidentLayers` has already emptied ``p.data``.
    """
    out = copy.deepcopy(mod)
    for p_src, p_dst in zip(mod.parameters(), out.parameters()):
        host = getattr(p_src, "_host", None)
        src = host if host is not None else p_src.data
        p_dst.data = src.detach().to(device).clone()
        # deepcopy dragged the pinned host buffer along; it is dead weight here.
        if hasattr(p_dst, "_host"):
            del p_dst._host
    return out.to(device).eval()


class ActivationProbe:
    """One probe step's worth of teacher-vs-student activation geometry.

    Lifecycle: the trainer sets ``probe.step`` and passes the probe to
    `distill_step`, which calls ``bind(sync_fn, student_norm)`` and ``begin(mask)``
    for the ``{(layer, phase): hidden}`` dict the teacher forward fills, then one
    ``record`` per (layer, phase); the trainer calls ``flush()`` afterwards.

    ``sync_fn(states, pre)`` must be the trainer's own sync adapter, so the
    rank-local reduce, `SyncBoundary.leaders` de-duplication and cross-rank
    ``_LeaderOnly`` share all behave exactly as they do in the objective.

    ``student_norm(layer, states, phase)`` returns the normed sublayer input for
    the same container type the walk uses.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        n_streams: int,
        stream_offset: int,
        shards_per_stream: int,
        rank: int,
        out_dir: str | Path,
        device,
        teacher_layers,
        sync_phase: str = "post-attn",
        block_walk: str = "tf",
        detail_layers: "set[int] | None" = None,
        detail_topk: int = 32,
    ):
        self.L = num_layers
        self.K = n_streams
        self.stream_offset = stream_offset
        self.rank = rank
        self.device = device
        self.sync_fn = None      # bound per step by `arm`
        self.student_norm = None
        # Row K is the merged/all-track row (written out as track = -1).
        self._buf = torch.full((num_layers, 2, n_streams + 1, _M), float("nan"),
                               device=device, dtype=torch.float32)
        # Σ_k ‖δ_k‖ summed over ALL ranks — the coherence denominator, and the one
        # quantity that needs a collective of its own. Batched into a single
        # all_reduce at flush rather than one per layer.
        self._dnorm_sum = torch.zeros((num_layers, 2), device=device, dtype=torch.float32)
        self._t_norms = {
            (i, ph): _clone_norm(
                getattr(teacher_layers[i],
                        "input_layernorm" if ph == "attn" else "post_attention_layernorm"),
                device,
            )
            for i in range(num_layers)
            for ph in PHASES
        }
        self.path = Path(out_dir) / f"rank{rank}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sync_phase = sync_phase
        self.block_walk = block_walk
        self._meta = {
            "rank": rank,
            "num_layers": num_layers,
            "streams_per_rank": n_streams,
            "shards_per_stream": shards_per_stream,
            "stream_offset": stream_offset,
            "sync_phase": sync_phase,
            "block_walk": block_walk,
            "detail_layers": sorted(detail_layers or ()),
            "metrics": list(METRICS),
        }
        self._wrote_meta = False
        self.step = -1
        self.mask = None
        self.caps: dict = {}
        # Overrides `attention_mask` when set: 1 only at the positions the metric
        # should read. Every reduction in this file goes through `_msum`/`_mmean`,
        # and `track_gram`'s masked path is proven equal to slicing
        # (tests/test_activation_probe.py), so this restricts the WHOLE panel to a
        # position set for free. Used to CONFIRM a culprit set that the detail dump
        # below has already found.
        self.score_mask: torch.Tensor | None = None
        # The probe batch's token ids, stamped into the meta once so a position
        # index in a detail row decodes to a token.
        self.input_ids: torch.Tensor | None = None
        # Layers to dump the un-reduced (B, T) and per-channel breakdown at.
        # Everything else in this file is an average over both axes, which can show
        # that an error is CONCENTRATED but never where.
        self.detail_layers = set(detail_layers or ())
        self.detail_topk = detail_topk
        self._detail: dict = {}
        # Free-running merged rows, scored against the same teacher captures as the
        # teacher-forced ones. No per-track dimension: the student's own walk hands
        # back synced states, not per-track ones.
        self._fr = torch.full((num_layers, 2, _M), float("nan"),
                              device=device, dtype=torch.float32)

    # -- lifecycle ---------------------------------------------------------

    def bind(self, sync_fn, student_norm) -> "ActivationProbe":
        """Adopt the trainer's own sync / norm adapters. They are built per step
        inside `distill_step` (they close over the track representation), so they
        are bound here rather than at construction — the probe must reduce exactly
        the way the objective does or its merged row means something else."""
        self.sync_fn = sync_fn
        self.student_norm = student_norm
        return self

    def begin(self, attention_mask: torch.Tensor | None,
              input_ids: torch.Tensor | None = None) -> dict:
        """Reset for a probe step; returns the dict the teacher forward fills.

        ``score_mask`` wins over ``attention_mask`` when set — see that attribute.
        """
        self.mask = self.score_mask if self.score_mask is not None else attention_mask
        if input_ids is not None:
            self.input_ids = input_ids
        self.caps = {}
        self._buf.fill_(float("nan"))
        self._fr.fill_(float("nan"))
        self._dnorm_sum.zero_()
        self._detail = {}
        return self.caps

    @torch.no_grad()
    def _record_detail(self, layer: int, phase: str, walk: str,
                       s: torch.Tensor, t: torch.Tensor) -> None:
        """Un-reduced (B, T) and per-channel breakdown for one merged pair.

        Only for layers named by ``--probe-detail``, and only on rank 0: `s` and `t`
        here are always MERGED states, which come out of an all-reduce and are
        therefore bit-identical on every rank, so one rank writing them loses
        nothing. (Per-TRACK tensors are rank-disjoint and must never be dumped this
        way.) The work itself is rank-uniform — it issues no collective, but neither
        does it skip one.
        """
        if layer not in self.detail_layers or self.rank != 0:
            return
        cos, ratio, err, den = pair_terms(s.detach(), t.detach())
        sd, td = s.detach().float(), t.detach().float()
        # Per channel: which hidden dims carry the error, and how big the teacher's
        # own activation is there. A few dims holding most of Σerr, at a position
        # where |t| is huge, is the massive-activation signature.
        axes = tuple(range(sd.ndim - 1))
        err_c = (sd - td).pow(2).sum(axes)
        t_absmax_c = td.abs().amax(dim=axes)
        k = min(self.detail_topk, err_c.numel())
        top_c = torch.topk(err_c, k)
        self._detail[(layer, phase, walk)] = {
            "err": err.flatten().cpu(),
            "den": den.flatten().cpu(),
            "cos": cos.flatten().cpu(),
            "nr": ratio.flatten().cpu(),
            "chan_idx": top_c.indices.cpu(),
            "chan_err": top_c.values.cpu(),
            "chan_tmax": t_absmax_c[top_c.indices].cpu(),
            "chan_err_total": float(err_c.sum()),
        }

    @torch.no_grad()
    def record_fr(self, caps: dict) -> None:
        """Score the student's own FREE-RUNNING states against the teacher's.

        The teacher-forced rows reset the carry to the teacher at every boundary, so
        each is that layer's OWN error — by construction they cannot see error
        ACCUMULATE, and at d1b they sit near the bf16 floor for most of the stack.
        Every downstream number is free-running. So the two walks measure different
        functions, and until this landed the probe only ever ran the first one (the
        probe step zeroes ``lambda_ce``, which skipped the free-running forward
        entirely). The ratio FR/TF at a depth is that layer's error AMPLIFICATION:
        how much it inflates the drift handed to it, over its error given a perfect
        input.

        ``caps`` is the student forward's own ``probe_capture``, so these are its
        SYNCED states at the schedule it deploys at — the same tensors, scored
        against the same teacher captures, under the same mask. Which depths appear
        follows `PTWrappedModel.sync_sets()`, i.e. the seam column of the phase.
        """
        for (i, ph), s in caps.items():
            if i < 0 or (i, ph) not in self.caps:
                continue
            t = self.caps[(i, ph)].detach()
            self._fr[i, PHASES.index(ph), 0:3] = torch.stack(
                pair_stats(s.detach(), t, self.mask)
            )
            self._record_detail(i, ph, "fr", s, t)

    def teacher_state(self, layer: int, phase: str) -> torch.Tensor:
        return self.caps[(layer, phase)]

    def _teacher_pre(self, layer: int, phase: str) -> torch.Tensor:
        """The teacher residual entering this sublayer: the previous layer's
        post-MLP for the mixer half, this layer's post-attn for the MLP half.
        Layer 0's predecessor is the embedding, stored under ``(-1, "mlp")``."""
        return self.caps[(layer - 1, "mlp")] if phase == "attn" else self.caps[(layer, "attn")]

    @torch.no_grad()
    def record(
        self,
        layer: int,
        phase: str,
        states,
        pres,
        merged: torch.Tensor | None = None,
        pre_shared: torch.Tensor | None = None,
    ) -> None:
        """Score one (layer, phase).

        ``states`` / ``pres``: the per-track post- and pre-sublayer residuals, in
        whichever container the walk uses. ``merged`` is the already-synced
        residual when the caller has one (the TF loop's boundary sync), else it is
        reconstructed with ``sync_fn(states, pre_shared)``.
        """
        p = PHASES.index(phase)
        t_state = self.caps[(layer, phase)].detach()
        t_pre = self._teacher_pre(layer, phase).detach()
        t_delta = t_state - t_pre
        mask = self.mask

        s_list = [x.detach() for x in _as_list(states)]
        p_list = [x.detach() for x in _as_list(pres)]
        if len(p_list) == 1 and len(s_list) > 1:
            p_list = p_list * len(s_list)  # a shared teacher-forced pre-state
        s_nin = _as_list(self.student_norm(layer, pres, phase))
        if len(s_nin) == 1 and len(s_list) > 1:
            s_nin = s_nin * len(s_list)
        t_nin = self._t_norms[(layer, phase)](t_pre)

        dnorm_acc = torch.zeros((), device=self._buf.device, dtype=torch.float32)
        for k, (s, pre, nin) in enumerate(zip(s_list, p_list, s_nin)):
            delta = s - pre
            self._buf[layer, p, k, 0:3] = torch.stack(pair_stats(s, t_state, mask))
            self._buf[layer, p, k, 3:6] = torch.stack(pair_stats(delta, t_delta, mask))
            self._buf[layer, p, k, 6:9] = torch.stack(pair_stats(nin.detach(), t_nin, mask))
            dn = _mmean(delta.float().pow(2).sum(-1).sqrt(), mask)
            self._buf[layer, p, k, 9] = dn
            dnorm_acc = dnorm_acc + dn
        self._dnorm_sum[layer, p] = dnorm_acc

        # The all-track sum of the sublayer deltas. Routed through the trainer's
        # own sync adapter against a ZERO pre-state so the deltas themselves are
        # what gets summed — `sync_fn(states, pre)` would additionally fold in
        # whatever separates ``pre`` from each track's own pre-state, which at
        # post-attn is the previous layer's un-summed MLP deltas.
        delta_list = [s - pre for s, pre in zip(s_list, p_list)]
        deltas = self._like(states, delta_list)
        gm, gx, gpr = track_gram(delta_list, mask)
        self._buf[layer, p, self.K, 10] = gm
        self._buf[layer, p, self.K, 11] = gx
        self._buf[layer, p, self.K, 12] = gpr
        zero = torch.zeros_like(t_pre)
        sum_delta = self.sync_fn(deltas, zero)
        if merged is None:
            merged = self.sync_fn(states, pre_shared if pre_shared is not None else t_pre)
        merged = merged.detach()

        self._buf[layer, p, self.K, 0:3] = torch.stack(pair_stats(merged, t_state, mask))
        self._record_detail(layer, phase, "tf", merged, t_state)
        self._buf[layer, p, self.K, 3:6] = torch.stack(pair_stats(sum_delta, t_delta, mask))
        self._buf[layer, p, self.K, 9] = _mmean(
            sum_delta.float().pow(2).sum(-1).sqrt(), mask
        )
        # nin_* stays NaN (→ null) on the merged row: there is no merged normed
        # input. Each track norms its OWN pre-state, and the walk never forms a
        # combined one — scoring norm(pre) of any single track here would read as
        # a measurement of something that does not exist.

    @staticmethod
    def _like(ref, tensors: list[torch.Tensor]):
        """Rebuild the walk's container type around a new per-track list."""
        return torch.stack(tensors, 0) if isinstance(ref, torch.Tensor) and ref.ndim == 4 else tensors

    @torch.no_grad()
    def flush(self) -> dict:
        """One collective, one host transfer, then append the step's rows.

        Returns a small ``{layer: (tf_attn, tf_mlp, fr_attn, fr_mlp)}`` digest of the
        merged relMSE for the caller's stdout line. The FR entries are NaN until a
        free-running forward has been scored (`record_fr`).
        """
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._dnorm_sum, op=dist.ReduceOp.SUM)
        buf = self._buf.cpu()
        dsum = self._dnorm_sum.cpu()
        fr = self._fr.cpu()

        rows = []
        for i in range(self.L):
            for p, phase in enumerate(PHASES):
                if torch.isnan(buf[i, p]).all() and torch.isnan(fr[i, p]).all():
                    continue  # a phase the schedule never reached
                # The free-running row: same layer, same phase, same teacher target,
                # the student's OWN walk instead of the teacher-forced one.
                if not torch.isnan(fr[i, p]).all():
                    rows.append({
                        "step": self.step, "rank": self.rank, "layer": i,
                        "phase": phase, "track": -1, "walk": "fr",
                        **{m: (None if v != v else round(float(v), 6))
                           for m, v in zip(METRICS, fr[i, p])},
                    })
                for k in range(self.K + 1):
                    vals = buf[i, p, k]
                    if torch.isnan(vals).all():
                        continue
                    merged = k == self.K
                    row = {
                        "step": self.step,
                        "rank": self.rank,
                        "layer": i,
                        "phase": phase,
                        "walk": "tf",
                        "track": -1 if merged else self.stream_offset + k,
                    }
                    # NaN → null: "not applicable here", never a fabricated 0.
                    row.update({
                        m: (None if v != v else round(float(v), 6))
                        for m, v in zip(METRICS, vals)
                    })
                    if merged:
                        denom = float(dsum[i, p])
                        row["coh"] = round(float(vals[9]) / denom, 6) if denom > 0 else None
                    rows.append(row)

        # Detail rows carry LISTS, not the METRICS columns, so they are tagged
        # `detail` and every existing view filters them out via `walk`/`track`.
        for (i, phase, walk), d in sorted(self._detail.items()):
            rows.append({
                "step": self.step, "rank": self.rank, "layer": i, "phase": phase,
                "walk": walk, "track": -1, "detail": "position",
                # err and den SEPARATELY: Σerr/Σden must reproduce this row's
                # res_relmse exactly, which is the rail on the whole dump.
                "err": [round(v, 9) for v in d["err"].tolist()],
                "den": [round(v, 6) for v in d["den"].tolist()],
                "cos": [round(v, 6) for v in d["cos"].tolist()],
                "nr": [round(v, 6) for v in d["nr"].tolist()],
            })
            rows.append({
                "step": self.step, "rank": self.rank, "layer": i, "phase": phase,
                "walk": walk, "track": -1, "detail": "channel",
                "chan_idx": d["chan_idx"].tolist(),
                "chan_err": [round(v, 9) for v in d["chan_err"].tolist()],
                "chan_tmax": [round(v, 6) for v in d["chan_tmax"].tolist()],
                "chan_err_total": round(d["chan_err_total"], 9),
            })

        if self.input_ids is not None and "input_ids" not in self._meta:
            self._meta["input_ids"] = self.input_ids.flatten().tolist()
        with self.path.open("a") as fh:
            if not self._wrote_meta:
                fh.write(json.dumps({"meta": self._meta}) + "\n")
                self._wrote_meta = True
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        return {
            i: (float(buf[i, 0, self.K, 2]), float(buf[i, 1, self.K, 2]),
                float(fr[i, 0, 2]), float(fr[i, 1, 2]))
            for i in range(self.L)
        }


def summary_lines(digest: dict, step: int, every: int = 8,
                  sync_phase: str = "post-attn", block_walk: str = "tf") -> list[str]:
    """A few lines of merged-row relMSE for the run log — the full grid is in the
    JSONL, this is only enough to see at a glance that the probe ran and that the
    second line is where it should be.

    ⚠ **The two columns SWAP ROLES with the phase.** Each phase has one column fed
    the teacher-exact residual (the RAIL — ≈0 while student weights == teacher
    weights, and a non-floor value means the probe is misaligned, not that the model
    is bad) and one fed OWN-CARRY (the SEAM — the sync this phase drops, structurally
    non-zero even with identical weights):

    | phase     | rail column | seam column |
    |-----------|-------------|-------------|
    | post-attn | post-mlp    | post-attn   |
    | post-mlp  | post-attn   | post-mlp    |

    So the cross-phase comparison is SEAM to SEAM — post-attn's `attn` column against
    post-mlp's `mlp` column — NOT column to column by name. (Measured untrained at
    32B/N=64/d1b: seam@L63 0.0210 post-attn vs 0.0514 post-mlp.) Printing post-mlp's
    seam as a rail read 2.43 at L0 and meant nothing; that value is the tiny early
    residual norm in the denominator, not misalignment.

    (At post-attn the final layer was exempt from the rail until 2026-08-18, when it
    became a full boundary.)

    ⚡ The **third line is the one to read**: the FREE-RUNNING seam, and `amp = FR/TF`.
    The TF rows above cannot see error ACCUMULATE — they hand every layer the teacher's
    residual — while every downstream number is free-running. `amp` is how much a layer
    inflates the drift it is handed over its error given a perfect input.
    """
    seam = 1 if sync_phase == "post-mlp" else 0   # which column is the own-carry one
    layers = sorted(digest)
    last = layers[-1]
    picked = [i for i in layers if i % every == 0 or i == last]
    body = " ".join(f"{i}:{digest[i][0]:.4f}/{digest[i][1]:.5f}" for i in picked)
    worst = max(digest[i][1] for i in layers)
    if sync_phase == "post-mlp":
        tail = (f"[probe] step {step} post-mlp seam: max merged relMSE over all "
                f"{len(layers)} layers = {worst:.2e} (the dropped sync — NOT a rail)")
    elif block_walk != "tf":
        # The rail only exists because the boundary MLP reads the TEACHER's residual.
        # Under a free-running carry it reads the student's own drift instead, so this
        # column measures that drift and has no floor to sit at.
        tail = (f"[probe] step {step} post-mlp seam: max merged relMSE over all "
                f"{len(layers)} layers = {worst:.2e} "
                f"(NOT a rail at block_walk={block_walk} — the carry is not teacher-exact)")
    else:
        tail = (f"[probe] step {step} alignment rail: max post-mlp merged relMSE over all "
                f"{len(layers)} layers = {worst:.2e} (≈0 while student==teacher weights)")
    lines = [
        f"[probe] step {step} merged relMSE attn/mlp @L{{{','.join(str(i) for i in picked)}}}: {body}",
        tail,
    ]
    # NaN when no free-running forward was scored (`record_fr` never called).
    fr = {i: digest[i][2 + seam] for i in layers if digest[i][2 + seam] == digest[i][2 + seam]}
    if fr:
        shown = [i for i in picked if i in fr]
        fr_body = " ".join(f"{i}:{fr[i]:.4f}" for i in shown)
        head = f"[probe] step {step} FREE-RUNNING seam @L{{{','.join(str(i) for i in shown)}}}: {fr_body}"
        if block_walk != "tf":
            # The block loop already carries the free-running trajectory, so the two
            # walks compute the same states and amp is 1 by construction. Printing it
            # as a ratio would dress an identity up as a measurement. (That the two
            # agree to 4 decimals IS worth something — it says the carry is wired to
            # the trajectory the model actually runs.)
            lines.append(f"{head} | amp N/A at block_walk={block_walk} (this IS the block walk)")
        else:
            amps = [fr[i] / digest[i][seam] for i in fr if digest[i][seam] > 0]
            lines.append(
                f"{head} | median amp {sorted(amps)[len(amps) // 2]:.0f}x" if amps else head
            )
    return lines
