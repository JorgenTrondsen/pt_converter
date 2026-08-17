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
    s = s.float()
    t = t.float()
    s_n = s.pow(2).sum(-1).sqrt()
    t_n = t.pow(2).sum(-1).sqrt()
    cos = (s * t).sum(-1) / (s_n * t_n).clamp(min=eps)
    ratio = s_n / t_n.clamp(min=eps)
    relmse = _msum((s - t).pow(2).sum(-1), mask) / _msum(t.pow(2).sum(-1), mask).clamp(min=eps)
    return _mmean(cos, mask), _mmean(ratio, mask), relmse


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
        m = mask.to(x.dtype)
        while m.ndim < x.ndim:
            m = m.unsqueeze(-1) if m.ndim > 1 else m.unsqueeze(0)
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
        self._meta = {
            "rank": rank,
            "num_layers": num_layers,
            "streams_per_rank": n_streams,
            "shards_per_stream": shards_per_stream,
            "stream_offset": stream_offset,
            "metrics": list(METRICS),
        }
        self._wrote_meta = False
        self.step = -1
        self.mask = None
        self.caps: dict = {}

    # -- lifecycle ---------------------------------------------------------

    def bind(self, sync_fn, student_norm) -> "ActivationProbe":
        """Adopt the trainer's own sync / norm adapters. They are built per step
        inside `distill_step` (they close over the track representation), so they
        are bound here rather than at construction — the probe must reduce exactly
        the way the objective does or its merged row means something else."""
        self.sync_fn = sync_fn
        self.student_norm = student_norm
        return self

    def begin(self, attention_mask: torch.Tensor | None) -> dict:
        """Reset for a probe step; returns the dict the teacher forward fills."""
        self.mask = attention_mask
        self.caps = {}
        self._buf.fill_(float("nan"))
        self._dnorm_sum.zero_()
        return self.caps

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

        Returns a small ``{layer: (attn_relmse, mlp_relmse)}`` digest of the merged
        rows for the caller's stdout line.
        """
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._dnorm_sum, op=dist.ReduceOp.SUM)
        buf = self._buf.cpu()
        dsum = self._dnorm_sum.cpu()

        rows = []
        for i in range(self.L):
            for p, phase in enumerate(PHASES):
                if torch.isnan(buf[i, p]).all():
                    continue  # a phase the schedule never reached
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

        with self.path.open("a") as fh:
            if not self._wrote_meta:
                fh.write(json.dumps({"meta": self._meta}) + "\n")
                self._wrote_meta = True
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        return {
            i: (float(buf[i, 0, self.K, 2]), float(buf[i, 1, self.K, 2]))
            for i in range(self.L)
        }


def summary_lines(digest: dict, step: int, every: int = 8) -> list[str]:
    """A few lines of merged-row relMSE for the run log — the full grid is in the
    JSONL, this is only enough to see at a glance that the probe ran and that the
    alignment rail is where it should be.

    The rail EXCLUDES the final layer. That is the one layer whose sync lands
    post-MLP (the LM head needs a full residual), so its mixer — and therefore its
    MLP — runs on the partial and it reads high by construction. Folding it into
    the max would make a healthy probe print a number that looks like a failure.
    """
    layers = sorted(digest)
    last = layers[-1]
    picked = [i for i in layers if i % every == 0 or i == last]
    body = " ".join(f"{i}:{digest[i][0]:.4f}/{digest[i][1]:.5f}" for i in picked)
    rail = max(digest[i][1] for i in layers if i != last)
    return [
        f"[probe] step {step} merged relMSE attn/mlp @L{{{','.join(str(i) for i in picked)}}}: {body}",
        f"[probe] step {step} alignment rail: max post-mlp merged relMSE over L<{last} "
        f"= {rail:.2e} (≈0 while student==teacher weights); "
        f"L{last} = {digest[last][1]:.3f}, partial-input by construction",
    ]
