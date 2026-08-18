"""The d1b-heal distillation step (lever-B / post-attn schedule).

Rebuilt from the pre-parallm-pivot trainer for the 27B N=8 pool-free program,
keeping the rail-validated recipe semantics and dropping the refuted-lever
surface (vocab-parallel, cross-head estimators, window-parallel, free-running
MSE). Three deliberate deviations from the 9B-record trainer, all forced by
27B memory or strictly signal-improving:

1. **Teacher = the frozen original slices on the EXACT schedule** (2 syncs per
   layer — bit-identical to the dense model by construction, the engine's
   dense rail) instead of a resident dense HF model: a dense 27B teacher is
   54 GB/rank and cannot exist next to the student on a 40 GB device, while
   the frozen-slice teacher costs one extra track slice per rank and shares
   the frozen embed/lm_head storage with the student.
2. **Replicated output loss** instead of vocab-parallel: every rank holds a
   frozen lm_head copy and computes the identical full-vocab CE, so each rank's
   (collective-free) backward carries the FULL loss signal into its own track —
   vocab-parallel delivered only the rank's shard slice of it.
3. **embed_tokens + lm_head are FROZEN** (they are exact dense copies; the
   heal targets slice function).  # ponytail: also saves ~7.6 GB of optimizer
   state on rank 0 — unfreeze via --train-embeddings if the Stage-1 gate fails.

Step structure:
  1. Teacher forward (no_grad): boundary targets — post-attn residual at
     boundaries, post-MLP at the final layer / mid-window taps.
  2. Teacher-forced post-attn block loop: per boundary segment, sync, tap
     block-MSE (normalized, clamped), backward, free the graph.
  3. Full free-running student forward (per-sublayer checkpointing) + chunked
     CE; ONE backward with the accumulated hidden grad. The SyncBoundary
     all-reduce is autograd-invisible, so this backward is collective-free on
     every rank (the record's gradient semantics).

The objective is ``lambda_block * block_mse + lambda_ce * ce``. Forward-KL,
centered logit-MSE and student forcing were all REMOVED 2026-07-30 as measured
no-ops — see `DistillConfig` and `distill_step` for the numbers, and `git log`
for the four-term version if an older recipe must be reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from parallm.model.pt_model import PTWrappedModel
from parallm.model.seam import checkpointed_halves
from parallm.train.losses import block_mse, block_split
from parallm.train.probe import ActivationProbe
from parallm.train.profile import PhaseTimer

# Which pre-norm a sublayer phase reads. Used only by the activation probe, which
# recomputes the normed input outside the compiled seam halves.
_NORM_ATTR = {"attn": "input_layernorm", "mlp": "post_attention_layernorm"}


@dataclass
class DistillConfig:
    """The objective: ``lambda_block * block_mse + lambda_ce * ce``.

    Forward-KL and centered logit-MSE were REMOVED 2026-07-30 after a
    leave-one-out sweep at D=2/N=24/F=3 on the 27B measured both as free (macro
    0.595 for block+ce vs 0.593 for all four terms) — and KL as actively
    inferior: the arm that did NOT optimize KL reached a LOWER KL (1.135) than
    the arm that did (1.168), because forward-KL and hard-label CE have
    near-parallel gradients and CE is the better-conditioned of the two.
    Re-tested 2026-08-18 on MATH rather than macro and refuted again.
    ``git log`` has the four-term version if an older recipe must be reproduced.
    """

    sync_layer_indices: tuple[int, ...] = field(default_factory=tuple)
    lambda_block: float = 4.0
    lambda_ce: float = 1.0
    lambda_mag: float | None = None
    intra_window_mse: bool = False
    ce_chunk_size: int = 256
    # Longest own-carry segment the TF loop holds uncheckpointed; see
    # TF_CKPT_MIN_SEGMENT. 0 restores the pre-2026-08-07 "always checkpoint at D>1"
    # behaviour, which is what a step-time control arm wants.
    tf_ckpt_min_segment: int = 2


def freeze_slice_teacher(model: PTWrappedModel) -> PTWrappedModel:
    """Configure the ORIGINAL track slices as the frozen exact-schedule teacher.

    ``model`` is a PTWrappedModel loaded from the raw convert output; in eval
    mode with ``requires_grad=False`` and ``sync_phase="exact"`` its forward is
    the dense teacher's up to fp summation order (the engine's dense rail), at
    one track slice of memory per rank.
    """
    model.eval()
    model.set_sync_phase("exact")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def teacher_forward(
    teacher: PTWrappedModel,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None,
    post_attn_layers: set[int],
    post_mlp_layers: set[int],
    probe_capture: dict | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """The post-norm final hidden (identical on every rank) and ``{layer: target}``
    captures: post-attn residual at ``post_attn_layers``, post-MLP hidden at
    ``post_mlp_layers`` — the lever-B target contract.

    ``probe_capture`` additionally retains BOTH phases at EVERY layer for the
    activation probe. The exact-schedule teacher already computes them, so this
    only holds references — at the cost of 2L resident ``(B, T, H)`` tensors
    instead of the handful the target contract needs, which is why it is passed
    only on probe steps.
    """
    return teacher(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_sync_hiddens=True,
        return_hidden_pre_lm_head=True,
        capture_post_attn=post_attn_layers,
        capture_post_mlp=post_mlp_layers,
        probe_capture=probe_capture,
    )


def capture_sets(
    sync_layer_indices, num_layers: int, intra_window_mse: bool = False
) -> tuple[set[int], set[int]]:
    """``(post_attn_layers, post_mlp_layers)`` the teacher captures: post-attn
    residual at every boundary, post-MLP at the final layer when the schedule does
    not make it one (plus mid-window taps under ``intra_window_mse``).

    ⚠ Never both for one layer: ``sync_hiddens`` is keyed by layer alone, so the
    post-MLP write would silently overwrite the post-attn target."""
    last = num_layers - 1
    post_attn = set(sync_layer_indices)
    post_mlp = set() if last in post_attn else {last}
    if intra_window_mse:
        post_mlp |= set(range(num_layers)) - post_attn - {last}
    return post_attn, post_mlp


# Default for `DistillConfig.tf_ckpt_min_segment`: the longest own-carry segment (in
# layers) the TF loop will hold uncheckpointed. A segment of n layers is 2n sublayers
# across K local tracks; at the F=1/N=64 shape (B=1, T=2048, H=2880, K=8) that is
# ~0.75 GB per layer of segment, so 2 costs ~1.5 GB against a 15.9 GiB peak — worth
# paying to delete a whole recompute of the TF forward. Raise it only with a
# `--mem-report` peak to back it up: rank 0 shares its card with a foreign job.
TF_CKPT_MIN_SEGMENT = 2


def _max_segment_layers(sync_attn_set: set[int], num_layers: int) -> int:
    """Layers in the longest stretch the TF loop carries between backwards.

    The loop flushes at every boundary and again at the final layer, so the live
    graph is one segment — `[0..b0]`, `(b0..b1]`, ... `(b_last..L-1]`. At D=1 every
    non-last layer is a boundary and every segment is 1; at D=2 they are all 2.
    """
    flushes = sorted(sync_attn_set | {num_layers - 1})
    prev, longest = -1, 0
    for f in flushes:
        longest = max(longest, f - prev)
        prev = f
    return longest


def ce_chunked(
    hidden: torch.Tensor,             # (B, T, H), grad-connected to student params
    lm_head: nn.Module,               # frozen head, identical on every rank
    labels: torch.Tensor,
    *,
    lambda_ce: float,
    chunk_size: int = 256,
    loss_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """λ_ce·CE in seq-chunks.

    Never materializes a (B, T, V) logits tensor: the student's logits are
    produced per chunk from its final hidden through the frozen ``lm_head``.
    Each chunk's gradient is taken against a detached ``h_anchor`` and
    accumulated into a (B, T, H) buffer the caller backwards through the real
    graph; the chunk's fp32 tensors are freed before the next (no retain_graph).

    Takes NO teacher hidden: since KL and logit-MSE were removed this objective
    is against hard labels only, which drops a full lm_head projection over the
    vocab plus two log-softmaxes per chunk.

    Returns ``(ce, grad_h)`` — detached scalar, UNSCALED (``loss_scale``
    multiplies only the accumulated gradient). Identical on every rank
    (replicated loss).
    """
    T = hidden.shape[1]
    device = hidden.device

    shift_labels = labels[:, 1:]
    ce_denom = (shift_labels != -100).sum().clamp(min=1).float()

    h_anchor = hidden.detach().requires_grad_(True)
    grad_h_accum = torch.zeros_like(h_anchor)
    ce_total = torch.zeros((), device=device)

    for t0 in range(0, T, chunk_size):
        t1 = min(t0 + chunk_size, T)
        with torch.enable_grad():
            # CE: logits[t] predicts labels[t+1]; the chunk's last position is
            # scored by the NEXT chunk's first label except at the seq end.
            lbl_hi = min(t1, T - 1)
            if lbl_hi <= t0:
                continue
            s_logits = lm_head(h_anchor[:, t0:t1, :]).float()
            ce_logits = s_logits[:, : lbl_hi - t0, :]
            ce_labels = labels[:, t0 + 1 : lbl_hi + 1]
            ce_chunk = F.cross_entropy(
                ce_logits.reshape(-1, ce_logits.size(-1)),
                ce_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ) / ce_denom
            ce_total += ce_chunk.detach()

            if lambda_ce != 0.0 and ce_chunk.requires_grad:
                (g,) = torch.autograd.grad(lambda_ce * ce_chunk * loss_scale, [h_anchor])
                grad_h_accum += g
            del s_logits

    return ce_total, grad_h_accum


def distill_step(
    student: PTWrappedModel,
    teacher: PTWrappedModel,
    lm_head: nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: DistillConfig,
    loss_scale: float = 1.0,
    prof: "PhaseTimer | None" = None,
    probe: "ActivationProbe | None" = None,
) -> dict[str, torch.Tensor]:
    """One heal step (backward done internally; caller syncs grads + steps).

    The TF block loop backwards per boundary segment (graph freed each flush,
    peak memory ≈ one segment); the output objective backwards once through
    the checkpointed full forward.

    The block loop is fully teacher-forced. Student forcing was REMOVED
    2026-07-30: an sf sweep at D=2/N=24/F=3 on the 27B found sf=0 / 0.25 / 0.5
    indistinguishable (0.589-0.594, a 0.005 spread), sf=0.75 worse by 0.031, and
    sf=1.0 divergent — a student trained with ZERO exposure to its own outputs
    scores the same evaluated FREE-RUNNING as one trained half on them, so
    exposure bias does not bind along the depth axis here.

    ``probe``: an `ActivationProbe` to score this step's per-layer / per-track
    activations against the teacher (see `train/probe.py`). Pass it only on probe
    steps, and only from a RANK-UNIFORM gate — it issues collectives. It adds no
    sublayer compute: every tensor it reads is already a local of the TF loop.
    """
    prof = prof if prof is not None else PhaseTimer()  # disabled null object
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]
    device = input_ids.device

    tm0 = student.text_models[0]
    L = len(tm0.layers)
    last = L - 1
    need_ce = cfg.lambda_ce != 0.0

    # Teacher targets: post-attn at every boundary (the final layer included), plus
    # mid-window post-MLP taps under --intra-window-mse.
    sync_attn_set, post_mlp_set = capture_sets(
        cfg.sync_layer_indices, L, cfg.intra_window_mse)
    # Only the per-boundary captures are consumed now — CE is against hard
    # labels, so the teacher's final hidden goes unused. Do NOT "clean that up"
    # by flipping teacher_forward's return_hidden_pre_lm_head to False: that
    # makes PTWrappedModel.forward materialize a (B, T, V) logits tensor on the
    # lm_head owner (~2 GB bf16 at T=2048, V=248320).
    with prof.phase("teacher_fwd"):
        _, t_caps = teacher_forward(
            teacher, input_ids, attention_mask,
            post_attn_layers=sync_attn_set, post_mlp_layers=post_mlp_set,
            probe_capture=probe.begin(attention_mask) if probe is not None else None,
        )
    # ----- Scaffolding (embed broadcast + masks + rotary, shared by the loop) -----
    prof_scaffold = prof.phase("scaffold")
    prof_scaffold.__enter__()
    inputs_embeds = student.embed(input_ids)
    position_ids, text_position_ids = tm0._resolve_position_ids(inputs_embeds, None)
    # The adapter owns the {layer_type: mask} mapping — same call the model's own
    # walk makes, so the two loops cannot drift on masking.
    layer_masks = student._adapter.build_masks(
        tm0.config, inputs_embeds, attention_mask, text_position_ids
    )
    position_embeddings = tm0.rotary_emb(inputs_embeds, position_ids)
    prof_scaffold.__exit__(None, None, None)

    def tap_loss(h_synced: torch.Tensor, t_target: torch.Tensor) -> torch.Tensor:
        if cfg.lambda_mag is not None:
            d, m = block_split(h_synced, t_target, attention_mask=attention_mask)
            return d + cfg.lambda_mag * m
        return block_mse(
            h_synced, t_target, attention_mask=attention_mask,
            normalize=True,  # always: the un-normalized form has no caller
        )

    # ----- Teacher-forced post-attn block loop (per-segment backward) -----
    block_loss_val = torch.zeros((), device=device)
    layer_relmse: dict[int, float] = {}
    block_start = inputs_embeds.detach()
    seg_loss = torch.zeros((), device=device)
    seg_count = 0

    def _flush(sl, sc):
        # One backward per boundary segment, so the graph is freed each time and
        # peak memory is ~one segment.
        if sc > 0 and cfg.lambda_block != 0.0 and sl.requires_grad:
            with prof.phase("tf_backward"):
                (cfg.lambda_block * (sl / sc) * loss_scale).backward()

    # Per-sublayer activation checkpointing bounds the peak graph to ONE
    # recomputed layer regardless of own-carry segment length. But the TF loop
    # backwards at EVERY boundary, so its live graph is one SEGMENT, not the whole
    # stack — and a short segment is cheap enough to hold uncheckpointed. Below the
    # threshold, checkpointing is pure recompute on the launch-bound hot path.
    # Measured share it comes out of: `tf_backward` is 36% of an F=1/N=64 step and
    # runs 4.7x its own forward where checkpoint+grad predicts ~3x.
    #
    # The FR forward below keeps the model's own checkpointing regardless — that one
    # holds all L layers and genuinely needs it.
    max_seg = _max_segment_layers(sync_attn_set, L)
    use_ckpt = (
        student.use_checkpoint
        and torch.is_grad_enabled()
        and max_seg > cfg.tf_ckpt_min_segment
    )

    # Sublayer/sync/share adapters, so ONE loop body serves both track
    # representations — this walk and `pt_model._run_post_attn_stack` must agree
    # sublayer for sublayer or the block targets are measured against a different
    # model, and two hand-mirrored loops is how that drifts. `mix`/`mlp` take the
    # per-sublayer INPUT as both the operand and the fuse pre-state, which every
    # call site here and in the model already satisfies.
    K = len(student.text_models)
    if student.exec_groups > 1:
        # Merged track run as G members: states are one [G, B, T, H] tensor, and
        # a shared [B, T, H] straight out of a sync broadcasts into the members.
        from parallm.model.batched import batched_halves

        mix, mlp = batched_halves(
            student.shadow, use_ckpt, position_embeddings, text_position_ids)
        def sync(xs, pre): return student.sync_module(xs, pre, stacked=True)
        def share(t): return t
        def norm_in(i, xs, phase):
            # OUTSIDE the compiled fold on purpose: `_fold_attn`/`_fold_mlp` are what
            # --compile captures, and a tap inside them is a graph break. The shadow's
            # skeleton carries the REAL norm modules (batched.MergedShadow.__init__
            # assigns them off the merged layer), so this reproduces exactly what the
            # fold normalizes, for one extra RMSNorm.
            return getattr(student.shadow.grid[0][i], _NORM_ATTR[phase])(xs)
    else:
        run_mixer, run_mlp = checkpointed_halves(
            use_ckpt, position_embeddings, text_position_ids)
        def mix(i, xs, mask):
            return student.sync_module.fuse(
                [run_mixer(tm.layers[i], xs[k], mask)
                 for k, tm in enumerate(student.text_models)], xs)
        def mlp(i, xs):
            return student.sync_module.fuse(
                [run_mlp(student.text_models[k].layers[i], xs[k]) for k in range(K)], xs)
        def sync(xs, pre): return student.sync_module(student.sync_module.leaders(xs), pre)
        def share(t): return [t] * K
        def norm_in(i, xs, phase):
            return [getattr(student.text_models[k].layers[i], _NORM_ATTR[phase])(xs[k])
                    for k in range(K)]

    block_input = share(inputs_embeds.detach())
    if probe is not None:
        probe.bind(sync, norm_in)

    prof_block = prof.phase("tf_block_loop")
    prof_block.__enter__()
    for i in range(L):
        layer_mask = layer_masks[tm0.config.layer_types[i]]
        h_attn = mix(i, block_input, layer_mask)
        is_boundary = i in sync_attn_set
        supervise = is_boundary or i == last or cfg.intra_window_mse
        if is_boundary:
            h_synced = sync(h_attn, block_start)  # post-attn, carried
        else:
            new_h = mlp(i, h_attn)
            # Final layer: post-MLP carried sync. Non-boundary: loss-only tap
            # (synced reconstruction; the forward keeps carrying the partial).
            h_synced = sync(new_h, block_start) if supervise else None
        if probe is not None:
            # Scored BEFORE `_flush` backwards this segment's graph. The mixer's
            # per-track pre-state is `block_input`, which at a boundary is the
            # PREVIOUS layer's own-carried post-MLP — that partial input is the
            # d1b seam, and it is what makes the post-attn row the interesting one.
            probe.record(i, "attn", h_attn, block_input,
                         merged=h_synced if is_boundary else None,
                         pre_shared=block_start)
            if not is_boundary:
                probe.record(i, "mlp", new_h, h_attn,
                             merged=h_synced, pre_shared=block_start)
        if supervise:
            t_l = t_caps.pop(i).detach()
            r = tap_loss(h_synced, t_l)
            seg_loss = seg_loss + r
            seg_count += 1
            block_loss_val = block_loss_val + r.detach()
            layer_relmse[i] = float(r.detach())
        if is_boundary:
            _flush(seg_loss, seg_count)
            seg_loss = torch.zeros((), device=device)
            seg_count = 0
            block_start = t_l  # teacher-forced carry (see the sf note above)
            block_input = mlp(i, share(t_l))
            if probe is not None:
                # Every track's MLP reads the SAME teacher-exact residual here, so
                # this row's only student-side error is the MLP weights themselves —
                # at step 0 the merged state must reproduce the teacher's post-MLP,
                # which is the probe's alignment rail.
                probe.record(i, "mlp", block_input, share(t_l))
        elif i == last:
            _flush(seg_loss, seg_count)
        else:
            block_input = new_h  # carry the partial
    prof_block.__exit__(None, None, None)

    n_taps = max(1, len(layer_relmse))
    block_loss_val = block_loss_val / n_taps

    # ----- Output objective: full free-running forward + chunked CE -----
    # The free-running output needs a direct output objective: teacher-forced
    # block-MSE alone left it untrained — reconstructing own-carry from teacher
    # inputs didn't transfer to free-running generation. The two are
    # COMPLEMENTARY, not redundant: dropping block made CE itself 50% worse
    # (3.262 vs 2.168) because block anchors the trajectory to a regime the
    # frozen lm_head can still decode, and neither alone clears macro ~0.55
    # while together they reach 0.595.
    ce_val = torch.zeros((), device=device)
    if need_ce:
        with prof.phase("fr_forward"):
            hidden, _ = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_sync_hiddens=False,
                return_hidden_pre_lm_head=True,
            )
        with prof.phase("ce_chunked"):
            ce_val, grad_h = ce_chunked(
                hidden, lm_head, labels,
                lambda_ce=cfg.lambda_ce,
                chunk_size=cfg.ce_chunk_size,
                loss_scale=loss_scale,
            )
        if grad_h is not None:
            # Backward through the checkpointed full forward: this RECOMPUTES every
            # sublayer, so it carries a whole extra forward's worth of work.
            with prof.phase("fr_backward"):
                torch.autograd.backward([hidden], [grad_h])

    total_val = cfg.lambda_block * block_loss_val + cfg.lambda_ce * ce_val
    return {
        "total": total_val,
        "block_mse": block_loss_val,
        "ce": ce_val,
        "layer_relmse": layer_relmse,  # detached floats, per supervised tap
    }
