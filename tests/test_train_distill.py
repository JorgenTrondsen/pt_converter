"""Rails for the rebuilt d1b-heal trainer (post-attn walk + distill step).

Rail 1 — exact schedule ≡ dense: ``sync_phase="exact"`` (2 syncs/layer) must
reproduce the dense forward at N=4 — the frozen-slice TEACHER's correctness.

Rail 2 — N=1 parity: every phase walk must reduce to the dense forward when
the SyncBoundary is a no-op.

Rail 3 — zero-loss identity: distill_step(student == teacher slices, N=1)
must read block_mse ≈ 0 (any target-phase misalignment between the teacher
captures and the student taps breaks this instantly).

Rail 4 — chunked CE ≡ direct full-logits computation (value and the gradient
w.r.t. the hidden state).

NOTE: the old rail 5 (sf=1.0 chain ≡ deployed forward) went with student
forcing when it was removed 2026-07-30 — the block loop is now always
teacher-forced, so there is no free-running variant of it left to compare.
That check has no replacement here; `git log` has it.

Rail 6 — gradients flow to every track's layer params through the step, at both
the d1b and a fixed-D=2 (gapped) schedule.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.engine import _submodule
from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.convert import slice_model_to_tracks
from parallm.train.distill import (
    DistillConfig,
    distill_step,
    freeze_slice_teacher,
    ce_chunked,
    teacher_forward,
)
from parallm.train.losses import block_mse


# Unchunked reference objective for rail 4 — the whole point of ce_chunked is
# that it never materializes these (B, T, V) logits, so they live only here.
def lm_cross_entropy(logits, labels, ignore_index=-100):
    shift = logits[:, :-1, :]
    return F.cross_entropy(shift.reshape(-1, shift.size(-1)).float(),
                           labels[:, 1:].reshape(-1), ignore_index=ignore_index)


def _tiny_config(n_layers: int = 8):
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        * (n_layers // 4),
        full_attention_interval=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )


def _build(n_tracks: int, sync_after=None, n_layers: int = 8, fuse_size: int = 1,
           merge_group: int = 1, exec_groups: int = 1):
    cfg = _tiny_config(n_layers)
    torch.manual_seed(42)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.manual_seed(7)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    dense.eval()

    n_shards = n_tracks * merge_group
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_shards, sync_block_depth=4, text_config_attr="config"
    )
    states = dict(enumerate(tracks))
    if merge_group > 1:
        from parallm.adapters import get_adapter_for_config
        from parallm.model.merge import merge_track_states

        states = merge_track_states(
            get_adapter_for_config(cfg), cfg, n_shards, states, merge_group
        )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=list(sync_after) if sync_after is not None else manifest.sync_layer_indices,
        track_group=None,
        fuse_size=fuse_size,
        merge_group=merge_group,
        exec_groups=exec_groups,
    )
    pt.eval()
    pt.load_track_state_dicts(states, strict=False)
    return cfg, dense, tracks, pt


def _dense_logits(dense, input_ids, attention_mask=None):
    with torch.no_grad():
        out = dense(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return dense.lm_head(out.last_hidden_state)


def _batch(cfg, B=1, T=16, seed=123):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones((B, T), dtype=torch.long),
    }


def test_exact_phase_matches_dense_n4():
    cfg, dense, _tracks, pt = _build(n_tracks=4)
    pt.set_sync_phase("exact")
    batch = _batch(cfg)
    with torch.no_grad():
        pt_logits, _ = pt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    ref = _dense_logits(dense, batch["input_ids"], batch["attention_mask"])
    max_abs = (ref - pt_logits).abs().max().item()
    assert max_abs < 5e-4, f"exact schedule drifts from dense by {max_abs}"


def test_post_attn_n1_matches_dense():
    cfg, dense, _tracks, pt = _build(n_tracks=1, sync_after=list(range(8)))
    pt.set_sync_phase("post-attn")
    batch = _batch(cfg)
    with torch.no_grad():
        pt_logits, _ = pt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    ref = _dense_logits(dense, batch["input_ids"], batch["attention_mask"])
    max_abs = (ref - pt_logits).abs().max().item()
    assert max_abs < 1e-4, f"post-attn N=1 drifts from dense by {max_abs}"


def test_distill_step_zero_loss_identity_n1():
    cfg, dense, tracks, pt = _build(n_tracks=1, sync_after=list(range(8)))
    pt.set_sync_phase("post-attn")
    pt.train()

    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=1, local_track_ids=(0,),
        sync_after_layers=list(range(8)), track_group=None,
    )
    teacher_pt.load_track_state_dicts({0: tracks[0]}, strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    dcfg = DistillConfig(sync_layer_indices=tuple(range(8)))
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg)
    assert losses["block_mse"].item() < 1e-8, f"block_mse {losses['block_mse'].item()} not ~0 at N=1"
    assert torch.isfinite(losses["ce"]), "ce not finite"


def test_ce_chunked_matches_direct():
    torch.manual_seed(0)
    B, T, H, V = 2, 13, 8, 31
    lm_head = nn.Linear(H, V, bias=False)
    lm_head.weight.requires_grad_(False)
    hidden = torch.randn(B, T, H, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    lam_ce = 0.3

    ce, grad_h = ce_chunked(hidden, lm_head, labels, lambda_ce=lam_ce, chunk_size=5)

    s_logits = lm_head(hidden)
    ce_ref = lm_cross_entropy(s_logits, labels)
    assert torch.allclose(ce, ce_ref, atol=1e-5), (ce.item(), ce_ref.item())

    (lam_ce * ce_ref).backward()
    assert torch.allclose(grad_h, hidden.grad, atol=1e-5)


def test_distill_step_grads_flow_n4():
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=list(range(8)))
    pt.set_sync_phase("post-attn")
    pt.train()
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(8)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    dcfg = DistillConfig(sync_layer_indices=tuple(range(8)))
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg)
    assert torch.isfinite(losses["total"])
    for k, tm in enumerate(pt.text_models):
        got = sum(1 for p_ in tm.layers.parameters() if p_.grad is not None and p_.grad.abs().sum() > 0)
        assert got > 0, f"track {k}: no layer param received gradient"
    # Teacher stayed frozen and untouched.
    assert all(not p_.requires_grad for p_ in teacher.parameters())


def test_distill_step_batched_merged_matches_looped_unfused():
    """The TF block loop must mirror `pt_model._run_post_attn_stack` sublayer for
    sublayer, or the block targets are measured against a different model than the
    one the free-running forward trains. The two walks share `mix`/`mlp`/`sync`
    adapters precisely so they cannot drift; this pins that they haven't.

    Checkpointing is ON so the batched path is exercised through recompute, which
    is where the two representations could disagree on saved-tensor order.
    """
    sched = [1, 3, 5, 7]
    cfg, _dense, tracks, looped = _build(n_tracks=4, sync_after=sched)
    _cfg2, _d2, _t2, merged = _build(n_tracks=1, sync_after=sched,
                                     merge_group=4, exec_groups=4)
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(8)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)  # exact schedule: layout-independent

    dcfg = DistillConfig(sync_layer_indices=tuple(sched))
    batch = _batch(cfg)
    out = []
    for pt in (looped, merged):
        pt.set_sync_phase("post-attn")
        pt.use_checkpoint = True
        pt.train()
        out.append(distill_step(pt, teacher, pt.lm_head, batch, dcfg))
    lo, me = out

    assert abs(lo["block_mse"].item() - me["block_mse"].item()) < 1e-5, (
        f"block_mse {lo['block_mse'].item()} vs {me['block_mse'].item()}")
    assert abs(lo["ce"].item() - me["ce"].item()) < 1e-4, (
        f"ce {lo['ce'].item()} vs {me['ce'].item()}")
    assert lo["layer_relmse"].keys() == me["layer_relmse"].keys()
    for i, r in lo["layer_relmse"].items():
        assert abs(r - me["layer_relmse"][i]) < 1e-5, f"tap {i}: {r} vs {me['layer_relmse'][i]}"
    # Non-vacuous: the taps carry real error, not ~0 from a degenerate walk.
    assert max(lo["layer_relmse"].values()) > 1e-6

    got = sum(1 for p_ in merged.text_models[0].layers.parameters()
              if p_.grad is not None and p_.grad.abs().sum() > 0)
    assert got > 0, "no gradient reached the merged weights"


def test_merged_shadow_tracks_live_parameters():
    """`MergedShadow.stacked` must read through to the parameter every call.

    Caching it looks free — `DenseShadow` does exactly that — but this provider's
    weights are being TRAINED, and `_regroup_qkv` returns a `cat`, i.e. a copy.
    A cached copy would pin the GDN qkv and conv weights at their step-0 values
    while the optimizer moved the real ones, and nothing in the loss would say so.
    Checked on the GDN params specifically, because they are the only ones whose
    unmerge is not a view.
    """
    _cfg, _dense, _tracks, merged = _build(n_tracks=1, sync_after=[1, 3, 5, 7],
                                           merge_group=4, exec_groups=4)
    for path in ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight",
                 "mlp.down_proj.weight", "self_attn.q_norm.weight"):
        li = 0 if "linear_attn" in path else 3  # layer 3 is the full-attention one
        before = merged.shadow.stacked(li, path).clone()
        with torch.no_grad():
            _submodule(merged.text_models[0].layers[li], path).add_(1.0)
        after = merged.shadow.stacked(li, path)
        assert torch.allclose(after, before + 1.0), f"{path} is stale — stacked() cached"


def test_distill_step_fixed_d2_schedule_n4():
    """The fixed-D=2 re-heal: student BUILT at the gapped schedule (own-carry at
    layers 0,2,4,6), checkpointing on. Grads must reach every track through both
    the own-carry TF loop and the free-running CE forward."""
    sched = [1, 3, 5, 7]
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=sched)
    pt.set_sync_phase("post-attn")
    pt.use_checkpoint = True  # the deployed FR-at-D>1 path is checkpointed
    pt.train()
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(8)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    dcfg = DistillConfig(sync_layer_indices=tuple(sched))
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg)
    assert torch.isfinite(losses["total"])
    assert losses["block_mse"].item() > 0 and losses["ce"].item() > 0
    for k, tm in enumerate(pt.text_models):
        got = sum(1 for p_ in tm.layers.parameters()
                  if p_.grad is not None and p_.grad.abs().sum() > 0)
        assert got > 0, f"track {k}: no grad at the fixed-D=2 schedule"


def test_direction_magnitude_separates_what_relmse_conflates():
    """`block_direction_magnitude` must split the two failures relMSE merges.

    relMSE ≈ 1 − 2·cos·r + r², so a state pointing the wrong way and one that
    points the right way but is short can score IDENTICALLY. They are not the
    same defect: a gain error is something the next layer can absorb, a
    direction error is missing content. The pair is only worth reporting if it
    tells those two apart where relMSE cannot — so construct exactly that case.
    """
    from parallm.train.losses import block_direction_magnitude

    torch.manual_seed(0)
    B, T, H = 2, 6, 32
    t = torch.randn(B, T, H)

    # (a) pure MAGNITUDE error: same direction, scaled by r.
    r = 0.8
    s_mag = t * r
    # (b) pure DIRECTION error: unit-norm-preserving, tuned so relMSE matches.
    # For an orthogonal perturbation of relative size d, relMSE = d²/(1+d²) at
    # matched norm... simplest is to search a mix that ties relMSE to (a)'s.
    noise = torch.randn(B, T, H)
    noise = noise - (noise * t).sum(-1, keepdim=True) / t.pow(2).sum(-1, keepdim=True) * t
    target_rel = block_mse(s_mag, t, normalize=True).item()
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cand = t + mid * noise * (t.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True))
        cand = cand * (t.norm(dim=-1, keepdim=True) / cand.norm(dim=-1, keepdim=True))
        if block_mse(cand, t, normalize=True).item() < target_rel:
            lo = mid
        else:
            hi = mid
    s_dir = t + lo * noise * (t.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True))
    s_dir = s_dir * (t.norm(dim=-1, keepdim=True) / s_dir.norm(dim=-1, keepdim=True))

    rel_mag = block_mse(s_mag, t, normalize=True).item()
    rel_dir = block_mse(s_dir, t, normalize=True).item()
    assert abs(rel_mag - rel_dir) < 0.02 * rel_mag, (
        f"fixture failed to tie relMSE: {rel_mag} vs {rel_dir}"
    )

    cos_mag, r_mag = block_direction_magnitude(s_mag, t)
    cos_dir, r_dir = block_direction_magnitude(s_dir, t)

    # Same relMSE, opposite anatomy — this is the whole point of the metric.
    # Closed forms at this fixture: pure scale gives rel=(1−r)², cos=1, ratio=r;
    # norm-preserving rotation gives rel=2(1−cos), ratio=1. So with r=0.8 both
    # score rel=0.04 while cos is 1.000 vs 0.980 and ratio is 0.800 vs 1.000.
    assert cos_mag.item() > 0.999, f"pure-scale error should keep direction: {cos_mag}"
    assert abs(r_mag.item() - r) < 1e-4, f"norm ratio should recover {r}: {r_mag}"
    assert abs(r_dir.item() - 1.0) < 1e-4, f"norm-preserving, got r={r_dir}"
    assert abs(cos_dir.item() - (1.0 - rel_dir / 2)) < 1e-3, (
        f"rotation cosine should be 1−rel/2={1 - rel_dir / 2:.4f}, got {cos_dir.item():.4f}"
    )
    # The claim that matters: relMSE ties to <2% while BOTH new coordinates
    # separate by orders of magnitude more than that tie tolerance.
    tie = 0.02 * rel_mag
    assert (cos_mag - cos_dir).item() > 10 * tie, "cosine did not separate the two"
    assert (r_dir - r_mag).item() > 10 * tie, "norm ratio did not separate the two"

    # Exact match is the degenerate anchor.
    cos_id, r_id = block_direction_magnitude(t, t)
    assert abs(cos_id.item() - 1.0) < 1e-6 and abs(r_id.item() - 1.0) < 1e-6

    # Padding is excluded, like block_mse: corrupting a masked position is a no-op.
    mask = torch.ones(B, T)
    mask[:, -2:] = 0
    s_pad = s_mag.clone()
    s_pad[:, -2:] = 1e3
    cos_a, r_a = block_direction_magnitude(s_mag, t, attention_mask=mask)
    cos_b, r_b = block_direction_magnitude(s_pad, t, attention_mask=mask)
    assert torch.allclose(cos_a, cos_b) and torch.allclose(r_a, r_b)


def _d2_step_fixture(sched=(1, 3, 5, 7)):
    """A D=2 student + exact-schedule frozen teacher, ready for `distill_step`."""
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=list(sched))
    pt.set_sync_phase("post-attn")
    pt.train()
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(cfg.num_hidden_layers)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    return cfg, pt, freeze_slice_teacher(teacher_pt), tuple(sched)


def test_lambda_mag_is_inert_by_default_and_live_when_set():
    """`--lambda-mag` must not perturb existing recipes at its default, and must
    actually change the step when set.

    It targets a measured defect in the relMSE tap (2026-08-04): the FIRST
    boundary at D=2 is cos 0.935 / r 5.4 = 99% magnitude while being ~93% of the
    block loss, yet magnitude is the failure training fixes on its own. What
    survives is direction, so the knob moves the gradient budget onto it.
    """
    cfg, pt, teacher, sync_after = _d2_step_fixture()
    batch = _batch(cfg)

    def run(**kw):
        for p in pt.parameters():
            if p.grad is not None:
                p.grad = None
        torch.manual_seed(0)
        out = distill_step(pt, teacher, pt.lm_head, batch,
                           DistillConfig(sync_layer_indices=sync_after, **kw))
        return out, {n: p.grad.detach().clone()
                     for n, p in pt.named_parameters() if p.grad is not None}

    base, g_base = run()
    off, g_off = run(lambda_mag=None)
    mag, g_mag = run(lambda_mag=0.1)

    # Inert at the default: bit-identical to not passing it at all.
    assert torch.equal(base["block_mse"], off["block_mse"])
    for n in g_base:
        assert torch.equal(g_base[n], g_off[n]), f"the default perturbed {n}"

    # Live when set — and it reaches the weights.
    assert not torch.equal(base["block_mse"], mag["block_mse"]), (
        "lambda_mag did not change the block loss"
    )
    changed = [n for n in g_base if not torch.equal(g_base[n], g_mag[n])]
    assert changed, "lambda_mag changed no gradient — the knob is a no-op"


def test_block_split_reconstructs_relmse_and_isolates_the_two_failures():
    """`block_split` must be the honest decomposition of the tap it replaces.

    relMSE ≈ 1 − 2·cos·r + r², so direction=2(1−cos) and magnitude=log(r)² have
    to track the same underlying quantities — otherwise the weighting decisions
    made from them are made from a different number than the one measured.
    """
    from parallm.train.losses import block_mse, block_split

    torch.manual_seed(0)
    t = torch.randn(2, 6, 32)

    # Pure scale error: direction term ~0, magnitude term = log(r)^2 exactly.
    r = 2.5
    d, m = block_split(t * r, t)
    assert d.item() < 1e-6, f"pure scale error leaked into direction: {d.item()}"
    assert abs(m.item() - torch.tensor(r).log().pow(2).item()) < 1e-5

    # Exact match is zero on both.
    d0, m0 = block_split(t, t)
    assert d0.item() < 1e-6 and m0.item() < 1e-6

    # The identity relMSE = 1 - 2·cos·r + r² is EXACT PER TOKEN. relMSE is a
    # ratio of sums while the split averages two nonlinear per-token functions,
    # so they only agree exactly when there is one token to average — check it
    # there, where any algebra error is unmissable.
    t1 = torch.randn(1, 1, 32)
    s1 = t1 + 0.4 * torch.randn_like(t1)
    d, m = block_split(s1, t1)
    cos = 1.0 - d.item() / 2.0
    ratio = float(torch.exp(torch.sqrt(m)))
    exact = 1.0 - 2.0 * cos * ratio + ratio**2
    rel1 = block_mse(s1, t1, normalize=True).item()
    assert abs(exact - rel1) < 1e-4 * max(rel1, 1.0), (
        f"per-token identity broken: {exact:.6f} vs {rel1:.6f}"
    )

    # In aggregate the two diverge by Jensen, but must stay the same order —
    # the weighting decisions in the plan are read off real profiles where the
    # gap measured 1-4%, so a wild divergence here would invalidate them.
    s = t + 0.4 * torch.randn_like(t)
    d, m = block_split(s, t)
    cos = 1.0 - d.item() / 2.0
    ratio = float(torch.exp(torch.sqrt(m)))
    approx = 1.0 - 2.0 * cos * ratio + ratio**2
    rel = block_mse(s, t, normalize=True).item()
    assert 0.5 * rel < approx < 2.0 * rel, (
        f"split and relMSE disagree by more than 2x: {approx:.4f} vs {rel:.4f}"
    )


def test_post_mlp_supervises_every_boundary_not_just_the_last_layer():
    """The mis-supervision regression: derive the targets from the BOUNDARY LIST and
    an empty attention set yields ONE target (at the final layer) instead of L.

    `capture_sets` now takes the two sets `sync_sets()` resolved, so a post-mlp
    schedule is supervised post-MLP at every boundary — and the TF loop taps the
    same depths, because it reads the same sets.
    """
    from parallm.train.distill import capture_sets

    L = 8
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=list(range(L)), n_layers=L)
    pt.set_sync_phase("post-mlp")
    pt.train()

    cap_attn, cap_mlp = capture_sets(*pt.sync_sets(), L)
    assert cap_attn == set()
    assert cap_mlp == set(range(L)), f"got {sorted(cap_mlp)}"

    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(L)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    losses = distill_step(pt, teacher, pt.lm_head, _batch(cfg),
                          DistillConfig(sync_layer_indices=tuple(range(L))))
    # One tap per boundary, not one for the whole stack.
    assert sorted(losses["layer_relmse"]) == list(range(L)), losses["layer_relmse"]
    assert torch.isfinite(losses["total"])
    assert losses["block_mse"].item() > 0
    for k, tm in enumerate(pt.text_models):
        got = sum(1 for p_ in tm.layers.parameters()
                  if p_.grad is not None and p_.grad.abs().sum() > 0)
        assert got > 0, f"track {k}: no grad at post-mlp"


def test_post_mlp_block_loop_taps_the_forward_it_deploys():
    """The TF loop and the model walk must agree on what a post-mlp boundary state
    IS. With student == teacher slices at N=1 the SyncBoundary is a no-op, so every
    tap has to land on its own target and block_mse collapses to ~0 — the same rail
    post-attn has, at the phase that used to have no trainer at all."""
    L = 8
    cfg, _dense, tracks, pt = _build(n_tracks=1, sync_after=list(range(L)), n_layers=L)
    pt.set_sync_phase("post-mlp")
    pt.train()
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=1, local_track_ids=(0,),
        sync_after_layers=list(range(L)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    losses = distill_step(pt, teacher, pt.lm_head, _batch(cfg),
                          DistillConfig(sync_layer_indices=tuple(range(L)), lambda_ce=0.0))
    assert losses["block_mse"].item() < 1e-8, losses["layer_relmse"]


def test_block_walk_fr_carries_the_students_own_state_detached():
    """`--block-walk fr` hands each window the residual it will really get.

    The tf carry is the teacher's target `t_l`; the fr carry is the student's own
    synced output. Both are detached, which is what keeps `_flush`'s per-segment
    backward — and the peak memory — identical between the two walks.
    """
    L = 8
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=list(range(L)), n_layers=L)
    pt.set_sync_phase("post-attn")
    pt.train()
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(L)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    kw = dict(sync_layer_indices=tuple(range(L)), lambda_ce=0.0)
    tf_out = distill_step(pt, teacher, pt.lm_head, _batch(cfg), DistillConfig(**kw))
    pt.zero_grad(set_to_none=True)
    fr_out = distill_step(pt, teacher, pt.lm_head, _batch(cfg),
                          DistillConfig(**kw, block_walk="fr"))

    # Same targets, different inputs => a different loss. If these matched, the flag
    # would be doing nothing.
    assert fr_out["block_mse"].item() != tf_out["block_mse"].item()
    # The student's own trajectory is WORSE than the teacher's, so every layer past
    # the first is handed a drifted input and the free-running loss must be larger.
    assert fr_out["block_mse"].item() > tf_out["block_mse"].item()
    # Both walks still supervise every boundary.
    assert sorted(fr_out["layer_relmse"]) == sorted(tf_out["layer_relmse"]) == list(range(L))
    # Gradients still reach every track through the fr walk.
    for k, tm in enumerate(pt.text_models):
        got = sum(1 for p_ in tm.layers.parameters()
                  if p_.grad is not None and p_.grad.abs().sum() > 0)
        assert got > 0, f"track {k}: no grad under block_walk=fr"


def test_block_walk_fr_is_identity_when_the_student_is_the_teacher():
    """At N=1 with student slices == teacher slices the SyncBoundary is a no-op, so
    the student's own trajectory IS the teacher's — the two walks must coincide and
    both read ~0. A carry wired to the wrong tensor breaks this immediately."""
    L = 8
    cfg, _dense, tracks, pt = _build(n_tracks=1, sync_after=list(range(L)), n_layers=L)
    pt.set_sync_phase("post-attn")
    pt.train()
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=1, local_track_ids=(0,),
        sync_after_layers=list(range(L)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    kw = dict(sync_layer_indices=tuple(range(L)), lambda_ce=0.0)
    for walk in ("tf", "fr"):
        out = distill_step(pt, teacher, pt.lm_head, _batch(cfg),
                           DistillConfig(**kw, block_walk=walk))
        assert out["block_mse"].item() < 1e-8, f"{walk}: {out['layer_relmse']}"
