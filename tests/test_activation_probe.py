"""Rails for the per-layer / per-track activation probe (`train/probe.py`).

The rail that matters is the ALIGNMENT one. An earlier per-layer fidelity panel
compared the student's post-attention residual against the teacher's post-MLP
hidden at the same index for months — same layer, different phase — and every
per-layer number it produced was void. Nothing in that panel could have caught
it, because a plausible-looking error is exactly what a phase shift produces.

This probe is falsifiable by construction, and these tests pin the two places it
must read ~0 or the wiring is wrong:

* **post-MLP** — every track's MLP reads the teacher's own post-attn residual
  through weights that ARE the teacher's, so the merged post-MLP state must
  reproduce the teacher's post-MLP hidden. Any (layer, phase) misalignment,
  off-by-one in the teacher pre-state, or wrong sync pre-state breaks it.
* **post-MLP normed input** — student and teacher normalize the IDENTICAL tensor,
  so cos = 1, ratio = 1, relMSE = 0 exactly while their norm weights agree.

And the counter-check that keeps those from being vacuous: **post-attn must NOT
be ~0** at N>1, because that row is the d1b seam — the mixer reads a residual
carrying only its own track's share of the previous layer's MLP.
"""
from __future__ import annotations

import json

import torch

torch.set_default_dtype(torch.float32)

from parallm.model.pt_model import PTWrappedModel
from parallm.train.distill import DistillConfig, distill_step, freeze_slice_teacher
from parallm.train.probe import METRICS, ActivationProbe

from tests.test_train_distill import _batch, _build

L = 8
SCHED = list(range(L))


def _teacher(cfg, tracks, n_tracks=4):
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=n_tracks, local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=SCHED, track_group=None,
    )
    pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    return freeze_slice_teacher(pt)


def _probe(pt, teacher, n_streams, out_dir, n_shards=4):
    return ActivationProbe(
        num_layers=L,
        n_streams=n_streams,
        stream_offset=0,
        shards_per_stream=n_shards // n_streams,
        rank=0,
        out_dir=out_dir,
        device=torch.device("cpu"),
        teacher_layers=teacher.text_models[0].layers,
    )


def _run(tmp_path, n_tracks=4, merge_group=1, exec_groups=1, seed_shift=False):
    """One probed d1b step with student slices == teacher slices."""
    cfg, _dense, tracks, pt = _build(
        n_tracks=n_tracks, sync_after=SCHED, n_layers=L,
        merge_group=merge_group, exec_groups=exec_groups,
    )
    n_shards = n_tracks * merge_group
    _cfg2, _d2, tracks_ref, _pt2 = _build(n_tracks=n_shards, sync_after=SCHED, n_layers=L)
    teacher = _teacher(cfg, tracks_ref, n_tracks=n_shards)
    pt.set_sync_phase("post-attn")
    pt.train()

    n_streams = exec_groups if exec_groups > 1 else n_tracks
    probe = _probe(pt, teacher, n_streams, tmp_path, n_shards=n_shards)
    probe.step = 0
    losses = distill_step(
        pt, teacher, pt.lm_head, _batch(cfg),
        DistillConfig(sync_layer_indices=tuple(SCHED)), probe=probe,
    )
    probe.flush()
    rows = [json.loads(l) for l in probe.path.read_text().splitlines()]
    return losses, [r for r in rows if "meta" not in r], n_streams


def _by(rows, phase, merged=None, walk="tf"):
    """Rows for one phase. ``walk`` defaults to the teacher-forced ones: the
    free-running walk emits its own merged row per (layer, phase), so selecting on
    ``track == -1`` alone now matches two rows, not one."""
    out = [r for r in rows
           if r["phase"] == phase and r["walk"] == walk and "detail" not in r]
    if merged is True:
        out = [r for r in out if r["track"] == -1]
    elif merged is False:
        out = [r for r in out if r["track"] != -1]
    return out


def test_post_mlp_reproduces_the_teacher_the_alignment_rail(tmp_path):
    """THE rail: with student weights == teacher weights, the merged post-MLP
    state must BE the teacher's post-MLP hidden — bit-exact, not merely close.

    Everything feeding that row is teacher-exact: at a boundary the MLP half
    reads the teacher's own post-attn tensor, shared by every track. So the only
    way it can be non-zero is a wiring error in the probe itself.

    NO LAYER IS EXEMPT. The final layer used to be — it was excluded from the
    post-attn sync set, so its mixer ran on the partial and this row read 0.345 at
    d1b/N=64. That was a defect, not a law: it is a full boundary now, and repairing
    it took that row to the bf16 floor and bought +0.2157 mmlu_pro_math_mc.
    """
    _losses, rows, _k = _run(tmp_path)
    merged = {r["layer"]: r for r in _by(rows, "mlp", merged=True)}
    assert len(merged) == L
    for i in range(L):
        assert merged[i]["res_relmse"] == 0.0, (
            f"L{i} post-mlp merged relMSE {merged[i]['res_relmse']:.3e} — misaligned"
        )
        assert abs(merged[i]["res_cos"] - 1.0) < 1e-6
        assert abs(merged[i]["res_nr"] - 1.0) < 1e-6


def test_post_attn_is_not_zero_or_the_rail_above_is_vacuous(tmp_path):
    """The counter-check. post-attn carries the d1b seam: the mixer reads a
    residual holding only this track's share of the previous layer's MLP delta,
    so it MUST diverge from the teacher — at every layer past the first, where
    there is no previous MLP yet."""
    _losses, rows, _k = _run(tmp_path)
    merged = _by(rows, "attn", merged=True)
    assert len(merged) == L
    assert merged[0]["res_relmse"] < 1e-8, (
        "layer 0's mixer reads the embedding, which is shared — it must be exact"
    )
    deeper = [r["res_relmse"] for r in merged if r["layer"] > 0]
    assert min(deeper) > 1e-6, (
        f"post-attn error vanished ({min(deeper):.3e}) — then the post-mlp rail "
        f"proves nothing, because every row would read ~0"
    )


def test_post_mlp_normed_input_is_a_pure_norm_drift_meter(tmp_path):
    """At a boundary, student and teacher hand the second norm the IDENTICAL
    tensor, so while their norm weights agree this row is exactly zero. That is
    what later makes it the size of the confound in the post-attn nin_* row,
    which mixes residual error with `input_layernorm.weight` drift.

    The final layer is included: it is a full boundary, so its MLP normalizes the
    teacher's residual like every other boundary's.
    """
    _losses, rows, _k = _run(tmp_path)
    scored = _by(rows, "mlp", merged=False)
    assert scored
    for r in scored:
        assert r["nin_relmse"] == 0.0, (r["layer"], r["track"], r["nin_relmse"])
        assert abs(r["nin_cos"] - 1.0) < 1e-6
        assert abs(r["nin_nr"] - 1.0) < 1e-6


def test_merged_rows_report_no_normed_input(tmp_path):
    """There is no such thing as a merged normed input: each track norms its own
    pre-state and the walk never forms a combined one. The merged row must say
    so with a null rather than inventing a number — a 0.0 there would read as
    'the merged input matches the teacher exactly', which is not a measurement.
    """
    _losses, rows, _k = _run(tmp_path)
    for r in [x for x in rows if x["track"] == -1]:
        assert r["nin_cos"] is None and r["nin_nr"] is None and r["nin_relmse"] is None
    for r in [x for x in rows if x["track"] != -1]:
        assert r["nin_relmse"] is not None


def test_grid_is_complete_and_coherence_is_a_ratio(tmp_path):
    _losses, rows, k = _run(tmp_path)
    # The teacher-forced grid: every layer x phase x (tracks + merged). The
    # free-running walk adds its own merged rows on top, counted separately below.
    tf_rows = [r for r in rows if r["walk"] == "tf"]
    assert len(tf_rows) == L * 2 * (k + 1), f"{len(tf_rows)} tf rows, want {L * 2 * (k + 1)}"
    for r in rows:
        assert set(METRICS) <= set(r)
        # Every value is either a real number or an explicit null; never a NaN
        # that survived into JSON, which json.loads would hand back as float nan.
        assert all(r[m] is None or r[m] == r[m] for m in METRICS), f"NaN in {r}"
    for r in [x for x in tf_rows if x["track"] == -1]:
        # ‖Σδ‖ / Σ‖δ‖: 1 when the tracks are redundant, →1/√N when orthogonal.
        assert 0.0 < r["coh"] <= 1.0 + 1e-6, r
    # A track's own delta is ~1/N of the teacher's, which is the whole reason
    # d_cos rather than d_nr is the per-track signal.
    per_track = _by(rows, "mlp", merged=False)
    assert max(r["d_nr"] for r in per_track) < 0.9


def test_track_gram_reports_effective_directions(tmp_path):
    """`gram_pr` must be the effective number of independent directions the tracks
    span — N when orthogonal, 1 when parallel — and it must live only on the merged
    row, since it is a property of the track SET, not of any one track.

    Tracks with a zero delta must be DROPPED, not counted: legacy `throughput`
    shards strand some with no MLP at all, and counting them would dilute the mean
    and cap `gram_pr` below the truth.
    """
    from parallm.train.probe import track_gram

    v = torch.randn(3, 4, 8)
    mean, mx, pr = track_gram([v, v * 2.0, v * 0.5], None)   # all one direction
    assert abs(mean - 1.0) < 1e-4 and abs(mx - 1.0) < 1e-4
    assert abs(pr - 1.0) < 1e-3, f"parallel tracks must span 1 direction, got {pr}"

    # gram_mean is SIGNED so it stays commensurable with coh/redun, which are net
    # sums; gram_max is absolute, for spotting a near-duplicate pair either way.
    mean, mx, _pr = track_gram([v, v * -1.0, v * -1.0], None)
    assert mean < 0.0 and abs(mx - 1.0) < 1e-4, (mean, mx)

    orth = list(torch.eye(3).reshape(3, 3, 1, 1) * torch.ones(1, 3, 4, 8))
    mean, mx, pr = track_gram(orth, None)
    assert abs(mean) < 1e-6 and abs(pr - 3.0) < 1e-3, f"orthogonal -> 3 dirs, got {pr}"

    # A dead track (no MLP slab at all) must not be counted as a direction.
    mean, mx, pr = track_gram(orth + [torch.zeros(3, 4, 8)], None)
    assert abs(pr - 3.0) < 1e-3, f"zero-delta track inflated the count: {pr}"

    _losses, rows, k = _run(tmp_path)
    # Merged TEACHER-FORCED rows only: the free-running walk has no per-track
    # dimension, so it has no Gram to report and leaves these null.
    for r in rows:
        if r["track"] == -1 and r["walk"] == "tf":
            assert r["gram_pr"] is not None and 1.0 <= r["gram_pr"] <= k + 1e-6
            # signed mean, absolute max: -1 <= mean <= |mean| <= max <= 1
            assert -1.0 <= r["gram_mean"] <= 1.0
            assert abs(r["gram_mean"]) <= r["gram_max"] <= 1.0 + 1e-6
        else:
            assert r["gram_pr"] is None, "gram is a set property, not a per-track one"


def test_looped_and_batched_exec_agree(tmp_path):
    """The batched fold drives the same streams as one [G,B,T,H] tensor. If the
    probe reads them differently from the looped list, every per-track number is
    representation-dependent and none of them mean anything."""
    _lo, rows_lo, k_lo = _run(tmp_path / "looped", n_tracks=4)
    _me, rows_me, k_me = _run(tmp_path / "merged", n_tracks=1,
                              merge_group=4, exec_groups=4)
    assert k_lo == k_me == 4
    key = lambda r: (r["layer"], r["phase"], r["track"])
    lo = {key(r): r for r in rows_lo}
    me = {key(r): r for r in rows_me}
    assert lo.keys() == me.keys()
    for k, a in lo.items():
        for m in METRICS:
            if a[m] is None:
                assert me[k][m] is None, f"{k} {m}: null vs {me[k][m]}"
                continue
            assert abs(a[m] - me[k][m]) < 1e-4, f"{k} {m}: {a[m]} vs {me[k][m]}"


def test_d2_schedule_teacher_forces_only_the_boundaries(tmp_path):
    """At D>1 the alignment rail MOVES, and it must move to exactly the right
    layers or every D>1 reading is misattributed.

    Only a BOUNDARY layer's MLP reads the teacher's residual (the loop resets the
    carry there). A non-boundary layer's MLP reads the partial its own mixer
    produced, so it must NOT be exact — the rail that says the probe is reading the
    D>1 walk correctly rather than silently re-deriving the d1b one.
    """
    sched = [1, 3, 5, 7]                      # D=2 over 8 layers; L-1 IS a boundary
    boundaries = set(sched)
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=sched, n_layers=L)
    teacher = _teacher(cfg, tracks)
    pt.set_sync_phase("post-attn")
    pt.train()
    probe = _probe(pt, teacher, 4, tmp_path)
    probe.step = 0
    distill_step(pt, teacher, pt.lm_head, _batch(cfg),
                 DistillConfig(sync_layer_indices=tuple(sched)), probe=probe)
    probe.flush()
    rows = [json.loads(l) for l in probe.path.read_text().splitlines() if "meta" not in l]

    tf_rows = [r for r in rows if r["walk"] == "tf"]
    assert len(tf_rows) == L * 2 * (4 + 1), "the grid must stay complete at D>1"
    merged = {r["layer"]: r for r in _by(rows, "mlp", merged=True)}
    for i in sorted(boundaries):
        assert merged[i]["res_relmse"] == 0.0, (
            f"L{i} is a boundary: its MLP reads the teacher's residual, so it must "
            f"be exact — got {merged[i]['res_relmse']:.3e}"
        )
    for i in range(L):
        if i not in boundaries:
            assert merged[i]["res_relmse"] > 1e-6, (
                f"L{i} is NOT a boundary: its MLP reads the partial, so it must not "
                f"be exact. If it is, the probe is teacher-forcing the whole walk."
            )


def test_probe_pass_is_a_pure_read_and_is_batch_deterministic(tmp_path):
    """Pure read: no gradients, and bit-identical rows for the same model on the same
    batch — which is what makes a difference between probe steps attributable to the
    MODEL rather than to batch difficulty.
    """
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    teacher = _teacher(cfg, tracks)
    pt.set_sync_phase("post-attn")
    pt.train()
    fixed = _batch(cfg, seed=999)
    quiet = DistillConfig(sync_layer_indices=tuple(SCHED),
                          lambda_block=0.0, lambda_ce=0.0)

    def probe_pass(tag):
        probe = _probe(pt, teacher, 4, tmp_path / tag)
        probe.step = 7
        with torch.no_grad():
            distill_step(pt, teacher, pt.lm_head, fixed, quiet, probe=probe)
        probe.flush()
        return [json.loads(l) for l in probe.path.read_text().splitlines()
                if "meta" not in l]

    for p in pt.parameters():
        p.grad = None
    a = probe_pass("a")
    assert all(p.grad is None for p in pt.parameters()), "the probe pass TRAINED"
    assert len([r for r in a if r["walk"] == "tf"]) == L * 2 * (4 + 1), \
        "the probe pass must still fill the whole grid"

    b = probe_pass("b")
    key = lambda r: (r["layer"], r["phase"], r["track"])
    assert {key(r): r for r in a} == {key(r): r for r in b}, (
        "same model, same batch, different numbers — cross-step probe readings "
        "would then be measuring noise, not the model"
    )


def test_probing_does_not_change_the_step(tmp_path):
    """The probe must be a pure observer: same loss, same gradients. It detaches
    every capture, but it also runs extra syncs — those must not touch the graph
    the TF loop backwards through."""
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    teacher = _teacher(cfg, tracks)
    pt.set_sync_phase("post-attn")
    pt.train()
    dcfg = DistillConfig(sync_layer_indices=tuple(SCHED))
    batch = _batch(cfg)

    def run(probe):
        for p in pt.parameters():
            p.grad = None
        torch.manual_seed(0)
        out = distill_step(pt, teacher, pt.lm_head, batch, dcfg, probe=probe)
        return out, {n: p.grad.detach().clone()
                     for n, p in pt.named_parameters() if p.grad is not None}

    base, g_base = run(None)
    probe = _probe(pt, teacher, 4, tmp_path)
    probe.step = 3
    probed, g_probed = run(probe)

    assert torch.equal(base["total"], probed["total"])
    assert base["layer_relmse"] == probed["layer_relmse"]
    assert g_base.keys() == g_probed.keys()
    for n in g_base:
        assert torch.equal(g_base[n], g_probed[n]), f"probe perturbed grad {n}"


def test_probe_survives_streamed_teacher_layers(tmp_path):
    """`HostResidentLayers.release` empties a teacher layer's parameter storage
    after the forward passes it, so the probe cannot call the LIVE norm modules —
    it clones them from ``p._host`` at construction. Simulate that here: park the
    teacher's params the way the streamer does and check the numbers are
    unchanged."""
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    teacher = _teacher(cfg, tracks)
    probe_ref = _probe(pt, teacher, 4, tmp_path / "a")

    for layer in teacher.text_models[0].layers:
        for p in layer.parameters():
            p._host = p.data.clone()
            p.data = torch.empty(0)
    probe_streamed = _probe(pt, teacher, 4, tmp_path / "b")

    for k, mod in probe_ref._t_norms.items():
        a = next(mod.parameters())
        b = next(probe_streamed._t_norms[k].parameters())
        assert a.numel() > 0 and torch.equal(a, b), f"{k}: norm clone lost its weights"


def test_track_gram_masks_the_right_axes():
    """Masking to a position must equal slicing it. Right-aligned padding put the
    (B, T) mask's T on x's B; the all-ones training mask kept that unreachable."""
    from parallm.train.probe import track_gram

    torch.manual_seed(0)
    x = torch.randn(3, 4, 5, 8)  # (K, B, T, H), B != T so a wrong axis raises
    keep = torch.zeros(4, 5, dtype=torch.long)
    keep[:, 2] = 1

    masked = track_gram(list(x.unbind(0)), keep)
    sliced = track_gram(list(x[:, :, 2:3, :].unbind(0)), None)
    assert all(abs(a - b) < 1e-4 for a, b in zip(masked, sliced)), (masked, sliced)
    assert abs(masked[0] - track_gram(list(x.unbind(0)), None)[0]) > 1e-6


def test_post_mlp_phase_records_both_columns_and_inverts_the_rail(tmp_path):
    """The probe at `sync_phase="post-mlp"`, where the two columns swap meaning.

    post-attn has no sync there, so the probe all-reduces `h_attn` itself
    (`merged=None`) and that column stays the comparable one across phases. The
    post-mlp column is now the SEAM — the boundary MLP reads an OWN-CARRY input,
    which is exactly the sync this phase drops — so it is non-zero at depth even
    with student weights == teacher weights. Reading it as an alignment rail is the
    mistake this rail exists to prevent.
    """
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    _cfg2, _d2, tracks_ref, _pt2 = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    teacher = _teacher(cfg, tracks_ref)
    pt.set_sync_phase("post-mlp")
    pt.train()

    probe = _probe(pt, teacher, 4, tmp_path)
    probe.step = 0
    losses = distill_step(
        pt, teacher, pt.lm_head, _batch(cfg),
        DistillConfig(sync_layer_indices=tuple(SCHED)), probe=probe,
    )
    probe.flush()
    rows = [json.loads(l) for l in probe.path.read_text().splitlines() if "meta" not in l]

    # Both columns are populated at every depth — the merged post-attn row exists
    # only because the probe syncs it itself.
    for phase in ("attn", "mlp"):
        assert len(_by(rows, phase, merged=True)) == L, phase

    attn = {r["layer"]: r for r in _by(rows, "attn", merged=True)}
    mlp = {r["layer"]: r for r in _by(rows, "mlp", merged=True)}
    # Layer 0's mixer reads the embedding, which is shared, so it is exact in BOTH
    # phases — the same anchor the post-attn run has.
    assert attn[0]["res_relmse"] < 1e-8
    # The seam: post-mlp own-carry diverges from the teacher at depth. It is NOT a
    # rail, and asserting it is ~0 (as the post-attn column is) would fail.
    deeper = [mlp[i]["res_relmse"] for i in range(1, L)]
    assert min(deeper) > 1e-6, (
        f"post-mlp seam vanished ({min(deeper):.3e}) — then this phase drops nothing"
    )
    # And the objective sees it: one supervised tap per boundary.
    assert sorted(losses["layer_relmse"]) == sorted(SCHED)


def test_free_running_rows_measure_what_the_teacher_forced_rows_cannot(tmp_path):
    """The probe's FREE-RUNNING walk — the gap the teacher-forced rows leave.

    Every TF row hands its layer the teacher's residual, so it reports that layer's
    own error and by construction CANNOT see error accumulate. Every downstream
    number is free-running. `record_fr` scores the student's own walk against the
    same teacher captures, so FR/TF at a depth is that layer's amplification.

    With DAMAGED tracks (N=4, a real seam) the free-running error must be strictly
    larger than the teacher-forced one at depth — if it is not, `record_fr` is
    scoring the wrong tensors.
    """
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    _cfg2, _d2, tracks_ref, _pt2 = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    teacher = _teacher(cfg, tracks_ref)
    pt.set_sync_phase("post-attn")
    pt.train()

    probe = _probe(pt, teacher, 4, tmp_path)
    probe.step = 0
    distill_step(pt, teacher, pt.lm_head, _batch(cfg),
                 DistillConfig(sync_layer_indices=tuple(SCHED), lambda_ce=0.0),
                 probe=probe)
    probe.flush()
    rows = [json.loads(l) for l in probe.path.read_text().splitlines() if "meta" not in l]

    fr = {r["layer"]: r for r in rows if r.get("walk") == "fr" and r["phase"] == "attn"}
    tf = {r["layer"]: r for r in rows
          if r.get("walk") == "tf" and r["phase"] == "attn" and r["track"] == -1}
    # Every row is tagged, and the FR walk covers the depths the schedule syncs at.
    assert all("walk" in r for r in rows)
    assert set(fr) == set(SCHED), f"got {sorted(fr)}"
    assert set(tf) >= set(SCHED)
    # FR rows are merged-only — the student's walk hands back synced states.
    assert all(r["track"] == -1 for r in rows if r["walk"] == "fr")
    # THE POINT: accumulated drift is strictly worse than per-layer error.
    # ⚠ layer 0's mixer reads the EMBEDDING, which is shared, so its teacher-forced
    # error is exactly 0 and the ratio is undefined there — the same guard
    # `report_probe.amplification` applies. Amplification only means anything where
    # there is a previous layer to have accumulated from.
    deep = [i for i in sorted(fr) if tf[i]["res_relmse"] > 0]
    assert len(deep) >= 2, f"nothing to compare: {[(i, tf[i]['res_relmse']) for i in sorted(fr)]}"
    amps = [fr[i]["res_relmse"] / tf[i]["res_relmse"] for i in deep]
    # Free-running is never BETTER than teacher-forced — the extra error is the drift
    # it was handed. ⚠ the shallowest boundary reads exactly 1.0: layer 0's mixer sees
    # the shared embedding and its sync sums every track's delta, so that state is
    # teacher-exact and the two walks have not yet parted. That is the rail's own
    # sanity check, not a miss.
    assert min(amps) >= 1.0 - 1e-6, f"free-running beat teacher-forced: {amps}"
    assert sorted(amps)[len(amps) // 2] > 1.5, f"no compounding to speak of: {amps}"


def test_free_running_and_teacher_forced_agree_when_there_is_no_error(tmp_path):
    """At N=1 the SyncBoundary is a no-op, so the student's own trajectory IS the
    teacher's: both walks must read ~0. This is what says the FR rows are scored
    against the right targets rather than merely being large."""
    cfg, _dense, tracks, pt = _build(n_tracks=1, sync_after=SCHED, n_layers=L)
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=1, local_track_ids=(0,),
        sync_after_layers=SCHED, track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)
    pt.set_sync_phase("post-attn")
    pt.train()

    probe = _probe(pt, teacher, 1, tmp_path, n_shards=1)
    probe.step = 0
    distill_step(pt, teacher, pt.lm_head, _batch(cfg),
                 DistillConfig(sync_layer_indices=tuple(SCHED), lambda_ce=0.0),
                 probe=probe)
    probe.flush()
    rows = [json.loads(l) for l in probe.path.read_text().splitlines() if "meta" not in l]
    fr = [r for r in rows if r.get("walk") == "fr"]
    assert fr, "no free-running rows"
    for r in fr:
        assert r["res_relmse"] < 1e-8, f"L{r['layer']} {r['phase']}: {r['res_relmse']}"


def test_block_walk_fr_probe_does_not_multiply_the_drift(tmp_path):
    """The merged row reduces as ``pre + Σ_k(state_k − pre)``, so ``pre`` must be the
    state the tracks ACTUALLY started from.

    At the boundary MLP row the call site used to let ``pre`` default to the
    TEACHER's post-attn state. Under ``block_walk="tf"`` the carry IS that tensor, so
    the default was right and the row was the alignment rail. Under ``"fr"`` it is
    not, and the offset ``(carry − teacher)`` gets summed once per track — at N=64
    that inflated the drift 64x and printed relMSE 190-1292, which reads as a
    diverging model rather than a mis-set pre-state.

    The check compares the two merged rows AT THE SAME LAYER. Under "fr" the boundary
    MLP reads the carry through the TEACHER's own MLP weights, so its error is the
    post-attn row's error carried one sublayer forward — the same order. The bug
    re-sums the offset K times, and since the offset IS the post-attn error that makes
    the post-MLP row ~K² times the post-attn one in relMSE (measured 4000x at N=64,
    ~16x at this toy's N=4). A magnitude bound would not catch the toy; this ratio does.
    """
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    _cfg2, _d2, tracks_ref, _pt2 = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    teacher = _teacher(cfg, tracks_ref)
    pt.set_sync_phase("post-attn")
    pt.train()

    probe = _probe(pt, teacher, 4, tmp_path)
    probe.step = 0
    distill_step(pt, teacher, pt.lm_head, _batch(cfg),
                 DistillConfig(sync_layer_indices=tuple(SCHED), lambda_ce=0.0,
                               block_walk="fr"),
                 probe=probe)
    probe.flush()
    rows = [json.loads(l) for l in probe.path.read_text().splitlines() if "meta" not in l]

    mlp = {r["layer"]: r["res_relmse"] for r in _by(rows, "mlp", merged=True)}
    attn = {r["layer"]: r["res_relmse"] for r in _by(rows, "attn", merged=True)}
    # Layer 0's post-attn row is exact (its mixer reads the shared embedding), so it
    # is not a usable denominator; every deeper boundary is.
    deep = [i for i in sorted(mlp) if attn.get(i, 0.0) > 1e-9]
    assert len(deep) >= 3, f"nothing to compare: {sorted(attn.items())}"
    ratios = [mlp[i] / attn[i] for i in deep]
    assert max(ratios) < 5.0, (
        f"post-mlp row is {max(ratios):.0f}x the post-attn row at the same layer — "
        f"the sync pre-state is the teacher's, not the carry, so the drift is being "
        f"summed once per track: {[(i, round(mlp[i], 6), round(attn[i], 6)) for i in deep]}"
    )
    # A summed offset also blows the magnitude up by ~K.
    assert max(r["res_nr"] for r in _by(rows, "mlp", merged=True)) < 3.0


def _detail_run(tmp_path, detail_layers, phase="post-attn"):
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    _cfg2, _d2, tracks_ref, _pt2 = _build(n_tracks=4, sync_after=SCHED, n_layers=L)
    teacher = _teacher(cfg, tracks_ref)
    pt.set_sync_phase(phase)
    pt.train()
    probe = ActivationProbe(
        num_layers=L, n_streams=4, stream_offset=0, shards_per_stream=1, rank=0,
        out_dir=tmp_path, device=torch.device("cpu"),
        teacher_layers=teacher.text_models[0].layers,
        sync_phase=phase, detail_layers=detail_layers,
    )
    probe.step = 0
    distill_step(pt, teacher, pt.lm_head, _batch(cfg),
                 DistillConfig(sync_layer_indices=tuple(SCHED), lambda_ce=0.0),
                 probe=probe)
    probe.flush()
    lines = probe.path.read_text().splitlines()
    meta = json.loads(lines[0])["meta"]
    return meta, [json.loads(l) for l in lines[1:]]


def test_probe_detail_default_off_is_byte_identical(tmp_path):
    """`--probe-detail` empty must change NOTHING — the recorded runs stay
    comparable to anything measured after it landed."""
    _m1, a = _detail_run(tmp_path / "a", set())
    _m2, b = _detail_run(tmp_path / "b", set())
    assert a == b
    assert not any("detail" in r for r in a), "detail rows leaked with the flag off"


def test_probe_detail_reconstructs_the_aggregate_it_explains(tmp_path):
    """THE rail on the dump: the per-position vectors must reproduce the very
    number they are there to explain.

    `res_relmse` is `Σ_p err_p / Σ_p den_p` — a NORM-WEIGHTED quantity — which is
    exactly why it can move 13x while the unweighted per-token means of cos and nr
    do not. Keeping err and den separate is what makes that reproducible; a dump
    that stored only their ratio could not be checked against anything.
    """
    layers = {2, 5}
    _meta, rows = _detail_run(tmp_path, layers)
    pos = {(r["layer"], r["phase"], r["walk"]): r for r in rows
           if r.get("detail") == "position"}
    chan = {(r["layer"], r["phase"], r["walk"]): r for r in rows
            if r.get("detail") == "channel"}
    agg = {(r["layer"], r["phase"], r["walk"]): r for r in rows
           if "detail" not in r and r["track"] == -1}
    assert pos, "no detail rows"
    assert {k[0] for k in pos} == layers, f"got layers {sorted({k[0] for k in pos})}"

    for key, d in pos.items():
        a = agg[key]
        got = sum(d["err"]) / sum(d["den"])
        assert abs(got - a["res_relmse"]) < 2e-3 * max(a["res_relmse"], 1e-6), (
            f"{key}: per-position Σerr/Σden {got:.6g} != res_relmse {a['res_relmse']:.6g}"
        )
        # cos and nr reduce by an UNWEIGHTED mean — the other half of the asymmetry.
        assert abs(sum(d["cos"]) / len(d["cos"]) - a["res_cos"]) < 1e-3
        assert abs(sum(d["nr"]) / len(d["nr"]) - a["res_nr"]) < 1e-3
        # The per-channel split must account for the same total error.
        assert abs(chan[key]["chan_err_total"] - sum(d["err"])) < 2e-3 * max(sum(d["err"]), 1e-6)


def test_probe_detail_carries_token_identity_and_only_merged_rows(tmp_path):
    """A position index is useless without the token it names, and a per-TRACK
    tensor must never be dumped this way — those are rank-disjoint, so one rank
    writing them would silently drop 7/8 of the model."""
    meta, rows = _detail_run(tmp_path, {3})
    detail = [r for r in rows if "detail" in r]
    assert meta["detail_layers"] == [3]
    assert "input_ids" in meta and len(meta["input_ids"]) == len(
        [r for r in detail if r["detail"] == "position"][0]["err"])
    assert all(r["track"] == -1 for r in detail), "a per-track row was dumped"


def test_detail_rows_do_not_displace_the_aggregate_rows(tmp_path):
    """A detail row carries `walk` and `track == -1` like an aggregate row, but holds
    LISTS instead of the METRICS columns. Any view keyed on (layer, phase) that does
    not exclude them silently REPLACES the real row — which would corrupt exactly the
    panel the detail rows exist to explain."""
    from scripts.report_probe import tf as tf_rows

    _meta, rows = _detail_run(tmp_path, {2, 5})
    kept = tf_rows(rows)
    assert kept, "the tf filter dropped everything"
    assert all("detail" not in r for r in kept)
    # The (layer, phase) -> row map a view builds must land on the metric-bearing row.
    m = {(r["layer"], r["phase"]): r for r in kept if r["track"] == -1}
    assert m, "no merged aggregate rows survived"
    for r in m.values():
        assert "res_relmse" in r and isinstance(r["res_relmse"], float)
