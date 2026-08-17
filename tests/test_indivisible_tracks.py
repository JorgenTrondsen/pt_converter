"""Exactness rails for the two N=24 mechanisms, plus track fusion on top of them.

N does not have to divide every dim: GDN key heads replicate, MLP width zero-pads.

Both mechanisms exist so Qwen3.5-27B can reach its N=24 ceiling (it has 24 q-heads)
despite 16 GDN key heads and intermediate_size=17408. They are only worth having if
they are EXACT, so the rail here is a whole forward at the `exact` sync schedule —
2 syncs/layer, equivalent to dense by construction — at an N that divides neither dim.

Scaled-down mirror of the 27B: 6 q-heads / 2 kv-heads, 4 GDN k-heads and 12 GDN
v-heads (ratio 3, as at 16/48), intermediate_size 17, all at N=6. So 4 % 6 != 0 and
17 % 6 != 0 — both mechanisms fire — while 6 % 6 == 0, 6 % 2 == 0 and 12 % 6 == 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.adapters import get_adapter_for_config
from parallm.model.merge import merge_track_states, split_track_state
from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.base import FusedSegmentColwise, GDNFusedQKV
from parallm.slicer.convert import slice_model_to_tracks
from parallm.utils.max_tracks import valid_track_counts

N_TRACKS = 6


def _indivisible_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=17,  # 17 % 6 != 0 -> zero-padded to 18 (3 per track)
        num_hidden_layers=8,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=4,  # 4 % 6 != 0 -> one k-head copy per v-head
        linear_num_value_heads=12,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        full_attention_interval=4,
        vocab_size=64,
        rms_norm_eps=1e-6,
    )


def _build_pt(cfg, tracks, sync_after_layers, *, phase, fuse_size, dense):
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=N_TRACKS,
        local_track_ids=tuple(range(N_TRACKS)),
        sync_after_layers=sync_after_layers,
        track_group=None,
        fuse_size=fuse_size,
    )
    pt.set_sync_phase(phase)
    pt.eval()
    pt.load_track_state_dicts({t: tracks[t] for t in range(N_TRACKS)}, strict=False)
    pt.lm_head.load_state_dict(dense.lm_head.state_dict())
    return pt


def test_exact_schedule_matches_dense_when_n_divides_neither_dim():
    cfg = _indivisible_config()
    assert N_TRACKS in valid_track_counts(cfg), "N=6 must be an accepted track count"

    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config"
    )
    per_track = manifest.per_track_param_shapes
    # 2 v-heads/track, each with its own k-head copy: 2*8 q + 2*8 k + 2*8 v.
    assert per_track["layers.0.linear_attn.in_proj_qkv.weight"] == (48, 64)
    assert per_track["layers.0.linear_attn.conv1d.weight"] == (48, 1, 2)
    assert per_track["layers.0.mlp.gate_proj.weight"] == (3, 64)  # ceil(17/6), too small to align
    assert per_track["layers.0.mlp.down_proj.weight"] == (64, 3)

    # All tracks in one process: SyncBoundary degenerates to a local sum.
    pt = _build_pt(cfg, tracks, manifest.sync_layer_indices,
                   phase="exact", fuse_size=1, dense=dense)

    torch.manual_seed(123)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        dense_out = dense(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        dense_logits = dense.lm_head(dense_out.last_hidden_state)
        pt_logits, _ = pt(input_ids=input_ids, attention_mask=attention_mask)

    max_abs_diff = (dense_logits - pt_logits).abs().max().item()
    assert max_abs_diff < 1e-4, f"exact-schedule drift {max_abs_diff} exceeds tolerance"


def test_track_fusion_preserves_exactness_and_collapses_the_schedule():
    """Track fusion (`fuse_size`) pools F rank-local tracks' partials at every
    sublayer no global sync covers, so between syncs the group computes as one
    F-wide track. Two rails, both against the dense forward:

    1. F=2 under the `exact` schedule — fusion must not double-count a delta
       (once via the group sum, once via the boundary's cross-track reduce).
    2. F=N with ONE boundary (the last layer) under `post-attn` — a single group
       means every fuse is a full cross-track sum, so the whole walk collapses
       to the exact schedule no matter how sparse the boundaries are. This is
       where the carry actually matters, so it is what pins `fuse`+`leaders`.
    """
    cfg = _indivisible_config()
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config"
    )
    last = cfg.num_hidden_layers - 1

    torch.manual_seed(123)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        dense_out = dense(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        dense_logits = dense.lm_head(dense_out.last_hidden_state)

    for phase, sync_layers, fuse in (
        ("exact", manifest.sync_layer_indices, 2),
        ("post-attn", [last], N_TRACKS),
    ):
        pt = _build_pt(cfg, tracks, sync_layers, phase=phase, fuse_size=fuse, dense=dense)
        with torch.no_grad():
            pt_logits, _ = pt(input_ids=input_ids, attention_mask=attention_mask)
        drift = (dense_logits - pt_logits).abs().max().item()
        assert drift < 1e-4, f"{phase} fuse_size={fuse} drift {drift} exceeds tolerance"


def test_fusion_is_off_by_default_and_not_a_no_op_when_on():
    """Guards the rails above from passing vacuously: fusion must be OFF unless
    asked for, and must actually change a sparse-schedule forward when asked."""
    cfg = _indivisible_config()
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config"
    )
    sync_layers = [3, cfg.num_hidden_layers - 1]  # sparse: most sublayers own-carry
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))

    def run(**kw):
        pt = PTWrappedModel(
            text_config=cfg, n_tracks=N_TRACKS, local_track_ids=tuple(range(N_TRACKS)),
            sync_after_layers=sync_layers, track_group=None, **kw,
        )
        pt.set_sync_phase("post-attn")
        pt.eval()
        pt.load_track_state_dicts({t: tracks[t] for t in range(N_TRACKS)}, strict=False)
        pt.lm_head.load_state_dict(dense.lm_head.state_dict())
        with torch.no_grad():
            return pt(input_ids=input_ids)[0]

    assert torch.equal(run(), run(fuse_size=1))
    assert not torch.allclose(run(), run(fuse_size=2))


def test_align_chunk_rounds_only_when_it_is_nearly_free():
    from parallm.slicer.base import align_chunk

    assert align_chunk(726) == 768    # the 27B/N=24 case: 101 -> 160 TFLOP/s
    assert align_chunk(59) == 64
    assert align_chunk(2176) == 2176  # N=8 already aligned -> shards unchanged
    assert align_chunk(3) == 3        # test-sized: rounding would inflate 21x


def test_every_track_gets_a_slab():
    """A convert slices the split dim as EVENLY as it can across n_tracks: the
    throughput rounding is given up whenever it would strand a track.

    Qwen3-32B at N=64 is the case. 25600/64 = 400 EXACTLY, yet rounding widens the
    slab to 448 and 25600/448 = 57.14 — so tracks 58-63 begin past the end of the
    tensor and hold nothing but zeros. The model stays exact (silu(0)*up = 0); it
    just spends 6 of its 64 tracks on nothing, and the tax GROWS with N, which is
    the opposite of what max-tracks wants. A track holding only an attention head
    is not a whole track.

    ⚠ This is a deliberate quality-for-architecture trade, not an optimization:
    the even slab measured −0.031 macro / −0.098 math (trained, n=2/arm) against
    the 448 one. Do not revert it by reading those numbers.
    """
    from parallm.slicer.base import align_chunk

    def slab(size, n):
        return align_chunk(-(-size // n), full_size=size, n_tracks=n)

    def covered(size, n):
        return min(n, -(-size // slab(size, n)))

    assert slab(25600, 64) == 400 and covered(25600, 64) == 64
    assert slab(25600, 32) == 800 and covered(25600, 32) == 32
    # 27B/N=24: 768 would cover only 23 of 24 tracks, so 726 stands — which is also
    # exactly what the pre-`align_chunk` shards on disk already hold.
    assert slab(17408, 24) == 726 and covered(17408, 24) == 24
    # Where rounding strands nobody it is still taken — the throughput win is given
    # up only when it actually costs a track.
    assert slab(25600, 16) == 1600                  # already a multiple of 64
    assert slab(25600, 24) == 1088                  # 24 slabs still cover 25600
    assert slab(3584, 8) == 448
    # An explicit `align` is a KERNEL requirement and outranks evenness.
    assert align_chunk(180, 8, full_size=2880, n_tracks=16) == 184


def test_narrow_shards_load_into_the_aligned_model_unchanged():
    """`align_chunk` widened the per-track MLP, so shards converted before it are
    short along one dim. The added lanes are zeros either way, so padding them on
    load must be the SAME function — that is what lets a healed 726-wide 27B
    checkpoint keep running without a re-convert."""
    cfg = _indivisible_config()
    cfg.intermediate_size = 350  # ceil(350/6)=59 -> aligned up to 64
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config"
    )
    assert manifest.per_track_param_shapes["layers.0.mlp.gate_proj.weight"] == (64, 64)

    # Re-slice under the PRE-alignment rule to stand in for an old convert. Only
    # the slicer's binding is patched, so the model still sizes itself at 64 —
    # exactly the mismatch a healed 27B checkpoint now presents.
    import parallm.slicer.base as slicer_base

    real_align = slicer_base.align_chunk
    slicer_base.align_chunk = lambda chunk, align=64, **kw: chunk
    try:
        old_tracks, old_manifest = slice_model_to_tracks(
            dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config"
        )
    finally:
        slicer_base.align_chunk = real_align
    assert old_manifest.per_track_param_shapes["layers.0.mlp.gate_proj.weight"] == (59, 64)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))
    attention_mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        dense_logits = dense.lm_head(
            dense(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            .last_hidden_state
        )

    # Both the fresh (64-wide) and the padded old (59-wide) shards must reproduce
    # dense — they partition the 350 real columns differently, and both are valid.
    for name, sd in (("fresh", tracks), ("padded-old", old_tracks)):
        pt = _build_pt(cfg, sd, manifest.sync_layer_indices,
                       phase="exact", fuse_size=1, dense=dense)
        with torch.no_grad():
            drift = (dense_logits - pt(input_ids=input_ids,
                                       attention_mask=attention_mask)[0]).abs().max().item()
        assert drift < 1e-4, f"{name} shards drift {drift}"


def test_merged_tracks_reproduce_dense_where_both_mechanisms_fire():
    """Merged tracks (`merge_group`) on the config that stresses both N=24 mechanisms.

    `test_track_fusion_equivalence` pins merged == summed fusion, but on a config
    where the GDN stays compact and the MLP divides. Here k-heads REPLICATE and the
    MLP ZERO-PADS, so the merge has to get the GDN's `[Q|K|V]` segment layout and
    the per-track pad width right — a plain cat of either would corrupt the slab.
    The `exact` schedule makes dense the ground truth, so this catches it outright.
    """
    cfg = _indivisible_config()
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config"
    )
    states = {t: tracks[t] for t in range(N_TRACKS)}

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))
    attention_mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        dense_logits = dense.lm_head(
            dense(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            .last_hidden_state
        )

    adapter = get_adapter_for_config(cfg)
    for fuse in (2, 3, N_TRACKS):
        logical = N_TRACKS // fuse
        merged = merge_track_states(adapter, cfg, N_TRACKS, states, fuse)
        pt = PTWrappedModel(
            text_config=cfg,
            n_tracks=logical,
            local_track_ids=tuple(range(logical)),
            sync_after_layers=manifest.sync_layer_indices,
            track_group=None,
            merge_group=fuse,
        )
        pt.set_sync_phase("exact")
        pt.eval()
        pt.load_track_state_dicts(merged, strict=True)
        pt.lm_head.load_state_dict(dense.lm_head.state_dict())
        with torch.no_grad():
            drift = (dense_logits - pt(input_ids=input_ids,
                                       attention_mask=attention_mask)[0]).abs().max().item()
        assert drift < 1e-4, f"merge_group={fuse} drift {drift} exceeds tolerance"


def test_batched_exec_on_the_replicating_indivisible_config():
    """The batched fold where BOTH N=24 mechanisms fire — the production config.

    `test_track_fusion_equivalence`'s batched rails run on `_nesting_config`, where
    the GDN stays COMPACT and the MLP divides. The 27B at N=24 is the other regime:
    16 k-heads over 24 tracks means one k-head copy per v-head, and 17408 zero-pads
    to 768 per track. `MergedShadow._regroup_qkv` derives its `[Q|K|V]` segment
    widths from the per-member config, so the two regimes give it different splits —
    and the one this program actually runs was, until now, the untested one.

    Both schedules, because they fail differently: `exact` checks the fold against
    DENSE ground truth (a mis-split of the qkv slab shows up immediately), and the
    sparse schedule checks it against the looped unfused walk, which is the only
    place per-member own-carry exists at all.
    """
    cfg = _indivisible_config()
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config"
    )
    states = {t: tracks[t] for t in range(N_TRACKS)}
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))
    merged_states = merge_track_states(
        get_adapter_for_config(cfg), cfg, N_TRACKS, states, N_TRACKS
    )

    def _batched(phase, sync_layers):
        pt = PTWrappedModel(
            text_config=cfg, n_tracks=1, local_track_ids=(0,),
            sync_after_layers=sync_layers, track_group=None,
            merge_group=N_TRACKS, exec_groups=N_TRACKS,
        )
        pt.set_sync_phase(phase)
        pt.eval()
        pt.load_track_state_dicts(merged_states, strict=True)
        pt.lm_head.load_state_dict(dense.lm_head.state_dict())
        with torch.no_grad():
            return pt(input_ids=input_ids)[0]

    with torch.no_grad():
        dense_logits = dense.lm_head(
            dense(input_ids=input_ids, use_cache=False).last_hidden_state)
    drift = (dense_logits - _batched("exact", manifest.sync_layer_indices)).abs().max().item()
    assert drift < 1e-4, f"batched exec at exact drifts from dense by {drift}"

    sparse = [1, 3, cfg.num_hidden_layers - 1]
    looped = _build_pt(cfg, tracks, sparse, phase="post-attn", fuse_size=1, dense=dense)
    with torch.no_grad():
        want = looped(input_ids=input_ids)[0]
    got = _batched("post-attn", sparse)
    drift = (want - got).abs().max().item()
    assert drift < 1e-4, f"batched exec != looped unfused at a sparse schedule: {drift}"
    # Non-vacuous: own-carry really is doing something at this schedule.
    assert (want - dense_logits).abs().max().item() > 1e-3


def test_merge_then_split_returns_the_original_shards():
    """A merged run saves by splitting back to N shards, so eval/serve never see the
    merged form. That only holds if `split(merge(x)) == x` for every key — including
    the OwnerOnly ones (present on track 0 alone) and the zero-padded MLP."""
    cfg = _indivisible_config()
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config"
    )
    states = {t: tracks[t] for t in range(N_TRACKS)}
    adapter = get_adapter_for_config(cfg)
    fuse = 3

    merged = merge_track_states(adapter, cfg, N_TRACKS, states, fuse)
    for g, msd in merged.items():
        back = split_track_state(adapter, cfg, N_TRACKS, msd, fuse, g * fuse)
        assert sorted(back) == [g * fuse + i for i in range(fuse)]
        for tid, sd in back.items():
            assert set(sd) == set(states[tid]), f"track {tid} key set changed"
            for key, val in sd.items():
                assert torch.equal(val, states[tid][key]), f"track {tid} {key} changed"


def test_gdn_compact_mode_is_a_plain_segment_split():
    """When the k-heads DO divide N (every N=8 dense and every MoE convert today),
    GDNFusedQKV must reproduce the old FusedSegmentColwise output byte-for-byte."""
    n_tracks = 4
    spec = GDNFusedQKV(num_k_heads=8, num_v_heads=16, head_k_dim=4, head_v_dim=4)
    old = FusedSegmentColwise(segments=(32, 32, 64))
    weight = torch.randn(128, 5)
    for t in range(n_tracks):
        assert torch.equal(spec.slice(weight, t, n_tracks), old.slice(weight, t, n_tracks))
    assert spec.per_track_shape((128, 5), n_tracks) == old.per_track_shape((128, 5), n_tracks)
