"""K=2 per rank, single-process (no NCCL): exercises the new lockstep sync path.

With n_tracks=2 and local_track_ids=(0,1) hosted in a single PTWrappedModel,
the SyncBoundary's local-sum-then-all-reduce degenerates to a pure local
sum (track_group=None skips the NCCL collective). This validates the
K>1 forward path end-to-end without needing a distributed launcher.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.convert import slice_model_to_tracks


def _tiny_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        full_attention_interval=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )


def test_k2_local_only_forward_is_finite_and_matches_manual_sync():
    cfg = _tiny_config()
    n_tracks = 2
    sync_block_depth = 4

    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=sync_block_depth, text_config_attr="config"
    )
    assert manifest.sync_layer_indices == [3, 7]

    # Single PTWrappedModel hosting both tracks (K=2, world_size=1).
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=(0, 1),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    ).eval()
    pt.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)

    with torch.no_grad():
        pt_logits, sync_hiddens = pt(
            input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=True
        )

    assert pt_logits is not None  # rank hosts track 0 (the owner)
    assert pt_logits.shape == (1, 16, cfg.vocab_size)
    assert torch.isfinite(pt_logits).all()
    assert set(sync_hiddens.keys()) == {3, 7}
    for h in sync_hiddens.values():
        assert h.shape == (1, 16, cfg.hidden_size)
        assert torch.isfinite(h).all()


def test_k2_intra_window_taps_observe_without_perturbing():
    """Mid-window taps add loss-only reconstructions at every non-boundary layer
    and must leave the carried state (boundary hiddens, logits) bit-identical."""
    cfg = _tiny_config()
    n_tracks = 2

    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=(0, 1),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    ).eval()
    pt.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)

    with torch.no_grad():
        base_logits, base_hiddens = pt(
            input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=True
        )
        tap_logits, tap_hiddens = pt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sync_hiddens=True,
            return_intra_window_hiddens=True,
        )

    # Every layer is reported: boundaries carry state, the rest are loss-only taps.
    assert set(tap_hiddens.keys()) == set(range(cfg.num_hidden_layers))
    for h in tap_hiddens.values():
        assert h.shape == (1, 16, cfg.hidden_size)
        assert torch.isfinite(h).all()
    # Observation must not perturb the forward.
    assert torch.equal(base_logits, tap_logits)
    for idx in manifest.sync_layer_indices:
        assert torch.equal(base_hiddens[idx], tap_hiddens[idx])


def test_k2_peer_rank_returns_no_logits():
    """A rank that does NOT own track 0 should have lm_head=None and emit logits=None."""
    cfg = _tiny_config()
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=4,
        local_track_ids=(2, 3),  # peer rank, no owner
        sync_after_layers=[3, 7],
        track_group=None,
    ).eval()
    assert pt.lm_head is None
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    with torch.no_grad():
        logits, _ = pt(input_ids=input_ids, attention_mask=attention_mask)
    assert logits is None


def test_post_attn_capture_sets_cover_every_metric_layer():
    """At post-attn the student records exactly the layers named in the capture
    sets, so `capture_sets` and the metric layers must agree.

    eval_fidelity passed neither set and got an EMPTY dict back, KeyError-ing on
    the first layer — its per-layer block_mse never worked at post-attn at all.
    Two things fixed that: it now passes the sets, and passing NEITHER no longer
    means "record nothing" but "record every state the walk syncs".
    """
    from parallm.train.distill import capture_sets

    cfg = _tiny_config()
    L = cfg.num_hidden_layers
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=2,
        local_track_ids=(0, 1),
        sync_after_layers=[1, 3, 5, 7],  # D=2
        track_group=None,
    ).eval()
    pt.set_sync_phase("post-attn")
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)

    cap_attn, cap_mlp = capture_sets(*pt.sync_sets(), L, False)
    with torch.no_grad():
        _, hiddens = pt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sync_hiddens=True,
            capture_post_attn=cap_attn,
            capture_post_mlp=cap_mlp,
        )
    assert set(hiddens) == {1, 3, 5, 7}, f"got {sorted(hiddens)}"

    # No capture sets: every layer the walk syncs at, which at post-attn is the
    # boundaries (post-attn state) and the final layer (post-MLP state).
    with torch.no_grad():
        _, defaulted = pt(
            input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=True
        )
    assert set(defaulted) == {1, 3, 5, 7}

    # MID-WINDOW taps are implemented at every phase now: the own-carry branch
    # reconstructs a synced post-MLP state at the depths in between. They are
    # loss-only, so the boundary states must be untouched.
    cap_attn, cap_mlp = capture_sets(*pt.sync_sets(), L, True)
    assert cap_attn | cap_mlp == set(range(L))
    with torch.no_grad():
        _, full = pt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sync_hiddens=True,
            return_intra_window_hiddens=True,
            capture_post_attn=cap_attn,
            capture_post_mlp=cap_mlp,
        )
    assert set(full) == set(range(L)), f"got {sorted(full)}"
    for i in (1, 3, 5, 7):
        assert torch.equal(full[i], hiddens[i]), f"L{i} perturbed by observation"


def test_final_layer_mlp_reads_the_synced_residual():
    """The final layer is a full boundary, so its MLP reads the post-attn SYNCED
    residual R and the head sees ``R + Σ_k mlp_k(R)``. K=2 is the minimum that can
    catch a wrong pre-state — `SyncBoundary` computes ``pre + Σ_k (h_k − pre)``, so
    at K=1 the short-circuit skips the sum entirely.
    """
    from parallm.model.seam import checkpointed_halves
    from parallm.train.distill import capture_sets

    torch.manual_seed(5)
    cfg = _tiny_config()
    L = cfg.num_hidden_layers
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=list(range(L)), track_group=None,
    ).eval()
    pt.set_sync_phase("post-attn")

    cap_attn, cap_mlp = capture_sets(*pt.sync_sets(), L, False)
    assert cap_attn == set(range(L)) and cap_mlp == set()

    with torch.no_grad():
        hidden, hiddens = pt(
            input_ids=torch.randint(0, cfg.vocab_size, (1, 8)),
            attention_mask=torch.ones((1, 8), dtype=torch.long),
            return_sync_hiddens=True, return_hidden_pre_lm_head=True,
            capture_post_attn=cap_attn, capture_post_mlp=cap_mlp,
        )
    assert set(hiddens) == set(range(L))

    _, run_mlp = checkpointed_halves(False, None, None)
    R = hiddens[L - 1]
    with torch.no_grad():
        per_track = [run_mlp(tm.layers[L - 1], R) for tm in pt.text_models]
        expected = pt.text_models[0].norm(pt.sync_module(per_track, R))
    torch.testing.assert_close(hidden, expected, rtol=1e-5, atol=1e-6)


def test_schedule_omitting_the_last_layer_still_syncs_it_post_mlp():
    """The own-carry branch is now only reachable at the last layer via a schedule
    that does not name it, so nothing else covers it."""
    from parallm.train.distill import capture_sets

    cfg = _tiny_config()
    L = cfg.num_hidden_layers
    sync = [1, 3, 5]  # deliberately no L-1
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=sync, track_group=None,
    ).eval()
    pt.set_sync_phase("post-attn")
    cap_attn, cap_mlp = capture_sets(*pt.sync_sets(), L, False)
    assert cap_attn == set(sync) and cap_mlp == {L - 1}
    with torch.no_grad():
        _, hiddens = pt(
            input_ids=torch.randint(0, cfg.vocab_size, (1, 8)),
            attention_mask=torch.ones((1, 8), dtype=torch.long),
            return_sync_hiddens=True,
            capture_post_attn=cap_attn, capture_post_mlp=cap_mlp,
        )
    assert set(hiddens) == set(sync) | {L - 1}


def test_sync_sets_maps_each_phase_to_two_sets():
    """The phase does not pick a WALK, it picks two SETS — and `last` is always in
    the post-MLP one, because the head needs a synced state to project."""
    cfg = _tiny_config()
    L = cfg.num_hidden_layers
    sched = [1, 3, 5, 7]
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=sched, track_group=None,
    ).eval()

    pt.set_sync_phase("post-attn")
    assert pt.sync_sets() == (set(sched), {L - 1})
    pt.set_sync_phase("post-mlp")
    assert pt.sync_sets() == (set(), set(sched))
    pt.set_sync_phase("exact")
    assert pt.sync_sets() == (set(range(L)), set(range(L)))

    # A schedule that does not name the final layer still gets a post-MLP sync there.
    pt2 = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=[1, 3, 5], track_group=None,
    ).eval()
    for phase in ("post-attn", "post-mlp", "exact"):
        pt2.set_sync_phase(phase)
        assert L - 1 in pt2.sync_sets()[1], phase


def test_post_mlp_walk_matches_a_whole_layer_reference_loop():
    """post-mlp = every track runs the WHOLE layer own-carry, then one all-reduce.

    The unified walk reaches that by splitting each layer into mixer/MLP halves and
    syncing on the MLP side only. That has to be the same function as calling the
    layer whole, which is what the deleted second walk did — so this rail is the
    reference loop written out longhand.
    """
    cfg = _tiny_config()
    n_tracks = 2
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )
    sync_set = set(manifest.sync_layer_indices)  # [3, 7]; 7 == last
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=n_tracks, local_track_ids=(0, 1),
        sync_after_layers=manifest.sync_layer_indices, track_group=None,
    ).eval()
    pt.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)
    pt.set_sync_phase("post-mlp")

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)

    with torch.no_grad():
        got, _ = pt(input_ids=input_ids, attention_mask=attention_mask)

        # --- the reference: whole-layer own-carry, sync after the layer ---
        tm0 = pt.text_models[0]
        h = pt.embed(input_ids)
        pos_ids, text_pos_ids = tm0._resolve_position_ids(h, None)
        masks = pt._adapter.build_masks(tm0.config, h, attention_mask, text_pos_ids)
        pos_emb = tm0.rotary_emb(h, pos_ids)

        block_start = h
        per_track = [h for _ in pt.text_models]
        for i in range(cfg.num_hidden_layers):
            per_track = [
                tm.layers[i](
                    per_track[k],
                    position_embeddings=pos_emb,
                    attention_mask=masks[tm.config.layer_types[i]],
                    position_ids=text_pos_ids,
                    past_key_values=None,
                    use_cache=False,
                )
                for k, tm in enumerate(pt.text_models)
            ]
            if i in sync_set:
                h = pt.sync_module(per_track, block_start)
                block_start = h
                per_track = [h for _ in pt.text_models]
        want = pt.lm_head(tm0.norm(h))

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    # Non-vacuous: post-attn on the same weights and schedule is a different network.
    pt.set_sync_phase("post-attn")
    with torch.no_grad():
        other, _ = pt(input_ids=input_ids, attention_mask=attention_mask)
    assert (other - want).abs().max().item() > 1e-3
