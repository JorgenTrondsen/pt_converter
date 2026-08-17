"""Qwen3 (dense) rails: round-trip reassembly, dense forward parity, and the
batched fold.

Two groups, and both matter for a different reason.

**Slicing / forward.** The N=4 test at `sync_phase="exact"` is the one that catches a
wrong spec: every sublayer syncs, so the PT walk IS the dense forward up to
floating-point summation order, and it exercises the per-head `q_proj` split, the
kv-group replication, the padded SwiGLU lanes and the mask map at once.

**The batched fold.** Qwen3 is the first family to run `engine._batched_attn` with
`AttnOps(gated_q=False, centered_norm=False)` — a plain `q_proj` (no `[q|gate]`
chunk, no output gate) and a plain RMSNorm weight (not `1 + w`). The equivalence
rails below drive the fold against the LOOPED path, which runs the real
`Qwen3Attention`/`Qwen3RMSNorm` modules, so a wrong flag (or a stale Qwen3.5-ism in
the shared rope helpers) fails here instead of silently producing wrong numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from parallm.adapters import get_adapter_for_config
from parallm.model.merge import merge_track_states, split_track_state
from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.base import Replicated
from parallm.slicer.convert import resolve_param_specs, slice_model_to_tracks
from parallm.slicer.qwen3 import decoder_layer_specs, top_level_specs


def _tiny_config(num_kv: int = 2, inter: int = 32, sliding: bool = False,
                 layers: int = 4) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=64,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=num_kv,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        use_sliding_window=sliding,
        sliding_window=3 if sliding else None,
        max_window_layers=2,  # with sliding on: layers 0-1 full, 2-3 sliding
    )


def _dense(cfg: Qwen3Config, seed: int = 42):
    torch.manual_seed(seed)
    dense = Qwen3Model(cfg).eval()
    # The per-head q_norm/k_norm ship as all-ones, and those are exactly the weights
    # `engine._rms_tracks` reimplements — an all-ones weight would let a wrong
    # `centered_norm` (w vs 1+w) pass on some paths. Randomize them.
    with torch.no_grad():
        for name, p in dense.named_parameters():
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                p.normal_(mean=1.0, std=0.05)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.manual_seed(7)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    return dense.eval()


def _build_pt(cfg, tracks, manifest, n_tracks: int, phase: str) -> PTWrappedModel:
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    )
    pt.eval()
    pt.load_track_state_dicts({t: tracks[t] for t in range(n_tracks)}, strict=True)
    pt.set_sync_phase(phase)
    return pt


def _dense_logits(dense, input_ids):
    with torch.no_grad():
        out = dense(input_ids=input_ids, use_cache=False)
        return dense.lm_head(out.last_hidden_state)


# --------------------------------------------------------------------------- #
# Adapter wiring
# --------------------------------------------------------------------------- #


def test_adapter_registered_and_layer_types():
    cfg = _tiny_config()
    adapter = get_adapter_for_config(cfg)
    assert adapter.model_type == "qwen3"
    assert set(adapter.valid_layer_types) == {"full_attention", "sliding_attention"}
    # No `get_layer_types` override needed — the default reads the config.
    assert adapter.layer_types_for(cfg) == list(cfg.layer_types)
    # Dense SwiGLU + plain self-attention: both the merge and the batched fold apply.
    assert adapter.supports_merged_tracks is True
    assert adapter.supports_batched_exec is True
    # The two facts `engine._batched_attn` cannot read off a module.
    assert (adapter.attn_ops.gated_q, adapter.attn_ops.centered_norm) == (False, False)


def test_build_masks_follows_the_configs_layer_types():
    """Only the types the stack actually has. With `use_sliding_window=False` HF sets
    `sliding_window = None`, so asking for a sliding mask has nothing to window."""
    adapter = get_adapter_for_config(_tiny_config())
    embeds = torch.zeros(1, 12, 64)
    pos = torch.arange(12).unsqueeze(0)

    plain = adapter.build_masks(_tiny_config(), embeds, None, pos)
    assert set(plain) == {"full_attention"}

    cfg = _tiny_config(sliding=True)
    assert set(cfg.layer_types) == {"full_attention", "sliding_attention"}
    assert set(adapter.build_masks(cfg, embeds, None, pos)) == {
        "full_attention", "sliding_attention"
    }

    # Under the default (SDPA) backend a plain causal mask is legitimately None —
    # SDPA takes `is_causal` instead — so compare the two under eager, where both
    # materialize and the sliding one must be strictly narrower.
    cfg._attn_implementation = "eager"
    masks = adapter.build_masks(cfg, embeds, None, pos)
    full, sliding = masks["full_attention"], masks["sliding_attention"]
    assert full is not None and sliding is not None
    assert (sliding == torch.finfo(sliding.dtype).min).sum() > (
        full == torch.finfo(full.dtype).min
    ).sum()


def test_valid_track_counts_for_the_shipped_shape():
    """The real Qwen3-32B shape: 64 q heads, 8 kv heads, 25600-wide MLP.

    `intermediate_size` is NOT a constraint — `Colwise(pad_full_size=...)` zero-pads
    it exactly — so max-tracks is 64 (one q-head per track)."""
    from parallm.utils.max_tracks import valid_track_counts

    cfg = Qwen3Config(
        hidden_size=5120, intermediate_size=25600, num_hidden_layers=64,
        num_attention_heads=64, num_key_value_heads=8, head_dim=128,
        vocab_size=151936,
    )
    assert valid_track_counts(cfg) == [64, 32, 16, 8]


# --------------------------------------------------------------------------- #
# Slicer
# --------------------------------------------------------------------------- #


def test_slice_reassemble_round_trip():
    cfg = _tiny_config()
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(dense, n_tracks=4, text_config_attr="config")

    assert manifest.model_type == "qwen3"
    assert manifest.layer_types == ["full_attention"] * cfg.num_hidden_layers
    assert len(tracks) == 4

    state = dense.state_dict()
    for li, layer_type in enumerate(manifest.layer_types):
        for sub_key, spec in decoder_layer_specs(cfg, layer_type).items():
            key = f"layers.{li}.{sub_key}"
            slices = [tracks[t][key] for t in range(4)]
            if type(spec).__name__ == "KVReplicatedColwise":
                # reassemble wants one unique slice per kv-group.
                tpg = 4 // spec.num_kv_heads
                slices = [slices[g * tpg] for g in range(spec.num_kv_heads)]
            got = spec.reassemble(slices)
            assert torch.allclose(got, state[key], atol=1e-6), key

    for canonical, spec in top_level_specs(cfg).items():
        got = spec.reassemble([tracks[t].get(canonical) for t in range(4)])
        assert torch.allclose(got, state[canonical], atol=1e-6), canonical


def test_per_track_shapes_at_n4():
    cfg = _tiny_config()
    tracks, _ = slice_model_to_tracks(_dense(cfg), n_tracks=4, text_config_attr="config")
    t0 = tracks[0]
    assert t0["layers.0.self_attn.q_proj.weight"].shape == (16, 64)  # 1 head * 16
    assert t0["layers.0.self_attn.k_proj.weight"].shape == (16, 64)  # 1 kv head of 2
    assert t0["layers.0.self_attn.v_proj.weight"].shape == (16, 64)
    assert t0["layers.0.self_attn.o_proj.weight"].shape == (64, 16)
    assert t0["layers.0.self_attn.q_norm.weight"].shape == (16,)  # per-head, replicated
    assert t0["layers.0.mlp.gate_proj.weight"].shape == (8, 64)  # 32/4
    assert t0["layers.0.mlp.down_proj.weight"].shape == (64, 8)
    # And the per-track config must agree with the slicer, or the shards won't load.
    adapter = get_adapter_for_config(cfg)
    assert adapter.build_per_track_text_config(cfg, 4).intermediate_size == 8


def test_kv_heads_are_replicated_within_a_group_and_differ_across_groups():
    """N=4 over 2 kv heads: tracks 0-1 share kv-group 0, tracks 2-3 share group 1."""
    cfg = _tiny_config(num_kv=2)
    tracks, _ = slice_model_to_tracks(_dense(cfg), n_tracks=4, text_config_attr="config")
    key = "layers.0.self_attn.k_proj.weight"
    assert torch.equal(tracks[0][key], tracks[1][key])
    assert torch.equal(tracks[2][key], tracks[3][key])
    assert not torch.equal(tracks[0][key], tracks[2][key])


def test_attention_bias_is_refused_rather_than_dropped():
    """No shipped Qwen3 sets it, and a silently-dropped `o_proj.bias` would be added
    N times by the sync's partial-sum semantics."""
    import pytest

    cfg = _tiny_config()
    cfg.attention_bias = True
    with pytest.raises(ValueError, match="attention_bias"):
        decoder_layer_specs(cfg, "full_attention")


# --------------------------------------------------------------------------- #
# Forward parity
# --------------------------------------------------------------------------- #


def test_n1_matches_dense_forward():
    """N=1: SyncBoundary is a no-op, so the PT model IS the dense model."""
    cfg = _tiny_config(num_kv=1)  # N=1 must be a multiple of num_kv
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=1, sync_block_depth=2, text_config_attr="config"
    )
    pt = _build_pt(cfg, tracks, manifest, 1, "post-mlp")
    pt.lm_head.weight = dense.lm_head.weight

    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        got, _ = pt(input_ids=ids)
    assert torch.allclose(got, _dense_logits(dense, ids), atol=2e-5, rtol=1e-4)


def test_n4_exact_schedule_matches_dense_forward():
    """N=4 at the exact schedule (sync after every sublayer) must reproduce the dense
    forward: each track holds a disjoint head set / MLP-width slab, and every partial
    is summed before the next sublayer reads it."""
    cfg = _tiny_config(num_kv=2)
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=4, sync_block_depth=1, text_config_attr="config"
    )
    pt = _build_pt(cfg, tracks, manifest, 4, "exact")
    pt.lm_head.weight = dense.lm_head.weight

    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        got, _ = pt(input_ids=ids)
    want = _dense_logits(dense, ids)
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-5, f"relL2 {rel:.3e}"


def test_n4_exact_schedule_matches_dense_with_padded_mlp_width():
    """30 lanes over 4 tracks of 8 -> 2 dead lanes. Zero-padding a SwiGLU lane is
    exact (silu(0)*up = 0), so the padded model is still the dense one."""
    cfg = _tiny_config(num_kv=2, inter=30)
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=4, sync_block_depth=1, text_config_attr="config"
    )
    assert tracks[0]["layers.0.mlp.gate_proj.weight"].shape == (8, 64)
    assert torch.count_nonzero(tracks[3]["layers.0.mlp.gate_proj.weight"][-2:]) == 0

    pt = _build_pt(cfg, tracks, manifest, 4, "exact")
    pt.lm_head.weight = dense.lm_head.weight
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        got, _ = pt(input_ids=ids)
    want = _dense_logits(dense, ids)
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-5, f"relL2 {rel:.3e}"


def test_n16_one_head_per_track_matches_dense_forward():
    """Exactness must not decay as N grows — the shipped 32B runs at N=64, where a
    kv-group spans 8 tracks and every sync sums 64 partials.

    On GPU in bf16 that regime is unmeasurable against zero (the reduction noise
    grows with N and swamps it, see the fidelity ledger), so it is pinned HERE in
    fp32 where the arithmetic is exact: 16 tracks, one q-head each, 4 tracks per
    kv-group.
    """
    cfg = Qwen3Config(
        hidden_size=64, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=4, head_dim=8,
        vocab_size=64, max_position_embeddings=64, rms_norm_eps=1e-6,
    )
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=16, sync_block_depth=1, text_config_attr="config"
    )
    # One q-head and a quarter of the MLP per track; 4 tracks share each kv head.
    assert tracks[0]["layers.0.self_attn.q_proj.weight"].shape == (8, 64)
    key = "layers.0.self_attn.k_proj.weight"
    assert torch.equal(tracks[0][key], tracks[3][key])
    assert not torch.equal(tracks[0][key], tracks[4][key])

    pt = _build_pt(cfg, tracks, manifest, 16, "exact")
    pt.lm_head.weight = dense.lm_head.weight
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        got, _ = pt(input_ids=ids)
    want = _dense_logits(dense, ids)
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-5, f"relL2 {rel:.3e}"


def test_n4_sliding_layers_match_dense_forward():
    """The `sliding_attention` half of the mask map, against ground truth."""
    cfg = _tiny_config(num_kv=2, sliding=True)
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=4, sync_block_depth=1, text_config_attr="config"
    )
    assert manifest.layer_types == ["full_attention"] * 2 + ["sliding_attention"] * 2
    pt = _build_pt(cfg, tracks, manifest, 4, "exact")
    pt.lm_head.weight = dense.lm_head.weight

    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        got, _ = pt(input_ids=ids)
    want = _dense_logits(dense, ids)
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-5, f"relL2 {rel:.3e}"


def test_n4_post_attn_schedule_runs_and_stays_close():
    """The training schedule (lever B, d1b): every layer a boundary. Not bit-exact —
    each layer's MLP reads the synced residual but carries its own delta — so this
    only guards that the walk runs and does not blow up."""
    cfg = _tiny_config(num_kv=2)
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=4, sync_block_depth=1, text_config_attr="config"
    )
    pt = _build_pt(cfg, tracks, manifest, 4, "post-attn")
    pt.lm_head.weight = dense.lm_head.weight

    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    last = cfg.num_hidden_layers - 1
    boundaries = set(manifest.sync_layer_indices) - {last}
    with torch.no_grad():
        got, sync_h = pt(
            input_ids=ids,
            return_sync_hiddens=True,
            capture_post_attn=boundaries,
            capture_post_mlp={last},
        )
    assert torch.isfinite(got).all()
    assert set(sync_h) == boundaries | {last}


# --------------------------------------------------------------------------- #
# Merged tracks + the batched fold
# --------------------------------------------------------------------------- #


def test_merged_config_is_f_aligned_slabs():
    """A merged track is F per-track slabs, not one F-wide slab — the distinction
    that makes the merged module loadable from the shards on disk."""
    from parallm.model.tracks.qwen3 import build_per_track_text_config

    cfg = _tiny_config()
    one = build_per_track_text_config(cfg, 4, fuse_size=1)
    two = build_per_track_text_config(cfg, 4, fuse_size=2)
    assert two.intermediate_size == one.intermediate_size * 2
    assert two.num_attention_heads == one.num_attention_heads * 2
    assert two.num_key_value_heads == one.num_key_value_heads * 2


def _build_merged(dense, cfg, n_tracks, sync_after, fuse_size=1, merge_group=1,
                  exec_groups=1, phase="post-attn"):
    """One loaded PT model over `n_tracks * merge_group` shards. Mirrors
    `tests/test_track_fusion_equivalence.py::_build`."""
    n_shards = n_tracks * merge_group
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_shards, sync_block_depth=1, text_config_attr="config"
    )
    states = {t: tracks[t] for t in range(n_shards)}
    if merge_group > 1:
        states = merge_track_states(
            get_adapter_for_config(cfg), cfg, n_shards, states, merge_group
        )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=sync_after,
        track_group=None,
        fuse_size=fuse_size,
        merge_group=merge_group,
        exec_groups=exec_groups,
    )
    pt.set_sync_phase(phase)
    pt.eval()
    pt.load_track_state_dicts(states, strict=False)
    pt.lm_head.load_state_dict(dense.lm_head.state_dict())
    return pt


def _run_merged(dense, cfg, n_tracks, sync_after, input_ids, **kw):
    pt = _build_merged(dense, cfg, n_tracks, sync_after, **kw)
    with torch.no_grad():
        return pt(input_ids=input_ids)[0]


def test_batched_exec_of_a_merged_track_equals_the_unfused_model():
    """The fold with `AttnOps(gated_q=False, centered_norm=False)`, against the
    LOOPED path — which runs the real `Qwen3Attention` and `Qwen3RMSNorm`. A stale
    Qwen3.5-ism (a `[q|gate]` chunk, a `1 + w` norm, the wrong eps attribute) shows
    up here as a large drift.

    Sparse schedule on purpose: at `exact` every sublayer syncs globally, so fused
    and unfused coincide and the rail would pass without executing member-wise.
    """
    cfg = _tiny_config(num_kv=2)
    dense = _dense(cfg)
    sync_after = [1, cfg.num_hidden_layers - 1]
    ids = torch.randint(0, cfg.vocab_size, (2, 16), generator=torch.manual_seed(123))

    unfused = _run_merged(dense, cfg, 4, sync_after, ids)
    batched = _run_merged(dense, cfg, 1, sync_after, ids, merge_group=4, exec_groups=4)
    drift = (batched - unfused).abs().max().item()
    assert drift < 1e-4, f"batched merged exec != looped unfused: max |dlogit| {drift}"

    # Non-vacuous: the same merged weights run as ONE wide track are the FUSED
    # model, and that is a materially different forward.
    fused = _run_merged(dense, cfg, 1, sync_after, ids, merge_group=4)
    assert (batched - fused).abs().max().item() > 1e-3


def test_batched_exec_at_intermediate_group_width_equals_looped_fusion():
    """G = K/F streams of F members: the only configuration where the diverged
    q_norm/k_norm have to be re-split ``[K, d] -> [G, F, d]``."""
    cfg = _tiny_config(num_kv=2)
    dense = _dense(cfg)
    sync_after = [1, cfg.num_hidden_layers - 1]
    ids = torch.randint(0, cfg.vocab_size, (2, 16), generator=torch.manual_seed(77))

    looped = _run_merged(dense, cfg, 4, sync_after, ids, fuse_size=2)   # K=4, F=2
    batched = _run_merged(dense, cfg, 1, sync_after, ids,
                          merge_group=4, exec_groups=2)                 # G=2, F=2
    drift = (batched - looped).abs().max().item()
    assert drift < 1e-4, f"batched G=2/F=2 != looped fuse_size=2: max |dlogit| {drift}"

    unfused = _run_merged(dense, cfg, 1, sync_after, ids, merge_group=4, exec_groups=4)
    fused = _run_merged(dense, cfg, 1, sync_after, ids, merge_group=4)
    assert (batched - unfused).abs().max().item() > 1e-3
    assert (batched - fused).abs().max().item() > 1e-3


def test_batched_exec_matches_dense_at_the_exact_schedule():
    """The fold from the other side: under `exact` the G members must sum back to
    the dense forward, so this compares against ground truth rather than another PT
    walk."""
    cfg = _tiny_config(num_kv=2)
    dense = _dense(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(5))
    sync_after = list(range(cfg.num_hidden_layers))

    ref = _dense_logits(dense, ids)
    batched = _run_merged(dense, cfg, 1, sync_after, ids, merge_group=4,
                          exec_groups=4, phase="exact")
    drift = (batched - ref).abs().max().item()
    assert drift < 5e-4, f"batched exec at the exact schedule drifts from dense by {drift}"


def test_batched_exec_routes_gradients_to_the_right_member_slabs():
    """Values agreeing is not enough: the merged parameter is ONE tensor, so a fold
    that read member k's slab for member j could still produce the right forward and
    corrupt every checkpoint."""
    cfg = _tiny_config(num_kv=2)
    dense = _dense(cfg)
    sync_after = [1, cfg.num_hidden_layers - 1]
    ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(9))
    adapter, n_shards = get_adapter_for_config(cfg), 4

    def _grads(model):
        model.train()
        logits, _ = model(input_ids=ids)
        logits.square().mean().backward()
        return model

    looped = _grads(_build_merged(dense, cfg, n_shards, sync_after))
    batched = _grads(_build_merged(dense, cfg, 1, sync_after,
                                   merge_group=n_shards, exec_groups=n_shards))

    merged_grads = {
        name: p.grad for name, p in batched.text_models[0].named_parameters()
        if p.grad is not None and name.startswith("layers.")
    }
    assert merged_grads, "no layer gradients on the merged track"
    per_member = split_track_state(adapter, cfg, n_shards, merged_grads, n_shards, 0)

    specs = resolve_param_specs(adapter, cfg)
    checked = 0
    for t in range(n_shards):
        for name, got in per_member[t].items():
            spec = specs[name]
            if isinstance(spec, Replicated) and spec.sync:
                continue  # one shared copy on the merged track: grad is the SUM
            want = looped.text_models[t].get_parameter(name).grad
            assert torch.allclose(got, want, atol=1e-5), (
                f"member {t} gradient mismatch at {name}: "
                f"max |d| {(got - want).abs().max().item()}"
            )
            checked += 1
    assert checked > 4 * n_shards, f"only {checked} per-member gradients compared"


def test_distill_step_batched_merged_matches_looped_unfused():
    """The whole training step through the fold, on the real trainer.

    The forward rails above pin `PTWrappedModel`'s walk; this one pins
    `train.distill`'s teacher-forced block loop, which mirrors that walk sublayer for
    sublayer and is what the F=1/max-tracks run actually executes. Checkpointing is ON
    so the batched path goes through recompute, where the two representations could
    disagree on saved-tensor order.
    """
    from parallm.train.distill import DistillConfig, distill_step, freeze_slice_teacher

    cfg = _tiny_config(num_kv=2)
    dense = _dense(cfg)
    L = cfg.num_hidden_layers
    sched = [1, L - 1]  # D=2-shaped: own-carry is where the fold can differ
    ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))
    batch = {"input_ids": ids, "labels": ids.clone(),
             "attention_mask": torch.ones_like(ids)}

    looped = _build_merged(dense, cfg, 4, sched)
    merged = _build_merged(dense, cfg, 1, sched, merge_group=4, exec_groups=4)
    # The teacher is the same frozen slices on the exact schedule (layout-independent).
    teacher = freeze_slice_teacher(_build_merged(dense, cfg, 4, list(range(L))))

    dcfg = DistillConfig(sync_layer_indices=tuple(sched))
    out = []
    for pt in (looped, merged):
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
    assert sum(1 for p in merged.text_models[0].layers.parameters()
               if p.grad is not None and p.grad.abs().sum() > 0) > 0


def test_eager_for_eval_releases_both_registries(monkeypatch):
    """An eval pass must un-compile the BATCHED fold too, not just the seam.

    The compiled units are built with ``dynamic=False``, so each distinct shape is
    its own dynamo cache entry. An lm-eval pass arrives with a different batch size
    and a fresh sequence length per batch, so leaving the fold compiled burns through
    ``recompile_limit`` (8) inside the FIRST eval — and once that limit is hit dynamo
    runs the code object EAGER for the rest of the process. A single missed registry
    therefore disables ``--compile`` for the whole remainder of training, which is
    invisible in the loss and shows up only as a step time nobody re-measures.
    Observed at 32B/N=64: 12 clean training steps, then all 16 warnings (2 halves x
    8 ranks) during the first eval.

    Same failure shape as `enable_seam_compile` reaching only the seam, which is why
    both directions are pinned.
    """
    from parallm.model import batched, seam

    monkeypatch.setattr(seam, "_COMPILED", {"mixer": object(), "mlp": object()})
    monkeypatch.setattr(batched, "_COMPILED", {"mixer": object(), "mlp": object()})
    before = (dict(seam._COMPILED), dict(batched._COMPILED))

    with seam.eager_for_eval():
        assert not seam._COMPILED, "seam still compiled during eval"
        assert not batched._COMPILED, "BATCHED FOLD still compiled during eval"
        # Both resolvers must hand back the eager functions while inside.
        assert seam._mixer_fn() is seam.seam_token_mixer
        assert batched._attn_fn() is batched._fold_attn
        assert batched._mlp_fn() is batched._fold_mlp

    # ...and the training graphs come back, or every eval would pay a recompile.
    assert seam._COMPILED == before[0]
    assert batched._COMPILED == before[1]


def test_compile_reaches_the_batched_fold_with_one_graph_per_layer_half(monkeypatch):
    """``--compile`` must compile the batched fold ONCE per half, not once per layer.

    The regression this pins is the reason the lever was inert for two sessions:
    `engine._batched_*` indexes its provider by the python int ``li``, dynamo
    SPECIALIZES on python ints, so handing that int to a compiled region recompiled
    the graph per layer, hit ``recompile_limit`` (8) and ran every later layer EAGER.
    A step-time matrix reports that as "inside the noise", never as a failure — so it
    has to fail here instead. 10 layers on purpose: more than the recompile limit, so
    the broken form would both blow the limit and miscount.

    ``backend="eager"``: the bug is a dynamo GUARD, so tracing is the entire test and
    inductor codegen would only make it slow.
    """
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    from parallm.model import batched, seam

    real_compile = torch.compile
    monkeypatch.setattr(
        torch, "compile", lambda fn, **kw: real_compile(fn, backend="eager", **kw)
    )
    # Fresh caches, restored by monkeypatch — these are module-level and shared.
    monkeypatch.setattr(seam, "_COMPILED", {})
    monkeypatch.setattr(batched, "_COMPILED", {})

    graphs = {}
    try:
        for n_layers in (4, 10):
            cfg = _tiny_config(num_kv=2, layers=n_layers)
            dense = _dense(cfg)
            sync_after = [1, n_layers - 1]
            ids = torch.randint(0, cfg.vocab_size, (1, 16),
                                generator=torch.manual_seed(31))
            eager = _run_merged(dense, cfg, 1, sync_after, ids,
                                merge_group=4, exec_groups=4)

            dynamo.reset()
            counters.clear()
            seam.enable_seam_compile("both")
            assert set(batched._COMPILED) == {"mixer", "mlp"}, (
                "enable_seam_compile must reach the batched fold: which "
                "representation runs is decided by exec_groups, not by the caller"
            )
            compiled = _run_merged(dense, cfg, 1, sync_after, ids,
                                   merge_group=4, exec_groups=4)
            graphs[n_layers] = counters["stats"]["unique_graphs"]
            # Non-vacuous: the compiled walk is the same walk.
            assert torch.equal(compiled, eager)
    finally:
        dynamo.reset()
        counters.clear()

    # THE invariant, and the one that does not rot with a torch upgrade: the graph
    # count is a property of the two SHAPES a residual arrives in (shared [B,T,H]
    # straight out of a sync, per-member [G,B,T,H] mid-window), never of the depth.
    assert graphs[4] == graphs[10], (
        f"graph count tracks layer count ({graphs[4]} at 4 layers, {graphs[10]} at 10) "
        f"— the compiled region is specializing per layer, which is the `li == k` bug"
    )
    # And it stays under `recompile_limit`, or later layers silently fall back to eager.
    assert graphs[10] <= 4, f"{graphs[10]} graphs per half-pair is more than the two " \
                            f"residual ranks explain"
