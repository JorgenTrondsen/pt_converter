"""torchrun entry point: measure how closely the trained PT student matches the teacher.

Loads a per-rank checkpoint (``track_*.safetensors`` + ``manifest.json``),
rebuilds the sharded teacher exactly as the training script does, and runs
both on a held-out stream. Reports:

  - KL(teacher‖student) (forward) and KL(student‖teacher) (reverse)
  - Top-1 / top-5 prediction agreement and top-5 set IoU
  - Per-sync-boundary hidden-state MSE
  - Student vs teacher perplexity (and gap)

The launch shape mirrors ``scripts/train_qwen3_5_9b.py``. The layout is uniform
(K = n_tracks // world_size); forward output is layout-independent (the
SyncBoundary all-reduce combines all tracks regardless of which rank hosts
them), so it need not match training. Eval uses the legacy full-logits student
path (full lm_head on the owner rank) — it loads a vocab-parallel-trained
checkpoint unchanged (track_0 carries the gathered full embed/lm_head) and
yields full-vocab logits for the fidelity metrics.

Single node, 8 GPUs, n_tracks=16, best checkpoint:

    torchrun --standalone --nproc-per-node=8 scripts/eval_fidelity.py \\
        --hf-model <teacher path> \\
        --checkpoint-dir ./pt_train_out/best \\
        --num-batches 200
"""
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from parallm.dist.fsdp_setup import wrap_teacher_with_fsdp
from parallm.dist.groups import build_groups
from parallm.eval.fidelity import fidelity_step
from parallm.model.merge import plan_track_layout
from parallm.model.pt_model import PTWrappedModel
from parallm.train.data import (
    DEFAULT_MIXTURE,
    CalibrationDataConfig,
    PackedTokenStream,
    parse_source_spec,
    preset_names,
    preset_sources,
)
from parallm.train.teacher import HookedTeacher, load_dense_reference
from parallm.utils.checkpoint import load_manifest, load_track, train_meta_arg


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Dense teacher model path (same as training)")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Per-rank checkpoint dir with track_*.safetensors and manifest.json "
                        "(e.g. ./pt_train_out/best, ./pt_train_out/final, or a step_N dir).")
    p.add_argument("--data-preset", default=DEFAULT_MIXTURE,
                   help=f"Streamed eval mixture: a name under configs/data (have: {preset_names()}) "
                        "or a path to any mixture JSON. The default matches the trainer's, so "
                        "fidelity is measured on the SAME distribution the student was distilled "
                        "on. Reads the front of the stream (--skip-docs 0). Overridden by "
                        "--data-source or --dataset-name.")
    p.add_argument("--data-source", action="append", default=None, metavar="NAME[:CONFIG[:KEY[:WEIGHT]]]",
                   help="Custom eval source(s), repeatable; if any given it REPLACES --data-preset.")
    p.add_argument("--dataset-name", default=None,
                   help="Legacy single-dataset override (e.g. 'Salesforce/wikitext' for the old "
                        "WikiText-103 comparator). If set, used instead of --data-preset.")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1",
                   help="Config for --dataset-name (ignored unless --dataset-name is set).")
    p.add_argument("--split", default="validation",
                   help="Split for --dataset-name (ignored unless --dataset-name is set).")
    p.add_argument("--text-key", default="text",
                   help="Text column for --dataset-name (ignored unless --dataset-name is set).")
    p.add_argument("--skip-docs", type=int, default=0,
                   help="Skip the first N docs of the preset/--data-source stream before packing. "
                        "0 reads the held-out front the default training val used; raise to match a "
                        "non-default --val-holdout-docs if you need strict disjointness from training.")
    p.add_argument("--seed", type=int, default=42,
                   help="Mixture interleave seed; keep equal across ranks (matches training).")
    p.add_argument("--num-batches", type=int, default=200,
                   help="Number of packed sequences to evaluate.")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=128,
                   help="Seq-chunk size for the vocab-wide fp32 expansion; matches "
                        "--kl-ce-chunk-size in training.")
    p.add_argument("--sync-indices", default=None,
                   help="Override the manifest's sync_layer_indices with a custom comma-separated "
                        "sorted list (e.g. '2,6,10,14,15,16,...,31'). Per-track weights are "
                        "schedule-independent, so this evaluates an arbitrary (non-uniform / "
                        "depth-tapered) sync schedule on the SAME slice with NO re-slice. Both the "
                        "student SyncBoundary placement and the teacher block_mse hooks follow it.")
    p.add_argument("--sync-phase", default=None, choices=["post-mlp", "post-attn", "exact"],
                   help="Where in a layer the sync fires. Default: leave the model's own "
                        "('post-mlp'). 'exact' = 2 syncs/layer, which is exactly equivalent "
                        "to dense — use it to verify a fresh convert (expect ~zero KL).")
    p.add_argument("--fuse-tracks", type=int, default=None,
                   help="F rank-local tracks pool their partials at every non-sync sublayer "
                        "(N/F-track behaviour on N shards). Default: read from the "
                        "checkpoint's train_meta.json, else 1.")
    p.add_argument("--intra-window-taps", action="store_true",
                   help="Also report block_mse/relmse at EVERY layer, not just the sync "
                        "boundaries. Mid-window rows are loss-only synced reconstructions "
                        "(the forward keeps feeding each track its PARTIAL residual — same "
                        "semantics as training's --intra-window-mse), so they localize where "
                        "inside a window the free-running error grows at D>=2. Hooks the "
                        "teacher at every layer and adds one all-reduce per mid-window layer.")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_count = torch.cuda.device_count()
    if local_rank >= gpu_count:
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} >= visible GPU count {gpu_count}. "
            f"Launch with --nproc-per-node <= #GPUs."
        )
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    # ----- Manifest + uniform layout (forward output is layout-independent). -----
    manifest = load_manifest(args.checkpoint_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    # Optional custom (non-uniform) sync schedule — weights are schedule-independent,
    # so this re-evaluates the SAME slice under a redistributed sync budget.
    if args.sync_indices is not None:
        sync_layers = [int(x) for x in args.sync_indices.split(",") if x.strip() != ""]
    elif manifest.sync_layer_indices is not None:
        sync_layers = list(manifest.sync_layer_indices)
    else:
        raise SystemExit(
            "[error] checkpoint manifest has no sync_layer_indices (a raw convert "
            "output carries none — the schedule is placed at train time). Pass "
            "--sync-indices, or point --checkpoint-dir at a trained checkpoint."
        )
    if args.fuse_tracks is None:
        args.fuse_tracks, _fuse_from = train_meta_arg(args.checkpoint_dir, "fuse_tracks", 1)
    else:
        _fuse_from = "flag"
    # Looped path (`allow_merge=False`); the plan is here for `fuse_ranks`, so a run
    # trained with F > tracks-per-rank is scored WITH its cross-rank groups.
    try:
        plan = plan_track_layout(
            manifest.n_tracks, dist.get_world_size(), args.fuse_tracks,
            allow_merge=False,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    if plan.fuse_ranks > 1:
        layout = build_groups(n_tracks=manifest.n_tracks, fuse_ranks=plan.fuse_ranks)
    _log(
        rank,
        f"[init] world={layout.world_size} n_tracks={manifest.n_tracks} "
        f"K={layout.tracks_per_rank} num_layers={manifest.num_layers} "
        f"fuse={args.fuse_tracks} (from {_fuse_from})",
    )
    _log(rank, f"[init] sync schedule: {len(sync_layers)} syncs at {sync_layers}"
               + ("  (OVERRIDE)" if args.sync_indices is not None else ""))
    _log(rank, f"[init] rank={rank} local_track_ids={layout.local_track_ids}")

    # ----- Teacher (frozen, FSDP-sharded across the world). -----
    cfg = AutoConfig.from_pretrained(args.hf_model)
    _log(rank, "[init] loading frozen dense teacher…")
    teacher_model, text_model = load_dense_reference(args.hf_model)
    # Mid-window taps need a teacher hidden at EVERY layer (same pattern as the
    # train script's --intra-window-mse); otherwise hook only the boundaries.
    metric_indices = (
        list(range(manifest.num_layers))
        if args.intra_window_taps
        else list(sync_layers)
    )
    teacher = HookedTeacher(
        text_model=text_model,
        lm_head=teacher_model.lm_head,
        sync_layer_indices=metric_indices,
    )
    # Shards layer-by-layer off CPU; moving the whole dense teacher across first
    # OOMs a 40 GB card at 27B (see wrap_teacher_with_fsdp).
    wrap_teacher_with_fsdp(text_model, teacher_model.lm_head)
    teacher_model = teacher_model.to(torch.cuda.current_device())

    # ----- Student. Same construction as the train script so the loaded
    # state_dict aligns 1:1 with the safetensors keys. -----
    _log(rank, f"[init] building PT student for tracks {layout.local_track_ids}…")
    # A VLM snapshot wraps the text config; a text-only family (gpt-oss, qwen3)
    # ships a FLAT one. Same guard as every other script.
    student = PTWrappedModel(
        text_config=getattr(cfg, "text_config", cfg),
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=sync_layers,
        track_group=layout.track_group,
        fuse_size=plan.fuse_size,
        fuse_group=layout.fuse_group,
        fuse_ranks=layout.fuse_ranks,
        fuse_rank=layout.fuse_rank,
    )
    if args.sync_phase is not None:
        student.set_sync_phase(args.sync_phase)
    track_states = {tid: load_track(args.checkpoint_dir, tid) for tid in layout.local_track_ids}
    student.load_track_state_dicts(track_states, strict=True)
    # Cast BEFORE the move (as train_cli and eval_lm_harness do): the tracks are
    # built in fp32, so moving first ships 2x the bytes across and peaks at fp32
    # size on the card before the cast frees it — 18 GiB rather than 9 for 4
    # tracks of the 27B, which OOMs once the sharded teacher is already resident.
    student = student.to(torch.bfloat16).to(torch.cuda.current_device())
    student.eval()

    # ----- Data. PackedTokenStream is reused as-is; switching dataset is just
    # a CalibrationDataConfig change. -----
    # Eval source priority: --data-source > --dataset-name (legacy single) > --data-preset.
    # Default is the qwen-mix training mixture's held-out front slice, so fidelity is
    # measured in-distribution and lines up with the training val_kl.
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    if args.data_source:
        data_cfg = CalibrationDataConfig(
            sources=[parse_source_spec(s) for s in args.data_source],
            seq_len=args.seq_len, seed=args.seed, skip_docs=args.skip_docs,
        )
        data_desc = "custom --data-source"
    elif args.dataset_name:
        data_cfg = CalibrationDataConfig.single(
            dataset_name=args.dataset_name, dataset_config=args.dataset_config,
            split=args.split, text_key=args.text_key, seq_len=args.seq_len, seed=args.seed,
        )
        data_desc = f"{args.dataset_name}/{args.dataset_config}/{args.split}"
    else:
        data_cfg = CalibrationDataConfig(
            sources=preset_sources(args.data_preset),
            seq_len=args.seq_len, seed=args.seed, skip_docs=args.skip_docs,
        )
        data_desc = f"preset:{args.data_preset} (held-out front, skip_docs={args.skip_docs})"
    _log(rank, f"[data] eval source: {data_desc}")
    ds = PackedTokenStream(tok, data_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    sync_indices = tuple(metric_indices)
    sums: dict[str, torch.Tensor] = {}
    n_batches = 0
    for batch in loader:
        if n_batches >= args.num_batches:
            break
        batch = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in batch.items()}
        if batch["input_ids"].ndim == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        m = fidelity_step(student, teacher, batch, sync_indices, args.chunk_size,
                          intra_window_taps=args.intra_window_taps)
        for name, val in m.items():
            if name not in sums:
                sums[name] = torch.zeros((), device=val.device, dtype=torch.float32)
            sums[name] = sums[name] + val.float()
        n_batches += 1

    # ----- Cross-rank reduction. fidelity_step emits zeros from non-owner
    # ranks, so SUM lands on the owner's value. Divide by batch count for the
    # per-token mean over the eval slice. -----
    for name in sums:
        dist.all_reduce(sums[name], op=dist.ReduceOp.SUM)
        sums[name] = sums[name] / max(1, n_batches)

    if rank == 0:
        s_nll = sums["student_nll"].item()
        t_nll = sums["teacher_nll"].item()
        s_ppl = math.exp(s_nll)
        t_ppl = math.exp(t_nll)
        kl_fwd = sums["kl_forward"].item()
        kl_rev = sums["kl_reverse"].item()
        print()
        print(f"===== Fidelity over {n_batches} batches "
              f"({data_desc}, seq_len={args.seq_len}) =====")
        print(f"  perplexity   : student={s_ppl:.4f}  teacher={t_ppl:.4f}  gap={s_ppl - t_ppl:+.4f}")
        print(f"  nll          : student={s_nll:.4f}  teacher={t_nll:.4f}  delta={s_nll - t_nll:+.4f}")
        print(f"  KL forward (t‖s) = {kl_fwd:.4f} nats")
        print(f"  KL reverse (s‖t) = {kl_rev:.4f} nats")
        print(f"  top1_agree   = {sums['top1_agree'].item():.4f}")
        print(f"  top5_agree   = {sums['top5_agree'].item():.4f}  (teacher top-1 ∈ student top-5)")
        print(f"  top5_set_iou = {sums['top5_set_iou'].item():.4f}")
        boundary_set = set(sync_layers)
        label = "per-layer" if args.intra_window_taps else "per-sync-boundary"
        print(f"  {label} block_mse (raw | rel=Σ(s−t)²/Σt² | cos | ‖s‖/‖t‖):")
        for layer_idx in sync_indices:
            raw = sums[f"block_mse_l{layer_idx}"].item()
            rel = sums[f"block_relmse_l{layer_idx}"].item()
            # cos/ratio split `rel` into its two factors (rel ≈ 1 − 2·cos·r + r²):
            # missing content points the wrong way, a gain error is only short.
            cos = sums[f"block_cos_l{layer_idx}"].item()
            ratio = sums[f"block_normratio_l{layer_idx}"].item()
            # "(mid)" rows are loss-only reconstructions at non-boundary depths,
            # not state the forward carries.
            tag = ""
            if args.intra_window_taps:
                tag = "      " if layer_idx in boundary_set else " (mid)"
            print(f"    layer {layer_idx:3d}{tag}: raw={raw:.6e}  rel={rel:.6e}  "
                  f"cos={cos:+.4f}  r={ratio:.4f}")
        print()

    teacher.remove_hooks()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
