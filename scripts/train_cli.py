"""torchrun entry point: the d1b heal / progressive-D distillation trainer.

Trains the per-track slices at a lever-B (post-attn) schedule against the
frozen-slice exact-schedule teacher (see ``parallm.train.distill``). One rank
per GPU, one PTWrappedModel per rank holding that rank's tracks.

Stage-1 d1b heal of the 27B (every layer a boundary):

    setsid nohup .venv/bin/torchrun --standalone --nproc-per-node=8 \\
        scripts/train_cli.py \\
        --hf-model <Qwen3.6-27B-NVFP4 snapshot dir> \\
        --checkpoint-dir convert_out/qwen3_6_27b_n8 \\
        --out-dir train_out/d1b_heal \\
        --max-steps 4001 --eval-every 500 > train_out/d1b_heal.log 2>&1 &

Progressive-D curriculum: warm-start ``--checkpoint-dir train_out/d1b_heal/best``
with ``--teacher-dir convert_out/qwen3_6_27b_n8`` (the teacher is ALWAYS the
original slices) and a ``--sync-indices`` subset.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from parallm.adapters import get_adapter_for_config
from parallm.dist.groups import build_groups
# Constants only — downstream.py is deliberately free of torch/lm_eval imports, so
# this stays cheap and a train-only install (no [eval] extra) still imports.
from parallm.eval.downstream import DEFAULT_TASKS, EVAL_TASK_PATH
from parallm.model.merge import plan_track_layout
from parallm.model.pt_model import PTWrappedModel
from parallm.train.probe import ActivationProbe, summary_lines
from parallm.train.profile import PhaseTimer
from parallm.train.data import (
    DEFAULT_MIXTURE,
    CalibrationDataConfig,
    PackedTokenStream,
    parse_source_spec,
    preset_names,
)
from parallm.train.distill import (
    DistillConfig,
    distill_step,
    freeze_slice_teacher,
)
from parallm.train.sync_grads import (
    assert_replicated_consistent,
    build_replication_plan,
    compute_global_grad_norm,
    sync_replicated_grads,
)
from parallm.utils.checkpoint import load_manifest, load_track, load_track_keys, save_manifest


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


@contextmanager
def _eval_compile_guard():
    """Force BOTH track representations back to eager for an lm-eval pass.

    Delegates to `seam.eager_for_eval`, which owns the pairing — this used to clear
    only `seam._COMPILED`, leaving the batched fold compiled, and an eval's shapes
    then exhausted dynamo's `recompile_limit` and dropped the fold to eager for the
    REST of training.
    """
    from parallm.model.seam import eager_for_eval

    with eager_for_eval():
        yield


class _KernelWindow:
    """`torch.profiler` over a fixed mid-run window, aggregated AFTER the loop.

    Starts at `_WARM` so the trace is of a settled step: with `--compile` the first
    steps carry inductor codegen and any dynamo recompiles, which would swamp the
    kernel counts. Ends after `_SPAN` steps — one step here issues ~700k kernels, and
    the trace holds every event in memory.

    Two things this got wrong the first time, both of which killed a run:

    **Every rank profiles, not just rank 0.** A rank-local slowdown desynchronizes
    the group — the ranks join a collective at every sync boundary, so whichever rank
    is paying for instrumentation makes the other seven wait. Profiling all of them
    keeps the cost symmetric; only rank 0 prints.

    **The aggregation happens after the training loop.** `key_averages()` over 2.09 M
    events took longer than the 600 s NCCL timeout, so ranks 1-7 aborted at the
    all-reduce while rank 0 was still summarising. Stopping the profiler mid-loop is
    cheap; turning it into a report is not, so that waits until no collective is
    pending.
    """

    _WARM, _SPAN = 8, 2

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.ctx = None
        self.t0 = 0.0
        self.wall = 0.0

    def maybe_start(self, step: int) -> None:
        if not self.enabled or self.ctx is not None or step != self._WARM:
            return
        from torch.profiler import ProfilerActivity, profile as torch_profile

        self.ctx = torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        self.ctx.__enter__()
        torch.cuda.synchronize()  # the window's wall must not inherit queued work
        self.t0 = time.perf_counter()

    def maybe_stop(self, step: int) -> None:
        """End the window. Cheap — no aggregation, so the ranks stay in lockstep."""
        # `step` is the one just FINISHED, so the window closes at _WARM+_SPAN-1.
        if not self.enabled or self.ctx is None or step + 1 < self._WARM + self._SPAN:
            return
        torch.cuda.synchronize()
        self.wall = time.perf_counter() - self.t0
        self.ctx.__exit__(None, None, None)
        self.enabled = False  # one window per run

    def final_report(self, rank: int) -> None:
        if self.ctx is None:
            return
        from parallm.train.profile import kernel_report

        # Every rank aggregated the same volume; only one of them needs to say so.
        report = kernel_report(self.ctx, self.wall, self._SPAN) if rank == 0 else ""
        self.ctx = None
        _log(rank, report)


def _describe_plan(plan, tracks_per_rank: int) -> str:
    """One-line account of how F was realised, so a run's log says which of the
    four grouping mechanisms is live rather than just the requested number."""
    parts = [f"{tracks_per_rank} shards/rank"]
    if plan.merge_group > 1:
        n_local = tracks_per_rank // plan.merge_group
        parts.append(
            f"merged {plan.merge_group}x"
            + (f" ({n_local} merged tracks/rank, looped)" if n_local > 1 else "")
        )
    if plan.exec_groups > 1:
        parts.append(
            f"run as {plan.exec_groups} streams x {plan.merge_group // plan.exec_groups}"
        )
    if plan.fuse_size > 1:
        parts.append(f"summed-fuse {plan.fuse_size}")
    if plan.fuse_ranks > 1:
        parts.append(f"CROSS-RANK over {plan.fuse_ranks} ranks")
    return ", ".join(parts) + f" => effective fuse {plan.effective_fuse}"


def _load_states(text_cfg, layout, ckpt_dir: str, merge_group: int):
    """This rank's per-(logical-)track state dicts, merged when merge_group > 1.

    Merged: logical track `t` is shards `[t*F, (t+1)*F)` of the on-disk convert,
    concatenated into one F-wide track — same function as fusing them, 1/F the
    kernels (see `parallm.model.merge`)."""
    if merge_group == 1:
        return {tid: load_track(ckpt_dir, tid) for tid in layout.local_track_ids}
    from parallm.model.merge import merge_track_states

    n_shards = layout.n_tracks * merge_group
    shards = {
        s: load_track(ckpt_dir, s)
        for tid in layout.local_track_ids
        for s in range(tid * merge_group, (tid + 1) * merge_group)
    }
    return merge_track_states(
        get_adapter_for_config(text_cfg), text_cfg, n_shards, shards, merge_group
    )


def _build_student(text_cfg, manifest, layout, ckpt_dir: str, sync_layers, sync_phase: str,
                   fuse_size: int = 1, merge_group: int = 1, exec_groups: int = 1):
    model = PTWrappedModel(
        text_config=text_cfg,
        n_tracks=layout.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=list(sync_layers),
        track_group=layout.track_group,
        fuse_size=fuse_size,
        merge_group=merge_group,
        exec_groups=exec_groups,
        fuse_group=layout.fuse_group,
        fuse_ranks=layout.fuse_ranks,
        fuse_rank=layout.fuse_rank,
    )
    states = _load_states(text_cfg, layout, ckpt_dir, merge_group)
    model.load_track_state_dicts(states, strict=True)
    model.set_sync_phase(sync_phase)
    # bf16 on the HOST first: the module is built in fp32 and the bf16 track weights
    # load into fp32 params, so moving before casting asks the device for 2x the
    # slice. Invisible at 35B (16.4 GiB transient still fit); it is what OOMs 122B.
    return model.to(torch.bfloat16).to(torch.cuda.current_device())


def _build_teacher_streamed(text_cfg, manifest, layout, ckpt_dir: str, sync_layers,
                            merge_group: int = 1):
    """Frozen exact-schedule teacher with its decoder layers paged from pinned
    host DRAM (``HostResidentLayers``): only the layers move off-device, which is
    what lets the teacher coexist with the student on a 40 GB card at 122 B.

    Merges alongside the student: at the `exact` schedule every sublayer syncs, so
    the merged group's members never diverge and one wide track is the same
    forward. The teacher is a large share of the step, so this is half the win."""
    from parallm.model.pt_model import HostResidentLayers

    model = PTWrappedModel(
        text_config=text_cfg,
        n_tracks=layout.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=list(sync_layers),
        track_group=layout.track_group,
        merge_group=merge_group,
    )
    model.load_track_state_dicts(
        _load_states(text_cfg, layout, ckpt_dir, merge_group), strict=True
    )
    model.set_sync_phase("exact")
    model = model.to(torch.bfloat16)  # still on the host
    for p in model.parameters():
        p.requires_grad_(False)
    model.layer_stream = HostResidentLayers(
        model.text_models, len(model.text_models[0].layers), torch.cuda.current_device()
    )
    return model.to(torch.cuda.current_device()), model.layer_stream


def _cpu_embed_hooks(embed, dev) -> None:
    """Embed on the host, in/out on the device. Hooks not a wrapper, so the
    state_dict keys (checkpoint format) are untouched; student and teacher share
    the weight through separate modules, so each module needs its own hooks."""
    embed.register_forward_pre_hook(lambda m, a: (a[0].to("cpu"),))
    embed.register_forward_hook(lambda m, a, out: out.to(dev))


def _extract_track_state(student: PTWrappedModel, k: int) -> dict[str, torch.Tensor]:
    """Inverse of ``load_track_state_dicts`` for local track index ``k``."""
    prefix = f"text_models.{k}."
    out: dict[str, torch.Tensor] = {}
    for key, val in student.state_dict().items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = val.detach().to("cpu", copy=True).contiguous()
        elif key == "lm_head.weight" and k == 0:
            out["lm_head.weight"] = val.detach().to("cpu", copy=True).contiguous()
    return out


def _save_checkpoint(student, manifest, layout, out_dir: Path, meta: dict, rank: int):
    """Weights-only per-track save: each rank writes its own tracks.

    Merged tracks are split back to `merge_group` shards, so what lands on disk is
    always an ordinary `manifest.n_tracks`-shard checkpoint — eval and serve never
    see the merged representation."""
    from safetensors.torch import save_file as save_safetensors

    F = getattr(student, "merge_group", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, tid in enumerate(layout.local_track_ids):
        merged = _extract_track_state(student, k)
        if F == 1:
            shards = {tid: merged}
        else:
            from parallm.model.merge import split_track_state

            shards = split_track_state(
                student._adapter, student.text_config, manifest.n_tracks,
                merged, F, tid * F,
            )
        for sid, sd in shards.items():
            path = out_dir / f"track_{sid}.safetensors"
            # Unlink first: a best/ overwrite otherwise holds old+new (~108 GB at
            # 27B) transiently, which busts the 191 GB home quota. The ckpt is a
            # re-creatable training artifact, so the seconds-wide corruption window
            # is acceptable.
            path.unlink(missing_ok=True)
            save_safetensors(sd, str(path))
    if rank == 0:
        # The saved manifest self-describes the trained schedule so eval picks
        # it up without --sync-indices.
        save_manifest(out_dir, replace(manifest, sync_layer_indices=list(meta["sync_layer_indices"])))
        (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    dist.barrier()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Original HF checkpoint dir (tokenizer + config)")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Slices the STUDENT starts from (raw convert output, or a previous best/ for the curriculum)")
    p.add_argument("--teacher-dir", default=None,
                   help="Slices the frozen exact-schedule teacher loads (default: --checkpoint-dir; "
                        "MUST be the original convert output when warm-starting the student from a heal)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--teacher-stream", action="store_true",
                   help="Page the frozen teacher's decoder layers in from pinned host DRAM one "
                        "at a time (~1 layer resident, not the whole teacher) — the memory-for-"
                        "PCIe trade that fits a 122 B teacher next to the student")
    p.add_argument("--optim-in-backward", action="store_true",
                   help="Step + free each param's grad inside backward, so the grad buffer is "
                        "~2 layers instead of the whole model. Requires --grad-accum-steps 1, "
                        "drops global grad-norm clipping (adafactor's clip_threshold covers it), "
                        "and gives one update per objective per batch instead of one summed step")
    p.add_argument("--cpu-embed", action="store_true",
                   help="Keep the frozen embed table on CPU (~8 MB/step of H2D) so rank 0 does "
                        "not carry embed+lm_head over its peers' budget")
    p.add_argument("--sync-indices", default=None,
                   help="Comma-separated boundary layers (default: every layer = the d1b schedule)")
    p.add_argument("--sync-phase", default="post-attn", choices=["post-attn", "post-mlp"],
                   help="WHICH sublayer's sync a boundary is. 'post-attn' (lever B, "
                        "the program's schedule): sync after the boundary layer's token "
                        "mixer, so its MLP reads the TRUE residual and its delta is "
                        "carried un-summed; the final layer keeps a post-MLP sync for the "
                        "head, so d1b costs L+1 all-reduces. 'post-mlp' (SPD's placement): "
                        "the ATTENTION sync is the one dropped and a whole layer runs "
                        "own-carry, so d1b costs L. The two are NOT budget-matched (65 vs "
                        "64 at 64 layers). See PTWrappedModel.sync_sets().")
    p.add_argument("--fuse-tracks", type=int, default=1,
                   help="F shards pool their partials at every non-sync sublayer, so "
                        "a group computes as one F-wide track between syncs (an "
                        "N/F-track model on N shards). Valid F = the divisors of "
                        "tracks-per-rank K, plus its multiples up to N: at or below K "
                        "a group is rank-local, above K it spans F/K whole ranks via a "
                        "subgroup all-reduce. How F is realised depends on the family "
                        "(see `parallm.model.merge.plan_track_layout`) — on a dense "
                        "family it is a semantic knob only, on an MoE it is also the "
                        "MERGE WIDTH, so step time improves with F. The [init] line "
                        "logs which mechanism is live.")
    p.add_argument("--no-merge-fused", action="store_true",
                   help="Run the rank's tracks as separate narrow modules instead of "
                        "merging their slabs into one wide track. Same function, ~2x "
                        "slower — for A/B'ing the merge on real weights.")
    p.add_argument("--intra-window-mse", action="store_true",
                   help="Loss-only synced taps at non-boundary layers (the D>1 curriculum wants this)")
    p.add_argument("--block-walk", choices=["tf", "fr"], default="tf",
                   help="Which trajectory the block objective's CARRY comes from. 'tf' "
                        "(default, and what every result on record was trained with): "
                        "each window starts from the TEACHER's residual, so every tap "
                        "measures that layer's own error. 'fr': each window starts from "
                        "the STUDENT's own synced state (detached), so it trains on the "
                        "input distribution it will actually see — the target is the "
                        "teacher either way. Detached, so peak memory and step time are "
                        "unchanged. ⚠ 'fr' is the old student-forcing at sf=1.0, deleted "
                        "2026-07-30 as divergent — a verdict VOIDED by the block_mse "
                        "clamp bug it ran under (see distill_step). ⚠ under 'fr' the "
                        "probe's post-attn-phase rail is no longer a rail.")
    # Recipe constants (the 9B-record defaults).
    p.add_argument("--max-steps", type=int, default=4001)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--optim", choices=["adamw", "adafactor"], default="adafactor",
                   help="adamw = fused bf16 default (27B/N=8, ~4 B/param state); adafactor "
                        "(factored 2nd moment, no 1st moment) ≈ 0 optimizer state (~0.5 GB/rank "
                        "vs 8 GB) — what fits N=16 K=2 on a 40 GB card (the 35B recipe) and "
                        "the memory-lean choice for scaling to 122 B.")
    p.add_argument("--lr-schedule-steps", type=int, default=None,
                   help="Cosine horizon for the LR schedule (default: --max-steps). Set it to the "
                        "REFERENCE run's --max-steps when comparing a short run against a "
                        "checkpoint taken part-way through a longer one; otherwise shortening "
                        "--max-steps silently anneals the LR and the two are not comparable")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lambda-block", type=float, default=4.0)
    p.add_argument("--lambda-ce", type=float, default=1.0)
    p.add_argument("--lambda-mag", type=float, default=None,
                   help="Switch every block tap from relMSE to "
                        "direction + LAMBDA_MAG*magnitude (losses.block_split). "
                        "relMSE = 1 - 2*cos*r + r^2 merges the two, and at D=2 the "
                        "FIRST boundary measures cos 0.935 / r 5.4 — 99%% magnitude, "
                        "direction already right — while being ~93%% of the block "
                        "loss. What actually survives training is direction (cos "
                        "0.93 -> 0.80 at r ~ 1.0). Unset = the historical relMSE.")
    p.add_argument("--ce-chunk-size", type=int, default=256)
    p.add_argument("--train-embeddings", action="store_true",
                   help="Unfreeze embed_tokens (lm_head stays frozen — the replicated loss "
                        "requires an identical head on every rank)")
    p.add_argument("--sync-attention-heads", action="store_true",
                   help="force_sync the diverged-by-default attention head params (legacy A/B)")
    # Data.
    p.add_argument("--data-preset", default=DEFAULT_MIXTURE,
                   help=f"Mixture name under configs/data (have: {preset_names()}) or a path to "
                        "any mixture JSON — pointing a run at different hub datasets needs no "
                        "code change. A source's weight is its share of training TOKENS; the "
                        "realized split is logged at startup.")
    p.add_argument("--data-source", action="append", default=None,
                   help="NAME[:CONFIG[:TEXT_KEY[:WEIGHT]]] (repeatable; overrides --data-preset). "
                        "Plain-text sources only — use a mixture JSON for chat/template formats.")
    # Eval: lm-evaluation-harness downstream macro, the checkpoint-selection
    # metric. val_kl is deliberately gone — it has been observed to invert the
    # downstream ranking outright, so selecting on it selects on noise.
    p.add_argument("--eval-tasks", default=DEFAULT_TASKS,
                   help="Comma-separated lm-eval tasks. The default four are the macro "
                        "convention (reasoning / math / code); changing them changes what "
                        "'macro' means, and the number stops being comparable to the ledger.")
    p.add_argument("--eval-task-path", default=EVAL_TASK_PATH,
                   help="Directory of extra lm-eval task YAMLs (default: the repo's "
                        "configs/eval_tasks). Must match the standalone eval script's, or the "
                        "in-loop macro and a standalone rescore stop meaning the same thing.")
    p.add_argument("--eval-limit", type=int, default=200,
                   help="Requests per task. 200 matches every standalone eval in the project "
                        "history (noise ~±0.015); lower it only for smokes.")
    p.add_argument("--eval-batch-size", type=int, default=8,
                   help="Requests per forward. The owner rank holds a (B, T, V) bf16 logits "
                        "tensor on top of the resident optimizer state and teacher — this is "
                        "the knob to drop if the first eval OOMs.")
    p.add_argument("--eval-max-length", type=int, default=2048)
    p.add_argument("--eval-num-fewshot", type=int, default=None,
                   help="Override fewshot count for all tasks (default: each task's standard).")
    # Cadence.
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--best-name", default="best")
    p.add_argument("--no-save", action="store_true",
                   help="Skip ALL checkpoint writes (for smokes — a 16-track MoE best/ "
                        "is ~68GB, pointless when only measuring fit/speed).")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable per-sublayer activation checkpointing — no backward "
                        "recompute (~1.5-2x faster, launch-bound). Needs the VRAM "
                        "headroom --optim adafactor frees.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mem-report", action="store_true")
    p.add_argument("--compile", choices=["off", "mixer", "mlp", "both"], default="off",
                   help="torch.compile the decoder-layer seam halves, on whichever track "
                        "representation is live (the looped seam AND the batched fold). "
                        "The profiled step "
                        "is ~75%% BACKWARD, and compiling the forward gives inductor the "
                        "backward too, so this is the lever that reaches the dominant "
                        "phase. 'mixer' skips the MLP, for a sparse-MoE whose "
                        "data-dependent expert sort will not hold a graph. Costs a "
                        "one-time warmup on the first steps.")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "max-autotune-no-cudagraphs"],
                   help="Inductor mode for --compile. The cudagraph modes are "
                        "absent because the track walk cannot use them: a graph's "
                        "memory pool is reused by the next per-layer invocation while "
                        "the backward still needs those tensors, and plain "
                        "'max-autotune' turns cudagraphs on.")
    p.add_argument("--tf-ckpt-min-segment", type=int, default=2,
                   help="Longest own-carry segment (in layers) the teacher-forced loop "
                        "holds WITHOUT activation checkpointing. That loop backwards at "
                        "every boundary, so its live graph is one segment and a short "
                        "one is cheaper to hold than to recompute — at D=2 (segment 2) "
                        "checkpointing saved no memory and cost a whole extra forward. "
                        "Numerically inert either way; 0 restores the old always-on "
                        "behaviour. Raise it only against a --mem-report peak.")
    p.add_argument("--moe-experts", choices=["auto", "dense", "grouped_mm"],
                   default="auto",
                   help="How a per-track MoE evaluates its experts. 'grouped_mm' sorts "
                        "tokens by expert; below SM90 that is a LOOP of E GEMMs plus "
                        "gathers whose backward was 39%% of all GPU time at N=64. "
                        "'dense' runs every expert on every token and lets the zero "
                        "routing weights drop the rest: E/top_k more FLOPs, but two "
                        "plain GEMMs and no indexing. 'auto' picks by per-expert width "
                        "(parallm.model.moe_dense.DENSE_WIDTH_MAX). Equivalent, not "
                        "bit-equal — the reduction order differs.")
    p.add_argument("--profile", action="store_true",
                   help="Device-synced per-phase step timers on --log-every steps "
                        "(teacher_fwd / tf_block_loop / tf_backward / fr_forward / "
                        "ce_chunked / fr_backward / grad_sync / optim). The syncs "
                        "serialize overlap, so a profiled step reads HIGHER than an "
                        "unprofiled one — compare phase SHARES between profiled runs "
                        "and take totals from unprofiled ones.")
    p.add_argument("--profile-kernels", action="store_true",
                   help="torch.profiler over steps 8-10 on rank 0: GPU-busy time vs "
                        "wall, kernels/step, mean kernel duration, and the top 20 by "
                        "self device time. Answers the question --profile cannot — "
                        "whether the step is starved for launches (cuda graphs) or "
                        "device-busy on kernels too small to fill the SMs "
                        "(concurrency). Supersedes --profile's syncs for that window.")
    p.add_argument("--ledge-probe", action="store_true",
                   help="Diagnostics for the block-loss crossing: per-tap relMSE "
                        "([taps]) and distance from the warm start ([dist]). The "
                        "latter snapshots theta_0 (~+7GB/rank at 27B/N=24), hence "
                        "opt-in. Emitted on --log-every steps only.")
    p.add_argument("--probe-steps", default="",
                   help="Comma list of steps at which to run the per-layer/per-track "
                        "activation probe (e.g. '0,50,250,500'); empty = off. Scores "
                        "every layer's post-attn AND post-mlp state, delta and normed "
                        "sublayer input against the exact-schedule teacher, per track. "
                        "Step 0 is the untrained-convert baseline and carries the "
                        "alignment rail, so include it. Costs ~2L extra resident "
                        "(B,T,H) teacher tensors and ~2 collectives/layer ON THOSE "
                        "STEPS ONLY.")
    p.add_argument("--probe-detail", default="",
                   help="Comma-separated layers to dump the UN-REDUCED breakdown at "
                        "(e.g. '8,42,43,44,45,63'). Every other probe number is an "
                        "average over the (B, T) axes, which can show that an error is "
                        "CONCENTRATED but never WHERE. At these layers the merged rows "
                        "additionally carry per-position err/den/cos/nr vectors and the "
                        "top-32 hidden channels by error contribution. Rank 0 only "
                        "(merged states come out of an all-reduce, so they are "
                        "rank-identical); ~80 KB per layer/phase/walk. Default off = "
                        "byte-identical output to before.")
    p.add_argument("--probe-dir", default=None,
                   help="Where --probe-steps writes rank{r}.jsonl (default "
                        "<out-dir>/probe). Keep it OFF '/' — that filesystem is a "
                        "9766M user quota.")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    torch.manual_seed(args.seed)

    manifest = load_manifest(args.checkpoint_dir)
    cfg_hf = AutoConfig.from_pretrained(args.hf_model)
    text_cfg = cfg_hf.text_config if hasattr(cfg_hf, "text_config") else cfg_hf
    adapter = get_adapter_for_config(text_cfg)
    # Merge the rank's WHOLE track set into one wide track (concatenated slabs)
    # whenever it holds more than one: 1/K the kernels and one full-width residual
    # stream instead of K (27B N=24: 6.4-7.7 -> ~3 s/step). `exec_groups` then says
    # how to RUN it — G=1 as one wide track (which sums the members' deltas, i.e.
    # fusion), G=K as K independent members (unfused, `parallm.model.batched`). The
    # step-time win is the merged STORAGE and belongs to both. --no-merge-fused
    # forces the looped path, which is how they are A/B'd on real weights.
    #
    # Merging is forward-equivalent but NOT optimizer-equivalent, and always was:
    # the K members share one nn.Parameter, so (a) adafactor keeps one factored
    # second moment for all of them, and (b) a Replicated(sync=True) weight — the
    # residual norms — has ONE copy per rank instead of K, so its replication group
    # is K times smaller and `sync_replicated_grads` divides by K less. Both are
    # per-parameter rescalings that adafactor/Adam largely absorb, and both already
    # applied to every merged run; they are why the looped path is the equivalence
    # reference and not a step-for-step trajectory reference.
    #
    # Families whose per-track slabs have no verified concatenation rule declare
    # `supports_merged_tracks=False` and keep the looped path — the alternative is
    # failing deep inside `build_per_track_text_config`. Families with no batched
    # fold (the MoE ones) declare `supports_batched_exec=False` and express F as the
    # MERGE WIDTH instead, looping over K/F merged tracks. `plan_track_layout` owns
    # the whole policy; see `parallm.model.merge`.
    world_size = dist.get_world_size()
    tracks_per_rank = manifest.n_tracks // world_size
    can_merge = adapter.supports_merged_tracks and not args.no_merge_fused
    if can_merge and tracks_per_rank > 1 and args.sync_attention_heads:
        raise SystemExit(
            "--sync-attention-heads cannot be merged: it keeps each kv-group's "
            "copies bit-identical across tracks, but merged they are one tensor. "
            "Pass --no-merge-fused."
        )
    try:
        plan = plan_track_layout(
            manifest.n_tracks, world_size, args.fuse_tracks,
            supports_merged_tracks=adapter.supports_merged_tracks,
            supports_batched_exec=adapter.supports_batched_exec,
            allow_merge=not args.no_merge_fused,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    merge_group, exec_groups, F = plan.merge_group, plan.exec_groups, plan.fuse_size
    layout = build_groups(
        n_tracks=plan.n_logical_tracks, fuse_ranks=plan.fuse_ranks
    )
    teacher_dir = args.teacher_dir or args.checkpoint_dir

    num_layers = text_cfg.num_hidden_layers
    if args.sync_indices is not None:
        sync_layers = sorted(int(x) for x in args.sync_indices.split(",") if x.strip())
    else:
        sync_layers = list(range(num_layers))  # d1b: every layer a boundary
    if sync_layers[-1] != num_layers - 1:
        sync_layers.append(num_layers - 1)  # the head needs the final post-MLP sync

    # post-attn: one all-reduce per boundary PLUS the head's post-MLP sync at the
    # final layer. post-mlp: one per boundary, the final layer already being one.
    # (`PTWrappedModel.sync_sets` is the authority; this only reports it before the
    # model is built.)
    n_syncs = len(sync_layers) + (1 if args.sync_phase == "post-attn" else 0)
    _log(rank, f"[init] n_tracks={manifest.n_tracks} world={layout.world_size} "
               f"boundaries={len(sync_layers)} phase={args.sync_phase} "
               f"syncs={n_syncs} "
               f"fuse={args.fuse_tracks} -> {_describe_plan(plan, tracks_per_rank)}")

    # Before any per-track config is built — `build_per_track_text_config` reads the
    # policy to stamp `_experts_implementation`, and the model is constructed below.
    from parallm.model.moe_dense import set_experts_policy

    set_experts_policy(args.moe_experts)

    if args.compile != "off":
        from parallm.model.seam import enable_seam_compile

        # Inductor codegen must not land on /dev/md0 (= "/", 9766M USER quota, and
        # it holds /tmp). A run with recompiles wrote 6.2 GB across 150k files
        # there, filled the quota, and broke the login shell itself — bash could
        # not write temp files, so every command exited 1. Redirect unless the
        # caller has deliberately pointed it somewhere off "/" already.
        cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
        if not cache or cache.startswith("/tmp"):
            cache = str(Path.home() / ".cache" / "inductor")
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
        Path(cache).mkdir(parents=True, exist_ok=True)
        _log(rank, f"[init] inductor cache: {cache}")
        enable_seam_compile(args.compile, inductor_mode=args.compile_mode)
        _log(rank, f"[init] torch.compile seam halves: {args.compile} "
                   f"(inductor mode={args.compile_mode})")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    # Data: identical stream on every rank (the SyncBoundary contract).
    if args.data_source:
        sources = [parse_source_spec(s) for s in args.data_source]
        data_cfg = CalibrationDataConfig(sources=sources, seq_len=args.seq_len, seed=args.seed)
    else:
        data_cfg = CalibrationDataConfig.from_preset(args.data_preset, seq_len=args.seq_len,
                                                     seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        PackedTokenStream(tokenizer, data_cfg), batch_size=args.batch_size
    )
    eval_tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]

    student = _build_student(text_cfg, manifest, layout, args.checkpoint_dir,
                             sync_layers, args.sync_phase, F, merge_group, exec_groups)
    student.use_checkpoint = not args.no_checkpoint
    student.train()
    # Report the RESOLVED choice, not the policy: under `auto` it depends on the
    # per-track expert width, which is only known once the config is built. A 2.8x
    # step-time lever that silently picks the other branch is worth one log line.
    _impl = getattr(student.text_models[0].config, "_experts_implementation", None)
    if _impl is not None:
        _log(rank, f"[init] MoE experts: {_impl} (--moe-experts {args.moe_experts}, "
                   f"per-track width {student.text_models[0].config.intermediate_size})")

    if args.teacher_stream:
        teacher_model, streamer = _build_teacher_streamed(
            text_cfg, manifest, layout, teacher_dir, sync_layers, merge_group)
        teacher = freeze_slice_teacher(teacher_model)
        _log(rank, f"[init] teacher layers streamed from pinned host DRAM: "
                   f"{streamer.host_bytes / 2**30:.2f} GiB/rank off-device")
    else:
        teacher_model = _build_student(text_cfg, manifest, layout, teacher_dir,
                                       sync_layers, "exact", 1, merge_group)
        teacher = freeze_slice_teacher(teacher_model)

    # Freeze the dense-copy pieces; tie the teacher's to the student's frozen
    # storage (2×2.54 GB back on rank 0 at 27B).
    tm0 = student.text_models[0]
    if tm0.embed_tokens is not None:
        tm0.embed_tokens.weight.requires_grad_(args.train_embeddings)
        if not args.train_embeddings:
            teacher_model.text_models[0].embed_tokens.weight = tm0.embed_tokens.weight
            if args.cpu_embed:
                # Shared weight to the host, then hooks on each module that reads
                # it — the teacher embeds through its OWN module.
                dev = torch.cuda.current_device()
                tm0.embed_tokens.to("cpu")
                _cpu_embed_hooks(tm0.embed_tokens, dev)
                t_embed = teacher_model.text_models[0].embed_tokens
                if t_embed is not None:
                    _cpu_embed_hooks(t_embed, dev)
                _log(rank, "[init] embed_tokens on CPU")
    if student.lm_head is not None:
        student.lm_head.weight.requires_grad_(False)  # ponytail: frozen always — replicated loss needs an identical head on every rank; re-broadcast per step if this ever unfreezes
        teacher_model.lm_head.weight = student.lm_head.weight

    # The loss head on every rank: owner uses the student's; peers load a
    # frozen copy of the same tensor from the teacher slices.
    if student.lm_head is not None:
        lm_head = student.lm_head
    else:
        w = load_track_keys(teacher_dir, 0, ["lm_head.weight"])["lm_head.weight"]
        lm_head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        lm_head.weight = torch.nn.Parameter(w, requires_grad=False)
        lm_head = lm_head.to(torch.bfloat16).to(torch.cuda.current_device())

    plan = build_replication_plan(
        student, adapter=adapter, text_cfg=text_cfg, layout=layout,
        force_sync=args.sync_attention_heads,
    )
    assert_replicated_consistent(plan)

    trainable = [p_ for p_ in student.parameters() if p_.requires_grad]
    n_train = sum(p_.numel() for p_ in trainable)
    _log(rank, f"[init] trainable params/rank ≈ {n_train/1e9:.2f}B; replication groups={len(plan)}")

    # --ledge-probe: distance from the warm start. The block loss sits at exactly
    # its warm-start value for hundreds of steps and then falls 6x in ~40, only
    # inside an LR window (2e-5 crosses at ~340, 3e-5 at ~700, >=4e-5 never) and
    # only with CE active. If the trigger is how far CE has transported the model
    # rather than LR itself, two LRs should cross at the SAME ||theta - theta_0||.
    # Grouped by parameter name with the layer index stripped, so the report says
    # attn vs MLP vs the per-track q_norm/k_norm rather than naming 64 layers.
    theta0: dict[str, torch.Tensor] = {}
    if args.ledge_probe:
        theta0 = {n_: p_.detach().clone()
                  for n_, p_ in student.named_parameters() if p_.requires_grad}
        _log(rank, f"[dist] theta_0 snapshot held: "
                   f"{sum(t.numel()*t.element_size() for t in theta0.values())/2**30:.2f}GiB/rank")

    def _dist_report() -> str:
        """Total and per-group L2 drift from theta_0 (rank-local shard)."""
        groups: dict[str, float] = {}
        total = 0.0
        for n_, p_ in student.named_parameters():
            t0_ = theta0.get(n_)
            if t0_ is None:
                continue
            # One param at a time: the fp32 temporary is a single tensor, not a
            # second copy of the whole shard.
            d2 = float((p_.detach() - t0_).float().pow_(2).sum())
            total += d2
            key = re.sub(r"\.\d+\.", ".*.", n_).rsplit(".", 1)[0]
            groups[key] = groups.get(key, 0.0) + d2
        top = sorted(groups.items(), key=lambda kv: -kv[1])[:5]
        return (f"total={total**0.5:.4f} "
                + " ".join(f"{k}={v**0.5:.4f}" for k, v in top))
    # fused: single-kernel step, no _foreach transients — the multi-tensor path
    # materializes a full extra copy of the second-moment states (~5.7 GB at
    # 27B/N=8), which is exactly the 40 GB card's missing headroom.
    def _make_optim(params):
        if args.optim == "adafactor":
            from transformers.optimization import Adafactor
            # We drive LR (lr_at cosine), so disable relative-step + param scaling;
            # beta1=None ⇒ no first moment ⇒ ~0 optimizer state (the 122 B lever).
            return Adafactor(params, lr=args.lr, weight_decay=0.0,
                             relative_step=False, scale_parameter=False, warmup_init=False)
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)

    cur_lr = [args.lr]
    gsq = torch.zeros((), device=torch.cuda.current_device())
    # Per-hooked-param step counts: how often each fired last batch, used to
    # normalize its lr (see _step_in_backward).
    fire_prev: dict[int, int] = {}
    fire_cur: "collections.Counter[int]" = collections.Counter()
    if args.optim_in_backward:
        # Step + free each param as its grad lands ⇒ grad buffer is ~2 layers, not
        # the whole model (the 122 B lever). distill_step backwards several times a
        # batch (per boundary segment + the free-running KL pass), so a param steps
        # once per contribution, not once on the sum — a recipe change, 35B-gated.
        assert args.grad_accum_steps == 1, \
            "--optim-in-backward requires --grad-accum-steps 1 (accumulation needs the buffer it frees)"
        # Replicated params (norm scales, KV rows) keep the resident path: their
        # grads all-reduce across ranks first, which a per-param hook can't order.
        replicated = {id(p_) for cg in plan for p_ in cg.local_params}
        hooked = [p_ for p_ in trainable if id(p_) not in replicated]
        hooked_optims = {id(p_): _make_optim([p_]) for p_ in hooked}

        def _step_in_backward(p_):
            gsq.add_(torch.linalg.vector_norm(p_.grad, dtype=torch.float32) ** 2)
            o = hooked_optims[id(p_)]
            # Adafactor moves ~lr/step; a param steps once per grad (2x/batch at
            # d1b), so undivided it travels 2x too far — measured 0.586 vs 0.700.
            # Divide by last batch's firing count to restore single-step distance.
            for g_ in o.param_groups:
                g_["lr"] = cur_lr[0] / fire_prev.get(id(p_), 1)
            o.step()
            p_.grad = None
            fire_cur[id(p_)] += 1

        for p_ in hooked:
            p_.register_post_accumulate_grad_hook(_step_in_backward)
        # None when there are no replicated params — torch rejects an empty group.
        rep_params = [p_ for p_ in trainable if id(p_) in replicated]
        optim = _make_optim(rep_params) if rep_params else None
        _log(rank, f"[init] optimizer={args.optim} in-backward: {len(hooked)} params stepped "
                   f"on arrival, {len(trainable) - len(hooked)} replicated params on the "
                   f"collective path; global grad-norm clip OFF "
                   f"(adafactor clip_threshold=1.0 covers it)")
    else:
        optim = _make_optim(trainable)
        _log(rank, f"[init] optimizer={args.optim}")

    # Cosine horizon decoupled from run length — else --max-steps silently anneals
    # the LR and a short run's step-N checkpoint isn't comparable to a long run's.
    horizon = args.lr_schedule_steps or args.max_steps

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        frac = (step - args.warmup_steps) / max(1, horizon - args.warmup_steps)
        cos = 0.5 * (1.0 + torch.cos(torch.tensor(frac * 3.141592653589793)).item())
        return args.lr * (args.lr_min_ratio + (1 - args.lr_min_ratio) * cos)

    dcfg = DistillConfig(
        sync_layer_indices=tuple(sync_layers),
        lambda_block=args.lambda_block,
        lambda_ce=args.lambda_ce,
        lambda_mag=args.lambda_mag,
        intra_window_mse=args.intra_window_mse,
        block_walk=args.block_walk,
        ce_chunk_size=args.ce_chunk_size,
        tf_ckpt_min_segment=args.tf_ckpt_min_segment,
    )
    # The block loop's carry is the difference between "reproduce the teacher from a
    # perfect input" and "reproduce it from your own drift" — worth one line, since
    # every recorded result is `tf` and a mislabelled arm is uncomparable.
    _log(rank, f"[init] objective: block_walk={args.block_walk} "
               f"lambda_block={args.lambda_block} lambda_ce={args.lambda_ce} "
               f"lambda_mag={args.lambda_mag}")
    out_dir = Path(args.out_dir)
    best_macro = float("-inf")
    step = 0
    accum = 0
    t0 = time.perf_counter()
    train_iter = iter(train_loader)

    probe_steps = {int(s) for s in args.probe_steps.split(",") if s.strip()}
    probe = None
    probe_batch = None
    if probe_steps:
        # One fixed batch for every probe step, so only the MODEL differs between
        # them; held out so a visit can't flatter a batch the model has fitted.
        # ⚠ Costs the stream its first batch — compare probed runs against probed.
        probe_batch = {k: v.to(torch.cuda.current_device())
                       for k, v in next(train_iter).items()}
        # A "stream" is one independent residual the walk drives: a batched fold
        # runs the merged track as `exec_groups` of them, the looped path runs one
        # per logical track. Numbering them globally (rank-major, contiguous — the
        # assignment `dist.groups.compute_contiguous_assignment` makes) is what lets
        # the per-rank JSONL files be concatenated without a gather.
        n_streams = exec_groups if exec_groups > 1 else len(layout.local_track_ids)
        shards_per_stream = manifest.n_tracks // (layout.world_size * n_streams)
        probe = ActivationProbe(
            num_layers=num_layers,
            n_streams=n_streams,
            stream_offset=(layout.local_track_ids[0] * merge_group) // shards_per_stream,
            shards_per_stream=shards_per_stream,
            rank=rank,
            out_dir=args.probe_dir or (out_dir / "probe"),
            device=torch.cuda.current_device(),
            teacher_layers=teacher.text_models[0].layers,
            sync_phase=args.sync_phase,
            block_walk=args.block_walk,
            detail_layers={int(x) for x in args.probe_detail.split(",") if x.strip()},
        )
        _log(rank, f"[init] activation probe at steps {sorted(probe_steps)} -> "
                   f"{args.probe_dir or (out_dir / 'probe')} "
                   f"({n_streams} streams/rank x {shards_per_stream} shards, "
                   f"fixed held-out batch)")

    def run_eval(tag: str) -> float:
        """The lm-evaluation-harness downstream macro on the live student.

        Same code path as ``scripts/eval_lm_harness.py``, and the student is
        scored at the schedule it trains on, so the in-loop number is directly
        comparable to a standalone run on the saved slices.
        """
        # Lazy: lm_eval is the optional [eval] extra, so a train-only install
        # still imports this script.
        from parallm.eval.downstream import MissingTasks, macro_metrics
        from parallm.eval.lm_eval_adapter import (
            is_lm_head_owner, make_student_forward_fn, run_lm_eval,
        )

        student.eval()
        # simple_evaluate reseeds the global torch/numpy/random RNGs identically
        # on every rank; the train stream is an already-constructed seeded
        # interleave and student-forcing draws from a per-step local RNG, so
        # neither is perturbed.
        #
        # RUN EVAL EAGER. Training has ONE shape (B x seq_len), so the compiled
        # seam holds a single graph; lm-eval feeds ragged contexts at
        # --eval-batch-size, which re-traces on every new (batch, length) pair.
        # Measured: that blows dynamo's recompile_limit of 8 within one eval and
        # then SIGSEGVs a rank. Nothing here is a step-time path, so compiling it
        # buys nothing even when it works.
        with _eval_compile_guard():
            results = run_lm_eval(
                make_student_forward_fn(student), tokenizer,
                tasks=eval_tasks, is_owner=is_lm_head_owner(student),
                limit=args.eval_limit, batch_size=args.eval_batch_size,
                max_length=args.eval_max_length, num_fewshot=args.eval_num_fewshot,
                seed=args.seed, log_samples=False,
                quiet=True,  # the training log gets the numbers, not progress bars
                include_path=args.eval_task_path,
            )
        student.train()
        try:
            per_task = macro_metrics(results, eval_tasks)  # populated on rank 0 only
            score = sum(per_task.values()) / len(per_task) if per_task else 0.0
        except MissingTasks as e:
            # A task that failed to load must NOT be averaged away: the tasks span
            # ~0.4 in accuracy, so a partial macro can beat a complete one and
            # promote a worse checkpoint. Score it unselectable and keep training —
            # losing the run to a hub blip would be the worse failure.
            _log(rank, f"{tag} INCOMPLETE — not eligible for best/: {e}")
            # -inf, not a low finite score: `best_macro` starts at -inf and the
            # selection test is a strict `>`, so this can never win, first eval or
            # last.
            per_task, score = {}, float("-inf")
        # _save_checkpoint is collective and ends in dist.barrier(), so a
        # rank-local "improved?" would hang the run — broadcast rank 0's number.
        t = torch.tensor([score], dtype=torch.float64, device=torch.cuda.current_device())
        dist.broadcast(t, src=0)
        _log(rank, f"{tag} macro={t.item():.4f}"
                   + ("".join(f" {k}={v:.4f}" for k, v in per_task.items())))
        return t.item()

    # A kernel trace wants phase NAMES in the trace but not the phase timers' syncs,
    # which serialize the very overlap the busy-vs-wall ratio is measuring.
    prof = PhaseTimer(enabled=args.profile and not args.profile_kernels,
                      record=args.profile_kernels)
    kprof = _KernelWindow(args.profile_kernels)

    while step < args.max_steps:
        kprof.maybe_start(step)
        prof.reset()
        prof.start_step()
        with prof.phase("data"):
            batch = next(train_iter)
            batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}
        cur_lr[0] = lr_at(step)  # the in-backward hooks read this as they fire
        gsq.zero_()
        # Rank-uniform gate — an asymmetric probe desyncs the next collective. Zeroing
        # both lambdas skips every backward and the CE half, reusing the training walk.
        if probe is not None and step in probe_steps:
            probe.step = step
            with torch.no_grad(), prof.phase("probe"):
                distill_step(student, teacher, lm_head, probe_batch,
                             replace(dcfg, lambda_block=0.0, lambda_ce=0.0),
                             probe=probe)
            for line in summary_lines(probe.flush(), step, sync_phase=args.sync_phase,
                                     block_walk=args.block_walk):
                _log(rank, line)
        losses = distill_step(
            student, teacher, lm_head, batch, dcfg,
            loss_scale=1.0 / args.grad_accum_steps,
            prof=prof,
        )
        accum += 1
        if accum < args.grad_accum_steps:
            continue
        accum = 0

        with prof.phase("grad_sync"):
            sync_replicated_grads(plan)
        if args.optim_in_backward:
            # Hooked params already stepped and freed, so no moment holds the whole
            # gradient to clip against; gsq is this rank's running sum, diagnostic
            # only (not deduplicated across replication groups).
            gnorm = gsq.sqrt()
        else:
            gnorm = compute_global_grad_norm(student, plan)
            clip = (args.max_grad_norm / (gnorm + 1e-6)).clamp(max=1.0)
            if clip < 1.0:
                for p_ in trainable:
                    if p_.grad is not None:
                        p_.grad.mul_(clip)
        if optim is not None:
            with prof.phase("optim"):
                for g in optim.param_groups:
                    g["lr"] = lr_at(step)
                optim.step()
                optim.zero_grad(set_to_none=True)
        if args.optim_in_backward:
            fire_prev = dict(fire_cur)  # this batch's counts normalize the next
            fire_cur.clear()

        kprof.maybe_stop(step)

        if step % args.log_every == 0:
            dt = (time.perf_counter() - t0) / max(1, step + 1)
            _log(rank, f"step {step} total={losses['total'].item():.4f} "
                       f"block={losses['block_mse'].item():.4f} "
                       f"ce={losses['ce'].item():.4f} "
                       + f"gnorm={gnorm.item():.2f} lr={lr_at(step):.2e} "
                         f"{dt:.2f}s/step")
            if args.profile:
                _log(rank, prof.report())
            if args.ledge_probe:
                # Per-tap relMSE is already computed and returned by distill_step;
                # gnorm is a rank-local non-deduplicated sum under
                # --optim-in-backward, so THIS is the trustworthy crossing signal.
                taps = losses.get("layer_relmse") or {}
                if taps:
                    _log(rank, f"[taps] step {step} "
                               + " ".join(f"{i}:{v:.4f}" for i, v in sorted(taps.items())))
                _log(rank, f"[dist] step {step} {_dist_report()}")
        if args.mem_report and step == 2:
            _log(rank, f"[mem] rank0 peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
                       f"resident={torch.cuda.memory_allocated()/2**30:.2f}GiB")

        step += 1
        if args.eval_every and step % args.eval_every == 0:
            macro = run_eval(f"[eval] step {step}")
            if macro > best_macro:
                best_macro = macro
                if not args.no_save:
                    _save_checkpoint(student, manifest, layout, out_dir / args.best_name,
                                     {"step": step, "macro": macro,
                                      "sync_layer_indices": sync_layers,
                                      "sync_phase": args.sync_phase,
                                      "args": vars(args)}, rank)
                    _log(rank, f"[save] best → {out_dir / args.best_name}")
        if args.save_every and not args.no_save and step % args.save_every == 0:
            _save_checkpoint(student, manifest, layout, out_dir / f"step_{step}",
                             {"step": step, "sync_layer_indices": sync_layers,
                              "sync_phase": args.sync_phase, "args": vars(args)}, rank)

    kprof.final_report(rank)  # after the loop: no collective is pending
    macro = run_eval(f"[final] step {step}")
    if macro > best_macro and not args.no_save:
        _save_checkpoint(student, manifest, layout, out_dir / args.best_name,
                         {"step": step, "macro": macro,
                          "sync_layer_indices": sync_layers,
                          "sync_phase": args.sync_phase, "args": vars(args)}, rank)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
