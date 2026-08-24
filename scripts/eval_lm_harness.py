"""torchrun entry point: run lm-evaluation-harness against the PT student and/or the dense teacher.

Default ``--target both`` loads both models, runs lm-eval on each with the
same task list / seeds, and prints side-by-side results — that's how you tell
whether the student "performs the same" as the teacher on the field-standard
benchmarks. Pass ``--target student`` or ``--target teacher`` to evaluate only
one (skips loading the other to save GPU memory).

Distributed pattern: every rank runs identical lm-eval code (same seed → same
request ordering). The student's non-owner ranks emit zero placeholders but
still execute the forward so cross-track SyncBoundary collectives match the
owner rank; only rank 0 prints / saves results. See
``src/parallm/eval/lm_eval_adapter.py``.

Single node, 8 GPUs, compare student vs teacher on the default task set:

    torchrun --standalone --nproc-per-node=8 scripts/eval_lm_harness.py \\
        --hf-model <teacher path> \\
        --checkpoint-dir ./pt_train_out/best \\
        --output-json ./pt_train_out/lm_eval.json

Smoke run (32 requests per task, fast pipeline check):

    torchrun --standalone --nproc-per-node=8 scripts/eval_lm_harness.py \\
        --hf-model <teacher path> \\
        --checkpoint-dir ./pt_train_out/best \\
        --tasks hellaswag,arc_easy \\
        --limit 32
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from parallm.dist.fsdp_setup import wrap_teacher_with_fsdp
from parallm.dist.groups import build_groups
from parallm.eval.downstream import DEFAULT_TASKS, EVAL_TASK_PATH, MissingTasks, macro_metrics
from parallm.eval.lm_eval_adapter import (
    is_lm_head_owner,
    make_student_forward_fn,
    make_teacher_forward_fn,
    run_lm_eval,
)
from parallm.model.merge import plan_track_layout
from parallm.model.pt_model import PTWrappedModel
from parallm.train.teacher import HookedTeacher, load_dense_reference
from parallm.utils.checkpoint import load_manifest, load_track, train_meta_arg


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _run_one_target(
    target: str,
    forward_fn,
    is_owner: bool,
    tokenizer,
    tasks: list[str],
    args,
    rank: int,
) -> dict:
    """Single lm-eval pass over the configured task list. Returns the raw
    ``simple_evaluate`` dict (or ``None`` off rank 0)."""
    _log(rank, f"[eval] running lm-eval on target={target}…")
    return run_lm_eval(
        forward_fn,
        tokenizer,
        tasks=tasks,
        is_owner=is_owner,
        limit=args.limit,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_fewshot=args.num_fewshot,
        seed=args.seed,
        include_path=args.eval_task_path,
    )


def _format_results(results: dict | None) -> str:
    if results is None or "results" not in results:
        return "  (no results)"
    lines = []
    for task_name, metrics in results["results"].items():
        pretty = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
            if not k.startswith("alias")
        )
        lines.append(f"  {task_name}: {pretty}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["student", "teacher", "both"], default="both",
                   help="Which model(s) to evaluate. Default 'both' loads student + teacher and "
                        "runs lm-eval on each with identical seeds for side-by-side comparison. "
                        "Use 'student' or 'teacher' alone to skip loading the other side.")
    p.add_argument("--hf-model", required=True,
                   help="Dense teacher model path (used for tokenizer in every mode; for teacher "
                        "weights in 'teacher' or 'both' mode).")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Per-rank checkpoint dir with track_*.safetensors and manifest.json. "
                        "In 'teacher' mode only the manifest is read (for n_tracks / sync layout).")
    p.add_argument("--tasks", default=DEFAULT_TASKS,
                   help="Comma-separated lm-eval task names — built-ins, or any YAML in "
                        "--eval-task-path. The default five are the macro convention. For an "
                        "unbiased math number use `--tasks mmlu_math_mc,mmlu_cs_mc --limit 0`: "
                        "cs is the subject-matched control, so read the two together. "
                        "mmlu_pro_math_mc is the old (10-way) math task, off-macro but still "
                        "scorable by name. Full MMLU is opt-in via --include-mmlu (slow).")
    p.add_argument("--eval-task-path", default=EVAL_TASK_PATH,
                   help="Directory of extra lm-eval task YAMLs (default: the repo's "
                        "configs/eval_tasks). Must match the trainer's --eval-task-path, or a "
                        "rescore here and the in-loop macro stop meaning the same thing.")
    p.add_argument("--include-mmlu", action="store_true",
                   help="Append `mmlu` to --tasks. Adds ~hours of GPU time on a single node.")
    p.add_argument("--num-fewshot", type=int, default=None,
                   help="Override fewshot count for all tasks. Default = each task's standard shot count.")
    p.add_argument("--limit", type=int, default=200,
                   help="Cap requests per task; 0 scores the full sets. Defaults to the "
                        "trainer's 200 so a rescore here is comparable to the in-loop macro "
                        "(noise ~±0.015). Lower it for smokes.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Requests scored per forward. Each forward pays a fixed set of NCCL "
                        "all-reduces (embed + one per sync boundary) regardless of batch size, "
                        "so batching amortizes them and fills the GPU — the biggest eval speedup. "
                        "The owner rank holds a (B, max_len, vocab) bf16 logits tensor, so lower "
                        "this for long-context / loglikelihood-rolling tasks (e.g. wikitext) to "
                        "avoid OOM; the short-context multiple-choice tasks are fine at 8+.")
    p.add_argument("--max-length", type=int, default=2048,
                   help="Tokenizer truncation length for lm-eval contexts. Matches the "
                        "trainer's --eval-max-length: a longer cap here would fit fewshot "
                        "prompts the in-loop macro truncated, and the two numbers would diverge.")
    p.add_argument("--output-json", default=None,
                   help="Optional path to dump per-target lm-eval results dict (rank 0 only). "
                        "In 'both' mode the file contains a top-level {student, teacher} dict.")
    p.add_argument("--seed", type=int, default=42,
                   help="lm-eval random/numpy/torch seed. Identical on every rank so request "
                        "ordering and fewshot sampling line up across ranks. Matches the "
                        "trainer's --seed so both draw the same limited subset of each task.")
    p.add_argument("--sync-indices", default=None,
                   help="Override the manifest sync schedule (comma-separated layer indices). "
                        "Required when the checkpoint is a raw convert output, which carries no "
                        "schedule.")
    p.add_argument("--sync-phase", default=None, choices=["post-mlp", "post-attn", "exact"],
                   help="Sync placement; DEFAULT reads it from the checkpoint's "
                        "train_meta.json (else post-attn). Scoring a model at the wrong phase "
                        "is a different network — a post-attn heal read as post-mlp measured "
                        "0.553 vs its true 0.700. 'post-attn'=lever B, L+1 syncs; "
                        "'post-mlp'=the attention sync dropped instead, L syncs; "
                        "'exact'=2 syncs/layer ≡ dense.")
    p.add_argument("--fuse-tracks", type=int, default=None,
                   help="F rank-local tracks pool their partials at every non-sync sublayer "
                        "(N/F-track behaviour on N shards). DEFAULT reads it from the "
                        "checkpoint's train_meta.json (else 1) — same trap as --sync-phase: "
                        "scoring a fused model unfused is a different network.")
    p.add_argument("--engine-replicas", default=None,
                   help="Path to a packed replica pool: score the student through the "
                        "sparse-replica ENGINE forward (parallm.engine.replay_chunk) instead "
                        "of the plain PT forward — the deployed decode math.")
    p.add_argument("--moe-experts", choices=["auto", "dense", "grouped_mm"], default="auto",
                   help="How a per-track MoE evaluates its experts (see train_cli). Exposed "
                        "here because scoring ONE checkpoint under both settings is the "
                        "equivalence check: lm-harness is bit-deterministic, so any macro "
                        "difference on identical weights is the implementation, not noise.")
    args = p.parse_args()

    from parallm.model.moe_dense import set_experts_policy

    set_experts_policy(args.moe_experts)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} >= visible GPU count {torch.cuda.device_count()}."
        )
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    # The manifest records the sync INDICES but not the PHASE or the track
    # fusion, so both come from the training metadata beside the weights.
    if args.sync_phase is None:
        args.sync_phase, _resolved_from = train_meta_arg(
            args.checkpoint_dir, "sync_phase", "post-attn")
    else:
        _resolved_from = "flag"
    if args.fuse_tracks is None:
        args.fuse_tracks, _fuse_from = train_meta_arg(args.checkpoint_dir, "fuse_tracks", 1)
    else:
        _fuse_from = "flag"

    manifest = load_manifest(args.checkpoint_dir)
    # Eval stays on the LOOPED path (`allow_merge=False`) — it is not the bottleneck
    # and the looped form is the equivalence reference. The plan is still needed for
    # `fuse_ranks`: a run trained with F > tracks-per-rank has cross-rank fusion
    # groups, and scoring it without them measures a different model.
    try:
        plan = plan_track_layout(
            manifest.n_tracks, dist.get_world_size(), args.fuse_tracks,
            allow_merge=False,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    layout = build_groups(n_tracks=manifest.n_tracks, fuse_ranks=plan.fuse_ranks)
    if args.sync_indices is not None:
        sync_layers = [int(x) for x in args.sync_indices.split(",") if x.strip() != ""]
    elif manifest.sync_layer_indices is not None:
        sync_layers = list(manifest.sync_layer_indices)
    elif args.sync_phase == "exact":
        sync_layers = list(range(manifest.num_layers))  # exact ignores the schedule
    else:
        raise SystemExit(
            "[error] checkpoint manifest has no sync_layer_indices — point "
            "--checkpoint-dir at a trained checkpoint (the schedule is placed at "
            "train time, so a raw convert output carries none), or pass --sync-indices."
        )
    _log(
        rank,
        f"[init] target={args.target} world={layout.world_size} "
        f"n_tracks={manifest.n_tracks} K={layout.tracks_per_rank} "
        f"sync_phase={args.sync_phase} (from {_resolved_from}) "
        f"fuse={args.fuse_tracks} (from {_fuse_from})",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    want_student = args.target in ("student", "both")
    want_teacher = args.target in ("teacher", "both")

    student = None
    teacher = None
    teardown_fns: list = []

    if want_student:
        cfg = AutoConfig.from_pretrained(args.hf_model)
        # The original snapshot wraps the text config (`cfg.text_config`); a
        # dense checkpoint saved from the text-only causal model (e.g. a healed
        # best/) IS the text config already.
        text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
        _log(rank, f"[init] building PT student for tracks {layout.local_track_ids}…")
        student = PTWrappedModel(
            text_config=text_cfg,
            n_tracks=manifest.n_tracks,
            local_track_ids=layout.local_track_ids,
            sync_after_layers=sync_layers,
            track_group=layout.track_group,
            fuse_size=plan.fuse_size,
            fuse_group=layout.fuse_group,
            fuse_ranks=layout.fuse_ranks,
            fuse_rank=layout.fuse_rank,
        )
        track_states = {tid: load_track(args.checkpoint_dir, tid) for tid in layout.local_track_ids}
        student.load_track_state_dicts(track_states, strict=True)
        student.set_sync_phase(args.sync_phase)
        # bf16 on the HOST first — see the same note in train_cli._build_student.
        student = student.to(torch.bfloat16).to(torch.cuda.current_device())
        student.eval()

    if want_teacher:
        _log(rank, "[init] loading frozen dense teacher…")
        teacher_model, text_model = load_dense_reference(args.hf_model)
        teacher = HookedTeacher(
            text_model=text_model,
            lm_head=teacher_model.lm_head,
            sync_layer_indices=sync_layers,
        )
        # Shards layer-by-layer off CPU; moving the whole dense teacher across
        # first OOMs a 40 GB card at 27B (see wrap_teacher_with_fsdp).
        wrap_teacher_with_fsdp(text_model, teacher_model.lm_head)
        teacher_model = teacher_model.to(torch.cuda.current_device())
        teardown_fns.append(teacher.remove_hooks)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.include_mmlu and "mmlu" not in tasks:
        tasks.append("mmlu")
    args.limit = args.limit or None  # 0 means "no cap", which lm-eval spells None
    _log(rank, f"[init] tasks={tasks} limit={args.limit}")

    all_results: dict[str, dict] = {}
    if want_student:
        student_fwd = make_student_forward_fn(student)
        if args.engine_replicas:
            from parallm.engine import PackedShadow, make_engine_forward_fn

            _log(rank, f"[init] loading packed pool {args.engine_replicas}…")
            shadow = PackedShadow(args.engine_replicas, args.checkpoint_dir,
                                  student, torch.cuda.current_device())
            student_fwd = make_engine_forward_fn(student, shadow, sync_layers)
        all_results["student"] = _run_one_target(
            "student",
            student_fwd,
            is_lm_head_owner(student),
            tokenizer, tasks, args, rank,
        )
    if want_teacher:
        all_results["teacher"] = _run_one_target(
            "teacher",
            make_teacher_forward_fn(teacher),
            True,  # FSDP all-gathers → real logits on every rank
            tokenizer, tasks, args, rank,
        )

    if rank == 0:
        print()
        for tgt, results in all_results.items():
            print(f"===== lm-evaluation-harness ({tgt}) =====")
            print(_format_results(results))
            try:
                m = macro_metrics(results, tasks)
                print(f"  macro={sum(m.values()) / len(m):.4f}")
            except MissingTasks as e:
                # Printed, never silently averaged over the survivors: the tasks
                # span ~0.4 in accuracy, so a partial macro is not a smaller macro,
                # it is a different (usually higher) number.
                print(f"  macro=INCOMPLETE — {e}")
            print()
        if args.output_json:
            Path(args.output_json).write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            print(f"full results → {args.output_json}")

    for fn in teardown_fns:
        fn()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
