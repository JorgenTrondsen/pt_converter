# ParaLLM

Slice an LLM into **parallel tracks** and give each track cheap **sparse
copies** of the others, so a distributed decode needs only a handful of
cross-track syncs per token instead of one per layer.

The premise (unchanged from the project's origin): **more tracks, fewer syncs.**
A dense model is sliced across `N` tracks (e.g. one attention head + an MLP slice
per track at `N=16`). Tracks run in lockstep; between **sync boundaries** each
track only sees its own partial residual update. The gap that opens between syncs
is what makes fewer syncs hard.

The result this repo is built around: that gap is recovered **comm-free** by
letting each track *recompute* the other tracks between syncs through cheap,
**activation-aware sparse copies** of their weights (Wanda / qwanda). At `N=16`
this holds ~90–99% of teacher downstream quality down to **4 sync events** on a
9B model (`D=8`), training-free, and composes with an already-quantized base. The
copies are static and stream from host DRAM behind the sync stalls, so they need
little resident HBM.

## The workflow

1. **Convert** — slice a dense model into `N` per-track shards
   (`scripts/convert_qwen3_5_9b.py` → `track_i.safetensors` + `manifest.json`).
   The sync schedule is *not* baked in; per-track weights are schedule-independent.
2. **Build cheap copies** — from one dense calibration pass, construct each
   track's sparse replica pool (`parallm.model.replica`: `collect_input_norms`
   then `degrade_track_layers` at a chosen `wanda`/`qwanda` sparsity).
3. **Serve** — a lightweight inference engine (sglang GPU backend, tracks spread
   across nodes at ~20 ms inter-node latency) runs the tracks in lockstep and
   replays the sparse copies between syncs, streaming the replica pool from host
   DRAM behind the sync stalls. *(This engine is the next build; the streaming
   account it implements is measured by `scripts/bench_stream_overlap.py`.)*

Optional **healing/training** of very sparse copies can reuse the primitives left
in `parallm.train` (`teacher`, `losses`, `data`) — note the project's own
experiments found plain wanda replicas are training-free and co-training them was
flat, so treat this as a skeleton, not a required step.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core: torch, transformers, safetensors, datasets, accelerate
pip install -e '.[eval]'    # + lm-eval, for the downstream harness
pip install -e '.[fast]' --no-build-isolation   # + FLA / causal-conv1d GPU kernels for the linear-attn layers
```

The 24 gated-delta linear-attention layers fall back to a slow pure-torch path
without the `[fast]` kernels; CPU tests force that path automatically.

## Convert

```bash
python scripts/convert.py \
    --hf-model <checkpoint dir (bf16, NVFP4, dense or MoE)> \
    --out-dir  ./pt_tracks \
    --n-tracks 16          # defaults to max-tracks for the model
```

Writes `track_0..N-1.safetensors` + `manifest.json`. One streaming converter for
every source — it dequantizes NVFP4/FP8 to bf16, drops the vision tower + MTP head,
and resolves the right adapter from `config.model_type`.

## Build the replica pool

```python
from parallm.model.replica import collect_input_norms, degrade_track_layers

# one dense calibration pass → per-input-channel norms (the Wanda criterion)
norms = collect_input_norms(dense_text_model, calib_batches, device="cuda")

# per track: deep-copied decoder layers with wanda-sparse (int4-quantized) weights
pool = degrade_track_layers(track_model, norms, n_tracks, track_id, frac=0.5, bits=4)
```

`frac` is the copy sparsity (0.5 is the depth-safe knee), `bits` the optional
int-quantized base (`qwanda`). See `parallm/model/replica.py`.

Or build + pack the whole pool to one file (family-general — bf16 / NVFP4 / MoE):

```bash
python scripts/build_replicas.py --tracks-dir ./pt_tracks --hf-model <checkpoint dir> \
    --config qwanda:4:0.5        # or wanda:0.5  |  q4mlp/q8mix:0.5 (NVFP4-mlp/FP8-mixer base)
```

## Evaluate

```bash
# fidelity: KL / top-k agreement / per-boundary hidden-MSE / ppl gap vs the teacher
torchrun --standalone --nproc-per-node=8 scripts/eval_fidelity.py \
    --hf-model <teacher> --checkpoint-dir ./pt_tracks --num-batches 200

# downstream retention (the metric that matters — proxies hid failures before).
# --tasks defaults to the 4-task macro: arc_easy, arc_challenge, mmlu_pro_math_mc,
# codemmlu_fim. Name any lm-eval built-in, or any YAML in configs/eval_tasks.
torchrun --standalone --nproc-per-node=8 scripts/eval_lm_harness.py \
    --hf-model <teacher> --checkpoint-dir ./pt_tracks

# the unbiased math number: the limit-200 prefix of that task reads high
torchrun --standalone --nproc-per-node=8 scripts/eval_lm_harness.py \
    --hf-model <teacher> --checkpoint-dir ./pt_tracks \
    --tasks mmlu_pro_math_mc --limit 0
```

Training data and eval tasks are both chosen at launch, no code change:
`--data-preset` takes a mixture name under [configs/data](configs/data) or a path to
any mixture JSON, and `--tasks` / `--eval-tasks` take any registered task name. A
new dataset is a new JSON; a new benchmark is a new YAML in
[configs/eval_tasks](configs/eval_tasks). In a mixture, a source's `weight` is its
share of training **tokens** — the realized token/document split is logged at startup.

## Layout

```
src/parallm/
├── slicer/         # dense → N per-track state dicts + PTManifest (base.py specs, convert.py engine, qwen3_5.py specs)
├── adapters/       # model-family registry (qwen3, qwen3_5, qwen3_5_moe, gpt_oss)
├── model/
│   ├── pt_model.py     # PTWrappedModel: per-rank lockstep forward over K tracks + SyncBoundary
│   ├── sync.py         # SyncBoundary: the one cross-track all-reduce
│   ├── replica.py      # the payload: wanda/qwanda copies + collect_input_norms + degrade_track_layers
│   └── tracks/qwen3_5.py  # per-track Qwen3.5 decoder
├── eval/           # fidelity.py, downstream.py, lm_eval_adapter.py
├── train/          # healing skeleton: teacher.py, losses.py, data.py
├── dist/           # groups.py (track/rank layout), fsdp_setup.py
└── utils/          # checkpoint.py (safetensors IO), max_tracks.py
scripts/            # convert / eval_fidelity / eval_lm_harness / bench_stream_overlap
docs/pt_state.md    # onboarding: the sparse-replica result and how it was reached
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

Core forward parity (`N=1` bit-equal to dense, `N=8`/`K=2` sync paths), slicer
round-trips, KV replication, and the replica rails (`test_replica.py`).
