# parallm — state and onboarding

Where the parallel-track (PT) conversion stands, the one result the codebase is
now built around, and where the code lives. Written to onboard a run without
re-deriving the months of dead ends behind it.

**Locked premise (do not violate):** *more tracks, fewer syncs.* Never "fix" a
quality gap by using fewer tracks, a lower `D`, or more syncs — those work but
defeat the point. Valid levers keep `N` high and the sync budget low.

## 1. Architecture

A dense model is sliced across `N` tracks (at `N=16`: one attention head + an MLP
slice per track). The forward runs all tracks lockstep. Between **sync
boundaries**, each track adds only its *own* partial residual update; at a
boundary one all-reduce recombines them — the only cross-track collective
(`SyncBoundary`, [model/sync.py](../src/parallm/model/sync.py)):

```
h_synced = h_pre_block + Σ_t (h_t − h_pre_block)
```

`D` = layers between syncs. Between two syncs a mid-window layer reads only its
track's residual (~`1/N` of the real update), so `D>1` cannot be recovered by any
purely *statistical* estimate of the missing content — that was proven, at
length, refuted (predictors, stale caches, fixed low-rank, geometry, on-policy
training, window-parallel rewiring, cross-head seams). Per-track weights are
schedule-independent; the schedule is chosen at serve/eval time, not baked into
the slice.

## 2. The result the code is built around — sparse-copy recomputation

The between-sync trajectory of any track is a **deterministic function of the last
synced residual and that track's weights**. So instead of *estimating* the missing
cross-track content, each rank **recomputes** the other tracks by replaying their
sublayers through **cheap copies** of their weights — comm-free.

The copies that hold quality are **activation-aware (Wanda) sparse** ones,
optionally over an int-quantized base (**qwanda**): score each weight
`|w_ij|·‖x_j‖` from one dense calibration pass and keep the top `1−frac` per
output row (survivors exact). Measured on the 9B slice, % of the `zero→oracle`
(dense) downstream headroom recovered:

| copy | D=2 (16 syncs) | D=4 (8) | D=8 (4 syncs) | bits/w |
|---|---|---|---|---|
| `wanda:0.5` | 98.0% | 97.7% | **99.4%** | 9 |
| `qwanda:4:0.5` (int4 + 50% sparse) | — | — | **89.8%** | 3 |

**`wanda:0.5` is depth-invariant at ~98–99% of dense quality down to 4 sync
events**, training-free, and its cheapness is *structural* (composes with an
already-quantized base). The frontier is a sparsity↔memory menu; a sub-1 GB
per-track pool is closed in this family (the replica must be a whole-network copy
— attn-only and MLP-only copies each collapse; both together = 99.4%).

The pool is static and accessed layer-sequentially, and every `D=8` pass stalls
≥20 ms at each of 4 sync boundaries, so it **streams from pinned host DRAM behind
the stalls**: a one-window ring (~626 MB) is as good as a full double buffer, cutting
resident HBM ~55% at +0 ms once a real multi-node sync costs ≥40 ms (measured,
[bench_stream_overlap.py](../scripts/bench_stream_overlap.py)).

Streaming is **window-granular and CUDA-graph-compatible** (2026-07-12): ring
slots sit at fixed device addresses the captured windows bake, and all
copy/event work runs in the eager boundary regions — streamed + graphed decode
is bit-identical to resident (`--residency streamed`, rails in
[tests/test_engine.py](../tests/test_engine.py)). On top of it, the codes plane
(the only compressible one: int4 codes carry 2.97/4 bits, the bitmap is AT the
entropy limit) can stream **entropy-coded** (`--pool-codec ent`,
[entropy_codec.py](../src/parallm/entropy_codec.py) +
[scripts/repack_replicas_entropy.py](../scripts/repack_replicas_entropy.py)):
a pool-wide static Huffman table, GPU-decoded into the ring during the stalls
(batched block-parallel triton decoder, ~26 GB/s on int4 pools — int8-heavy
pools drop to ~13 GB/s, the LUT gather scatters), lossless ⇒ tokens stay
bit-identical. Measured on the 9B `qwanda:4:0.5` floor pool (2.50 GB → 2.11 GB):
per-window copies 49.7 → 40.8 ms on a 12.7 GB/s shared link. On a dedicated
PCIe4 node (~26 GB/s) the ent window (~528 MB) ≈ the 20 ms stall budget —
streaming is ~free at S=20 where raw (+4 ms/window) is not, and both are free
at S≥25 or PCIe5.

## 3. Where it lives

- **Convert:** [slicer/](../src/parallm/slicer/) + [scripts/convert.py](../scripts/convert.py) (one streaming converter for bf16 / NVFP4 / FP8 / MXFP4, dense / MoE) → per-track `safetensors` + manifest.
- **Model families:** [adapters/](../src/parallm/adapters/) — `qwen3_5_text`, `qwen3_5_moe_text`, `gpt_oss` (20b + 120b), `qwen3` (dense 8B/14B/32B). Adding one is three small files: `slicer/<family>.py` (per-param `SlicerSpec`s + the `{layer_type: (prefix, spec_fn)}` map + `build_masks`), `model/tracks/<family>.py` (three HF module classes + a per-track config builder), `adapters/<family>.py` (the `ModelAdapter` + one import line in `adapters/__init__.py`). Nothing outside those may name a family. Rails to copy: [tests/test_gpt_oss_slice.py](../tests/test_gpt_oss_slice.py) — reassembly round-trip, N=1 dense parity, and **N>1 parity at `sync_phase="exact"`**, which is the one that catches a wrong spec. A family that also wants the batched fold (`supports_batched_exec=True`) declares `attn_ops=AttnOps(...)` — the two things `engine._batched_attn` cannot read off a module, a `[q|gate]` doubled `q_proj` and a zero-centered (`1 + w`) RMSNorm — and copies the equivalence rails in [tests/test_qwen3_slice.py](../tests/test_qwen3_slice.py), which drive the fold against the LOOPED path so a wrong flag fails instead of returning plausible numbers.
- **Copies (the payload):** [model/replica.py](../src/parallm/model/replica.py) — `collect_input_norms` (dense calibration), `wanda_prune_weight` / `fake_quant_weight` / `block_wanda_prune_weight` (the per-weight transforms), `degrade_track_layers` (build a track's replica pool). Rails: [tests/test_replica.py](../tests/test_replica.py).
- **Forward:** [model/pt_model.py](../src/parallm/model/pt_model.py) `PTWrappedModel` (lockstep window iteration + `SyncBoundary`).
- **Track grouping (`--fuse-tracks F`):** F shards compute as one track between syncs. [model/merge.py](../src/parallm/model/merge.py) `plan_track_layout` owns the whole policy and is the one place to read; it returns the four mechanisms and asserts `effective_fuse == F`. Valid F = the divisors of K=N/world **plus its multiples up to N** — at or below K a group is rank-local, above K it spans F/K whole ranks through a subgroup all-reduce (`SyncBoundary.fuse` + `_LeaderOnly`), so **F is no longer capped by the GPU count**. Merging (concatenated slabs) is the step-time lever and is now on for the MoE families too: it widens each expert and leaves the expert *count* alone — **4.38x end-to-end at gpt-oss N=64 (15.3 → 3.5 s/step), 5.25x on the expert GEMM alone**. **The win is WIDTH, not fewer dispatches**: batching the group dispatches while keeping members separate is only 1.12x, so at F=1 (max tracks, 1 attention head and a 48-wide expert slab per track) there is nothing to widen and step time stays on the looped floor — speed and fusion are physically coupled at max tracks. `exec_groups` (the batched fold, [model/batched.py](../src/parallm/model/batched.py)) is dense-only — under G>1 each stream routes its own top-k, so there is no single `grouped_mm`; MoE families set `supports_batched_exec=False` and express F as the merge width instead. Rails: [tests/test_track_layout_plan.py](../tests/test_track_layout_plan.py), [tests/test_moe_merge_equivalence.py](../tests/test_moe_merge_equivalence.py), [tests/test_cross_rank_fusion.py](../tests/test_cross_rank_fusion.py) (4-rank gloo; the only assignment-sensitive one).
- **Step time at F=1** (max tracks, where merging has nothing to widen): the step is ~100% overhead — gpt-oss-20b activates 3.61 B of 21 B params, so a rank runs ~16.6 TFLOP in a 10.0 s step ≈ **0.5% of an A100's bf16 peak**. **Measure the split before optimising anything**: halve `--seq-len` with no profiler attached — launch COUNT is unchanged and device work roughly halves, so the fit is the answer. Measured 512/1024/2048 → **6.89 / 7.88 / 10.01 s/step = ~2.05 ms per token plus a 5.8 s FIXED FLOOR**, i.e. 58% of the step does not care how much data flows through it. Do this rather than trusting `--profile-kernels`' busy-vs-wall ratio, which the profiler's own per-op cost inflates by seconds at 321k ops/step. Levers, in [train/profile.py](../src/parallm/train/profile.py) + [model/seam.py](../src/parallm/model/seam.py) + [model/moe_dense.py](../src/parallm/model/moe_dense.py): `--compile {mixer,mlp,both}` compiles the two seam *functions* (**1.24x**; the mixer half alone is only 1.07x — the MoE half is where the launches are), dropping `--optim-in-backward` is a further 1.13x, and `--tf-ckpt-min-segment` stops checkpointing own-carry segments too short to be worth a recompute. **`--moe-experts` is the one that attacks the floor**: `torch._grouped_mm` has no true grouped kernel below SM90 and LOOPS over experts, so at N=64 one MoE call issued ~192 skinny GEMM launches plus a sort and four 47 MB gathers whose backward was 39% of all GPU time — dense evaluation replaces the lot with two plain GEMMs (**4.70x on the isolated call, 398 → 56 launches**). ⚠⚠ **Two answers this area used to give are REFUTED and must not be re-derived**: CUDA graphs (built, 3 blockers, `seam.enable_seam_compile` raises) and multi-stream tracks (refuted unbuilt — concurrency cannot help a device that is already idle). **THE TWO CONFIGS WORTH RUNNING (F=1/N=64, all measured in one bracketed matrix): `--moe-experts dense --optim-in-backward --compile both` = 4.84 s/step at 10.85 GiB when memory binds, or drop `--optim-in-backward` for 3.85 s/step at 15.89 GiB when it does not** — oib is a ~5 GiB memory lever costing ~1.13x speed, not a speed tax to avoid. ❌ AOTAutograd's min-cut partitioner (`--recompute partitioner`) was built here and REMOVED once dense landed: it reached only 4.33 s/step at 14.86 GiB, where dropping `--optim-in-backward` is faster for ~1 GiB more — **a lever is worth what it buys on the CURRENT step, not the one it was tuned on**. ⚠ the logged `s/step` is a CUMULATIVE average — de-average before comparing, and measure on an idle box ([scratchpad/steptime_matrix.sh](../scratchpad/steptime_matrix.sh) refuses otherwise), bracketed by two reference arms — rank 0 shares its GPU with a bursty foreign job.
- **Step time on the BATCHED fold, where the same flag inverts** (qwen3-32B, N=64/F=1, seq 1024, 8×A100): `--compile` reached only the two seam *functions*, so it compiled **nothing** for a family running the batched path (`exec_groups > 1`) — that is every `supports_batched_exec` family, at every F, and it is why an earlier `--compile mlp` arm read "inside the noise". It now also compiles [model/batched.py](../src/parallm/model/batched.py)'s halves, through a per-layer `_LayerFold` that resolves the weights **before** the compiled region: dynamo SPECIALIZES ON PYTHON INTS, so handing the compiled callable the layer index recompiled per layer, blew `recompile_limit` (8) and ran the remaining 56 layers EAGER — slower than not compiling. Measured against two eager bracket arms at 3.48 / 3.55: **3.52 → 2.70 s/step = 1.30x, peak 17.71 → 17.27 GiB** (memory-NEGATIVE, as on gpt-oss). **The mixer half is the entire win here (2.88 alone) and the MLP half alone buys nothing (3.60 = the bracket)** — the exact inverse of the gpt-oss attribution above, because a dense SwiGLU is already 3 batched GEMMs while the attention half carries two fp32 `_rms_tracks` chains plus the rope and reshapes. ⚠⚠ **the "fixed floor" the bullet above quotes is a CONTENDED-BOX number**: this same config read 6.0 s/step while a foreign job shared GPU 0 and **3.38 s/step once it left** (43% of the step), so re-measure the baseline on an idle box before pricing any lever against it. Post-compile phases: `teacher_fwd` 30%, `fr_backward` 28%, `tf_backward` 25%, `tf_block_loop` 9%, `fr_forward` 5% — and `grad_sync` is now 0-1% where it was 20% of pure straggler wait. Numerics judged against a CONTROL, not zero: two identical eager arms differ 0.014 in step-0 loss and the compiled arm lands inside that spread. Matrix: [logs/qwen3/steptime_matrix_compile.txt](../logs/qwen3/steptime_matrix_compile.txt); rail against the recompile bug: `tests/test_qwen3_slice.py` asserts the graph count is independent of LAYER COUNT.
- **Eval:** [eval/fidelity.py](../src/parallm/eval/fidelity.py) (KL/ppl), [eval/downstream.py](../src/parallm/eval/downstream.py) + [eval/lm_eval_adapter.py](../src/parallm/eval/lm_eval_adapter.py). **Judge recovery by downstream retention** (arc_challenge / winogrande / piqa), not KL/ppl — the proxy hid a real failure once (KL ~85% while hard-reasoning was ~22–33%).

## 4. The inference engine (built)

[engine.py](../src/parallm/engine.py) + [scripts/serve_cli.py](../scripts/serve_cli.py):
lockstep decode, track-as-batch forward, CUDA-graphed windows, simulated
inter-node link (`--latency-ms`), streamed/entropy-coded pool residency (§2).
27B decode ≈ 108 ms/tok at S=20 (5 rounds × 20 ms + ~8 ms) resident; streamed
adds `Σ_w max(0, window_bytes/BW − S)` — on the shared-PCIe sim box BW is
~12.7 GB/s/rank (pairs of GPUs share host links; a real one-rank-per-node
deployment keeps the full link). Inference centralizes embed + lm_head on a
head node; per track ≈ own bf16 blocks + a streamed replica ring + KV/state.

**Node-envelope fit (measured 2026-07-13, 9B/N=16 ent-streamed, S=20).** The
serve panel prints a per-rank HBM ledger. Against the deployment envelope of
8 GB VRAM / 16 GB DRAM per node:

| node | VRAM steady | breakdown | host DRAM |
|---|---|---|---|
| track node (1 track, projected from measured peers) | **4.04 GB** | 0.83 own blocks + 1.42 ring(ent) + 1.79 KV/graphs/act | 2.11 GB pinned |
| head node (embed+lm_head add 3.79 GB) | **~7.8 GB** | tight; shed the head's own track via non-uniform `tracks_per_rank_list` if it overflows | 2.11 GB pinned |

Speed on an uncontended link (world=1 arm; the engine realizes ~21–23 GB/s of
the box's 25.9 GB/s PCIe4): ent copies 25.0 ms/window → **+19.8 ms/tok** at
S=20 (raw: 27.3 ms → +29.3); at S=40 measured **+0.0 — fully hidden**; the
crossover is S ≈ 25 ms, and full PCIe4 or PCIe5 puts ent at ≈ the S=20 stall
already. Verdict: the 9B-class config fits the 8/16 node with ~2× VRAM
headroom on track nodes, and stall-hiding follows the law exactly — free from
S≥25 on this link, S=20 on full PCIe4/PCIe5. Anything above the ~2.1–2.5 GB
pool class cannot hide at S=20/PCIe4: 27B (9.14 GB ent wire) adds ~+270 ms/tok
and its own bf16 slice alone breaks 8 GB; GLM-class needs the MoE
active-expert tier account (`docs/glm_sizing.md`).

## 5. The refuted program (recovering `D>1` quality without recomputation)

Everything that tried to *estimate* the missing cross-track content — statistical
predictors, stale/temporal caches, fixed low-rank buses, permutation/rotation
geometry, staggered cross-attention, Jacobi refinement, phased/post-attn and
window-parallel sync placement, on-policy reverse-KL training, cross-head seams,
low-rank co-trained replicas, L+S decompositions — was refuted end-to-end. That
code and its distillation trainer were removed in the parallm pivot; the full
negative record lives in the project memory and in git at tag
`pre-parallm-pivot`. Do not re-run any of it without a genuinely new idea.

## 6. Draft-verify decode (the >27B / 100B-class arm) — 2026-07-16

Where the pool cannot fit the node (100B+: 23–28 GB > 16 GB DRAM; a ≤1×track
budget closes ALL weight-space copies by arithmetic — b/15 bits/param vs the
~2.3 bits/param measured floor), the surviving lever is **amortized exact
input**: block draft-verify IN TRACK MODE. A small same-tokenizer drafter
(head rank only — zero bytes on track nodes) proposes k tokens; ONE lockstep
verify chunk over the k+1 positions gives every layer real synced residuals;
syncs/token = boundaries/τ. Engine: `generate_draft_verify`
([engine.py](../src/parallm/engine.py)) — k+1-token cached chunks, GDN
conv/recurrent rollback to the accepted prefix, one tiny accept broadcast per
block; rails in `test_draft_verify_matches_plain_greedy`. Measured on the 27B
(D=16 + q4mlp/q8mix pool): **1.17–1.54 syncs/token on prose, 0.24 on code
(on-gen τ=16.33 @k32)**, streamed pool wire ÷ τ ⇒ 0.56 GB/token on code —
27B streamed reopened for code-class decode. **τ is domain-dependent** (0.8B
drafter vs dense 27B: prose 2.7 saturated by k=16; code 7→16, k-capped) —
the drafter, not the schedule or the degraded verifier, is the binding lever;
see `docs/draft_verify_sizing.md` for the full gates + the 100B pool-free d1b
ledger and the two named engineering items (graphed verify windows, fast
drafter loop).

**2026-07-17 wall-clock closed (≤100 ms/token):** resident 95.9/86.5-steady
ms/token prose (k=8) and 36.7/25.8 code (k=32), streamed+ent code 70.6 —
draft-verify now beats the plain 107 ms baseline on both domains resident and
on code streamed. The profiler corrected the tax attribution: the packed GEMV
had chunk positions on the launch grid, so each verify position re-read the
whole pool (925 ms busy/pass); fixed by a tl.dot M-tile (1.7–2.1× the M=1
cost, M=1 body bit-identical). Verify windows are CUDA-graphed at T=k+1
(graphed ≡ eager emitted ids), the drafter runs through the engine itself at
N=1 (`serve_cli.py EngineDrafter`, 5.2 ms/step, single-chain fast path), the
accept broadcast rides the draft time (5 rounds/block), and the GDN rollback
batches all layers into one kernel call. All bit-exact; 119 tests green.
Streamed prose (414 ms/token) is pool-wire physics (pool/τ) — the drafter
lever, same as 100B prose.
