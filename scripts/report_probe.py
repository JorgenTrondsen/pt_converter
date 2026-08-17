"""Read the activation probe's JSONL and print it as tables.

The probe (`parallm.train.probe`, `--probe-steps`) writes one row per
(step, layer, phase, track) per rank. That is ~9k rows a step at the 32B/N=64
shape — greppable, but not readable. This turns it into the three views worth
looking at:

  A. per-layer merged profile — where along the stack each phase loses the
     teacher, and how the two phases compare at the same depth
  B. per-track spread — whether the tracks agree with each other or one is
     carrying the error
  C. step-over-step — what training actually moved

Read the merged post-mlp column FIRST. While the student's weights are still the
teacher's it must be ~0 at every layer except the last; a non-zero one means the
probe was misaligned and nothing else in the file can be trusted. (The final
layer is exempt by construction — it is the one layer whose sync lands post-MLP,
so its mixer runs on the partial residual.)

    python scripts/report_probe.py <out-dir>/probe
    python scripts/report_probe.py <out-dir>/probe --metric d_cos --step 0
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from parallm.train.probe import METRICS, PHASES


def load(probe_dir: Path) -> tuple[list[dict], dict]:
    """Every rank's rows plus the shared meta header. The files are independent
    (no gather at write time), so this concatenation IS the reassembly."""
    rows, meta = [], {}
    files = sorted(probe_dir.glob("rank*.jsonl"))
    if not files:
        raise SystemExit(f"no rank*.jsonl under {probe_dir}")
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "meta" in rec:
                meta = rec["meta"]
            else:
                rows.append(rec)
    return rows, meta


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v, spec=".4f"):
    return "     -" if v is None else format(v, spec)


def redundancy(merged: dict, tracks: list[dict]) -> float | None:
    """``coh`` divided by the value it would take if the tracks' deltas were
    exactly mutually orthogonal — i.e. how REDUNDANT the tracks are.

    ``coh`` alone cannot be read against ``1/√N``: that floor only holds when
    every track's delta has the same norm, and they do not. The honest floor is
    ``√(Σ‖δ_k‖²) / Σ‖δ_k‖`` from the per-track norms this file already carries.

      1.0  → exactly orthogonal; no track can supply what another does
      >1   → partially aligned; the tracks duplicate each other's content
      <1   → cancelling (has not been observed; treat as an artifact of combining
             means-of-norms into a Pythagorean identity, which is approximate)
    """
    d = [t["dnorm"] for t in tracks if t.get("dnorm")]
    if not d or merged.get("coh") is None:
        return None
    return merged["coh"] / (sum(v * v for v in d) ** 0.5 / sum(d))


def profile(rows: list[dict], step: int) -> None:
    """View A — the merged (all-track) row at each depth, both phases side by
    side. relMSE is what the objective optimizes; cos and r say what it is made
    of (relMSE ≈ 1 − 2·cos·r + r²), and a magnitude deficit is a gain the next
    layer can absorb while a direction error is missing content."""
    m = {(r["layer"], r["phase"]): r for r in rows
         if r["step"] == step and r["track"] == -1}
    per = defaultdict(list)
    for r in rows:
        if r["step"] == step and r["track"] != -1:
            per[(r["layer"], r["phase"])].append(r)
    layers = sorted({k[0] for k in m})
    head = f"{'relMSE':>10}  {'cos':>7}  {'r':>6}  {'coh':>6}  {'redun':>5}"
    w = len(head)
    print(f"\n===== A. merged profile @ step {step} =====")
    print(f"  {'':5} | {'post-attn'.center(w)} | {'post-mlp'.center(w)}")
    print(f"  layer | {head} | {head}")
    for i in layers:
        cells = []
        for ph in PHASES:
            r = m.get((i, ph))
            if r is None:
                cells.append(" " * w)
                continue
            cells.append(f"{_fmt(r['res_relmse'], '.3e'):>10}  {_fmt(r['res_cos'], '+.4f'):>7}  "
                         f"{_fmt(r['res_nr']):>6}  {_fmt(r.get('coh')):>6}  "
                         f"{_fmt(redundancy(r, per[(i, ph)]), '.2f'):>5}")
        print(f"  {i:5d} | {cells[0]} | {cells[1]}")
    print("  redun = coh / its exactly-orthogonal value; 1.00 = orthogonal tracks, "
          ">1 = redundant")


def spread(rows: list[dict], step: int, metric: str) -> None:
    """View B — the per-track distribution of one metric at each depth.

    ``d_nr`` is deliberately uninformative here: one track carries ~1/N of the
    teacher's update by construction, so it reports the track count, not a
    defect. ``d_cos`` is the per-track signal — whether this track's own
    contribution points along the true update.

    ⚠ Tracks with ``dnorm == 0`` are EXCLUDED and counted in ``n`` instead. They
    are not near-orthogonal contributors, they are absent ones: `slicer.base
    .align_chunk` rounds a per-track slab up to a multiple of 64 for GEMM
    throughput, and when that overshoots, the trailing tracks get slabs that are
    pure zero padding. At Qwen3-32B/N=64 the MLP's 25600 lanes go out 448 at a
    time, so SIX tracks hold no MLP at all — folding their cos of 0 into the min
    would report "some track points nowhere" every single layer.
    """
    buckets = defaultdict(list)
    absent = defaultdict(int)
    for r in rows:
        if r["step"] != step or r["track"] == -1:
            continue
        if r.get("dnorm") == 0.0:
            absent[(r["layer"], r["phase"])] += 1
            continue
        buckets[(r["layer"], r["phase"])].append(r[metric])
    layers = sorted({k[0] for k in list(buckets) + list(absent)})
    head = f"{'n':>3}  {'min':>7}  {'mean':>7}  {'max':>7}"
    w = len(head)
    print(f"\n===== B. per-track {metric} @ step {step} "
          f"(over CONTRIBUTING tracks; n = how many) =====")
    print(f"  {'':5} | {'post-attn'.center(w)} | {'post-mlp'.center(w)}")
    print(f"  layer | {head} | {head}")
    for i in layers:
        cells = []
        for ph in PHASES:
            vals = [v for v in buckets.get((i, ph), []) if v is not None]
            if not vals:
                cells.append(" " * w)
                continue
            cells.append(f"{len(vals):>3}  {min(vals):+.4f}  "
                         f"{sum(vals) / len(vals):+.4f}  {max(vals):+.4f}")
        print(f"  {i:5d} | {cells[0]} | {cells[1]}")
    dropped = max(absent.values(), default=0)
    if dropped:
        print(f"  ({dropped} track(s) excluded per row: zero delta — pure "
              f"align_chunk zero padding, no MLP slab)")


def over_steps(rows: list[dict], metric: str) -> None:
    """View C — did training move it? Averaged over layers, per phase.

    ⚠ A move here is a description, not a lever. Per the signal-gate rule a probe
    metric only becomes something to optimize after it has ranked two KNOWN
    artifacts with the lever toggled at scoring time.
    """
    steps = sorted({r["step"] for r in rows})
    if len(steps) < 2:
        return
    print(f"\n===== C. {metric} over steps (mean over layers, merged row) =====")
    print("   step |   post-attn    post-mlp")
    for s in steps:
        cells = [
            _mean([r[metric] for r in rows
                   if r["step"] == s and r["track"] == -1 and r["phase"] == ph])
            for ph in PHASES
        ]
        print(f"  {s:5d} | {_fmt(cells[0], '.4e'):>11}  {_fmt(cells[1], '.4e'):>11}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("probe_dir", type=Path)
    p.add_argument("--step", type=int, default=None,
                   help="which probe step to profile (default: the last one)")
    p.add_argument("--metric", default="d_cos", choices=list(METRICS),
                   help="metric for the per-track spread and the step table")
    args = p.parse_args()

    rows, meta = load(args.probe_dir)
    steps = sorted({r["step"] for r in rows})
    step = args.step if args.step is not None else steps[-1]
    if step not in steps:
        raise SystemExit(f"step {step} not in {steps}")

    tracks = sorted({r["track"] for r in rows if r["track"] != -1})
    print(f"probe: {len(rows)} rows, steps {steps}, {len(tracks)} tracks, "
          f"{meta.get('num_layers')} layers, "
          f"{meta.get('shards_per_stream')} shard(s) per track")

    profile(rows, step)
    spread(rows, step, args.metric)
    over_steps(rows, args.metric)
    over_steps(rows, "res_relmse")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
