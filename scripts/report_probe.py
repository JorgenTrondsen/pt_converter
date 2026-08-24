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

⚠ **The two columns SWAP ROLES with the run's `sync_phase`** (printed in the header,
and recorded in the JSONL meta). Each phase has one column fed the teacher-exact
residual — the RAIL — and one fed OWN-CARRY — the SEAM, the sync that phase drops:

    phase      | rail column | seam column
    -----------|-------------|------------
    post-attn  | post-mlp    | post-attn
    post-mlp   | post-attn   | post-mlp

Read the RAIL column FIRST. While the student's weights are still the teacher's it
must be ~0 at every layer; a non-zero one means the probe was misaligned and nothing
else in the file can be trusted. (At post-attn the last layer is exempt by
construction — it is the one layer whose sync lands post-MLP, so its mixer runs on
the partial residual.)

The SEAM column is structurally non-zero even with identical weights — that is the
measurement, not a defect. Its value at shallow depths is dominated by the tiny
early residual norm in the denominator (post-mlp reads 2.43 at L0 untrained), so
compare it at depth. Across phases, compare SEAM to SEAM — post-attn's post-attn
column against post-mlp's post-mlp column — never column to column by name.

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
                # Files written before the free-running walk landed carry no `walk`
                # key, and every row in them is teacher-forced.
                rec.setdefault("walk", "tf")
                rows.append(rec)
    return rows, meta


def tf(rows: list[dict]) -> list[dict]:
    """The teacher-forced AGGREGATE rows — the only ones with a per-track dimension.

    Detail rows also carry ``walk`` and ``track == -1`` but hold LISTS instead of the
    METRICS columns, so every view that reads a metric by name must exclude them or
    it will silently overwrite the real row for that (layer, phase).
    """
    return [r for r in rows if r["walk"] == "tf" and "detail" not in r]


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
    rows = tf(rows)
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
    for r in tf(rows):
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


def amplification(rows: list[dict], step: int, sync_phase: str) -> None:
    """View D — the free-running seam, and how much each layer AMPLIFIES drift.

    The teacher-forced rows above hand every layer the teacher's residual, so each
    is that layer's error given a PERFECT input. The free-running row is the same
    layer scored on the student's own walk, where the error handed to it is whatever
    the previous layers accumulated. Their ratio is that layer's amplification:

        amp = FR / TF   — >1 means this depth inflates the drift it is given

    This is the per-layer map the SCHEDULE levers want. A layer with a small TF error
    and a large amp is one where a sync buys a lot; a large TF error with amp ≈ 1 is
    a layer that is simply mis-fit and that training, not placement, should fix.

    Only the SEAM phase is shown — the other column is the rail (its input is
    teacher-exact by construction, so it has no free-running counterpart worth a
    ratio).
    """
    seam = "mlp" if sync_phase == "post-mlp" else "attn"
    by = {(r["walk"], r["layer"]): r for r in rows
          if r["step"] == step and r["track"] == -1 and r["phase"] == seam
          and "detail" not in r}
    layers = sorted(i for (w, i) in by if w == "fr")
    if not layers:
        print(f"\n===== D. amplification @ step {step}: no free-running rows "
              f"(run predates `record_fr`) =====")
        return
    print(f"\n===== D. post-{seam} SEAM: teacher-forced vs FREE-RUNNING @ step {step} =====")
    print(f"  layer | {'TF relMSE':>10}  {'FR relMSE':>10}  {'amp':>8}  {'FR cos':>8}")
    amps = []
    for i in layers:
        f = by[("fr", i)]
        t = by.get(("tf", i))
        a = (f["res_relmse"] / t["res_relmse"]
             if t and t.get("res_relmse") and f.get("res_relmse") is not None else None)
        if a is not None:
            amps.append(a)
        print(f"  {i:5d} | {_fmt(t and t.get('res_relmse'), '.3e'):>10}  "
              f"{_fmt(f.get('res_relmse'), '.3e'):>10}  "
              f"{_fmt(a, '.1f'):>8}  {_fmt(f.get('res_cos'), '+.4f'):>8}")
    if amps:
        amps.sort()
        print(f"  median amp {amps[len(amps) // 2]:.1f}x   max {amps[-1]:.1f}x   "
              f"(1.0 = this layer passes drift through without inflating it)")


def detail(rows: list[dict], step: int, meta: dict, tokenizer=None, top: int = 12) -> None:
    """View E — WHERE the error is, on the axes every other view averages away.

    `res_relmse` is `Σ_p err_p / Σ_p den_p`, a norm-WEIGHTED quantity, while `cos`
    and `nr` are unweighted per-token means. So a handful of high-‖t‖ positions can
    move relMSE by an order of magnitude while leaving cos and nr flat — which is
    what a concentration index `relMSE / (1 − 2·cos·nr + nr²)` ≫ 1 detects, and what
    this view localizes. `--probe-detail` populates it.

    Read the cumulative share first. A few positions holding most of Σerr is the
    massive-activation signature; an even spread means the concentration reading was
    wrong and no amount of per-channel detail will rescue it.
    """
    det = [r for r in rows if r.get("detail") and r["step"] == step]
    if not det:
        return
    ids = meta.get("input_ids") or []
    pos = {(r["layer"], r["phase"], r["walk"]): r for r in det if r["detail"] == "position"}
    chan = {(r["layer"], r["phase"], r["walk"]): r for r in det if r["detail"] == "channel"}
    print(f"\n===== E. per-position / per-channel detail @ step {step} =====")
    for key in sorted(pos):
        layer, phase, walk = key
        d = pos[key]
        err, den = d["err"], d["den"]
        tot = sum(err) or 1.0
        order = sorted(range(len(err)), key=lambda p: -err[p])
        cum = lambda n: sum(err[p] for p in order[:n]) / tot
        print(f"\n  L{layer} post-{phase} [{walk}]  relMSE {tot / (sum(den) or 1.0):.4e}"
              f"   top-1 {cum(1):.1%} | top-8 {cum(8):.1%} | top-32 {cum(32):.1%}"
              f"  of {len(err)} positions")
        print(f"    {'pos':>6} {'err share':>10} {'‖t_p‖':>10} {'cos_p':>9} {'nr_p':>8}  token")
        for p in order[:top]:
            tok = ""
            if p < len(ids):
                tok = f"{ids[p]}"
                if tokenizer is not None:
                    tok += f"  {tokenizer.convert_ids_to_tokens([ids[p]])[0]!r}"
            print(f"    {p:>6} {err[p] / tot:>10.1%} {den[p] ** 0.5:>10.3f} "
                  f"{d['cos'][p]:>+9.4f} {d['nr'][p]:>8.4f}  {tok}")
        c = chan.get(key)
        if c:
            ctot = c["chan_err_total"] or 1.0
            share = [f"{i}:{e / ctot:.1%}(|t|{m:.1f})"
                     for i, e, m in list(zip(c["chan_idx"], c["chan_err"], c["chan_tmax"]))[:8]]
            print(f"    top channels: " + "  ".join(share))
            print(f"      top-32 channels hold {sum(c['chan_err']) / ctot:.1%} of the error")


def over_steps(rows: list[dict], metric: str) -> None:
    """View C — did training move it? Averaged over layers, per phase.

    ⚠ A move here is a description, not a lever. Per the signal-gate rule a probe
    metric only becomes something to optimize after it has ranked two KNOWN
    artifacts with the lever toggled at scoring time.
    """
    rows = tf(rows)
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
    p.add_argument("--hf-model", default=None,
                   help="tokenizer source, so a detail row's position decodes to a token")
    args = p.parse_args()

    rows, meta = load(args.probe_dir)
    steps = sorted({r["step"] for r in rows})
    step = args.step if args.step is not None else steps[-1]
    if step not in steps:
        raise SystemExit(f"step {step} not in {steps}")

    tracks = sorted({r["track"] for r in rows if r["track"] != -1})
    # Runs written before the phase was recorded are post-attn: it was the only
    # phase the trainer would accept.
    phase = meta.get("sync_phase", "post-attn")
    print(f"probe: {len(rows)} rows, steps {steps}, {len(tracks)} tracks, "
          f"{meta.get('num_layers')} layers, "
          f"{meta.get('shards_per_stream')} shard(s) per track, "
          f"sync_phase={phase}")
    rail, seam = ("attn", "mlp") if phase == "post-mlp" else ("mlp", "attn")
    print(f"  rail column = post-{rail} (~0 at every layer while untrained); "
          f"seam column = post-{seam} (the dropped sync — compare seam to seam)")

    tok = None
    if args.hf_model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf_model)

    profile(rows, step)
    amplification(rows, step, phase)
    detail(rows, step, meta, tok)
    spread(rows, step, args.metric)
    over_steps(rows, args.metric)
    over_steps(rows, "res_relmse")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
