"""Slice a HF checkpoint into N per-track shards — one converter for every source.

Streams the checkpoint's text weights to bf16 (NVFP4/FP8 dequantized, vision + MTP
dropped) and feeds them to the model-agnostic slicer. Works for plain-bf16
(Qwen3.5-9B), NVFP4 (Qwen3.6-27B-NVFP4), and MoE (Qwen3.6-35B-A3B) checkpoints
alike — the right adapter is resolved from ``config.model_type``.

The sync schedule is NOT chosen here (per-track weights are schedule-independent);
the train script places boundaries dynamically on the most sensitive layers.

    python scripts/convert.py --hf-model <checkpoint dir> \
        --out-dir convert_out/<name> --n-tracks 16
"""
from __future__ import annotations

import argparse
import sys

from transformers import AutoConfig

from parallm.slicer.convert import slice_model_to_tracks
from parallm.slicer.loader import StateDictModel, stream_text_state_dict
from parallm.utils.checkpoint import save_tracks
from parallm.utils.max_tracks import valid_track_counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Path to the checkpoint dir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-tracks", type=int, default=None, help="Defaults to max-tracks")
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(args.hf_model)
    tcfg = getattr(cfg, "text_config", cfg)  # VLM sub-config, else the config itself
    valid = valid_track_counts(tcfg)  # descending; valid[0] == max-tracks
    n = args.n_tracks if args.n_tracks is not None else valid[0]
    if n not in valid:
        # Check the whole set, not just the upper bound — an invalid smaller N
        # would otherwise only fail deep in the slicer, after the full stream.
        print(f"[error] n_tracks={n} invalid for this model; valid: {valid}", file=sys.stderr)
        return 2

    print(f"[info] streaming text weights to bf16 (n_tracks={n}); dropping vision + MTP…", flush=True)
    text_sd = stream_text_state_dict(args.hf_model)

    print(f"[info] slicing {len(text_sd)} text weights into {n} tracks…", flush=True)
    tracks, manifest = slice_model_to_tracks(
        StateDictModel(text_sd, tcfg), n_tracks=n, text_config_attr="config",
    )
    save_tracks(args.out_dir, tracks, manifest)
    # Say how the MLP actually landed. `align_chunk` keeps every track covered, so
    # this should always read N/N — it is here because the failure is silent
    # otherwise: zero-padded lanes are exact (silu(0)*up = 0), so a convert whose
    # MLP spans fewer tracks than N looks perfectly healthy.
    slab = manifest.per_track_param_shapes.get("layers.0.mlp.gate_proj.weight")
    if slab:
        inter = int(getattr(tcfg, "intermediate_size", 0))
        covered = min(n, -(-inter // slab[0])) if inter else n
        note = "" if covered == n else f"  ⚠ {n - covered} track(s) hold NO MLP"
        print(f"[info] MLP slab {slab[0]} x {n} tracks covers {inter} lanes "
              f"({covered}/{n} tracks carry MLP){note}")
    print(f"[ok] wrote {len(tracks)} track shards + manifest to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
