"""Shared plumbing for the replica-pool build scripts.

The per-model *selection policy* differs (bf16 downstream-anchor table vs NVFP4
per-family recon proxy), but the calibration-batch source, the per-track weight
iteration, and the calibrated candidate scoring are identical — they live here so
each build script is just its loader + policy.
"""
from __future__ import annotations

import torch
from safetensors import safe_open

from parallm.model.replica import _WANDA_KEY
from parallm.slicer.loader import stream_text_state_dict


def load_calib_text_model(hf_model: str, adapter, text_cfg):
    """Instantiate the family's full (unsliced) text model from a bf16-dequantized
    stream of the checkpoint and dispatch it across the GPUs for calibration.

    Works for every base (bf16 / NVFP4 / FP8) and family, because
    ``stream_text_state_dict`` normalizes to bf16 with the text model's own
    ``layers.*`` key namespace and ``adapter.full_text_model_cls`` gives the class.
    """
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights

    if adapter.full_text_model_cls is None:
        raise ValueError(f"adapter {adapter.model_type!r} has no full_text_model_cls for calibration")
    with init_empty_weights():
        model = adapter.full_text_model_cls(text_cfg)
    model.load_state_dict(stream_text_state_dict(hf_model, drop_lm_head=True), strict=False, assign=True)
    model.eval()
    layer_cls = type(model.layers[0]).__name__
    n = max(1, torch.cuda.device_count())
    dmap = infer_auto_device_map(
        model, max_memory={i: "34GiB" for i in range(n)}, no_split_module_classes=[layer_cls],
    )
    return dispatch_model(model, device_map=dmap)


def get_calib_batches(tokenizer, n_batches: int, seq_len: int):
    """Real-text calibration batches from the wikitext preset; random-token
    fallback if the dataset can't be reached (flow still runs, norms weaker)."""
    try:
        from torch.utils.data import DataLoader

        from parallm.train.data import CalibrationDataConfig, PackedTokenStream
        cfg = CalibrationDataConfig.single(seq_len=seq_len, seed=0)  # defaults to wikitext-103
        loader = DataLoader(PackedTokenStream(tokenizer, cfg), batch_size=1)
        out = []
        for b in loader:
            ids = b["input_ids"]
            out.append({"input_ids": ids if ids.ndim == 2 else ids.unsqueeze(0),
                        "attention_mask": b.get("attention_mask")})
            if len(out) >= n_batches:
                break
        if out:
            print(f"[calib] {len(out)} wikitext batches × {seq_len} tokens", flush=True)
            return out
    except Exception as e:  # noqa: BLE001
        print(f"[calib] wikitext unavailable ({e}); falling back to random tokens", flush=True)
    V = tokenizer.vocab_size if tokenizer is not None else 150000
    g = torch.Generator().manual_seed(0)
    return [{"input_ids": torch.randint(0, V, (1, seq_len), generator=g),
             "attention_mask": None} for _ in range(n_batches)]


def iter_track_linears(shard_path: str):
    """Yield ``(layer_idx, rel, weight)`` for every 2-D decoder Linear in a track
    shard whose input space Wanda calibrates (``rel in _WANDA_KEY``)."""
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if not (key.startswith("layers.") and key.endswith(".weight")):
                continue
            parts = key.split(".")
            rel = ".".join(parts[2:-1])
            if rel not in _WANDA_KEY:
                continue
            w = f.get_tensor(key)
            if w.ndim == 2:  # skip conv / norm weights that slip through
                yield int(parts[1]), rel, w


def iter_track_experts(shard_path: str):
    """Yield ``(layer_idx, name, slab)`` for the fused 3-D MoE expert Parameters
    (``mlp.experts.gate_up_proj`` / ``.down_proj``). Empty for a dense shard."""
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith("layers.") or not key.endswith(
                (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
            ):
                continue
            parts = key.split(".")
            w = f.get_tensor(key)
            if w.ndim == 3:
                yield int(parts[1]), ".".join(parts[2:]), w
