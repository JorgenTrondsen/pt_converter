"""Calibration / fine-tune data loader.

Perplexity-recovery distillation matches a frozen teacher via KL / CE / block-MSE,
so the student only learns to match the teacher on inputs it *actually sees*. To
recover quality across the teacher's code/math-heavy distribution — not just
encyclopedic English — the default is a streamable **mixture**, weighted-interleaved
on the fly.

Mixtures are JSON under ``configs/data``, not code: pointing a run at different hub
datasets is ``--data-preset <name-or-path>``, no edit here. See ``preset_sources``
for the schema.

All sources stream (``streaming=True``): nominal dataset size is irrelevant — only
the tokens actually consumed are fetched and then discarded, nothing is downloaded
in full (a 4k-step run at seq=4096 touches ~tens of millions of tokens regardless
of whether the source nominally holds 600M or 600B).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

# Mixture JSONs live at the repo root, resolved off THIS file rather than cwd:
# ranks launch via torchrun from varying working directories.
MIXTURE_DIR = Path(__file__).resolve().parents[3] / "configs" / "data"

DEFAULT_MIXTURE = "cascade2"

# Documents sampled per source to price its mean length (see `token_weights`).
# 64 is enough to separate sources that differ by the 50x that actually matters
# here, and is paid once at startup.
DOC_PROBE_DOCS = 64


@dataclass
class DataSourceSpec:
    """One streaming source in a (possibly interleaved) mixture.

    ``weight`` is this source's share of training **tokens** (normalized across
    the mixture). It is NOT the interleave probability — see ``token_weights``.

    ``format`` selects how a row becomes text:
      * ``"text"``     — take ``text_key`` verbatim.
      * ``"template"`` — ``template.format(**row)``, e.g. ``"{problem}\\n\\n{solution}"``.
      * ``"chat"``     — run ``field`` (default ``messages``) through the tokenizer's
        chat template. This is what lets a ``messages``-format SFT corpus be trained on.

    ``shuffle_buffer`` (0 = off, the default) reservoir-shuffles this source and
    randomizes its shard order. **It is off by default because turning it on changes
    which documents a run trains on, and every mixture recorded so far read its
    sources in file order** — enabling it on an existing preset voids comparability
    with the runs already in ``logs/``. Turn it on for a NEW preset whose source is
    ordered: a run consumes only ``steps * batch * seq_len`` tokens (~2M, i.e. ~100
    documents of a long-form SFT corpus), so a corpus grouped by sub-source or
    subset would otherwise contribute only its first group.

    Sources must be parquet-native (standard-format) datasets — modern ``datasets``
    no longer supports script-based loaders (e.g. ``codeparrot/github-code-clean``);
    use parquet mirrors instead.
    """

    dataset_name: str
    dataset_config: str | None = None
    split: str = "train"
    text_key: str = "text"
    weight: float = 1.0
    format: str = "text"
    template: str | None = None
    field: str | None = None
    shuffle_buffer: int = 0

    def label(self) -> str:
        return f"{self.dataset_name}:{self.dataset_config}" if self.dataset_config else self.dataset_name


def _valid_messages(messages) -> bool:
    """Structural check on a chat row: a list of ``{role: str, content: str}``.

    Deliberately structural, not truthy. Cascade-2 opens most rows with
    ``{"role": "system", "content": ""}``, and rejecting empty content would
    silently drop whole subsets — a source that renders to nothing looks exactly
    like a source that is merely down-weighted.
    """
    if not isinstance(messages, (list, tuple)) or not messages:
        return False
    return all(
        isinstance(m, dict)
        and isinstance(m.get("role"), str) and m["role"]
        and isinstance(m.get("content"), str)
        for m in messages
    )


def render_fn(spec: DataSourceSpec, tokenizer):
    """Build the ``row -> {"text": str}`` mapper for one source, per ``spec.format``.

    Rows that cannot be rendered yield ``""`` and are dropped by the packer's
    empty-text check rather than raising, so one malformed row does not kill a run.
    """
    if spec.format == "text":
        key = spec.text_key
        return lambda row: {"text": row.get(key) or ""}

    if spec.format == "template":
        if not spec.template:
            raise ValueError(f"format='template' needs a template: {spec.label()}")
        tmpl = spec.template
        return lambda row: {"text": tmpl.format(**row)}

    if spec.format == "chat":
        key = spec.field or "messages"

        def render(row):
            messages = row.get(key)
            if not _valid_messages(messages):
                return {"text": ""}
            return {"text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )}

        return render

    raise ValueError(f"unknown source format {spec.format!r} (text|template|chat)")


def _load_rendered(spec: DataSourceSpec, tokenizer, seed: int = 42):
    """Stream one source and normalize it to a single ``"text"`` column.

    Dropping the original columns is not cosmetic: ``interleave_datasets`` requires
    a shared feature schema, so this is what lets a chat corpus and a plain-text
    corpus sit in one mixture.

    ``seed`` only matters when the spec opts into ``shuffle_buffer``; it is the
    config seed so every rank draws the identical stream.
    """
    from datasets import load_dataset  # local: avoid hard dep at module load

    ds = load_dataset(
        spec.dataset_name, spec.dataset_config, split=spec.split, streaming=True,
    )
    if spec.shuffle_buffer:
        # Shuffles shard ORDER as well as buffering rows, which is the half that
        # matters for a corpus grouped by sub-source across shards.
        ds = ds.shuffle(seed=seed, buffer_size=spec.shuffle_buffer)
    return ds.map(render_fn(spec, tokenizer), remove_columns=list(ds.column_names or []))


def mean_doc_tokens(spec: DataSourceSpec, tokenizer, stream=None) -> float:
    """Mean tokens per *rendered* document, over the first ``DOC_PROBE_DOCS`` docs.

    ``stream`` is an ALREADY-RENDERED (``{"text": ...}``-keyed) source; omit it and
    the spec is loaded and rendered here. Either way it must be re-iterable (a hub
    ``IterableDataset`` or a list): this probe walks it from the start and the
    training stream then walks it again, so a one-shot generator would arrive at
    the packer already drained.
    """
    if stream is None:
        stream = _load_rendered(spec, tokenizer)
    total, n = 0, 0
    for row in stream:
        text = row.get("text", "")
        if not text:
            continue
        total += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        n += 1
        if n >= DOC_PROBE_DOCS:
            break
    if not n:
        raise ValueError(f"{spec.label()} rendered no non-empty documents in "
                         f"{DOC_PROBE_DOCS} rows — check `format`/`field`/`text_key`")
    return total / n


def token_weights(sources: list[DataSourceSpec], lengths: list[float]) -> list[float]:
    """Token shares -> interleave (per-document) probabilities.

    ``interleave_datasets`` draws one **document** at a time, so a source's token
    share is its document share times its mean document length. Dividing out the
    length is what makes a declared 30% actually arrive as 30% of tokens; without
    it, the longest source takes almost everything (Cascade-2 spans 922 to 48,340
    tokens/doc, a 52x spread).
    """
    raw = [s.weight / max(n, 1e-9) for s, n in zip(sources, lengths)]
    total = sum(raw)
    if total <= 0:
        raise ValueError("mixture weights sum to zero")
    return [r / total for r in raw]


def _log_mixture(sources, lengths, probs) -> None:
    """Report the REALIZED split, which is the only one that matters.

    Printed because nominal weights and realized weights came apart badly once and
    nothing in the run surfaced it: a nominal 34/33/33 trained as 66/32/2.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    tok_total = sum(p * n for p, n in zip(probs, lengths))
    for s, n, p in zip(sources, lengths, probs):
        share = p * n / tok_total if tok_total else 0.0
        print(f"[data] {s.label()}: {share:.1%} of tokens = {p:.1%} of documents "
              f"@ {n:,.0f} tok/doc", flush=True)


def _mixture_path(name: str) -> Path:
    """Resolve a mixture name (under ``configs/data``) or a path to a JSON file."""
    cand = Path(name)
    if cand.suffix == ".json" or cand.is_file():
        return cand
    return MIXTURE_DIR / f"{name}.json"


def preset_names() -> list[str]:
    """Names of the mixtures shipped in ``configs/data`` (for help strings)."""
    return sorted(p.stem for p in MIXTURE_DIR.glob("*.json"))


def preset_sources(name: str) -> list[DataSourceSpec]:
    """Load a mixture by name (``configs/data/<name>.json``) or by path.

    JSON shape: ``{"sources": [{"dataset": ..., "weight": ..., "format": ...}, ...]}``.
    ``dataset`` is the HuggingFace hub id and ``config`` its subset name; every other
    key mirrors a ``DataSourceSpec`` field and is optional. Adding a dataset is a new
    JSON file, never an edit to this module.

    For a one-off plain-text source, prefer ``--data-source NAME[:CONFIG[:KEY[:WEIGHT]]]``
    (repeatable) over writing a JSON — only chat/template formats need a mixture file.
    """
    path = _mixture_path(name)
    if not path.is_file():
        raise KeyError(f"unknown data mixture {name!r} ({path}); have {preset_names()}")
    out = []
    for src in json.loads(path.read_text())["sources"]:
        src = dict(src)
        out.append(DataSourceSpec(
            dataset_name=src.pop("dataset"),
            dataset_config=src.pop("config", None),
            **src,
        ))
    return out


def parse_source_spec(spec: str) -> DataSourceSpec:
    """Parse a CLI ``NAME[:CONFIG[:TEXT_KEY[:WEIGHT]]]`` source string.

    Empty CONFIG / TEXT_KEY fields fall back to defaults (None / "text"), so
    ``name::code:0.2`` sets text_key + weight while leaving config unset. Plain-text
    sources only — chat/template formats need a mixture JSON.
    """
    parts = spec.split(":")
    if not parts[0]:
        raise ValueError(f"empty dataset name in --data-source spec: {spec!r}")
    name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    text_key = parts[2] if len(parts) > 2 and parts[2] else "text"
    weight = float(parts[3]) if len(parts) > 3 and parts[3] else 1.0
    return DataSourceSpec(name, config, text_key=text_key, weight=weight)


@dataclass
class CalibrationDataConfig:
    sources: list[DataSourceSpec] = field(
        default_factory=lambda: preset_sources(DEFAULT_MIXTURE)
    )
    seq_len: int = 4096
    # Fixed interleave seed: the loader is consumed with num_workers=0 and NO
    # DistributedSampler, so every rank must read the identical stream (under
    # vocab-parallel every rank backwards the same batch and the SyncBoundary
    # all-reduce assumes identical inputs). Keep this equal across ranks.
    seed: int = 42
    # Held-out boundary: discard the first ``skip_docs`` raw documents of the
    # (interleaved) stream before packing. Used to carve a disjoint val set out of
    # the SAME mixture: the train stream sets ``skip_docs=N`` while a mirror val set
    # reads the front (``skip_docs=0``) of the identical seeded sequence, so the two
    # cover non-overlapping document ranges. 0 = read from the start (legacy).
    skip_docs: int = 0

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "CalibrationDataConfig":
        return cls(sources=preset_sources(name), **kwargs)

    @classmethod
    def single(
        cls,
        dataset_name: str = "Salesforce/wikitext",
        dataset_config: str | None = "wikitext-103-raw-v1",
        split: str = "train",
        text_key: str = "text",
        **kwargs,
    ) -> "CalibrationDataConfig":
        """One-source config (e.g. the held-out validation set)."""
        return cls(
            sources=[DataSourceSpec(dataset_name, dataset_config, split, text_key)],
            **kwargs,
        )


def _interleave(streams: list, weights: list[float], seed: int):
    """Single stream → itself; multiple → seeded weighted interleave.

    ``stopping_strategy="all_exhausted"`` keeps the combined stream alive until
    every source is exhausted (the huge streaming sources never exhaust within a
    run); the seed makes the source-choice sequence deterministic across ranks.
    """
    if len(streams) == 1:
        return streams[0]
    from datasets import interleave_datasets  # local: avoid hard dep at import

    total = sum(weights)
    probs = [w / total for w in weights]
    return interleave_datasets(
        streams, probabilities=probs, seed=seed, stopping_strategy="all_exhausted"
    )


class PackedTokenStream(IterableDataset):
    """Streams one or more HF datasets, tokenizes, and packs into fixed-length sequences.

    Multiple sources are weighted-interleaved with the config's fixed seed so the
    stream is identical on every rank — do NOT add per-node sharding here (see the
    note on ``CalibrationDataConfig.seed``).

    ``_streams`` is a test seam: pre-built (already "text"-keyed) iterables to use
    instead of ``load_dataset``, so the packing / interleave logic can be exercised
    without network. They must be re-iterable — the length probe walks them first.
    """

    def __init__(self, tokenizer, cfg: CalibrationDataConfig, *, _streams: list | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg

        if _streams is not None:
            streams = list(_streams)
            sources = cfg.sources
        else:
            sources = cfg.sources
            streams = [_load_rendered(s, tokenizer, cfg.seed) for s in sources]

        # Price each source in tokens before interleaving. Skipped for a single
        # source, where the weight is unused anyway.
        if len(streams) > 1:
            lengths = []
            for s, ds in zip(sources, streams):
                lengths.append(mean_doc_tokens(s, tokenizer, stream=ds))
            weights = token_weights(sources, lengths)
            _log_mixture(sources, lengths, weights)
        else:
            weights = [1.0] * len(streams)

        self.ds = _interleave(streams, weights, cfg.seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buf: list[int] = []
        seq_len = self.cfg.seq_len
        # Skip the held-out prefix: counted on EVERY raw example (before the
        # empty-text check below) so the boundary is content-independent and
        # identical across ranks — the train stream (skip_docs=N) and a mirror
        # val stream (skip_docs=0) then read disjoint document ranges of the same
        # seeded sequence. The train loader is iterated once per run, so this is a
        # one-time advance; a mirror val (skip_docs=0) pays nothing.
        skip_remaining = self.cfg.skip_docs
        for example in self.ds:
            if skip_remaining > 0:
                skip_remaining -= 1
                continue
            text = example.get("text", "")
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            while len(buf) >= seq_len + 1:
                chunk = buf[: seq_len + 1]
                buf = buf[seq_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                # labels aligned with input_ids (HF convention): labels[i] == input_ids[i].
                # The fidelity NLL consumer does the next-token shift internally, so labels
                # MUST NOT be pre-shifted here — pre-shifting double-shifts and scores each
                # prediction against token p+2 (near-random ppl for any model).
                labels = input_ids.clone()
                yield {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones(seq_len, dtype=torch.long)}
