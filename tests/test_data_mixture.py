"""Unit tests for the streaming data mixture (train/data.py).

Network-free: the packing tests inject plain in-memory iterables via the
``_streams`` test seam; only the interleave-determinism test needs the optional
``datasets`` package (skipped if absent).
"""
from __future__ import annotations

import pytest
import torch

import json

from parallm.train.data import (
    DEFAULT_MIXTURE,
    MIXTURE_DIR,
    CalibrationDataConfig,
    DataSourceSpec,
    PackedTokenStream,
    mean_doc_tokens,
    parse_source_spec,
    preset_names,
    preset_sources,
    render_fn,
    token_weights,
)


class _CharTok:
    """Deterministic stand-in tokenizer: one id per character."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages)


# ----- mixtures are JSON on disk, not code -----

def test_shipped_mixtures_and_default():
    # Only mixtures needing a chat/template render are shipped as files; a plain-text
    # source is a --data-source flag, not a JSON.
    assert DEFAULT_MIXTURE == "cascade2"
    assert DEFAULT_MIXTURE in preset_names()


def test_every_shipped_mixture_loads():
    for name in preset_names():
        assert preset_sources(name), name


def test_default_mixture_is_cascade2():
    srcs = preset_sources("cascade2")
    assert {s.dataset_config for s in srcs} == {
        "math", "science", "swe", "chat", "instruction_following"}
    assert all(s.dataset_name == "nvidia/Nemotron-Cascade-2-SFT-Data" for s in srcs)
    assert all(s.format == "chat" and s.field == "messages" for s in srcs)
    assert pytest.approx(sum(s.weight for s in srcs), rel=1e-6) == 1.0


def test_calib_default_is_still_wikitext():
    # replica_build.get_calib_batches builds its calibration batches from this,
    # and does NOT want the huge chat mixture.
    (src,) = CalibrationDataConfig.single().sources
    assert src.dataset_name == "Salesforce/wikitext"
    assert src.dataset_config == "wikitext-103-raw-v1"
    assert src.format == "text"


def test_mixture_loads_from_an_arbitrary_path(tmp_path):
    """A path, not just a name — so a one-off mixture needs no file in the repo."""
    p = tmp_path / "custom.json"
    p.write_text(json.dumps({"sources": [
        {"dataset": "some/ds", "config": "sub", "weight": 0.4, "text_key": "body"},
    ]}))
    (src,) = preset_sources(str(p))
    assert (src.dataset_name, src.dataset_config, src.text_key) == ("some/ds", "sub", "body")
    assert src.weight == pytest.approx(0.4)


def test_preset_sources_returns_independent_copies():
    a = preset_sources("cascade2")
    a[0].weight = 999.0
    assert preset_sources("cascade2")[0].weight != 999.0


def test_unknown_mixture_raises_and_names_the_alternatives():
    with pytest.raises(KeyError) as e:
        preset_sources("does-not-exist")
    assert "cascade2" in str(e.value)


def test_mixture_dir_resolves_off_the_package_not_cwd(monkeypatch, tmp_path):
    # Ranks launch via torchrun from varying working directories.
    monkeypatch.chdir(tmp_path)
    assert MIXTURE_DIR.is_dir()
    assert preset_sources("cascade2")


# ----- rendering: what lets a chat corpus be trained on at all -----

def test_render_text_format():
    spec = DataSourceSpec("d", format="text", text_key="body")
    assert render_fn(spec, _CharTok())({"body": "hello"}) == {"text": "hello"}
    assert render_fn(spec, _CharTok())({"body": None}) == {"text": ""}


def test_render_template_format():
    spec = DataSourceSpec("d", format="template", template="{q}\n\n{a}")
    got = render_fn(spec, _CharTok())({"q": "2+2?", "a": "4"})
    assert got == {"text": "2+2?\n\n4"}


def test_render_template_without_a_template_raises():
    with pytest.raises(ValueError):
        render_fn(DataSourceSpec("d", format="template"), _CharTok())


def test_render_chat_format_runs_the_chat_template():
    spec = DataSourceSpec("d", format="chat", field="messages")
    row = {"messages": [{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "<think>x</think>yo"}]}
    assert render_fn(spec, _CharTok())(row) == {"text": "<user>hi<assistant><think>x</think>yo"}


def test_chat_rows_with_an_empty_system_message_are_kept():
    """Cascade-2 opens most rows with {"role": "system", "content": ""}. A truthiness
    check on content would silently drop whole subsets."""
    spec = DataSourceSpec("d", format="chat")
    row = {"messages": [{"role": "system", "content": ""},
                        {"role": "user", "content": "hi"}]}
    assert render_fn(spec, _CharTok())(row)["text"] == "<system><user>hi"


@pytest.mark.parametrize("messages", [
    None, [], "not a list",
    [{"role": "user"}],                        # no content key
    [{"role": "", "content": "x"}],            # empty role
    [{"role": "user", "content": None}],       # content not a str
])
def test_malformed_chat_rows_render_empty_rather_than_raising(messages):
    # One bad row must not kill a run; the packer drops empty text.
    spec = DataSourceSpec("d", format="chat")
    assert render_fn(spec, _CharTok())({"messages": messages}) == {"text": ""}


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        render_fn(DataSourceSpec("d", format="parquet"), _CharTok())


# ----- token-vs-document weighting -----

def test_mean_doc_tokens_measures_rendered_length():
    docs = [{"text": "a" * 10}, {"text": "b" * 20}, {"text": ""}]
    assert mean_doc_tokens(DataSourceSpec("d"), _CharTok(), stream=docs) == pytest.approx(15.0)


def test_mean_doc_tokens_on_an_all_empty_source_raises():
    # Otherwise a source with a wrong `field` weights in at length ~0 and swallows
    # the mixture, which is much harder to spot than a crash at startup.
    with pytest.raises(ValueError):
        mean_doc_tokens(DataSourceSpec("d"), _CharTok(), stream=[{"text": ""}])


def test_token_weights_divide_out_document_length():
    # Equal token shares over a 50x length spread must NOT be equal document shares.
    srcs = [DataSourceSpec("short", weight=0.5), DataSourceSpec("long", weight=0.5)]
    probs = token_weights(srcs, [1_000.0, 50_000.0])
    assert probs[0] == pytest.approx(50 * probs[1])
    assert sum(probs) == pytest.approx(1.0)


def test_equal_token_shares_over_a_length_spread_yield_equal_token_counts():
    """The property the weighting exists for: nominal 50/50 arrives as 50/50 of
    TOKENS, not of documents. Read as document probabilities instead, the long
    source would take 98% of the tokens."""
    lengths = [1_000.0, 50_000.0]
    probs = token_weights([DataSourceSpec("a", weight=0.5), DataSourceSpec("b", weight=0.5)], lengths)
    tokens = [p * n for p, n in zip(probs, lengths)]
    assert tokens[0] == pytest.approx(tokens[1])


def test_cascade2_weights_realize_as_the_recorded_split():
    """Reproduces the realized split logged by the raw arm of the Cascade-2 A/B
    (logs/qwen3/32b_d1b_nemo_ab.log): 30% of math's tokens is 5.2% of documents,
    20% of swe's is 1.4%, and instruction_following takes 55.3% of the draws."""
    srcs = preset_sources("cascade2")
    measured = {"math": 19724.0, "science": 3832.0, "swe": 48340.0,
                "chat": 2735.0, "instruction_following": 922.0}
    probs = token_weights(srcs, [measured[s.dataset_config] for s in srcs])
    by_config = {s.dataset_config: p for s, p in zip(srcs, probs)}
    assert by_config["math"] == pytest.approx(0.052, abs=5e-4)
    assert by_config["swe"] == pytest.approx(0.014, abs=5e-4)
    assert by_config["instruction_following"] == pytest.approx(0.553, abs=5e-4)


def test_zero_total_weight_raises():
    with pytest.raises(ValueError):
        token_weights([DataSourceSpec("a", weight=0.0)], [10.0])


# ----- --data-source spec parsing -----

@pytest.mark.parametrize(
    "spec, expected",
    [
        ("ds", DataSourceSpec("ds", None, text_key="text", weight=1.0)),
        ("ds:cfg", DataSourceSpec("ds", "cfg", text_key="text", weight=1.0)),
        ("ds:cfg:body", DataSourceSpec("ds", "cfg", text_key="body", weight=1.0)),
        ("ds:cfg:body:0.3", DataSourceSpec("ds", "cfg", text_key="body", weight=0.3)),
        # empty config / text_key fields fall back to defaults
        ("ds::code:0.2", DataSourceSpec("ds", None, text_key="code", weight=0.2)),
    ],
)
def test_parse_source_spec(spec, expected):
    got = parse_source_spec(spec)
    assert got.dataset_name == expected.dataset_name
    assert got.dataset_config == expected.dataset_config
    assert got.text_key == expected.text_key
    assert got.weight == pytest.approx(expected.weight)


def test_parse_source_spec_empty_name_raises():
    with pytest.raises(ValueError):
        parse_source_spec(":cfg")


# ----- packing -----

def test_pack_shapes_labels_and_contiguous_tiling():
    seq_len = 4
    # 3 docs → 30 chars → 30 tokens → 7 full chunks of seq_len (28 tokens used).
    docs = [{"text": "abcdefghij"}, {"text": "klmnopqrst"}, {"text": "uvwxyz0123"}]
    cfg = CalibrationDataConfig(sources=[DataSourceSpec("x")], seq_len=seq_len)
    stream = PackedTokenStream(_CharTok(), cfg, _streams=[docs])

    all_tokens = [ord(c) for d in docs for c in d["text"]]
    out = list(stream)
    assert len(out) == len(all_tokens) // seq_len  # contiguous tiling, stride == seq_len

    flat = []
    for item in out:
        assert item["input_ids"].shape == (seq_len,)
        assert item["input_ids"].dtype == torch.long
        # labels are NOT pre-shifted (consumers shift internally).
        assert torch.equal(item["labels"], item["input_ids"])
        assert torch.equal(item["attention_mask"], torch.ones(seq_len, dtype=torch.long))
        flat.extend(item["input_ids"].tolist())
    # yielded chunks reconstruct the token stream in order, no gaps/overlap.
    assert flat == all_tokens[: len(out) * seq_len]


def test_empty_documents_are_skipped():
    seq_len = 4
    docs = [{"text": ""}, {"text": "abcdefgh"}, {"text": ""}, {"text": "ijkl"}]
    cfg = CalibrationDataConfig(sources=[DataSourceSpec("x")], seq_len=seq_len)
    out = list(PackedTokenStream(_CharTok(), cfg, _streams=[docs]))
    expected = [ord(c) for c in "abcdefghijkl"]
    flat = [t for item in out for t in item["input_ids"].tolist()]
    assert flat == expected[: len(out) * seq_len]


# ----- held-out skip_docs (disjoint train/val on the same mixture) -----

def _flat(stream):
    return [t for item in stream for t in item["input_ids"].tolist()]


def test_skip_docs_holds_out_front_disjoint():
    """skip_docs=N drops the first N docs; the train (N) and val (0) streams over
    the same source cover disjoint document ranges."""
    seq_len = 4
    docs = [{"text": c * 4} for c in "abcdef"]  # 6 docs × 4 tokens
    full = _flat(PackedTokenStream(
        _CharTok(), CalibrationDataConfig(sources=[DataSourceSpec("x")], seq_len=seq_len),
        _streams=[list(docs)]))
    skipped = _flat(PackedTokenStream(
        _CharTok(), CalibrationDataConfig(sources=[DataSourceSpec("x")], seq_len=seq_len, skip_docs=2),
        _streams=[list(docs)]))

    held_out = [ord(c) for d in docs[:2] for c in d["text"]]   # first 2 docs reserved for "val"
    kept = [ord(c) for d in docs[2:] for c in d["text"]]       # what "train" sees
    # val (full, reads the front) covers the held-out region; train (skipped) does not.
    assert full[: len(held_out)] == held_out
    assert skipped == kept[: len(skipped)]
    # disjoint: no held-out token leaks into the skipped stream.
    assert not (set(skipped) & set(held_out))


def test_skip_docs_counts_empty_documents():
    """The boundary is content-independent: empty docs count toward skip_docs so the
    cut is identical regardless of which docs happen to be empty (and across ranks)."""
    seq_len = 4
    docs = [{"text": ""}, {"text": "aaaa"}, {"text": "bbbb"}, {"text": "cccc"}]
    # skip_docs=2 discards the empty doc + "aaaa", leaving "bbbb"/"cccc".
    skipped = _flat(PackedTokenStream(
        _CharTok(), CalibrationDataConfig(sources=[DataSourceSpec("x")], seq_len=seq_len, skip_docs=2),
        _streams=[list(docs)]))
    expected = [ord(c) for c in "bbbbcccc"]
    assert skipped == expected[: len(skipped)]


# ----- interleave determinism (needs `datasets`) -----

def _build_two_source_stream(seed):
    from datasets import Dataset

    a = Dataset.from_dict({"text": [chr(ord("a") + i) * 12 for i in range(20)]}).to_iterable_dataset()
    # second source uses a DIFFERENT text key to exercise rename → "text".
    b_raw = Dataset.from_dict({"body": [chr(ord("A") + i) * 12 for i in range(20)]}).to_iterable_dataset()
    b = b_raw.rename_column("body", "text").select_columns(["text"])
    cfg = CalibrationDataConfig(
        sources=[DataSourceSpec("a", weight=0.5), DataSourceSpec("b", text_key="body", weight=0.5)],
        seq_len=4,
        seed=seed,
    )
    return PackedTokenStream(_CharTok(), cfg, _streams=[a, b])


def test_interleave_is_deterministic_across_ranks():
    pytest.importorskip("datasets")
    # Two independent builds with the same seed == two ranks: identical batches.
    s1, s2 = _build_two_source_stream(123), _build_two_source_stream(123)
    out1 = [item["input_ids"] for item, _ in zip(s1, range(8))]
    out2 = [item["input_ids"] for item, _ in zip(s2, range(8))]
    assert len(out1) == 8
    for a, b in zip(out1, out2):
        assert torch.equal(a, b)


def test_interleave_seed_changes_order():
    pytest.importorskip("datasets")
    s1 = _build_two_source_stream(1)
    s2 = _build_two_source_stream(2)
    out1 = [item["input_ids"].tolist() for item, _ in zip(s1, range(8))]
    out2 = [item["input_ids"].tolist() for item, _ in zip(s2, range(8))]
    # Different seeds should give a different interleave (mixed-source content differs).
    assert out1 != out2
