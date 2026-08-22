"""The repo's custom lm-eval tasks (configs/eval_tasks) actually register and are
shaped for PTLM.

Guards the failure that is otherwise only visible mid-run: a task named in
DEFAULT_TASKS that lm-eval cannot resolve, or one whose output_type PTLM does not
implement. Both surface at the FIRST eval — after the model is built and warmed.

The doc-formatter tests need nothing but dicts; the registration test needs the
optional [eval] extra.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from parallm.eval.downstream import DEFAULT_TASKS, EVAL_TASK_PATH

TASK_DIR = Path(EVAL_TASK_PATH)


def _load_utils():
    spec = importlib.util.spec_from_file_location("eval_task_utils", TASK_DIR / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


utils = _load_utils()


class _TaskLoader(yaml.SafeLoader):
    """SafeLoader that tolerates lm-eval's ``!function utils.foo`` tag.

    lm-eval resolves it by importing from the task's own directory; here we only
    need the surrounding keys, so keep it as the plain string.
    """


_TaskLoader.add_constructor("!function", lambda loader, node: node.value)


def _yamls():
    return sorted(TASK_DIR.glob("*.yaml"))


def _read(path) -> dict:
    return yaml.load(path.read_text(), Loader=_TaskLoader)


# ----- the YAMLs -----

def test_task_dir_resolves_off_the_package(monkeypatch, tmp_path):
    # Ranks launch via torchrun from varying working directories.
    monkeypatch.chdir(tmp_path)
    assert TASK_DIR.is_dir()
    assert _yamls()


def test_shipped_yamls_cover_the_non_builtin_default_tasks():
    shipped = {_read(p)["task"] for p in _yamls()}
    assert {"mmlu_math_mc", "codemmlu_fim"} <= shipped
    # arc_* are lm-eval built-ins and deliberately NOT redefined here.
    assert shipped.isdisjoint({"arc_easy", "arc_challenge"})


@pytest.mark.parametrize("path", _yamls(), ids=lambda p: p.stem)
def test_every_task_is_loglikelihood_shaped(path):
    """PTLM does not implement generate_until — a generative task would fail at the
    first request, which is why the built-in mmlu_pro_math is not usable here."""
    cfg = _read(path)
    assert cfg["output_type"] in {"multiple_choice", "loglikelihood"}
    assert [m["metric"] for m in cfg["metric_list"]] == ["acc"]
    assert path.stem == cfg["task"]  # include_path resolves by filename


def test_math_tasks_are_two_shot_and_code_task_is_zero_shot():
    # Pinned in the YAMLs, not on the command line: a YAML-level num_fewshot cannot
    # be overridden by --num-fewshot, and codemmlu shots would be whole code files.
    cfgs = {_read(p)["task"]: _read(p) for p in _yamls()}
    assert cfgs["mmlu_math_mc"]["num_fewshot"] == 2
    assert cfgs["mmlu_cs_mc"]["num_fewshot"] == 2
    assert cfgs["mmlu_pro_math_mc"]["num_fewshot"] == 2
    assert cfgs["codemmlu_fim"]["num_fewshot"] == 0


def test_mmlu_cs_differs_from_mmlu_math_in_the_SUBJECT_AND_NOTHING_ELSE():
    """The control is only a control if every other knob matches. A stray difference
    in shots/options/renderer would confound 'CS vs math' with 'task B vs task A'."""
    math_cfg, cs_cfg = _read(TASK_DIR / "mmlu_math_mc.yaml"), _read(TASK_DIR / "mmlu_cs_mc.yaml")
    differing = {k for k in set(math_cfg) | set(cs_cfg) if math_cfg.get(k) != cs_cfg.get(k)}
    assert differing == {"task", "process_docs"}


@pytest.mark.parametrize("task_name, subjects", [
    ("mmlu_math_mc", "MMLU_MATH_SUBJECTS"),
    ("mmlu_cs_mc", "MMLU_CS_SUBJECTS"),
])
def test_mmlu_slices_draw_their_shots_from_the_same_subjects(task_name, subjects):
    """The shots come out of a 57-subject dev split, so they are on-subject only
    because lm-eval defaults fewshot_config.process_docs to the task-level
    process_docs. Needs the [eval] extra: it is that default, not our YAML, under test."""
    tasks = pytest.importorskip("lm_eval.tasks")
    task = tasks.TaskManager(include_path=EVAL_TASK_PATH).load(task_name)["tasks"][task_name]
    shots = list(task.fewshot_docs())[: task.config.num_fewshot]
    assert shots and all(d["subject"] in getattr(utils, subjects) for d in shots)


# ----- the doc formatters -----

def test_letter_choices_and_target():
    doc = {"options": ["a", "b", "c"], "answer": "B"}
    assert utils.letter_choices(doc) == ["A", "B", "C"]
    assert utils.letter_target(doc) == 1


def test_letter_target_accepts_an_integer_answer():
    # cais/mmlu stores `answer` as the 0-based index, not a letter.
    assert utils.letter_target({"choices": ["a", "b", "c", "d"], "answer": 2}) == 2


def test_letter_target_tolerates_whitespace_and_case():
    assert utils.letter_target({"choices": ["a", "b"], "answer": " b "}) == 1


def test_mmlu_pro_supports_ten_options():
    """MMLU-Pro is 10-way, so the chance floor is 0.10 — the LETTERS table must
    reach J or the gold answer can be unrepresentable."""
    doc = {"question": "q", "options": [str(i) for i in range(10)], "answer": "J"}
    assert utils.letter_choices(doc)[-1] == "J"
    assert utils.letter_target(doc) == 9
    text = utils.mmlu_doc_to_text(doc)
    assert text.startswith("q\n")
    assert text.endswith("Answer:")  # single-letter continuation, the cheapest shape


def test_mmlu_filters_are_the_original_taxonomy_subcategories():
    # Taxonomy subcategories, not ad-hoc picks — that is what makes cs a like-for-like
    # control for math rather than another benchmark.
    assert utils.MMLU_MATH_SUBJECTS == {
        "abstract_algebra", "college_mathematics", "elementary_mathematics",
        "high_school_mathematics", "high_school_statistics"}
    assert utils.MMLU_CS_SUBJECTS == {
        "college_computer_science", "computer_security",
        "high_school_computer_science", "machine_learning"}
    assert not (utils.MMLU_MATH_SUBJECTS & utils.MMLU_CS_SUBJECTS)


@pytest.mark.parametrize("fn, subjects", [
    (utils.filter_mmlu_math, utils.MMLU_MATH_SUBJECTS),
    (utils.filter_mmlu_cs, utils.MMLU_CS_SUBJECTS),
])
def test_mmlu_filters_shuffle_deterministically(fn, subjects):
    """--limit/--eval-limit take an ordered PREFIX and cais/mmlu is sorted by
    subject, so an unshuffled filter would score one or two subjects at limit 200.
    The seed is fixed so the prefix is the same sample on every run."""
    datasets = pytest.importorskip("datasets")
    ordered = sorted(subjects) + ["anatomy"]  # subject-sorted, as it ships
    rows = [{"subject": s, "question": f"{s}-{i}"} for s in ordered for i in range(20)]
    got = fn(datasets.Dataset.from_list(rows))

    assert {d["subject"] for d in got} == set(subjects)  # anatomy dropped
    order = [d["question"] for d in got]
    assert order != [r["question"] for r in rows if r["subject"] != "anatomy"]  # shuffled
    assert len(set(d["subject"] for d in got.select(range(20)))) > 1  # prefix spans subjects
    stable = fn(datasets.Dataset.from_list(rows))
    assert order == [d["question"] for d in stable]  # and the same shuffle every time


def test_mmlu_renders_a_four_way_doc():
    """cais/mmlu docs carry `choices`, not `options`, and 4 of them — chance 0.25."""
    doc = {"question": "q", "choices": ["w", "x", "y", "z"], "answer": 3}
    assert utils.letter_choices(doc) == ["A", "B", "C", "D"]
    assert utils.mmlu_doc_to_text(doc) == "q\nA. w\nB. x\nC. y\nD. z\nAnswer:"


def test_codemmlu_includes_the_problem_description_when_present():
    doc = {"problem_description": "  desc  ", "question": "code", "choices": ["x", "y"]}
    text = utils.codemmlu_doc_to_text(doc)
    assert text.startswith("desc\n\ncode")
    assert text.endswith("Answer:")


def test_codemmlu_omits_a_missing_problem_description():
    text = utils.codemmlu_doc_to_text({"question": "code", "choices": ["x", "y"]})
    assert text.startswith("code")


# ----- registration (needs the [eval] extra) -----

def test_default_tasks_all_resolve_through_the_include_path():
    lm_eval_tasks = pytest.importorskip("lm_eval.tasks")
    available = set(lm_eval_tasks.TaskManager(include_path=EVAL_TASK_PATH).all_tasks)
    missing = [t for t in DEFAULT_TASKS.split(",") if t not in available]
    assert not missing, f"DEFAULT_TASKS unresolvable: {missing}"
