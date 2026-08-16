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
    assert {"mmlu_pro_math_mc", "codemmlu_fim"} <= shipped
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


def test_math_task_is_two_shot_and_code_task_is_zero_shot():
    # Pinned in the YAMLs, not on the command line: a YAML-level num_fewshot cannot
    # be overridden by --num-fewshot, and codemmlu shots would be whole code files.
    cfgs = {_read(p)["task"]: _read(p) for p in _yamls()}
    assert cfgs["mmlu_pro_math_mc"]["num_fewshot"] == 2
    assert cfgs["codemmlu_fim"]["num_fewshot"] == 0


# ----- the doc formatters -----

def test_letter_choices_and_target():
    doc = {"options": ["a", "b", "c"], "answer": "B"}
    assert utils.letter_choices(doc) == ["A", "B", "C"]
    assert utils.letter_target(doc) == 1


def test_letter_target_tolerates_whitespace_and_case():
    assert utils.letter_target({"choices": ["a", "b"], "answer": " b "}) == 1


def test_mmlu_pro_supports_ten_options():
    """MMLU-Pro is 10-way, so the chance floor is 0.10 — the LETTERS table must
    reach J or the gold answer can be unrepresentable."""
    doc = {"question": "q", "options": [str(i) for i in range(10)], "answer": "J"}
    assert utils.letter_choices(doc)[-1] == "J"
    assert utils.letter_target(doc) == 9
    text = utils.mmlu_pro_doc_to_text(doc)
    assert text.startswith("q\n")
    assert text.endswith("Answer:")  # single-letter continuation, the cheapest shape


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
