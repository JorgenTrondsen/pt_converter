"""Downstream (lm-eval) scoring for the student.

Collapses an lm-eval ``simple_evaluate`` results dict into the program's reported
numbers, and owns the task set itself so the trainer's in-loop macro and a
standalone rescore cannot drift apart. Pure dict arithmetic, so it is unit-testable
on CPU without the ``[eval]`` extra or a GPU — keep this module free of top-level
``torch`` / ``lm_eval`` imports.
"""
from __future__ import annotations

from pathlib import Path

# Repo-root ``configs/eval_tasks`` — extra lm-eval task YAMLs, resolved off the
# package rather than cwd (ranks launch via torchrun from varying directories).
# Passed to lm-eval as a TaskManager include_path, which is what lets a task over
# an arbitrary hub dataset be added by dropping in a YAML, with no code change.
EVAL_TASK_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "eval_tasks")

# The macro task set, shared by the trainer's in-loop eval and the standalone
# script so the two report the same number: reasoning (arc_easy / arc_challenge),
# math (mmlu_pro_math_mc) and code (codemmlu_fim).
#
# Changing this changes what "macro" means and silently voids comparability with
# every macro= already recorded in logs/ — re-baseline rather than compare across
# a change. Pass --tasks/--eval-tasks for a one-off instead of editing this.
DEFAULT_TASKS = "arc_easy,arc_challenge,mmlu_pro_math_mc,codemmlu_fim"


class MissingTasks(KeyError):
    """An expected task produced no score. Never silently averaged away."""


def _acc(metrics: dict) -> "float | None":
    """This task's ``acc``, ignoring lm-eval's ``,<filter>`` key suffix.

    Always ``acc``, never ``acc_norm``. Not because the tasks are uniform —
    arc_easy/arc_challenge score full-sentence continuations and DO report
    acc_norm — but because taking it would silently give a different number than
    every result recorded in logs/. The two custom tasks score a single letter,
    where length normalization is meaningless and only ``acc`` is emitted.
    """
    for key, val in metrics.items():
        if key.split(",")[0] == "acc" and isinstance(val, (int, float)):
            return float(val)
    return None


def macro_metrics(
    results: "dict | None", tasks: "str | list[str] | None" = None
) -> "dict[str, float]":
    """Each EXPECTED task's ``acc``, keyed by task name.

    Iterates the expected task list rather than the results table, for two reasons.
    A group task puts its subtasks in the table alongside the group, and averaging
    those would swamp the macro. And a task that failed to load would otherwise
    vanish without trace — which is not a neutral failure: dropping
    ``mmlu_pro_math_mc``, reliably the lowest scorer, *raises* the macro by ~0.07,
    so a hub outage would read as a win and could promote a worse checkpoint.

    ``results`` is ``None`` on every rank but global rank 0 (``simple_evaluate``
    returns nothing elsewhere), which yields ``{}``.
    """
    if not results:
        return {}
    table = results.get("results", {})
    if isinstance(tasks := (DEFAULT_TASKS if tasks is None else tasks), str):
        tasks = tasks.split(",")
    out, missing = {}, []
    for task in (t.strip() for t in tasks if t.strip()):
        val = _acc(table.get(task, {}))
        if val is None:
            missing.append(task)
        else:
            out[task] = val
    if missing:
        raise MissingTasks(f"expected tasks did not score: {missing} (got {sorted(table)})")
    return out
