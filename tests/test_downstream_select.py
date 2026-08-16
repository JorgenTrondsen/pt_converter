"""Unit tests for the downstream scorer (eval/downstream.py).

Only the pure piece (no lm_eval, no GPUs): the metric aggregation that turns an
lm-eval ``simple_evaluate`` dict into the reported macro.
"""
import pytest

from parallm.eval.downstream import DEFAULT_TASKS, MissingTasks, macro_metrics


def macro(results, tasks=None):
    m = macro_metrics(results, tasks)
    return sum(m.values()) / len(m) if m else 0.0


def _results(**task_metrics):
    """Build a minimal lm-eval-shaped results dict."""
    return {"results": dict(task_metrics)}


def _ledger(arc_easy, arc_challenge, mmlu_pro_math_mc, codemmlu_fim):
    """The four-task table as lm-eval emits it, ``,none`` filter suffix and all."""
    return _results(
        arc_easy={"acc,none": arc_easy, "acc_norm,none": arc_easy + 0.02},
        arc_challenge={"acc,none": arc_challenge, "acc_norm,none": arc_challenge + 0.02},
        mmlu_pro_math_mc={"acc,none": mmlu_pro_math_mc},
        codemmlu_fim={"acc,none": codemmlu_fim},
    )


# ----- the task set -----

def test_default_tasks_are_the_recorded_macro():
    assert DEFAULT_TASKS.split(",") == [
        "arc_easy", "arc_challenge", "mmlu_pro_math_mc", "codemmlu_fim"]


def test_task_spec_accepts_a_string_or_a_list():
    full = _ledger(0.7550, 0.5500, 0.4350, 0.8450)
    assert macro(full, "arc_easy,arc_challenge") == pytest.approx(0.6525)
    assert macro(full, ["arc_easy", " arc_challenge "]) == pytest.approx(0.6525)


# ----- aggregation -----

def test_macro_reproduces_the_cascade2_ab_ledger():
    """Pins the convention against a real recorded eval.

    logs/qwen3/32b_d1b_nemo_ab.log, raw arm, step 625:
      macro=0.6462 arc_easy=0.7550 arc_challenge=0.5500
      mmlu_pro_math_mc=0.4350 codemmlu_fim=0.8450
    """
    assert macro(_ledger(0.7550, 0.5500, 0.4350, 0.8450)) == pytest.approx(0.6462, abs=1e-4)


def test_acc_is_the_metric_not_acc_norm():
    # Every task here scores a single-letter continuation, so length normalization
    # is meaningless — taking acc_norm would report a different number than the
    # ledger. The ``,none`` filter suffix must be stripped.
    m = macro_metrics(_ledger(0.7550, 0.5500, 0.4350, 0.8450))
    assert m == {"arc_easy": 0.7550, "arc_challenge": 0.5500,
                 "mmlu_pro_math_mc": 0.4350, "codemmlu_fim": 0.8450}


def test_macro_off_rank0_is_zero():
    # simple_evaluate returns None on every rank but global rank 0; those ranks
    # take the broadcast value instead of this one.
    assert macro(None) == 0.0
    assert macro({}) == 0.0
    assert macro_metrics(None) == {}


def test_a_partial_result_raises_instead_of_averaging_the_survivors():
    """The failure mode this guards: dropping the lowest scorer RAISES the macro.

    mmlu_pro_math_mc sits ~0.2 below the rest, so averaging over "whatever scored"
    turns a hub outage into an apparent +0.07 win — and best/ would promote it.
    """
    full = _ledger(0.7550, 0.5500, 0.4350, 0.8450)
    partial = _results(**{k: v for k, v in full["results"].items() if k != "mmlu_pro_math_mc"})
    with pytest.raises(MissingTasks) as e:
        macro_metrics(partial)
    assert "mmlu_pro_math_mc" in str(e.value)
    # and the number it would have reported is HIGHER than the true macro.
    survivors = [0.7550, 0.5500, 0.8450]
    assert sum(survivors) / len(survivors) > macro(full)


def test_empty_table_raises_only_when_results_are_present():
    # `None`/`{}` mean "not rank 0" and yield {}; a populated dict missing every
    # expected task is a real failure.
    with pytest.raises(MissingTasks):
        macro_metrics(_results(something_else={"acc,none": 0.5}))


def test_group_subtasks_do_not_leak_into_the_macro():
    """A group task puts its subtasks in the table next to the group; iterating the
    table instead of the expected list would average ~19 extra rows."""
    res = _ledger(0.7550, 0.5500, 0.4350, 0.8450)
    res["results"].update({f"mmlu_stem_sub{i}": {"acc,none": 0.01} for i in range(19)})
    assert macro(res) == pytest.approx(0.6462, abs=1e-4)


def test_an_explicit_task_subset_is_scored_alone():
    # `--tasks mmlu_pro_math_mc --limit 0`, the unbiased math rescore.
    res = _ledger(0.7550, 0.5500, 0.4350, 0.8450)
    assert macro(res, "mmlu_pro_math_mc") == pytest.approx(0.4350)


def test_unlisted_tasks_default_to_acc():
    assert macro(_results(mmlu={"acc,none": 0.55}), "mmlu") == pytest.approx(0.55)
