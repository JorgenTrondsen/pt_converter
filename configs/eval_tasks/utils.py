"""Doc formatters for the repo's custom lm-eval tasks.

Every task here is a **letter** multiple-choice task: the context ends in
"Answer:" and each choice is a single letter, so the scored continuation is one
token. That is the cheapest loglikelihood shape there is, which matters because
``PTLM`` re-runs the whole track-sharded forward (with its NCCL all-reduces) for
every request.

Plain functions over dicts — no lm-eval imports — so they unit-test offline.
"""
LETTERS = "ABCDEFGHIJ"


def _options(doc) -> list[str]:
    return doc["options"] if "options" in doc else doc["choices"]


def letter_choices(doc) -> list[str]:
    """``["A", "B", ...]``, one per option present on this doc.

    Sized per doc, not per task: MMLU-Pro questions carry anywhere up to 10
    options, so a fixed 10-letter list would offer choices that do not exist.
    """
    return list(LETTERS[: len(_options(doc))])


def letter_target(doc) -> int:
    """Index of the gold letter in ``letter_choices``.

    ``answer`` is a letter on MMLU-Pro/CodeMMLU but already an integer index on
    cais/mmlu, so accept both rather than fork the formatter.
    """
    answer = doc["answer"]
    if isinstance(answer, int):
        return answer
    return LETTERS.index(answer.strip().upper())


def _lettered(prompt: str, doc) -> str:
    for letter, opt in zip(LETTERS, _options(doc)):
        prompt += f"{letter}. {opt.strip()}\n"
    return prompt + "Answer:"


# ----- the MMLU-family tasks -----
# One rendering over three slices, so a difference between them is the SUBJECT and
# nothing else: cais/mmlu math (the macro's math slot) and cais/mmlu computer
# science (its control), both 4 options / chance 0.25; plus TIGER-Lab/MMLU-Pro math
# (up to 10, chance 0.10), off-macro so a checkpoint can be scored on both sources.

def mmlu_doc_to_text(doc) -> str:
    return _lettered(f"{doc['question']}\n", doc)


def filter_mmlu_pro_math(dataset):
    return dataset.filter(lambda doc: doc["category"] == "math")


# Subcategories of the original MMLU taxonomy (hendrycks et al.'s categories.py),
# not ad-hoc subject picks — which is what makes `math` the counterpart of
# MMLU-Pro's math category, and `cs` a like-for-like control against it.
MMLU_MATH_SUBJECTS = frozenset({
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
})  # 1064 test docs

MMLU_CS_SUBJECTS = frozenset({
    "college_computer_science",
    "computer_security",
    "high_school_computer_science",
    "machine_learning",
})  # 412 test docs


SHUFFLE_SEED = 1234


def shuffled(dataset):
    """Fixed-seed shuffle — the `process_docs` every task in the macro needs.

    The shuffle is load-bearing, not cosmetic. lm-eval's ``--limit`` / the trainer's
    ``--eval-limit`` take an ordered PREFIX of the docs (``instances[:limit]`` in
    ``lm_eval/api/task.py``), NOT a random sample, so an unshuffled limit-N scores
    whatever the dataset happens to ship first. Measured cost of leaving it off:
    arc_easy's prefix is systematically hard, and its limit-200 accuracy ran
    **0.063 BELOW** its full-set value on the teacher, 0.066 below on a trained
    student — 2x the nominal sigma, and in the same direction for every model, so it
    does not cancel in a model-vs-model comparison the way noise would.

    The seed is fixed, so the prefix is the same sample on every run and at every
    step (arms stay comparable, and the in-loop curve is not resampled per eval).
    ``--limit 0`` scores the same docs either way — accuracy is a mean, so order
    cannot change it.
    """
    return dataset.shuffle(seed=SHUFFLE_SEED)


def _subject_slice(dataset, subjects):
    """Docs in ``subjects`` only, in a fixed shuffled order.

    cais/mmlu ships its test split sorted by subject, so an unshuffled limit-200 of
    the math slice would score abstract_algebra + college_mathematics and nothing
    else, missing all 864 elementary/high-school docs. See ``shuffled``.
    """
    return shuffled(dataset.filter(lambda doc: doc["subject"] in subjects))


def filter_mmlu_math(dataset):
    return _subject_slice(dataset, MMLU_MATH_SUBJECTS)


def filter_mmlu_cs(dataset):
    return _subject_slice(dataset, MMLU_CS_SUBJECTS)


# ----- codemmlu_fim (Fsoft-AIC/CodeMMLU, fill_in_the_middle) -----

def codemmlu_doc_to_text(doc) -> str:
    """Problem statement (when present) + the holed code + lettered candidates."""
    header = (doc.get("problem_description") or "").strip()
    prompt = f"{header}\n\n" if header else ""
    return _lettered(f"{prompt}{doc['question']}\n\nWhich line fills the blank?\n", doc)
