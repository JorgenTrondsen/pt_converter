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
    """Index of the gold letter in ``letter_choices``."""
    return LETTERS.index(doc["answer"].strip().upper())


def _lettered(prompt: str, doc) -> str:
    for letter, opt in zip(LETTERS, _options(doc)):
        prompt += f"{letter}. {opt.strip()}\n"
    return prompt + "Answer:"


# ----- mmlu_pro_math_mc (TIGER-Lab/MMLU-Pro, category == "math") -----

def filter_math(dataset):
    return dataset.filter(lambda doc: doc["category"] == "math")


def mmlu_pro_doc_to_text(doc) -> str:
    return _lettered(f"{doc['question']}\n", doc)


# ----- codemmlu_fim (Fsoft-AIC/CodeMMLU, fill_in_the_middle) -----

def codemmlu_doc_to_text(doc) -> str:
    """Problem statement (when present) + the holed code + lettered candidates."""
    header = (doc.get("problem_description") or "").strip()
    prompt = f"{header}\n\n" if header else ""
    return _lettered(f"{prompt}{doc['question']}\n\nWhich line fills the blank?\n", doc)
