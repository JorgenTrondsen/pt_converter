"""lm-evaluation-harness adapter for the sharded PT student (and the dense teacher).

Why a custom adapter instead of subclassing ``HFLM``:

- The student is a ``PTWrappedModel`` (track-sharded across ranks), not a
  ``transformers.PreTrainedModel``. HFLM's ``__init__`` assumes a HF model and
  does device placement / batch-size autodetection that doesn't apply here.
- Only the rank that owns track 0 (and therefore ``lm_head``) produces logits.
  Peer ranks must still execute the forward in lockstep so the per-block
  ``SyncBoundary`` all-reduces match — but their per-request scores are
  meaningless and get discarded.

Distributed pattern (all ranks run identical lm-eval code):

- Every rank runs ``lm_eval.simple_evaluate`` with the same task list, same
  fewshot seeds, same model arg → same request stream in the same order.
- Every rank calls the model forward at the same time → SyncBoundary collectives
  match across ranks.
- Owner rank computes real loglikelihoods. Peer ranks compute zero placeholders.
- Only rank 0 prints / serializes results — peers' garbage is thrown away.
- No explicit broadcast needed: lm-eval is deterministic given identical seeds.

The teacher (dense, FSDP-wrapped) returns real logits on every rank, so the
same adapter class drives both — only the ``model_fn`` differs.
"""
from __future__ import annotations

import contextlib
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


ForwardFn = Callable[[torch.Tensor, torch.Tensor], "torch.Tensor | None"]
"""(input_ids, attention_mask) -> logits (B, T, V) or None on non-owner ranks."""


class PTLM(TemplateLM):
    """``TemplateLM`` over a callable forward function.

    Use with either the sharded student (``forward_fn`` returns None on peer
    ranks) or the dense teacher (``forward_fn`` returns logits on every rank).
    """

    def __init__(
        self,
        forward_fn: ForwardFn,
        tokenizer,
        *,
        max_length: int = 4096,
        batch_size: int = 1,
        device: "torch.device | str" = "cuda",
        is_owner_rank: bool = True,
    ):
        super().__init__()
        self._forward_fn = forward_fn
        self.tokenizer = tokenizer
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._is_owner_rank = is_owner_rank
        # lm-eval uses model.rank / model.world_size to data-parallel-shard
        # requests across processes. We're model-parallel (every rank sees every
        # request), so keep the defaults from LM.__init__ (rank=0, world_size=1).

    # ----- TemplateLM abstract surface -----

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        **kwargs,
    ) -> list[int]:
        special = False if add_special_tokens is None else add_special_tokens
        return self.tokenizer.encode(string, add_special_tokens=special)

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> "torch.Tensor | None":
        """One forward pass. All ranks call this together; peer ranks may get None."""
        with torch.no_grad():
            return self._forward_fn(input_ids, attention_mask)

    # ----- Core scoring -----

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        """Score (context_enc, continuation_enc) pairs.

        Requests are sorted by total length desc, batched, padded left, run
        through the forward, and the continuation log-probs are summed.

        On peer ranks (``self._is_owner_rank=False``) the forward still happens
        — required to keep cross-track SyncBoundary collectives aligned with
        the owner rank — but the returned scores are zero placeholders.
        """
        # Sort longest first so each batch's pad waste is bounded by within-
        # batch variance, not by the longest request overall.
        order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][1]) + len(requests[i][2])))
        results: list[tuple[float, bool] | None] = [None] * len(requests)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.eot_token_id

        try:
            from tqdm.auto import tqdm  # local import
            it = tqdm(
                range(0, len(order), self._batch_size),
                desc="loglikelihood",
                disable=disable_tqdm or not self._is_owner_rank,
            )
        except ImportError:
            it = range(0, len(order), self._batch_size)

        for batch_start in it:
            batch_indices = order[batch_start : batch_start + self._batch_size]
            batch = [requests[i] for i in batch_indices]

            # Truncate from the left to fit in max_length; the continuation is
            # always preserved intact (lm-eval relies on this for scoring).
            prepped: list[tuple[list[int], int, int]] = []  # (full_ids, ctx_len_used, cont_len_used)
            for (_ctx_str, _cont_str), ctx_enc, cont_enc in batch:
                cont_used = cont_enc
                if len(cont_used) >= self._max_length:
                    # Drop earliest cont tokens too — pathological, very long continuation.
                    cont_used = cont_used[-(self._max_length - 1):]
                budget = self._max_length - len(cont_used)
                ctx_used = ctx_enc[-budget:] if len(ctx_enc) > budget else ctx_enc
                full = ctx_used + cont_used
                prepped.append((full, len(ctx_used), len(cont_used)))

            max_len = max(len(p[0]) for p in prepped)
            padded = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long, device=self._device
            )
            attn_mask = torch.zeros(
                (len(batch), max_len), dtype=torch.long, device=self._device
            )
            for i, (full, _ctx_len, _cont_len) in enumerate(prepped):
                pad_len = max_len - len(full)
                padded[i, pad_len:] = torch.tensor(full, dtype=torch.long, device=self._device)
                attn_mask[i, pad_len:] = 1

            logits = self._model_call(padded, attn_mask)

            if logits is None or not self._is_owner_rank:
                # Peer rank — emit zero placeholders. The forward already
                # participated in the SyncBoundary collectives so the owner
                # rank's run is correct; only the owner's scores are reported.
                for idx in batch_indices:
                    results[idx] = (0.0, False)
                continue

            # logits[i, t, :] predicts padded[i, t+1].
            # Continuation tokens occupy padded[i, pad_len + ctx_len : pad_len + ctx_len + cont_len]
            # so the predicting logit positions are [pad_len + ctx_len - 1 .. pad_len + ctx_len + cont_len - 2].
            for i, (idx, (full, ctx_len, cont_len)) in enumerate(zip(batch_indices, prepped)):
                pad_len = max_len - len(full)
                start = pad_len + ctx_len - 1
                end = start + cont_len  # exclusive; end - start == cont_len
                cont_logits = logits[i, start:end, :].float()
                cont_tokens = torch.tensor(
                    full[ctx_len : ctx_len + cont_len],
                    dtype=torch.long,
                    device=cont_logits.device,
                )
                log_probs = F.log_softmax(cont_logits, dim=-1)
                picked = log_probs.gather(dim=-1, index=cont_tokens.unsqueeze(-1)).squeeze(-1)
                logprob_sum = picked.sum().item()
                greedy = log_probs.argmax(dim=-1)
                is_greedy = bool((greedy == cont_tokens).all().item())
                results[idx] = (logprob_sum, is_greedy)

        # No `None` should remain — every request was filled.
        return [r if r is not None else (0.0, False) for r in results]

    def loglikelihood_rolling(
        self, requests: list, disable_tqdm: bool = False
    ) -> list[float]:
        """Rolling-window log-likelihood for perplexity-style tasks (wikitext, etc).

        Tokenizes each request's text, slides a ``max_length``-token window
        over it (disjoint windows), and scores each window as a single
        ``(context, continuation)`` pair via ``_loglikelihood_tokens``.
        """
        loglikelihoods: list[float] = []
        try:
            from tqdm.auto import tqdm
            req_iter = tqdm(
                [req.args[0] for req in requests],
                desc="loglikelihood_rolling",
                disable=disable_tqdm or not self._is_owner_rank,
            )
        except ImportError:
            req_iter = [req.args[0] for req in requests]

        for string in req_iter:
            tokens = self.tok_encode(string, add_special_tokens=False)
            # Disjoint windows of size max_length; the first window's context is
            # a single prefix token (BOS/EOS), subsequent windows use the prior
            # window's tail as context.
            windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=tokens,
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self._max_length,
                        context_len=1,
                    ),
                )
            )

            # Convert to _loglikelihood_tokens' request format. The (str, str)
            # placeholder isn't used downstream — only ctx_enc / cont_enc are.
            tokens_requests = [(("", ""), list(ctx), list(cont)) for ctx, cont in windows]
            scored = self._loglikelihood_tokens(tokens_requests, disable_tqdm=True)
            loglikelihoods.append(sum(s for s, _ in scored))

        return loglikelihoods

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Generation tasks (e.g. GSM8K, HumanEval) are not supported yet.

        The initial task set in ``scripts/eval_lm_harness.py`` is all
        loglikelihood-based, so this path is intentionally unimplemented.
        """
        raise NotImplementedError(
            "PTLM does not implement generate_until. Use only loglikelihood-based "
            "tasks (hellaswag, arc_*, winogrande, piqa, mmlu)."
        )


def make_student_forward_fn(student) -> ForwardFn:
    """Wrap a PTWrappedModel as a ``ForwardFn`` for PTLM.

    Returns None on ranks that don't own ``lm_head``, matching PTLM's contract.
    """
    def _fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> "torch.Tensor | None":
        logits, _ = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sync_hiddens=False,
        )
        return logits

    return _fn


def make_teacher_forward_fn(teacher) -> ForwardFn:
    """Wrap a HookedTeacher as a ``ForwardFn`` for PTLM.

    Teacher is FSDP-wrapped → returns real logits on every rank.
    """
    def _fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits, _captures = teacher.forward(input_ids, attention_mask=attention_mask)
        return logits

    return _fn


def is_lm_head_owner(student) -> bool:
    """Whether this rank's student instance owns the shared lm_head."""
    return getattr(student, "lm_head", None) is not None


@contextlib.contextmanager
def _tqdm_off():
    """Force every tqdm bar off for the duration of the block.

    lm-eval draws bars from places with incompatible knobs: ``build_all_requests``
    calls bare ``tqdm(...)`` with no disable parameter at all, while
    ``TemplateLM.loglikelihood`` and our own scoring loops pass ``disable=``
    explicitly. Overriding the argument (not just its default — an explicit
    ``disable=False`` would win over that) is the one thing that silences all of
    them. The original ``__init__`` is restored on the way out.
    """
    from tqdm.std import tqdm as _tqdm

    saved = _tqdm.__init__

    def _disabled(self, *args, **kwargs):
        saved(self, *args, **{**kwargs, "disable": True})

    _tqdm.__init__ = _disabled
    try:
        yield
    finally:
        _tqdm.__init__ = saved


def run_lm_eval(
    forward_fn: ForwardFn,
    tokenizer,
    *,
    tasks: list[str],
    is_owner: bool,
    limit: int | None = None,
    batch_size: int = 8,
    max_length: int = 4096,
    num_fewshot: int | None = None,
    seed: int = 0,
    log_samples: bool = True,
    quiet: bool = False,
    include_path: "str | None" = None,
) -> "dict | None":
    """One lm-eval pass over ``forward_fn``. Every rank must call this.

    All four seeds are pinned to ``seed`` so each rank builds an identical
    request stream in an identical order — that is what keeps the cross-track
    ``SyncBoundary`` collectives inside the forward in lockstep.

    ``include_path`` is a directory of extra lm-eval task YAMLs (this repo ships
    ``configs/eval_tasks``, see ``downstream.EVAL_TASK_PATH``), registered
    alongside the built-in registry. That is how a task over an arbitrary
    HuggingFace hub dataset is added without a code change: drop a YAML in that
    directory and name it in ``tasks``.

    Returns ``simple_evaluate``'s dict on global rank 0 and ``None`` on every
    other rank (lm-eval gates the return on ``LOCAL_RANK == 0``), so callers
    that act on the score collectively must broadcast it.
    """
    from lm_eval import simple_evaluate  # deferred: heavy import
    from lm_eval.tasks import TaskManager

    lm = PTLM(
        forward_fn=forward_fn,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        device=torch.cuda.current_device(),
        is_owner_rank=is_owner,
    )
    with _tqdm_off() if quiet else contextlib.nullcontext():
        return simple_evaluate(
            model=lm,
            tasks=tasks,
            task_manager=TaskManager(include_path=include_path) if include_path else None,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            log_samples=log_samples,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
        )
