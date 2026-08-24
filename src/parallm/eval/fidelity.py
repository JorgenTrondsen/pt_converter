"""Logit-level fidelity metrics for a trained PT student vs the dense teacher.

The training loop only measures student-only LM CE on a held-out batch. After
distillation we want to know how closely the student *matches the teacher* on
the same input — KL, top-k agreement, per-sync-boundary hidden MSE, and
perplexity gap. This module provides one entry point, `fidelity_step`, that
produces the full metric panel.

All work runs under `torch.no_grad()`. The chunked vocab pass mirrors
`distill.ce_chunked`'s memory-bounding pattern (no backward, so no
`retain_graph` bookkeeping).

Aggregation contract: only the rank that owns track 0 (and therefore `lm_head`)
emits non-zero values for every metric — peer ranks emit zero placeholders.
This lets the caller `all_reduce(SUM)` across the world uniformly without
caring about the world layout. Hidden-state MSE numbers are emitted from the
same rank for the same reason (the synced hidden is identical on every rank
that participated in the SyncBoundary, so taking it once avoids a divide-by-
world_size).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from parallm.model.pt_model import PTWrappedModel
from parallm.train.losses import block_direction_magnitude, block_mse
from parallm.train.teacher import HookedTeacher


_LOGIT_METRIC_NAMES = (
    "kl_forward",
    "kl_reverse",
    "top1_agree",
    "top5_agree",
    "top5_set_iou",
    "student_nll",
    "teacher_nll",
)


def _fidelity_logit_metrics(
    student_logits: torch.Tensor,  # (B, T, V) — owner only
    teacher_logits: torch.Tensor,  # (B, T, V)
    labels: torch.Tensor,          # (B, T)
    attention_mask: torch.Tensor | None,  # (B, T) or None
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    B, T, V = student_logits.shape
    device = student_logits.device

    if attention_mask is not None:
        token_mask = attention_mask.float()
    else:
        token_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    n_kl_tokens = token_mask.sum().clamp(min=1)

    # NLL uses next-token shifting: logits at [0..T-2] predict labels at [1..T-1].
    shift_labels = labels[:, 1:]
    shift_mask = (shift_labels != -100).float()
    if attention_mask is not None:
        shift_mask = shift_mask * attention_mask[:, 1:].float()
    n_nll_tokens = shift_mask.sum().clamp(min=1)

    zero = torch.zeros((), device=device, dtype=torch.float32)
    kl_fwd = zero.clone()
    kl_rev = zero.clone()
    top1 = zero.clone()
    top5_in = zero.clone()
    top5_iou_sum = zero.clone()
    s_nll_sum = zero.clone()
    t_nll_sum = zero.clone()

    for t0 in range(0, T, chunk_size):
        t1 = min(t0 + chunk_size, T)
        s_chunk = student_logits[:, t0:t1, :].float()
        t_chunk = teacher_logits[:, t0:t1, :].float()
        m_chunk = token_mask[:, t0:t1]  # (B, chunk)

        # Distribution-level metrics.
        s_logp = F.log_softmax(s_chunk, dim=-1)
        t_logp = F.log_softmax(t_chunk, dim=-1)
        s_p = s_logp.exp()
        t_p = t_logp.exp()
        per_kl_fwd = (t_p * (t_logp - s_logp)).sum(dim=-1)  # (B, chunk)
        per_kl_rev = (s_p * (s_logp - t_logp)).sum(dim=-1)
        kl_fwd = kl_fwd + (per_kl_fwd * m_chunk).sum()
        kl_rev = kl_rev + (per_kl_rev * m_chunk).sum()

        # Argmax / top-5 agreement.
        s_argmax = s_chunk.argmax(dim=-1)
        t_argmax = t_chunk.argmax(dim=-1)
        top1 = top1 + ((s_argmax == t_argmax).float() * m_chunk).sum()

        s_top5 = s_chunk.topk(5, dim=-1).indices       # (B, chunk, 5)
        t_top5 = t_chunk.topk(5, dim=-1).indices
        teacher_top1_in_student_top5 = (
            (s_top5 == t_argmax.unsqueeze(-1)).any(dim=-1).float() * m_chunk
        )
        top5_in = top5_in + teacher_top1_in_student_top5.sum()

        # |A ∩ B| where A, B are the 5-element top-5 sets, then IoU = i / (10 - i).
        intersect = (
            (s_top5.unsqueeze(-1) == t_top5.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()
        )
        iou = intersect / (10.0 - intersect).clamp(min=1.0)
        top5_iou_sum = top5_iou_sum + (iou * m_chunk).sum()

        # Per-model NLL (shifted). Logits [t0..ce_t1) pair with labels [t0+1..ce_t1+1).
        ce_t1 = min(t1, T - 1)
        if t0 < ce_t1:
            n_ce = ce_t1 - t0
            s_ce = s_chunk[:, :n_ce, :].reshape(-1, V)
            t_ce = t_chunk[:, :n_ce, :].reshape(-1, V)
            ce_lbl = shift_labels[:, t0:ce_t1].reshape(-1)
            ce_mask = shift_mask[:, t0:ce_t1].reshape(-1)
            s_per = F.cross_entropy(s_ce, ce_lbl, ignore_index=-100, reduction="none")
            t_per = F.cross_entropy(t_ce, ce_lbl, ignore_index=-100, reduction="none")
            s_nll_sum = s_nll_sum + (s_per * ce_mask).sum()
            t_nll_sum = t_nll_sum + (t_per * ce_mask).sum()

    return {
        "kl_forward": kl_fwd / n_kl_tokens,
        "kl_reverse": kl_rev / n_kl_tokens,
        "top1_agree": top1 / n_kl_tokens,
        "top5_agree": top5_in / n_kl_tokens,
        "top5_set_iou": top5_iou_sum / n_kl_tokens,
        "student_nll": s_nll_sum / n_nll_tokens,
        "teacher_nll": t_nll_sum / n_nll_tokens,
    }


@torch.no_grad()
def fidelity_step(
    student: PTWrappedModel,
    teacher: HookedTeacher,
    batch: dict[str, torch.Tensor],
    sync_layer_indices: Sequence[int],
    chunk_size: int = 128,
    intra_window_taps: bool = False,
) -> dict[str, torch.Tensor]:
    """One eval batch worth of teacher-vs-student metrics.

    Runs the teacher (frozen, FSDP-sharded) and the student (sharded across
    tracks) on the same input. The student forward is *not* teacher-forced —
    measures end-to-end accumulated drift rather than per-block error.

    Returns a flat dict of scalar tensors. Block-MSE metrics are emitted as
    ``block_mse_l{idx}`` for each sync layer index. Logit metrics follow the
    names in ``_LOGIT_METRIC_NAMES``. Peer ranks (without ``lm_head``) emit
    zero placeholders so the caller can ``all_reduce(SUM)`` uniformly across
    the world.

    With ``intra_window_taps`` the student also emits loss-only synced
    reconstructions at the non-boundary depths (the forward keeps feeding each
    track its partial residual — same semantics as training's
    ``--intra-window-mse``); pass every layer index the teacher was hooked at
    via ``sync_layer_indices`` to get a block-MSE row per layer.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]

    teacher_logits, teacher_hiddens = teacher.forward(
        input_ids, attention_mask=attention_mask
    )
    # The student only records a hidden for layers named in the capture sets —
    # passing neither yields an EMPTY dict and every metric lookup below KeyErrors.
    # Reuse the trainer's own helper, fed the student's own schedule, so fidelity
    # taps land on exactly the depths training supervises.
    from parallm.train.distill import capture_sets

    cap_attn, cap_mlp = capture_sets(
        *student.sync_sets(),
        len(student.text_models[0].layers),
        intra_window_taps,
    )
    student_logits, student_sync_hiddens = student(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_sync_hiddens=True,
        return_intra_window_hiddens=intra_window_taps,
        capture_post_attn=cap_attn,
        capture_post_mlp=cap_mlp,
    )

    device = input_ids.device
    metrics: dict[str, torch.Tensor] = {}

    if student_logits is not None:
        for layer_idx in sync_layer_indices:
            # Raw masked-mean MSE (absolute) AND the scale-free relative ratio
            # Σ(s−t)²/Σt². The residual-stream norm grows with depth, so the raw
            # value inflates with depth even at a constant relative error; the
            # normalized ratio divides that norm out, so the two together separate
            # "deep layers are genuinely worse" from "deep layers are just bigger".
            metrics[f"block_mse_l{layer_idx}"] = block_mse(
                student_sync_hiddens[layer_idx],
                teacher_hiddens[layer_idx],
                attention_mask=attention_mask,
            ).detach().float()
            metrics[f"block_relmse_l{layer_idx}"] = block_mse(
                student_sync_hiddens[layer_idx],
                teacher_hiddens[layer_idx],
                attention_mask=attention_mask,
                normalize=True,
            ).detach().float()
            # relMSE conflates "points the wrong way" with "right way, wrong
            # length" (relMSE ≈ 1 − 2ρr + r²). Those are different failures with
            # different remedies, and which one the D>1 deficit is has never been
            # measured at the boundaries — only at the logits.
            cos, ratio = block_direction_magnitude(
                student_sync_hiddens[layer_idx],
                teacher_hiddens[layer_idx],
                attention_mask=attention_mask,
            )
            metrics[f"block_cos_l{layer_idx}"] = cos.detach().float()
            metrics[f"block_normratio_l{layer_idx}"] = ratio.detach().float()
        metrics.update(
            _fidelity_logit_metrics(
                student_logits, teacher_logits, labels, attention_mask, chunk_size
            )
        )
    else:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        for layer_idx in sync_layer_indices:
            metrics[f"block_mse_l{layer_idx}"] = zero.clone()
            metrics[f"block_relmse_l{layer_idx}"] = zero.clone()
            metrics[f"block_cos_l{layer_idx}"] = zero.clone()
            metrics[f"block_normratio_l{layer_idx}"] = zero.clone()
        for name in _LOGIT_METRIC_NAMES:
            metrics[name] = zero.clone()

    return metrics
