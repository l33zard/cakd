"""Loss functions for the knowledge-distillation experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def vanilla_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.5,
    tau: float = 4.0,
) -> torch.Tensor:
    """Return the standard hard-label plus softened-logit KD loss.

    ``teacher_logits`` should already be detached or computed inside a
    ``torch.no_grad()`` context by the training loop.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")

    ce_loss = F.cross_entropy(student_logits, targets)

    # KL divergence expects log-probabilities as its first argument and
    # probabilities as its target. The class dimension is dimension 1.
    student_log_probs = F.log_softmax(student_logits / tau, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / tau, dim=1)
    soft_target_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )

    return (1.0 - alpha) * ce_loss + alpha * (tau**2) * soft_target_loss
