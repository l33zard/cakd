"""Differentiable intermediate-feature alignment for cross-architecture KD."""

from __future__ import annotations

import torch
from torch import nn


def differentiable_linear_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return centered linear CKA while preserving gradients through inputs.

    Both inputs have shape ``(observations, features)``.  Observations are
    centered along dimension 0, and the feature-space Frobenius-norm formula
    is evaluated without detaching either input.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("differentiable_linear_cka expects two 2D tensors")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of observations")
    if x.shape[0] == 0:
        raise ValueError("CKA requires at least one observation")
    if eps <= 0:
        raise ValueError("eps must be positive")

    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    cross = x_centered.transpose(0, 1) @ y_centered
    x_gram = x_centered.transpose(0, 1) @ x_centered
    y_gram = y_centered.transpose(0, 1) @ y_centered

    numerator = torch.linalg.matrix_norm(cross, ord="fro").square()
    denominator = (
        torch.linalg.matrix_norm(x_gram, ord="fro")
        * torch.linalg.matrix_norm(y_gram, ord="fro")
    )
    return numerator / (denominator + eps)


class TeacherStudentAlignment(nn.Module):
    """Frozen ResNet-50 teacher and trainable DeiT-Tiny student alignment.

    Teacher layer3 features are converted from ``B x 1024 x 8 x 8`` to
    ``(B*64) x 1024``.  Student block-8 features are converted from
    ``B x 65 x 192`` to ``(B*64) x 192`` after removing the CLS token.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        feat_dim_teacher: int = 1024,
        feat_dim_student: int = 192,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.projection = nn.Linear(feat_dim_teacher, feat_dim_student)
        self._teacher_feat: torch.Tensor | None = None
        self._student_feat: torch.Tensor | None = None
        self._teacher_hook = self.teacher.layer3.register_forward_hook(
            self._store_teacher_feature
        )
        self._student_hook = self.student.blocks[7].register_forward_hook(
            self._store_student_feature
        )

    def _store_teacher_feature(self, _module, _inputs, output) -> None:
        self._teacher_feat = output[0] if isinstance(output, (tuple, list)) else output

    def _store_student_feature(self, _module, _inputs, output) -> None:
        self._student_feat = output[0] if isinstance(output, (tuple, list)) else output

    def _teacher_tokens(self) -> torch.Tensor:
        if self._teacher_feat is None:
            raise RuntimeError("Teacher feature hook did not capture a tensor")
        teacher_features = self._teacher_feat
        if teacher_features.ndim != 4:
            raise RuntimeError("Expected teacher features with shape B x C x H x W")
        teacher_features = teacher_features.permute(0, 2, 3, 1)
        return teacher_features.reshape(-1, teacher_features.shape[-1])

    def _student_tokens(self) -> torch.Tensor:
        if self._student_feat is None:
            raise RuntimeError("Student feature hook did not capture a tensor")
        student_features = self._student_feat
        if student_features.ndim != 3:
            raise RuntimeError("Expected student features with shape B x tokens x features")
        return student_features[:, 1:, :].reshape(-1, student_features.shape[-1])

    def forward_teacher(self, images: torch.Tensor) -> torch.Tensor:
        """Run the teacher without gradients and return spatial tokens."""
        with torch.no_grad():
            self.teacher(images)
        return self._teacher_tokens()

    def forward_teacher_with_logits(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the teacher once and return detached logits and spatial tokens."""
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        return teacher_logits, self._teacher_tokens()

    def forward_student(self, images: torch.Tensor) -> torch.Tensor:
        """Run the student and return patch tokens without the CLS token."""
        self.student(images)
        return self._student_tokens()

    def forward_student_with_logits(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the student once and return logits and patch tokens."""
        student_logits = self.student(images)
        return student_logits, self._student_tokens()

    def alignment_loss(
        self,
        teacher_tokens: torch.Tensor,
        student_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``1 - CKA(projected teacher tokens, student tokens)``."""
        projected_teacher = self.projection(teacher_tokens)
        return 1.0 - differentiable_linear_cka(projected_teacher, student_tokens)

    def remove_hooks(self) -> None:
        """Remove the feature hooks when the alignment module is no longer used."""
        self._teacher_hook.remove()
        self._student_hook.remove()

