"""Transparent linear Centered Kernel Alignment utilities."""

from __future__ import annotations

import torch


def linear_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute centered linear CKA for two observation-by-feature matrices.

    Args:
        x: Tensor with shape ``(observations, x_features)``.
        y: Tensor with shape ``(observations, y_features)``.
        eps: Small positive safeguard for a near-zero denominator.

    Returns:
        A scalar tensor.  For non-degenerate representations, the value is
        normally between zero and one.

    The observations are centered before computing the feature-space formula:

        ||X^T Y||_F^2 /
        (||X^T X||_F ||Y^T Y||_F + eps)
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("linear_cka expects two 2D observation matrices")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of observations")
    if x.shape[0] == 0:
        raise ValueError("linear_cka requires at least one observation")
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
    if denominator.abs() <= eps:
        # CKA is undefined for a constant representation.  Returning zero is
        # safer for analysis than producing NaN or an arbitrary large value.
        return torch.zeros((), dtype=x.dtype, device=x.device)
    return numerator / (denominator + eps)


def run_sanity_checks() -> None:
    """Run small numerical checks for the CKA implementation."""
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(32, 8, generator=generator)
    same = linear_cka(x, x)
    scaled = linear_cka(x, 7.5 * x)

    assert torch.isclose(same, torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(scaled, torch.tensor(1.0), atol=1e-5)
    print(f"CKA(X, X): {same.item():.6f}")
    print(f"CKA(X, 7.5 X): {scaled.item():.6f}")
    print("CKA sanity checks passed.")


if __name__ == "__main__":
    run_sanity_checks()
