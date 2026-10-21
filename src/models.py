"""Model factories for the CIFAR-100 distillation experiments.

This module defines the teacher and student only.  Knowledge-distillation and
feature-alignment losses will be added separately.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet50

import timm


def create_teacher(num_classes: int = 100) -> nn.Module:
    """Create a randomly initialized CIFAR-compatible ResNet-50 teacher.

    The ImageNet-style ResNet stem is changed for 32x32 CIFAR images:

    - the first 7x7, stride-2 convolution becomes a 3x3, stride-1 convolution;
    - the initial max-pooling layer is replaced with an identity operation;
    - the final fully connected layer predicts ``num_classes`` classes.
    """
    teacher = resnet50(weights=None)
    teacher.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    teacher.maxpool = nn.Identity()
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    return teacher


def create_student(
    num_classes: int = 100,
    image_size: int = 32,
    patch_size: int = 4,
) -> nn.Module:
    """Create a randomly initialized CIFAR-compatible DeiT-Tiny student.

    ``timm`` provides the DeiT-Tiny architecture.  Its embedding dimension,
    depth, and number of attention heads remain the standard 192, 12, and 3;
    only the input image and patch sizes are overridden for CIFAR-100.
    """
    student = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=image_size,
        patch_size=patch_size,
    )

    if student.embed_dim != 192:
        raise RuntimeError(f"Expected DeiT-Tiny embedding dimension 192, got {student.embed_dim}")
    if len(student.blocks) != 12:
        raise RuntimeError(f"Expected 12 transformer blocks, got {len(student.blocks)}")
    if student.blocks[0].attn.num_heads != 3:
        raise RuntimeError(
            f"Expected 3 attention heads, got {student.blocks[0].attn.num_heads}"
        )
    if tuple(student.patch_embed.patch_size) != (patch_size, patch_size):
        raise RuntimeError(
            "The timm DeiT patch-size override did not produce the requested "
            f"{patch_size}x{patch_size} patches"
        )
    return student


def _parameter_count(model: nn.Module) -> int:
    """Return the number of trainable and non-trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def run_model_diagnostics() -> None:
    """Validate the model interfaces and print diagnostic information.

    The forward hooks are temporary diagnostic instrumentation.  They verify
    the feature shapes needed by the later alignment experiment, but they are
    not part of either model's training architecture.
    """
    teacher = create_teacher()
    student = create_student()
    dummy_images = torch.randn(2, 3, 32, 32)
    expected_input_shape = (2, 3, 32, 32)
    expected_logits_shape = (2, 100)

    assert tuple(dummy_images.shape) == expected_input_shape

    assert tuple(teacher.conv1.kernel_size) == (3, 3)
    assert tuple(teacher.conv1.stride) == (1, 1)
    assert isinstance(teacher.maxpool, nn.Identity)
    assert teacher.fc.out_features == 100

    assert student.embed_dim == 192
    assert len(student.blocks) == 12
    assert student.blocks[0].attn.num_heads == 3
    assert tuple(student.patch_embed.patch_size) == (4, 4)
    assert tuple(student.patch_embed.grid_size) == (8, 8)
    assert student.patch_embed.num_patches == 64
    assert tuple(student.pos_embed.shape) == (1, 65, 192)

    teacher_features: list[torch.Tensor] = []
    student_features: list[torch.Tensor] = []

    teacher_hook = teacher.layer3.register_forward_hook(
        lambda _module, _inputs, output: teacher_features.append(output)
    )
    student_hook = student.norm.register_forward_hook(
        lambda _module, _inputs, output: student_features.append(output)
    )

    teacher.eval()
    student.eval()
    try:
        with torch.no_grad():
            teacher_logits = teacher(dummy_images)
            student_logits = student(dummy_images)
    finally:
        teacher_hook.remove()
        student_hook.remove()

    assert tuple(teacher_logits.shape) == expected_logits_shape
    assert tuple(student_logits.shape) == expected_logits_shape
    assert len(teacher_features) == 1
    assert len(student_features) == 1
    assert tuple(teacher_features[0].shape) == (2, 1024, 8, 8)
    assert tuple(student_features[0].shape) == (2, 65, 192)

    print(f"teacher logits shape: {tuple(teacher_logits.shape)}")
    print(f"teacher parameter count: {_parameter_count(teacher):,}")
    print(f"teacher layer3 shape: {tuple(teacher_features[0].shape)}")
    print(f"student logits shape: {tuple(student_logits.shape)}")
    print(f"student parameter count: {_parameter_count(student):,}")
    print(f"student pre-classifier feature shape: {tuple(student_features[0].shape)}")


if __name__ == "__main__":
    run_model_diagnostics()
