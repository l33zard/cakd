# CISC 473 Deep Learning: Cross-Architecture Knowledge Distillation

Reproduction code for P22 — ResNet-50 teacher -> DeiT-Tiny student on
CIFAR-100, with CKA-based projected feature alignment.

## Quickstart

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python -c "from torchvision.datasets import CIFAR100; CIFAR100(root='data', download=True)"

Download the four checkpoints from Google Drive:
https://drive.google.com/drive/folders/1J2HeM9IKQTrtBiroXC0C_Mx8FZVZoYu9

Place them in `checkpoints/`:
    teacher_best.pt
    student_seed42_best.pt
    kd_student_seed42_best.pt
    kd_student_seed42_aligned_best.pt

## Reproduce main results (Section 4.2)

    python -m scripts.evaluate_test --checkpoint checkpoints/teacher_best.pt --data-dir data --split-seed 42 --arch teacher --output-json results/teacher_test.json
    python -m scripts.evaluate_test --checkpoint checkpoints/student_seed42_best.pt --data-dir data --split-seed 42 --arch student --output-json results/student_test.json
    python -m scripts.evaluate_test --checkpoint checkpoints/kd_student_seed42_best.pt --data-dir data --split-seed 42 --arch student --output-json results/kd_student_test.json
    python -m scripts.evaluate_test --checkpoint checkpoints/kd_student_seed42_aligned_best.pt --data-dir data --split-seed 42 --arch aligned --output-json results/kd_student_aligned_test.json

Expected test top-1: 75.35 / 55.05 / 61.55 / 61.91.

## Reproduce CKA before/after (Section 4.6)

    python -m scripts.cka_diagnostic --teacher-checkpoint checkpoints/teacher_best.pt --aligned-checkpoint checkpoints/kd_student_seed42_aligned_best.pt --unaligned-student-checkpoint checkpoints/kd_student_seed42_best.pt --data-dir data --n-samples 512 --split-seed 42 --device cpu --output-json results/cka_before.json

    python -m scripts.cka_diagnostic --teacher-checkpoint checkpoints/teacher_best.pt --aligned-checkpoint checkpoints/kd_student_seed42_aligned_best.pt --data-dir data --n-samples 512 --split-seed 42 --device cpu --output-json results/cka_after.json

Expected: raw CKA 0.269 -> 0.260, projected CKA 0.348 -> 0.839.

## Reproduce analysis figures (Sections 4.4, 4.5, Appendix A.2)

    python scripts/confusion_matrix.py
    python scripts/tsne_features.py
    python scripts/plot_teacher_training.py

## Repository layout

    scripts/     Training, evaluation, and analysis entry points
    src/         Shared Python package code (models, data, CKA, alignment)
    results/     Generated metrics and figures
    checkpoints/ Model checkpoints (downloaded from Google Drive)
    final/       Final report LaTeX source and compiled PDF

## Notes

- All results are single-seed (42); see the report's Limitations section.
- Hardware: Google Colab free CPU tier (T4 unavailable for the runs reported).
