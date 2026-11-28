#!/usr/bin/env bash
# Reproduce all main results (Table 1, Table 3, figures) from the four
# checkpoints in checkpoints/. Usage: bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
DATA=data
CK=checkpoints

python -c "from torchvision.datasets import CIFAR100; CIFAR100(root='$DATA', download=True)"

python -m scripts.evaluate_test --checkpoint $CK/teacher_best.pt --data-dir $DATA --split-seed 42 --arch teacher --output-json results/teacher_test.json
python -m scripts.evaluate_test --checkpoint $CK/student_seed42_best.pt --data-dir $DATA --split-seed 42 --arch student --output-json results/student_test.json
python -m scripts.evaluate_test --checkpoint $CK/kd_student_seed42_best.pt --data-dir $DATA --split-seed 42 --arch student --output-json results/kd_student_test.json
python -m scripts.evaluate_test --checkpoint $CK/kd_student_seed42_aligned_best.pt --data-dir $DATA --split-seed 42 --arch aligned --output-json results/kd_student_aligned_test.json

python -m scripts.cka_diagnostic --teacher-checkpoint $CK/teacher_best.pt --aligned-checkpoint $CK/kd_student_seed42_aligned_best.pt --unaligned-student-checkpoint $CK/kd_student_seed42_best.pt --data-dir $DATA --n-samples 512 --split-seed 42 --device cpu --output-json results/cka_before.json
python -m scripts.cka_diagnostic --teacher-checkpoint $CK/teacher_best.pt --aligned-checkpoint $CK/kd_student_seed42_aligned_best.pt --data-dir $DATA --n-samples 512 --split-seed 42 --device cpu --output-json results/cka_after.json

python scripts/confusion_matrix.py
python scripts/tsne_features.py
python scripts/plot_teacher_training.py

echo "Done. Expected test top-1: 75.35 / 55.05 / 61.55 / 61.91"
