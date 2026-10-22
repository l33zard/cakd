# CISC 473 Deep Learning: Cross-Architecture Knowledge Distillation

This repository contains the implementation for the CISC 473 project on
cross-architecture knowledge distillation:

**ResNet-50 teacher -> DeiT-Tiny student on CIFAR-100**

The primary experiment will compare vanilla logit-only knowledge distillation
with a projected intermediate-feature alignment loss.  The current repository
is only a basic scaffold; model and training code will be added incrementally.

## Repository layout

```text
configs/       Experiment configuration files
src/           Shared Python package code
scripts/       Later command-line experiment scripts
results/       Locally generated metrics and analysis outputs
checkpoints/   Locally generated model checkpoints
proposal/      Project proposal and its compiled PDF
```

## Planned progression

1. CIFAR-100 data pipeline
2. ResNet-50 teacher
3. DeiT-Tiny student trained from scratch
4. Vanilla knowledge distillation
5. CKA measurement
6. Projected CKA feature alignment
7. Focused layer-pair and loss-weight ablations

CRD, ReviewKD, DKD, TinyImageNet, the ResNet-110 -> MobileNetV2 pair, and
attention-map alignment are currently out of scope.

## Trained checkpoints

Model checkpoints are too large for GitHub (over 100 MB) and are shared on
Google Drive instead:

**https://drive.google.com/drive/folders/1J2HeM9IKQTrtBiroXC0C_Mx8FZVZoYu9?usp=sharing**

Download `teacher_best.pt` into `checkpoints/` so the path is
`checkpoints/teacher_best.pt` (the default `--checkpoint-path` of
`scripts/train_teacher.py`). To reproduce it from scratch instead, run
`python scripts/train_teacher.py` (100 epochs, seed 42).

CIFAR-100 is downloaded automatically by torchvision on first run.

## Environment setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

No training or model entry points exist yet.  They will be documented here as
they are added.
