"""Plot teacher train/val top-1 curves from results/teacher_metrics.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/teacher_metrics.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(df["epoch"], df["train_top1"], label="Teacher train", color="#002452")
ax.plot(df["epoch"], df["val_top1"], label="Teacher validation", color="#B90E31")
ax.set_xlabel("Epoch"); ax.set_ylabel("Top-1 accuracy (%)")
ax.set_xlim(1, 100); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("results/teacher_training.png", dpi=150)
print("wrote results/teacher_training.png")
