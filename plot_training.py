"""Parse training log and plot loss curves."""
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_DIR = Path("/Users/3i-a1-2022-062/Sean/workspace/RE-ID/logs")

# Pick the most recent log if not specified
if len(sys.argv) > 1:
    log_path = Path(sys.argv[1])
else:
    logs = sorted(LOG_DIR.glob("train_*.log"))
    if not logs:
        raise FileNotFoundError(f"No log files found in {LOG_DIR}")
    log_path = logs[-1]

print(f"Parsing: {log_path}")

# ── Parse log ────────────────────────────────────────────────
epochs, train_total, train_triplet, train_arcface = [], [], [], []
train_dap, train_dan = [], []
val_triplet, val_dap, val_dan = [], [], []

train_re = re.compile(
    r"\[Train\] Loss:\s*([\d.]+)\s+Triplet:\s*([\d.]+)\s+ArcFace:\s*([\d.]+)"
    r"\s+d_ap:\s*([\d.]+)\s+d_an:\s*([\d.]+)"
)
val_re = re.compile(
    r"\[Val\]\s+Triplet:\s*([\d.]+)\s+d_ap:\s*([\d.]+)\s+d_an:\s*([\d.]+)"
)
epoch_re = re.compile(r"Epoch\s+(\d+)/\d+")

current_epoch = None
with open(log_path) as f:
    for line in f:
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            continue
        m = train_re.search(line)
        if m and current_epoch is not None:
            epochs.append(current_epoch)
            train_total.append(float(m.group(1)))
            train_triplet.append(float(m.group(2)))
            train_arcface.append(float(m.group(3)))
            train_dap.append(float(m.group(4)))
            train_dan.append(float(m.group(5)))
            continue
        m = val_re.search(line)
        if m:
            val_triplet.append(float(m.group(1)))
            val_dap.append(float(m.group(2)))
            val_dan.append(float(m.group(3)))

best_epoch = epochs[val_triplet.index(min(val_triplet))]
best_val   = min(val_triplet)
print(f"Best val triplet loss: {best_val:.4f} at epoch {best_epoch}")

# ── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Horse Re-ID Training — Batch Hard Triplet + ArcFace", fontsize=14, fontweight="bold")

TRAIN_COL = "#2196F3"
VAL_COL   = "#FF5722"
BEST_COL  = "#4CAF50"

# 1. Triplet loss (train vs val)
ax = axes[0, 0]
ax.plot(epochs, train_triplet, color=TRAIN_COL, label="Train", linewidth=1.8)
ax.plot(epochs, val_triplet,   color=VAL_COL,   label="Val",   linewidth=1.8)
ax.axvline(best_epoch, color=BEST_COL, linestyle="--", linewidth=1.2, label=f"Best (ep {best_epoch})")
ax.scatter([best_epoch], [best_val], color=BEST_COL, zorder=5, s=60)
ax.set_title("Triplet Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. ArcFace loss (train only)
ax = axes[0, 1]
ax.plot(epochs, train_arcface, color="#9C27B0", linewidth=1.8)
ax.axvline(best_epoch, color=BEST_COL, linestyle="--", linewidth=1.2)
ax.set_title("ArcFace Loss (Train)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True, alpha=0.3)

# 3. d_ap vs d_an (train)
ax = axes[1, 0]
ax.plot(epochs, train_dap, color="#FF9800", label="d_ap (same horse)", linewidth=1.8)
ax.plot(epochs, train_dan, color="#009688", label="d_an (diff horse)", linewidth=1.8)
ax.fill_between(epochs, train_dap, train_dan,
                where=[an > ap for ap, an in zip(train_dap, train_dan)],
                alpha=0.12, color="#009688", label="Margin (d_an − d_ap)")
ax.axvline(best_epoch, color=BEST_COL, linestyle="--", linewidth=1.2)
ax.set_title("Embedding Distances (Train)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Distance")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. d_ap vs d_an (val)
ax = axes[1, 1]
ax.plot(epochs, val_dap, color="#FF9800", label="d_ap (same horse)", linewidth=1.8)
ax.plot(epochs, val_dan, color="#009688", label="d_an (diff horse)", linewidth=1.8)
ax.fill_between(epochs, val_dap, val_dan,
                where=[an > ap for ap, an in zip(val_dap, val_dan)],
                alpha=0.12, color="#009688", label="Margin (d_an − d_ap)")
ax.axvline(best_epoch, color=BEST_COL, linestyle="--", linewidth=1.2)
ax.set_title("Embedding Distances (Val)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Distance")
ax.legend()
ax.grid(True, alpha=0.3)

for ax in axes.flat:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

plt.tight_layout()

out_path = log_path.with_suffix(".png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
