"""
Horse Re-ID model test.
Compares embeddings of:
  - Image A1  } same horse (val identity, unseen during training)
  - Image A2  }
  - Image B   different horse (different val identity)
"""
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — save to file without opening a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Imports from horse_reid_triplet ──────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from horse_reid_triplet import HorseReIDModel, Config

# ── Config ───────────────────────────────────────────────────
CKPT  = "/Users/3i-a1-2022-062/Sean/workspace/RE-ID/checkpoints/best_horse_reid.pth"
DATA  = Path("/Users/3i-a1-2022-062/Sean/workspace/RE-ID/test_img/test2")
cfg   = Config()

# ── Load checkpoint ──────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=cfg.DEVICE)
train_ids = ckpt["train_ids"]
val_ids   = ckpt["val_ids"]
print(f"Checkpoint epoch : {ckpt['epoch'] + 1}")
print(f"Val loss         : {ckpt['val_loss']:.4f}")
print(f"Val identities   : {val_ids}")

# ── Build model ───────────────────────────────────────────────
model = HorseReIDModel(
    backbone_name=cfg.BACKBONE,
    embedding_dim=cfg.EMBEDDING_DIM,
    pretrained=False,
    num_ids=len(train_ids),
    use_arcface=cfg.USE_ARCFACE,
    arcface_s=cfg.ARCFACE_S,
    arcface_m=cfg.ARCFACE_M,
).to(cfg.DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Model loaded.\n")

# ── Pick test images ──────────────────────────────────────────
img_a1_path = DATA / "input.png"
img_a2_path = DATA / "positive.png"
img_b_path  = DATA / "negative.png"

print(f"Input    (query)  : {img_a1_path.name}")
print(f"Positive (same)   : {img_a2_path.name}")
print(f"Negative (diff)   : {img_b_path.name}")

# ── Transform & embed ─────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize(cfg.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def embed(path):
    img = Image.open(path).convert("RGB")
    t   = tf(img).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        emb = model(t)
    return emb.squeeze(0)

emb_a1 = embed(img_a1_path)
emb_a2 = embed(img_a2_path)
emb_b  = embed(img_b_path)

sim_same = F.cosine_similarity(emb_a1.unsqueeze(0), emb_a2.unsqueeze(0)).item()
sim_diff = F.cosine_similarity(emb_a1.unsqueeze(0), emb_b.unsqueeze(0)).item()

print(f"\nCosine similarity  A1 vs A2 (same horse) : {sim_same:+.4f}")
print(f"Cosine similarity  A1 vs B  (diff horse) : {sim_diff:+.4f}")
print(f"Gap (same − diff)                        : {sim_same - sim_diff:+.4f}")
verdict = "✅ CORRECT" if sim_same > sim_diff else "❌ WRONG"
print(f"Verdict                                  : {verdict}")

# ── Visualise ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Horse Re-ID — Cosine Similarity Test\n"
             "(val identities, never seen during training)",
             fontsize=13, fontweight="bold")

SAME_COLOR = "#2196F3"  # blue
DIFF_COLOR = "#F44336"  # red

def show(ax, path, title, border_color, sim_label, sim_val):
    img = Image.open(path).convert("RGB")
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight="bold", color=border_color)
    ax.set_xlabel(f"{sim_label}\n{sim_val:+.4f}",
                  fontsize=12, color=border_color, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)

show(axes[0], img_a1_path,
     "Input (query)", SAME_COLOR,
     "— reference —", 1.0)

show(axes[1], img_a2_path,
     "Positive (same horse)", SAME_COLOR,
     "Cosine sim (same horse)", sim_same)

show(axes[2], img_b_path,
     "Negative (different horse)", DIFF_COLOR,
     "Cosine sim (diff horse)", sim_diff)

# Bracket annotation: same vs diff gap
y = 1.06
for ax, col, txt in [
    (axes[1], SAME_COLOR, f"sim={sim_same:+.4f}  ↑ higher = more similar"),
    (axes[2], DIFF_COLOR, f"sim={sim_diff:+.4f}  ↓ lower  = more different"),
]:
    ax.annotate(txt, xy=(0.5, y), xycoords="axes fraction",
                ha="center", fontsize=9.5, color=col,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.2))

plt.tight_layout()
out = "/Users/3i-a1-2022-062/Sean/workspace/RE-ID/reid_test_result_test2.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.show()
