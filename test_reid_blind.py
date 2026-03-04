"""
Blind Re-ID test — model predicts which image is the same horse.
No positive/negative labels given; result is purely model-driven.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from horse_reid_triplet import HorseReIDModel, Config

CKPT    = "./checkpoints/best_horse_reid.pth"
TEST_DIR = Path("./test_imgs/test4")
cfg     = Config()

# ── Load model ────────────────────────────────────────────────
ckpt  = torch.load(CKPT, map_location=cfg.DEVICE)
model = HorseReIDModel(
    backbone_name=cfg.BACKBONE,
    embedding_dim=cfg.EMBEDDING_DIM,
    pretrained=False,
    num_ids=len(ckpt["train_ids"]),
    use_arcface=cfg.USE_ARCFACE,
    arcface_s=cfg.ARCFACE_S,
    arcface_m=cfg.ARCFACE_M,
).to(cfg.DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tf = transforms.Compose([
    transforms.Resize(cfg.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def embed(path):
    img = Image.open(path).convert("RGB")
    t = tf(img).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        return model(t).squeeze(0)

# ── Images ────────────────────────────────────────────────────
input_path = TEST_DIR / "IMG_0238-0001.png"
img_a_path = TEST_DIR / "IMG_0238-0003.png"
img_b_path = TEST_DIR / "IMG_0238-0005.png"

emb_input = embed(input_path)
emb_a     = embed(img_a_path)
emb_b     = embed(img_b_path)

sim_a = F.cosine_similarity(emb_input.unsqueeze(0), emb_a.unsqueeze(0)).item()
sim_b = F.cosine_similarity(emb_input.unsqueeze(0), emb_b.unsqueeze(0)).item()

pred_same = "Image A" if sim_a > sim_b else "Image B"
pred_diff = "Image B" if sim_a > sim_b else "Image A"
sim_same  = sim_a if sim_a > sim_b else sim_b
sim_diff  = sim_b if sim_a > sim_b else sim_a

print(f"Input  vs Image A ({img_a_path.name}): {sim_a:+.4f}")
print(f"Input  vs Image B ({img_b_path.name}): {sim_b:+.4f}")
print(f"\nModel predicts → {pred_same} is the SAME horse as input")
print(f"                → {pred_diff} is a DIFFERENT horse")
print(f"Gap: {abs(sim_a - sim_b):+.4f}")

# ── Visualise ─────────────────────────────────────────────────
SAME_COL  = "#1976D2"
DIFF_COL  = "#D32F2F"

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Horse Re-ID — Blind Test (test3)\nModel predicts same vs different without labels",
             fontsize=13, fontweight="bold")

entries = [
    (input_path, "Input (query)",          "#555",    "— reference —",     None),
    (img_a_path, f"Image A\n{img_a_path.name}", SAME_COL if sim_a > sim_b else DIFF_COL,
     f"sim = {sim_a:+.4f}", "SAME ✓" if sim_a > sim_b else "DIFF ✗"),
    (img_b_path, f"Image B\n{img_b_path.name}", SAME_COL if sim_b > sim_a else DIFF_COL,
     f"sim = {sim_b:+.4f}", "DIFF ✗" if sim_a > sim_b else "SAME ✓"),
]

for ax, (path, title, col, sim_label, verdict) in zip(axes, entries):
    img = Image.open(path).convert("RGB")
    ax.imshow(img)
    ax.set_title(title, fontsize=10, fontweight="bold", color=col)
    xlabel = sim_label if verdict is None else f"{sim_label}   →   {verdict}"
    ax.set_xlabel(xlabel, fontsize=11, color=col, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(4)

plt.tight_layout()
out = TEST_DIR / "reid_result_test3.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
