"""
Horse Re-ID test — test1, test2 (pre-cropped), test3 (YOLO horse detection).
Generates reid_test_result_all.png
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

CKPT     = "/Users/3i-a1-2022-062/Sean/workspace/RE-ID/checkpoints/best_horse_reid.pth"
TEST_IMG  = Path("/Users/3i-a1-2022-062/Sean/workspace/RE-ID/test_img")
OUT_PATH  = "/Users/3i-a1-2022-062/Sean/workspace/RE-ID/reid_test_result_all2.png"
TEST_DIRS_YOLO = ["test3", "test4"]  # raw photos → YOLO crop
HORSE_CLS = 17  # COCO class index for horse

cfg = Config()

# ── Load Re-ID model ──────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=cfg.DEVICE, weights_only=False)
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
print(f"Re-ID model loaded  (epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f})")

tf = transforms.Compose([
    transforms.Resize(cfg.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def embed(img: Image.Image) -> torch.Tensor:
    t = tf(img.convert("RGB")).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        return model(t).squeeze(0)

# ── YOLO horse detector ───────────────────────────────────────
from ultralytics import YOLO
yolo = YOLO("yolo11n.pt")

def detect_horse(path: Path) -> Image.Image:
    """Run YOLO, return the highest-confidence horse crop (or full image if none found)."""
    full = Image.open(path).convert("RGB")
    results = yolo(str(path), classes=[HORSE_CLS], conf=0.15, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print(f"  No horse detected in {path.name} — using full image")
        return full
    # pick highest confidence
    confs = boxes.conf.cpu()
    best  = int(confs.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().int().tolist()
    crop = full.crop((x1, y1, x2, y2))
    print(f"  Horse crop {path.name}: [{x1},{y1},{x2},{y2}]  conf={confs[best]:.2f}")
    return crop

# ── Test cases ────────────────────────────────────────────────
# test1 & test2: already cropped horse images
# test3:         raw photos → YOLO crop first

print("\n── Test1 & Test2 (pre-cropped) ──")
tests_cropped = []
for name in ["test1", "test2"]:
    d = TEST_IMG / name
    q_img = Image.open(d / "input.png").convert("RGB")
    p_img = Image.open(d / "positive.png").convert("RGB")
    n_img = Image.open(d / "negative.png").convert("RGB")
    emb_q = embed(q_img)
    emb_p = embed(p_img)
    emb_n = embed(n_img)
    sim_p = F.cosine_similarity(emb_q.unsqueeze(0), emb_p.unsqueeze(0)).item()
    sim_n = F.cosine_similarity(emb_q.unsqueeze(0), emb_n.unsqueeze(0)).item()
    correct = sim_p > sim_n
    print(f"  {name}: sim_pos={sim_p:+.4f}  sim_neg={sim_n:+.4f}  {'CORRECT' if correct else 'WRONG'}")
    tests_cropped.append(dict(label=name, q=q_img, p=p_img, n=n_img,
                               sim_p=sim_p, sim_n=sim_n, correct=correct, yolo=False))

for yolo_dir in TEST_DIRS_YOLO:
    print(f"\n── {yolo_dir} (YOLO horse detection) ──")
    td = TEST_IMG / yolo_dir
    td_files = sorted([f for f in td.glob("*.png") if not f.name.startswith("reid_")])
    print(f"  Found {len(td_files)} images: {[f.name for f in td_files]}")

    crops = [detect_horse(p) for p in td_files[:3]]
    embs  = [embed(c) for c in crops]
    sim_ab = F.cosine_similarity(embs[0].unsqueeze(0), embs[1].unsqueeze(0)).item()
    sim_ac = F.cosine_similarity(embs[0].unsqueeze(0), embs[2].unsqueeze(0)).item()
    print(f"  sim(q,A)={sim_ab:+.4f}  sim(q,B)={sim_ac:+.4f}")
    print(f"  Model predicts: {'Image A' if sim_ab >= sim_ac else 'Image B'} is the same horse as query")

    tests_cropped.append(dict(
        label=f"{yolo_dir} [YOLO]",
        q=crops[0], p=crops[1], n=crops[2],
        sim_p=sim_ab, sim_n=sim_ac,
        correct=None,
        yolo=True,
    ))

# ── Plot ──────────────────────────────────────────────────────
BLUE  = "#1976D2"
GREEN = "#388E3C"
RED   = "#D32F2F"
GRAY  = "#757575"

n_rows = len(tests_cropped)
fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4.5 * n_rows))
if n_rows == 1:
    axes = [axes]

fig.patch.set_facecolor("#F7F7F7")
n_correct = sum(1 for t in tests_cropped if t["correct"] is True)
n_labeled = sum(1 for t in tests_cropped if t["correct"] is not None)

fig.suptitle(
    f"Horse Re-ID — MobileNetV4 + ArcFace + Batch-Hard Triplet\n"
    f"Checkpoint epoch {ckpt['epoch']+1}  |  Val loss {ckpt['val_loss']:.4f}  |  "
    f"Labeled tests: {n_correct}/{n_labeled} correct",
    fontsize=13, fontweight="bold", y=1.01
)

col_headers = ["Input (query)", "Positive / Image A", "Negative / Image B"]

for row, t in enumerate(tests_cropped):
    sim_p, sim_n, correct = t["sim_p"], t["sim_n"], t["correct"]

    # pick colors
    if correct is True:
        c_pos, c_neg = GREEN, RED
    elif correct is False:
        c_pos, c_neg = RED, GREEN
    else:
        # blind: color by which is higher
        c_pos = BLUE if sim_p >= sim_n else GRAY
        c_neg = BLUE if sim_n > sim_p else GRAY

    for col, (img, sim_val, color, xlabel) in enumerate([
        (t["q"],  None,    BLUE,  "— reference —"),
        (t["p"],  sim_p,   c_pos, None),
        (t["n"],  sim_n,   c_neg, None),
    ]):
        ax = axes[row][col]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(4)

        # column header only on first row
        if row == 0:
            ax.set_title(col_headers[col], fontsize=10, color=color, fontweight="bold", pad=5)

        # xlabel
        if sim_val is None:
            ax.set_xlabel(xlabel, fontsize=10, color=color, labelpad=4)
        else:
            if correct is None:
                verdict = "← SAME" if (col == 1 and sim_p >= sim_n) or (col == 2 and sim_n > sim_p) else "← DIFF"
                ax.set_xlabel(f"sim = {sim_val:+.4f}   {verdict}", fontsize=10,
                              color=color, fontweight="bold", labelpad=4)
            else:
                ax.set_xlabel(f"sim = {sim_val:+.4f}", fontsize=10,
                              color=color, fontweight="bold", labelpad=4)

    # row label on left
    yolo_tag = "  [YOLO]" if t["yolo"] else ""
    axes[row][0].set_ylabel(t["label"] + yolo_tag, fontsize=10, fontweight="bold",
                             rotation=90, labelpad=8, color="#333")

    # verdict box on right of row
    if correct is not None:
        verdict_str = f"CORRECT\ngap={sim_p-sim_n:+.4f}" if correct else f"WRONG\ngap={sim_p-sim_n:+.4f}"
        v_color = GREEN if correct else RED
    else:
        winner = "A" if sim_p >= sim_n else "B"
        verdict_str = f"→ Image {winner}\n(blind test)\ngap={sim_p-sim_n:+.4f}"
        v_color = BLUE
    axes[row][2].text(1.04, 0.5, verdict_str, transform=axes[row][2].transAxes,
                      fontsize=9, va="center", ha="left", color=v_color, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=v_color, lw=1.5))

plt.tight_layout(pad=2.0)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {OUT_PATH}")
