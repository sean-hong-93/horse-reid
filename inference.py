"""
Horse Re-ID Inference
=====================
Given a query image and a gallery folder, detects horses with YOLO,
embeds them with the Re-ID model, and returns top-K matches.

Usage:
    # Query vs gallery folder
    python inference.py --query path/to/query.jpg --gallery path/to/gallery/

    # Query vs multiple specific images
    python inference.py --query query.jpg --gallery img1.jpg img2.jpg img3.jpg

    # Adjust top-K results (default 5)
    python inference.py --query query.jpg --gallery gallery/ --topk 3

    # Skip saving output image
    python inference.py --query query.jpg --gallery gallery/ --no-save
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from horse_reid_triplet import HorseReIDModel, Config

# ── Constants ─────────────────────────────────────────────────
HORSE_CLS  = 17   # COCO class index for horse
CROP_PAD   = 0.12  # fractional padding around YOLO bounding box
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CKPT_PATH  = "/Users/3i-a1-2022-062/Sean/workspace/RE-ID/checkpoints/best_horse_reid.pth"
YOLO_MODEL = "yolo11n.pt"


# ── Model loading ─────────────────────────────────────────────

def load_reid_model(ckpt_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = Config()
    model = HorseReIDModel(
        backbone_name=cfg.BACKBONE,
        embedding_dim=cfg.EMBEDDING_DIM,
        pretrained=False,
        num_ids=len(ckpt["train_ids"]),
        use_arcface=cfg.USE_ARCFACE,
        arcface_s=cfg.ARCFACE_S,
        arcface_m=cfg.ARCFACE_M,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Re-ID model loaded  (epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f})")
    return model, cfg


def load_yolo(yolo_model: str):
    from ultralytics import YOLO
    return YOLO(yolo_model)


# ── Image utilities ───────────────────────────────────────────

def detect_horse_crop(yolo, image_path: Path, pad: float = CROP_PAD) -> Image.Image:
    """Run YOLO and return the highest-confidence horse crop (with padding).
    Falls back to the full image if no horse is detected."""
    full = Image.open(image_path).convert("RGB")
    results = yolo(str(image_path), classes=[HORSE_CLS], conf=0.15, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return full  # fallback

    best  = int(boxes.conf.cpu().argmax())
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().int().tolist()
    w, h = x2 - x1, y2 - y1
    W, H = full.size
    x1 = max(0, x1 - int(w * pad))
    y1 = max(0, y1 - int(h * pad))
    x2 = min(W, x2 + int(w * pad))
    y2 = min(H, y2 + int(h * pad))
    return full.crop((x1, y1, x2, y2))


def embed_image(model, crop: Image.Image, cfg: Config, device: torch.device) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    t = tf(crop.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t).squeeze(0)


def collect_images(sources: list[str]) -> list[Path]:
    """Expand a mix of files and directories into a flat image path list."""
    paths = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            paths.extend(sorted(f for f in p.iterdir()
                                 if f.suffix.lower() in IMG_EXTS and '._' not in f.name))
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
        else:
            print(f"  [warn] skipping {src} — not an image or directory")
    return paths


# ── Visualisation ─────────────────────────────────────────────

def save_results(query_crop: Image.Image, query_path: Path,
                 results: list[dict], out_path: Path):
    topk = len(results)
    fig, axes = plt.subplots(1, topk + 1, figsize=(4 * (topk + 1), 5))
    fig.patch.set_facecolor("#F5F5F5")
    fig.suptitle("Horse Re-ID — Top-K Results", fontsize=13, fontweight="bold")

    BLUE = "#1976D2"
    GREEN = "#388E3C"
    RED   = "#D32F2F"

    # Query
    axes[0].imshow(query_crop)
    axes[0].set_title("Query", fontsize=11, fontweight="bold", color=BLUE)
    axes[0].set_xlabel(query_path.name, fontsize=8, color="#555")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for sp in axes[0].spines.values():
        sp.set_edgecolor(BLUE); sp.set_linewidth(4)

    # Top-K matches
    for i, r in enumerate(results):
        ax = axes[i + 1]
        ax.imshow(r["crop"])
        sim   = r["sim"]
        color = GREEN if i == 0 else (RED if sim < 0.5 else "#FB8C00")
        ax.set_title(f"#{i+1}  sim={sim:+.4f}", fontsize=10,
                     fontweight="bold", color=color)
        ax.set_xlabel(r["path"].name, fontsize=7, color="#555")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Horse Re-ID inference")
    parser.add_argument("--query",   required=True,        help="Query image path")
    parser.add_argument("--gallery", required=True, nargs="+",
                        help="Gallery: folder(s) or image file(s)")
    parser.add_argument("--topk",    type=int, default=5,  help="Number of top matches (default 5)")
    parser.add_argument("--ckpt",    default=CKPT_PATH,    help="Re-ID checkpoint path")
    parser.add_argument("--yolo",    default=YOLO_MODEL,   help="YOLO model weights")
    parser.add_argument("--no-save", action="store_true",  help="Skip saving result image")
    args = parser.parse_args()

    cfg    = Config()
    device = cfg.DEVICE
    print(f"Device: {device}")

    # Load models
    model, cfg = load_reid_model(args.ckpt, device)
    yolo = load_yolo(args.yolo)

    # Collect gallery images
    gallery_paths = collect_images(args.gallery)
    gallery_paths = [p for p in gallery_paths if p.resolve() != Path(args.query).resolve()]
    print(f"Gallery: {len(gallery_paths)} images")

    if not gallery_paths:
        print("No gallery images found. Exiting.")
        return

    # Embed query
    query_path = Path(args.query)
    print(f"\nProcessing query: {query_path.name}")
    query_crop = detect_horse_crop(yolo, query_path)
    query_emb  = embed_image(model, query_crop, cfg, device)

    # Embed gallery
    print("Embedding gallery...")
    gallery = []
    for p in gallery_paths:
        try:
            crop = detect_horse_crop(yolo, p)
            emb  = embed_image(model, crop, cfg, device)
            sim  = F.cosine_similarity(query_emb.unsqueeze(0), emb.unsqueeze(0)).item()
            gallery.append({"path": p, "crop": crop, "emb": emb, "sim": sim})
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")

    # Sort by similarity descending
    gallery.sort(key=lambda x: x["sim"], reverse=True)
    topk = min(args.topk, len(gallery))
    top_results = gallery[:topk]

    # Print results
    print(f"\nTop-{topk} matches for '{query_path.name}':")
    print(f"{'Rank':<6} {'Similarity':>10}  {'File'}")
    print("-" * 50)
    for i, r in enumerate(top_results):
        print(f"  #{i+1:<4} {r['sim']:>+.4f}      {r['path'].name}")

    # Save visualisation
    if not args.no_save:
        out_path = Path(args.query).parent / f"reid_result_{query_path.stem}.png"
        save_results(query_crop, query_path, top_results, out_path)


if __name__ == "__main__":
    main()
