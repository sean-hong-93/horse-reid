"""
Horse Re-ID Video Inference
============================
Detects horses with YOLO frame-by-frame, assigns persistent Re-ID labels,
and writes an annotated output video.

New horses that appear mid-video are automatically added to the gallery.

Usage:
    python inference.py --source video.mp4
    python inference.py --source video.mp4 --threshold 0.65 --output out.mp4
    python inference.py --source video.mp4 --threshold 0.65 --skip 2  # process every 2nd frame
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from horse_reid_triplet import HorseReIDModel, Config

# ── Constants ─────────────────────────────────────────────────
HORSE_CLS  = 0       # Class id for horse in custom YOLO model
CROP_PAD   = 0.10    # padding fraction around YOLO box
CKPT_PATH  = "checkpoints/best_horse_reid.pth"
YOLO_MODEL = "yolo11n_480_kd_diff_hyp_best_ATH.pt"

# Distinct colors per horse ID (BGR for OpenCV)
ID_COLORS = [
    (255, 100,  50),   # blue-ish
    ( 50, 200,  50),   # green
    ( 50,  50, 255),   # red
    (255, 200,   0),   # cyan
    (200,   0, 255),   # magenta
    (  0, 200, 255),   # yellow
    (150, 100, 255),   # pink
    (  0, 180, 180),   # olive
]

def get_color(horse_id: int):
    return ID_COLORS[horse_id % len(ID_COLORS)]


# ── Re-ID Gallery / Tracker ───────────────────────────────────

class HorseTracker:
    """
    Maintains a gallery of known horse embeddings.
    Each new detection is compared to the gallery:
      - sim >= threshold  →  matched to existing ID (gallery updated via EMA)
      - sim <  threshold  →  new horse, new ID added to gallery
    """

    def __init__(self, sim_threshold: float = 0.65, ema_alpha: float = 0.2):
        self.gallery: dict[int, torch.Tensor] = {}   # id → L2-normalized embedding
        self.next_id = 0
        self.sim_threshold = sim_threshold
        self.ema_alpha = ema_alpha                   # gallery update rate

    def assign(self, emb: torch.Tensor) -> tuple[int, float]:
        """Return (horse_id, best_similarity). Adds new ID if no match found."""
        if not self.gallery:
            return self._new_id(emb), 0.0

        # Compute cosine similarity to every known horse
        gallery_ids  = list(self.gallery.keys())
        gallery_embs = torch.stack([self.gallery[i] for i in gallery_ids])  # [N, D]
        sims = F.cosine_similarity(emb.unsqueeze(0), gallery_embs).cpu()    # [N]

        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        best_id  = gallery_ids[best_idx]

        if best_sim >= self.sim_threshold:
            # EMA update: keeps gallery fresh without overwriting
            updated = (1 - self.ema_alpha) * self.gallery[best_id] + self.ema_alpha * emb
            self.gallery[best_id] = F.normalize(updated, dim=0)
            return best_id, best_sim
        else:
            return self._new_id(emb), best_sim

    def _new_id(self, emb: torch.Tensor) -> int:
        horse_id = self.next_id
        self.gallery[horse_id] = emb
        self.next_id += 1
        print(f"  [tracker] New horse detected → ID #{horse_id}  "
              f"(gallery size: {len(self.gallery)})")
        return horse_id

    @property
    def num_ids(self):
        return len(self.gallery)


# ── Model utilities ───────────────────────────────────────────

def load_reid(ckpt_path: str, device: torch.device):
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
    print(f"Re-ID loaded  (epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f})")
    return model, cfg


def embed(model, crop_bgr: np.ndarray, cfg: Config, device: torch.device) -> torch.Tensor:
    """Embed a BGR numpy crop into an L2-normalized 128-dim vector."""
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tf  = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    t = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t).squeeze(0)


# ── Drawing ───────────────────────────────────────────────────

def draw_box(frame: np.ndarray, x1, y1, x2, y2,
             horse_id: int, sim: float, conf: float):
    color = get_color(horse_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"Horse #{horse_id}"
    if sim > 0:
        label += f"  sim={sim:.2f}"
    sub   = f"det={conf:.2f}"

    # Background for label
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, sub, (x1 + 2, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_gallery_overlay(frame: np.ndarray, tracker: HorseTracker):
    """Small overlay in top-left showing current gallery IDs."""
    lines = [f"Gallery: {tracker.num_ids} horse(s)"]
    for hid in tracker.gallery:
        lines.append(f"  Horse #{hid}")
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        cv2.putText(frame, line, (10, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    get_color(hid) if i > 0 else (200, 200, 200), 1)


# ── Crop helper ───────────────────────────────────────────────

def get_crop(frame: np.ndarray, x1, y1, x2, y2, pad=CROP_PAD) -> np.ndarray:
    H, W = frame.shape[:2]
    dw, dh = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1c = max(0, x1 - dw)
    y1c = max(0, y1 - dh)
    x2c = min(W, x2 + dw)
    y2c = min(H, y2 + dh)
    return frame[y1c:y2c, x1c:x2c]


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Horse Re-ID video inference")
    parser.add_argument("--source",    required=True,           help="Input video path")
    parser.add_argument("--output",    default=None,            help="Output video path (default: source_reid.mp4)")
    parser.add_argument("--ckpt",      default=CKPT_PATH,       help="Re-ID checkpoint")
    parser.add_argument("--yolo",      default=YOLO_MODEL,      help="YOLO weights")
    parser.add_argument("--threshold", type=float, default=0.65,help="Re-ID similarity threshold")
    parser.add_argument("--conf",      type=float, default=0.7,help="YOLO detection confidence")
    parser.add_argument("--skip",      type=int,   default=1,   help="Process every N-th frame (default 1 = all)")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output) if args.output else source.parent / f"{source.stem}_reid.mp4"

    cfg    = Config()
    device = cfg.DEVICE
    print(f"Device  : {device}")
    print(f"Source  : {source}")
    print(f"Output  : {output}")
    print(f"Threshold: {args.threshold}  |  YOLO conf: {args.conf}  |  Skip: {args.skip}")

    # Load models
    reid_model, cfg = load_reid(args.ckpt, device)

    from ultralytics import YOLO
    yolo = YOLO(args.yolo)

    tracker = HorseTracker(sim_threshold=args.threshold)

    # Open video
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"Error: cannot open {source}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video   : {W}x{H} @ {fps:.1f}fps  ({total_frames} frames)")

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.skip,
        (W, H)
    )

    frame_idx = 0
    processed = 0
    is_first_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.skip != 0:
            continue

        processed += 1
        if processed % 30 == 0:
            print(f"  Frame {frame_idx}/{total_frames}  |  IDs so far: {tracker.num_ids}")

        # YOLO detect horses
        results = yolo(frame, classes=[HORSE_CLS], conf=args.conf, verbose=False)
        boxes   = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            if is_first_frame:
                # First frame: register every detection as a new horse without
                # comparing within the same frame (avoids merging similar horses)
                print(f"  [first frame] Seeding gallery with {len(boxes)} detection(s)")
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().int().tolist()
                    det_conf = float(box.conf[0].cpu())

                    crop = get_crop(frame, x1, y1, x2, y2)
                    if crop.size == 0:
                        continue

                    emb = embed(reid_model, crop, cfg, device)
                    horse_id = tracker._new_id(emb)
                    draw_box(frame, x1, y1, x2, y2, horse_id, 0.0, det_conf)
                is_first_frame = False
            else:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().int().tolist()
                    det_conf = float(box.conf[0].cpu())

                    crop = get_crop(frame, x1, y1, x2, y2)
                    if crop.size == 0:
                        continue

                    emb = embed(reid_model, crop, cfg, device)
                    horse_id, sim = tracker.assign(emb)
                    draw_box(frame, x1, y1, x2, y2, horse_id, sim, det_conf)

        draw_gallery_overlay(frame, tracker)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\nDone — {processed} frames processed, {tracker.num_ids} unique horse(s) found.")
    print(f"Saved : {output}")


if __name__ == "__main__":
    main()
