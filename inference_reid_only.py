"""
Horse Re-ID Video Inference — PCB model (ReID only, no IoU fallback)
=====================================================================
Usage:
    python inference_reid_only.py --source video.mp4
    python inference_reid_only.py --source video.mp4 --threshold 0.5 --output out.mp4
    python inference_reid_only.py --source video.mp4 --skip 2
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from horse_reid_triplet import HorseReIDModel, Config
from horse_reid_pcb import HorseReIDModelPCB

# ── Constants ──────────────────────────────────────────────────
HORSE_CLS        = 0
CROP_PAD         = 0.10
CKPT_PATH        = "checkpoints/best_horse_reid_v6_pcb_3parts_sideview.pth"
YOLO_MODEL       = "yolo11n_480_kd_diff_hyp_best_ATH.pt"
MAX_GALLERY_SIZE = 8

NUM_PARTS = 3
PART_DIM  = 512

ID_COLORS = [
    (255, 100,  50),
    ( 50, 200,  50),
    ( 50,  50, 255),
    (255, 200,   0),
    (200,   0, 255),
    (  0, 200, 255),
    (150, 100, 255),
    (  0, 180, 180),
]

def get_color(horse_id: int):
    return ID_COLORS[horse_id % len(ID_COLORS)]


# ── PCB embedding helpers ──────────────────────────────────────

def split_parts(emb: torch.Tensor) -> list[torch.Tensor]:
    return emb.split(PART_DIM, dim=0)


def part_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    parts_a = split_parts(emb_a)
    parts_b = split_parts(emb_b)
    sims = [float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
            for a, b in zip(parts_a, parts_b)]
    return sum(sims) / len(sims)


# ── Gallery / Tracker (ReID only) ─────────────────────────────

class PCBHorseTracker:
    """
    Gallery-based tracker using ReID only — no IoU fallback.
    Each horse ID stores a ring buffer of up to max_gallery embeddings.
    Match score = max over gallery entries of avg per-part cosine similarity.
    """

    def __init__(self, sim_threshold: float = 0.5, max_gallery: int = MAX_GALLERY_SIZE):
        self.gallery: dict[int, deque[torch.Tensor]] = {}
        self.next_id      = 0
        self.sim_threshold = sim_threshold
        self.max_gallery   = max_gallery

    def assign(self, emb: torch.Tensor) -> tuple[int, float, str]:
        """Return (horse_id, best_sim, method) where method is 'reid' or 'new'."""
        if not self.gallery:
            return self._new_id(emb), 0.0, "new"

        best_id, best_sim = self._best_match(emb)

        if best_sim >= self.sim_threshold:
            self.gallery[best_id].append(emb)
            return best_id, best_sim, "reid"

        print(f"  [tracker] No match — best_reid={best_sim:.3f} (need>={self.sim_threshold})")
        return self._new_id(emb), best_sim, "new"

    def _best_match(self, emb: torch.Tensor) -> tuple[int, float]:
        best_id, best_sim = -1, -1.0
        for hid, emb_buf in self.gallery.items():
            sim = max(part_similarity(emb, g) for g in emb_buf)
            if sim > best_sim:
                best_sim, best_id = sim, hid
        return best_id, best_sim

    def _new_id(self, emb: torch.Tensor) -> int:
        hid = self.next_id
        self.gallery[hid] = deque([emb], maxlen=self.max_gallery)
        self.next_id += 1
        print(f"  [tracker] New horse → ID #{hid}  (gallery: {len(self.gallery)} horses)")
        return hid

    @property
    def num_ids(self):
        return len(self.gallery)


# ── Model loading ──────────────────────────────────────────────

def load_pcb(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = Config()
    model = HorseReIDModelPCB(
        backbone_name=cfg.BACKBONE,
        num_parts=NUM_PARTS,
        part_dim=PART_DIM,
        pretrained=False,
        num_ids=len(ckpt["train_ids"]),
        use_arcface=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"PCB Re-ID loaded  (epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f})")
    print(f"  embedding: {NUM_PARTS} parts × {PART_DIM}d = {NUM_PARTS*PART_DIM}d")
    return model, cfg


def embed(model, crop_bgr: np.ndarray, cfg: Config, device: torch.device) -> torch.Tensor:
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tf  = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    t = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t).squeeze(0)


# ── Crop / Draw helpers ────────────────────────────────────────

def get_crop(frame: np.ndarray, x1, y1, x2, y2, pad=CROP_PAD) -> np.ndarray:
    H, W = frame.shape[:2]
    dw = int((x2 - x1) * pad)
    dh = int((y2 - y1) * pad)
    return frame[max(0, y1-dh):min(H, y2+dh),
                 max(0, x1-dw):min(W, x2+dw)]


def draw_box(frame, x1, y1, x2, y2, horse_id, sim, conf, method="reid"):
    color = get_color(horse_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"Horse #{horse_id}"
    if sim > 0:
        label += f"  {method}={sim:.2f}"
    sub = f"det={conf:.2f}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, sub, (x1 + 2, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_overlay(frame, tracker):
    lines = [f"Horses tracked: {tracker.num_ids}"]
    for hid in tracker.gallery:
        n = len(tracker.gallery[hid])
        lines.append(f"  #{hid}  ({n} embs)")
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, line, (10, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    get_color(i - 1) if i > 0 else (200, 200, 200), 1)


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Horse Re-ID (PCB, ReID only) video inference")
    parser.add_argument("--source",      required=True)
    parser.add_argument("--output",      default=None)
    parser.add_argument("--ckpt",        default=CKPT_PATH)
    parser.add_argument("--yolo",        default=YOLO_MODEL)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--conf",        type=float, default=0.4)
    parser.add_argument("--skip",        type=int,   default=1)
    parser.add_argument("--max-gallery", type=int,   default=MAX_GALLERY_SIZE)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output) if args.output else source.parent / f"{source.stem}_reid.mp4"

    cfg    = Config()
    device = cfg.DEVICE
    print(f"Device    : {device}")
    print(f"Source    : {source}")
    print(f"Output    : {output}")
    print(f"Threshold : {args.threshold}  |  YOLO conf: {args.conf}  |  Skip: {args.skip}")
    print(f"Gallery   : max {args.max_gallery} embeddings/horse")

    reid_model, cfg = load_pcb(args.ckpt, device)

    from ultralytics import YOLO
    yolo = YOLO(args.yolo)

    tracker = PCBHorseTracker(sim_threshold=args.threshold, max_gallery=args.max_gallery)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"Error: cannot open {source}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    if rotation in (90, 270):
        W, H = H, W

    print(f"Video     : {W}×{H} @ {fps:.1f}fps  ({total} frames)  rotation={rotation}°")

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.skip,
        (W, H),
    )

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        frame_idx += 1
        if frame_idx % args.skip != 0:
            continue

        processed += 1
        if processed % 30 == 0:
            print(f"  Frame {frame_idx}/{total}  |  IDs: {tracker.num_ids}")

        results = yolo(frame, classes=[HORSE_CLS], conf=args.conf, verbose=False)
        boxes   = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().int().tolist()
                det_conf = float(box.conf[0])

                crop = get_crop(frame, x1, y1, x2, y2)
                if crop.size == 0:
                    continue

                emb = embed(reid_model, crop, cfg, device)
                horse_id, sim, method = tracker.assign(emb)
                draw_box(frame, x1, y1, x2, y2, horse_id, sim, det_conf, method)

        draw_overlay(frame, tracker)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\nDone — {processed} frames, {tracker.num_ids} unique horse(s).")
    print(f"Saved : {output}")


if __name__ == "__main__":
    main()
