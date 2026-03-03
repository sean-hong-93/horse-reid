# Horse Re-ID

Horse re-identification using **MobileNetV4** backbone with **Batch-Hard Triplet Loss + ArcFace**.
Given a query horse image, the model retrieves the same horse from a gallery by comparing L2-normalized embeddings.

---

## Architecture

```
Input image (480×480)
    └─ MobileNetV4-Conv-Small (pretrained backbone)
        └─ Embedding head: Linear(1280→512) → BN → ReLU → Dropout → Linear(512→128)
            └─ L2 normalization → 128-dim embedding
                ├─ Batch-Hard Triplet Loss  (training)
                └─ ArcFace head            (training)
```

**Loss = Triplet Loss + ArcFace Loss**
- **Batch-Hard Triplet**: mines the hardest positive/negative pair in each batch
- **ArcFace**: adds angular margin (m=0.5) between identity classes for tighter clusters

---

## Dataset

Dataset path: `/Volumes/skuley/3i/videos/workspace/dataset/horse_feature_extractor/`

```
horse_feature_extractor/
├── train/   # 12 identities, ~84k images
├── val/     # 2 identities, ~23k images  (brownhorse1, grayhorse1)
└── test/    # 2 identities, ~19k images  (brownhorse2, brownhorse3-1)
```

Images are video frames named `{identity}_frame_{N}.jpg`.
macOS resource fork files (`._*`) and files with `._` in the name are automatically filtered.

---

## Setup

```bash
conda activate reid   # or any env with torch, timm, torchvision, matplotlib, ultralytics
pip install torch torchvision timm ultralytics matplotlib pillow
```

---

## Training

```bash
python horse_reid_triplet.py
```

Key config in `Config` class (`horse_reid_triplet.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BACKBONE` | `mobilenetv4_conv_small` | Pretrained backbone |
| `EMBEDDING_DIM` | 128 | Output embedding size |
| `IMG_SIZE` | (480, 480) | Input resolution |
| `P` | 4 | Identities per batch (PK sampling) |
| `K` | 8 | Images per identity → batch size = P×K = 32 |
| `NUM_EPOCHS` | 150 | Total epochs |
| `LEARNING_RATE` | 1e-4 | AdamW LR |
| `MARGIN` | 0.3 | Triplet loss margin |
| `RESUME` | True | Resume from best checkpoint |

**Augmentation** (training only):
- RandomHorizontalFlip, RandomPerspective, RandomAffine
- ColorJitter (brightness/contrast/saturation/hue)
- RandomGrayscale, RandomErasing

Training logs are saved to `logs/train_YYYYMMDD_HHMMSS.log`.
Best checkpoint is saved to `checkpoints/best_horse_reid.pth`.

### Plot training curves

```bash
python plot_training.py                        # auto-picks latest log
python plot_training.py logs/train_XXX.log     # specific log
```

---

## Testing

### With YOLO horse detection (test1, test2 pre-cropped + test3/test4 raw photos)

```bash
conda run -n rfdetr python test_reid_all.py
```

Output: `reid_test_result_all.png`

Each row shows: **Query | Positive (same horse) | Negative (different horse)**
with cosine similarity scores and CORRECT/WRONG verdict.

YOLO (`yolo11n.pt`) detects horses (COCO class 17) from raw photos before embedding.

### Simple triplet test (test_img/test2)

```bash
python test_reid.py
```

---

## Results

| Run | IMG_SIZE | K | Best Val Triplet Loss | Notes |
|-----|----------|---|----------------------|-------|
| Run 1 | 224×224 | 4 | 0.1878 (ep 5) | Baseline |
| Run 2 | 480×480 | 4 | 0.2103 (ep 15) | Larger input |
| Run 3 | 480×480 | 8 | 0.2602 (ep 28) | Strong augmentation |
| Run 4 | 480×480 | 8 | in progress | Resume + P=4 + WarmRestarts |

**Test accuracy (labeled test1 & test2): 2/2 correct** across all runs.

### Key metrics (Run 3 — best generalization)

| Test | sim_pos | sim_neg | Gap | Result |
|------|---------|---------|-----|--------|
| test1 (pre-cropped) | +0.919 | +0.582 | +0.337 | CORRECT |
| test2 (rider, angles) | +0.797 | +0.492 | +0.305 | CORRECT |
| test4 (YOLO crop) | +0.588 | +0.628 | −0.040 | blind |

---

## File Structure

```
RE-ID/
├── horse_reid_triplet.py       # Main training script
├── plot_training.py            # Parse log → loss curve plots
├── test_reid_all.py            # Full test with YOLO detection
├── test_reid.py                # Simple triplet test
├── test_reid_blind.py          # Blind test (no ground truth labels)
├── test_img/
│   ├── test1/                  # Pre-cropped horse pairs
│   ├── test2/                  # Horses with riders, different angles
│   ├── test3/                  # Raw high-res photos (43MB each)
│   └── test4/                  # Resized version of test3
├── logs/                       # Training logs + curve plots
└── checkpoints/
    └── best_horse_reid.pth     # Best model checkpoint
```

---

## Notes

- **PK Sampling**: With 12 train identities and P=4, each epoch yields 3 batches — enough gradient steps for stable convergence. Previously P=8 yielded only 1 batch/epoch causing high loss variance.
- **CosineAnnealingWarmRestarts**: LR resets every 30 epochs (T_0=30) to escape local minima during long training runs.
- **macOS metadata files**: The dataset volume contains `._` resource fork files. These are filtered at dataset init time via `'._' not in p.name`.
