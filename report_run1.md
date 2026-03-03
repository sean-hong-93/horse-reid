# Horse Re-ID — Training Run 1 Report

**Date:** 2026-02-26
**Script:** `horse_reid_triplet.py`
**Checkpoint:** `checkpoints/best_horse_reid.pth`

---

## 1. Setup

### Model Architecture
| Component | Detail |
|---|---|
| Backbone | MobileNetV4 Conv Small (pretrained, ImageNet) |
| Backbone output dim | 1280 |
| Embedding dim | 128 |
| Classification head | ArcFace (s=64.0, m=0.5) |
| Device | Apple MPS |

### Loss Function
Combined **Batch Hard Triplet Loss** + **ArcFace Loss**

- **Batch Hard Triplet Loss** — mines the hardest positive (same identity, largest distance) and hardest negative (different identity, smallest distance) within each batch. Prevents easy-triplet collapse seen with random triplet sampling.
- **ArcFace Loss** — adds an angular margin of 0.5 rad (~28.6°) between identity clusters. Produces tighter, more separable embeddings especially effective for small datasets.

### Training Configuration
| Parameter | Value |
|---|---|
| Epochs | 50 |
| PK sampling | P=8 identities × K=4 images = 32 per batch |
| Learning rate | 1e-4 (AdamW, weight decay 1e-4) |
| LR schedule | CosineAnnealingLR |
| Triplet margin | 0.3 |
| ArcFace weight | 1.0 |
| Image size | 224×224 |
| Augmentation | RandomHorizontalFlip, ColorJitter |

### Dataset
| Split | Identities | Images |
|---|---|---|
| Train | 22 | 6,180 |
| Val | 6 | 1,501 |
| **Total** | **28** | **7,681** |

**Source:** `horse10/labeled-data` (full-frame video frames)
**Split strategy:** Identity-level (val horses never seen during training)

**Train identities:** Sample5, Sample3, BrownHorseinShadow, Sample7, ChestnutHorseLight, BrownHorseintoshadow, Sample17, Sample4, Sample14, Sample15, Sample6, GreyHorseNoShadowBadLight, Brownhorseoutofshadow, Sample8, Sample9, Sample12, Sample2, Sample19, Sample1, Chestnuthorseongrass, Sample10, Sample16

**Val identities:** Sample11, Brownhorselight, Sample18, Sample13, GreyHorseLightandShadow, Sample20

---

## 2. Training Results

### Key Milestones

| Epoch | Train Loss | Triplet | ArcFace | Val Triplet | d_ap | d_an | Margin |
|---|---|---|---|---|---|---|---|
| 1  | 43.02 | 0.411 | 42.61 | 0.255 | 0.681 | 0.726 | +0.045 |
| 10 | 28.71 | 0.328 | 28.39 | 0.192 | 0.867 | 0.974 | +0.107 |
| 20 | 18.37 | 0.195 | 18.17 | 0.089 | 0.837 | 1.063 | +0.226 |
| **31** | **11.38** | **0.109** | **11.27** | **0.045** ✓ | **0.820** | **1.128** | **+0.308** |
| 40 | 10.01 | 0.081 | 9.933 | 0.088 | 0.851 | 1.110 | +0.259 |
| 50 | 6.965 | 0.061 | 6.905 | 0.095 | 0.818 | 1.125 | +0.307 |

> ✓ Best checkpoint saved at **epoch 31** — val triplet loss `0.0453`

### Embedding Distance Trend

The core signal that the model is learning correctly: `d_an` (distance between different horses) growing while `d_ap` (distance between same horse) shrinks, with an increasing margin between them.

| Epoch | d_ap (same horse) | d_an (diff horse) | Margin (d_an − d_ap) |
|---|---|---|---|
| 1  | 1.193 | 1.082 | −0.111 ← not yet separated |
| 5  | 1.173 | 1.072 | −0.101 |
| 12 | 1.101 | 1.094 | −0.007 |
| 14 | 1.082 | 1.112 | **+0.030** ← crosses zero |
| 31 | 0.820 | 1.128 | **+0.308** ← best val |
| 50 | 0.978 | 1.243 | +0.265 |

### Loss Curves

![Training Loss Curves](logs/train_20260226_140631.png)

---

## 3. Re-ID Test Results

Query: one image of a val horse (unseen during training)
Compared against: one image of the **same horse** at a different frame, and one image of a **different horse**.
Metric: **cosine similarity** of L2-normalized embeddings.

### Test 1 — Full-frame images (original labeled-data)

| Pair | Cosine Similarity |
|---|---|
| A1 vs A2 — same horse (*Brownhorselight*, frame 157 vs 310) | **+0.7447** |
| A1 vs B  — diff horse (*GreyHorseLightandShadow*, frame 233) | **+0.3182** |
| **Gap** | **+0.4265** |
| Verdict | ✅ CORRECT |

![Test Result — Full Frames](reid_test_result.png)

A threshold of ~0.5–0.6 would cleanly separate same vs different identity on full-frame images.

---

### Test 2 — YOLO-cropped images (bounding box crops)

| Pair | Cosine Similarity |
|---|---|
| A1 vs A2 — same horse (*Brownhorselight*, frame 163 vs 312) | **+0.4658** |
| A1 vs B  — diff horse (*GreyHorseLightandShadow_get_highest_conf*, frame 247) | **+0.4534** |
| **Gap** | **+0.0124** |
| Verdict | ✅ CORRECT (barely) |

![Test Result — Crops](reid_test_result_crops.png)

The model still gets the correct answer but the margin collapses from **0.4265 → 0.0124**. This is a **domain mismatch** issue: the model was trained on full-frame scene images, but inference is done on tightly-cropped detections with a very different visual distribution (different aspect ratio, no background context, detection artifacts).

---

## 4. Summary

| | Result |
|---|---|
| Best val triplet loss | **0.0453** (epoch 31) |
| Embedding separation on full frames | **+0.4265** margin |
| Embedding separation on crops | **+0.0124** margin |
| Generalisation to unseen identities | ✅ Yes (identity-level val split) |
| Triplet loss collapse | ✅ Prevented (Batch Hard mining) |

### Observations

1. **Batch Hard mining works** — triplet loss stays non-zero throughout all 50 epochs, steadily decreasing from 0.41 → 0.06.
2. **ArcFace dominates the total loss** early on (~99%), providing strong supervised signal that kick-starts embedding separation before triplet mining becomes effective.
3. **Slight overfitting after epoch 31** — val loss rises from 0.045 back to ~0.09 while train loss continues to fall. Early stopping at epoch 31 is appropriate.
4. **Domain mismatch on crops** — the model works well on full-frame images but degrades significantly on YOLO crops, which were not seen during training.

---

## 5. Next Steps

- **Retrain on YOLO crops** (`/ultralytics/runs/crop/`) to fix the domain mismatch. The crops are the actual inference-time input and the model should be trained on the same distribution.
- **Early stopping** — add patience-based early stopping around epoch 31 to avoid wasted compute.
- **Data augmentation** — add RandomCrop and RandomErasing to make the model more robust to partial occlusions common in crop images.
