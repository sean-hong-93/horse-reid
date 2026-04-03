"""
Horse Re-ID using Triplet Loss with MobileNetV4 Small backbone
DistributedDataParallel (DDP) version — uses all available GPUs.

Launch:
    torchrun --nproc_per_node=8 horse_reid_triplet_ddp.py
"""

import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
import wandb
from horse_reid_pcb import HorseReIDModelPCB


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Experiment tracking
    EXPERIMENT = "v6_pcb_3parts_sideview"
    EXPERIMENT_DESC = "PCB 3-part (3x512=1536), 397-id sideview, blur aug (GaussianBlur+MotionBlur), random pos + hardest neg, ArcFace(0.2), constant LR freeze(50ep), cosine after unfreeze, early stop"

    # Data — flat directory (identity folders), split by ratio
    DATA_ROOT = "/home/jovyan/sean/sam2/reid_dataset_gallery_sideview"
    SPLIT_JSON = None
    EXCLUDE_FOLDERS = [
        "1C510989-70E8-4EFD-BECD-F8CFA6D5AC99",
        "20240112_131009787",
        "20241110_181636852",
        "20241130_095000281",
        "20241208_093908143",
        "20241213_104846789",
        "20241230_125951941~3",
        "2025030501_2Horses(4k).MOV",
    ]
    VAL_RATIO = 0.1  # 10% identities for validation

    # Model
    BACKBONE = "mobilenetv4_conv_small"
    EMBEDDING_DIM = 1536  # 3 parts × 512
    NUM_PARTS = 3
    PART_DIM = 512
    PRETRAINED = True

    # Training — PK sampling: P identities × K images per batch
    P = 16              # Identities per batch per GPU
    K = 8               # Images per identity  →  effective batch size per GPU = P*K = 128
    NUM_EPOCHS = 5000
    LEARNING_RATE = 1e-4
    WARMUP_EPOCHS = 10  # Linear warmup from 1e-6 to LEARNING_RATE

    # Resume
    RESUME = True
    RESUME_CKPT = "checkpoints/best_horse_reid_v6_pcb_3parts_sideview.pth"
    MARGIN = 0.5       # Semi-hard triplet margin

    # ArcFace — re-enabled as anti-collapse regularizer
    USE_ARCFACE = True
    ARCFACE_S = 64.0
    ARCFACE_M = 0.5
    ARCFACE_WEIGHT = 0.2

    # Gradual unfreezing
    FREEZE_BACKBONE_EPOCHS = 0  # Freeze backbone longer to prevent overfitting

    # Curriculum augmentation
    HARD_AUG_EPOCH = 0  # Switch from mild to hard augmentation

    # Early stopping
    EARLY_STOP_PATIENCE = 1000  # Stop if val doesn't improve for N epochs

    # Image
    IMG_SIZE = (224, 224)

    # DDP
    WORLD_SIZE = 8
    GPU_OFFSET = 0
    MASTER_PORT = "29501"

    # Save
    SAVE_DIR = "./checkpoints"

    # Wandb
    WANDB_PROJECT = "horse-reid"
    WANDB_RUN_NAME = None  # Auto-generated if None


# =============================================================================
# DDP Setup
# =============================================================================

def setup_ddp(rank, world_size):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = Config.MASTER_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    gpu_id = rank + Config.GPU_OFFSET
    torch.cuda.set_device(gpu_id)


def cleanup_ddp():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


# =============================================================================
# Blur augmentations
# =============================================================================

class MotionBlur:
    """
    Applies directional motion blur to a PIL image.

    Simulates camera pan / fast-moving subject blur by convolving with a
    line kernel at a random angle. Kernel length controls blur intensity.

    Args:
        kernel_size: length of the blur line (pixels). Larger = more blur.
        angle_range: (min_deg, max_deg) range to randomly sample angle from.
                     0° = horizontal (left-right camera pan),
                     90° = vertical (up-down pan).
    """

    def __init__(self, kernel_size: int = 15, angle_range: tuple = (0, 180)):
        self.kernel_size = kernel_size
        self.angle_range = angle_range

    def __call__(self, img: Image.Image) -> Image.Image:
        import numpy as np
        from scipy.ndimage import convolve

        angle = random.uniform(*self.angle_range)
        k = self.kernel_size

        # Build a line kernel at the given angle
        kernel = np.zeros((k, k), dtype=np.float32)
        cx = cy = k // 2
        for i in range(k):
            x = int(round(cx + (i - cx) * np.cos(np.radians(angle))))
            y = int(round(cy + (i - cx) * np.sin(np.radians(angle))))
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1.0
        kernel /= kernel.sum()

        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = convolve(arr, kernel)
        else:
            for c in range(arr.shape[2]):
                arr[:, :, c] = convolve(arr[:, :, c], kernel)

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class RandomMotionBlur:
    """Applies MotionBlur with probability p."""

    def __init__(self, p: float = 0.5, kernel_size: int = 15,
                 angle_range: tuple = (0, 180)):
        self.p = p
        self.blur = MotionBlur(kernel_size=kernel_size, angle_range=angle_range)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return self.blur(img)
        return img


# =============================================================================
# Dataset
# =============================================================================

class HorseReIDDataset(Dataset):
    """
    Flat (image, label) dataset for Batch Hard Triplet + ArcFace training.

    Returns individual (image, label) pairs — no pre-formed triplets.
    Triplets are mined online inside SoftMarginBatchHardTripletLoss using the full
    pairwise distance matrix of the batch.

    Use with PKSampler to guarantee each batch contains P identities × K images,
    which is required for effective hard mining.
    """

    def __init__(self, root: str, exclude_folders: List[str],
                 identity_list: List[str] = None, transform=None):
        self.root = Path(root)
        self.transform = transform

        all_identities: Dict[str, List[Path]] = {}
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.'):
                continue
            if folder.name in exclude_folders:
                continue
            images = [str(p) for p in Path(folder).glob('*.jpg') if '._' not in p.name]
            if len(images) >= 2:
                all_identities[folder.name] = images

        # Support identity-level train/val split
        if identity_list is not None:
            self.identity_list = identity_list
        else:
            self.identity_list = list(all_identities.keys())

        self.identity_to_label = {name: i for i, name in enumerate(self.identity_list)}

        self.samples: List[tuple] = []  # (path, label)
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)

        for name in self.identity_list:
            label = self.identity_to_label[name]
            for img_path in all_identities[name]:
                idx = len(self.samples)
                self.samples.append((img_path, label))
                self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = None
        for _ in range(20):
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except Exception:
                alt_idx = random.choice(self.label_to_indices[label])
                img_path, label = self.samples[alt_idx]
                idx = alt_idx
        if img is None:
            raise RuntimeError(f"Failed to load any valid image for label {label} after 20 retries")
        if self.transform:
            img = self.transform(img)
        return img, label, idx

    @property
    def labels(self):
        return [label for _, label in self.samples]

    @property
    def num_ids(self):
        return len(self.identity_list)


# =============================================================================
# Distributed PK Sampler
# =============================================================================

class DistributedPKSampler(Sampler):
    """
    Distributed PK Sampler for DDP training.

    When enough identities exist (>= P * world_size), partitions them uniquely
    across ranks so each GPU sees different identities per step.

    When identities are scarce (< P * world_size), each rank independently
    samples P identities using a rank-specific seed — identities may overlap
    across GPUs but each rank still gets valid PK batches for triplet mining.

    Call set_epoch(epoch) before each epoch for proper shuffling.
    """

    def __init__(self, label_to_indices: Dict[int, List[int]], P: int, K: int,
                 rank: int, world_size: int):
        self.P = P
        self.K = K
        self.rank = rank
        self.world_size = world_size
        self.label_to_indices = label_to_indices
        self.labels = list(label_to_indices.keys())
        self.epoch = 0

        if len(self.labels) < P:
            raise ValueError(
                f"DistributedPKSampler needs at least P={P} identities, "
                f"got {len(self.labels)}"
            )

        self.exclusive = len(self.labels) >= P * world_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.exclusive:
            yield from self._iter_exclusive()
        else:
            yield from self._iter_shared()

    def _iter_exclusive(self):
        """Each rank gets a unique, non-overlapping slice of P identities."""
        g = torch.Generator()
        g.manual_seed(self.epoch)
        perm = torch.randperm(len(self.labels), generator=g).tolist()
        labels = [self.labels[i] for i in perm]

        total_P = self.P * self.world_size
        for start in range(0, len(labels) - total_P + 1, total_P):
            rank_start = start + self.rank * self.P
            batch_labels = labels[rank_start: rank_start + self.P]
            for label in batch_labels:
                idxs = self.label_to_indices[label].copy()
                random.shuffle(idxs)
                selected = idxs[:self.K] if len(idxs) >= self.K else random.choices(idxs, k=self.K)
                yield from selected

    def _iter_shared(self):
        """Each rank independently samples P identities (overlap allowed)."""
        rng = random.Random(self.epoch * 1000 + self.rank)
        labels = self.labels.copy()
        rng.shuffle(labels)

        # Number of batches: same across all ranks for DDP sync
        num_batches = max(len(labels) // self.P, 1)
        for b in range(num_batches):
            batch_labels = rng.sample(labels, self.P) if len(labels) >= self.P else labels
            for label in batch_labels:
                idxs = self.label_to_indices[label].copy()
                rng.shuffle(idxs)
                selected = idxs[:self.K] if len(idxs) >= self.K else [idxs[rng.randint(0, len(idxs)-1)] for _ in range(self.K)]
                yield from selected

    def __len__(self):
        if self.exclusive:
            total_P = self.P * self.world_size
            num_batches = len(self.labels) // total_P
        else:
            num_batches = max(len(self.labels) // self.P, 1)
        return num_batches * self.P * self.K


# =============================================================================
# Model
# =============================================================================

class HorseReIDModel(nn.Module):
    """
    Re-ID model using MobileNetV4 backbone with embedding head.

    When use_arcface=True, call get_arcface_logits(embeddings, labels) during
    training to obtain ArcFace logits for the combined loss. The main forward()
    always returns L2-normalized embeddings (unchanged interface).
    """

    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int,
        pretrained: bool = True,
        num_ids: int = None,
        use_arcface: bool = True,
        arcface_s: float = 64.0,
        arcface_m: float = 0.5,
    ):
        super().__init__()
        self.use_arcface = use_arcface and (num_ids is not None)

        # Backbone (remove classifier)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        # Get backbone output dimension
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[-1]
            self.backbone.train()

        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )
        self.embedding_dim = embedding_dim

        # ArcFace head (training only; not used at inference)
        if self.use_arcface:
            self.arcface = ArcFaceHead(embedding_dim, num_ids, s=arcface_s, m=arcface_m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized embeddings (used for triplet loss and inference)."""
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)

    def get_arcface_logits(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Returns scaled cosine logits with angular margin for the ArcFace loss."""
        return self.arcface(embeddings, labels)

    def freeze_backbone(self):
        """Freeze backbone parameters so only embedding + ArcFace heads train."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# =============================================================================
# ArcFace Head
# =============================================================================

class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss head.

    Replaces standard linear classifier by normalizing both features and
    weights, then adding angular margin m to the target class logit:
        cos(θ + m)  instead of  cos(θ)

    Args:
        feat_dim   : dimension of input embeddings (must be L2-normalized)
        num_classes: number of horse identities
        s          : feature scale (default 64.0)
        m          : angular margin in radians (default 0.5 ≈ 28.6°)
    """
    def __init__(self, feat_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(x, F.normalize(self.weight, p=2, dim=1))

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi  = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine

        return self.s * output


# =============================================================================
# Loss Functions
# =============================================================================

class RandomPosHardNegTripletLoss(nn.Module):
    """
    Triplet Loss with random positive + hardest negative mining.

    For each anchor:
      - random positive: random same-identity image in the batch
      - hardest negative: different identity, smallest distance

    loss = max(0, d_ap - d_an + margin)

    Random positive gives stable gradients matching EMA inference behavior.
    Hardest negative focuses learning capacity on separating identities.
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        dist_mat = self._pairwise_euclidean(embeddings)

        B = embeddings.size(0)
        labels_row = labels.view(B, 1)

        pos_mask = (labels_row == labels.view(1, B)) & \
                   ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        neg_mask = labels_row != labels.view(1, B)

        # Random positive per anchor (random same-identity image)
        pos_weights = pos_mask.float()
        pos_weights = pos_weights / pos_weights.sum(dim=1, keepdim=True).clamp(min=1)
        rp_indices = torch.multinomial(pos_weights, 1).squeeze(1)
        dist_ap = dist_mat[torch.arange(B, device=embeddings.device), rp_indices]

        # Hardest negative per anchor (smallest distance)
        dist_an = dist_mat.clone()
        dist_an[~neg_mask] = float('inf')
        dist_an = dist_an.min(dim=1)[0]

        # Hard margin: max(0, d_ap - d_an + margin)
        loss = F.relu(dist_ap - dist_an + self.margin).mean()

        return loss, dist_ap.mean().item(), dist_an.mean().item()

    @staticmethod
    def _pairwise_euclidean(x: torch.Tensor) -> torch.Tensor:
        m = x.size(0)
        xx = (x ** 2).sum(1, keepdim=True).expand(m, m)
        dist = xx + xx.t() - 2 * x.mm(x.t())
        return dist.clamp(min=1e-12).sqrt()


# =============================================================================
# Debug Triplet Visualization
# =============================================================================

_DEBUG_MEAN = np.array([0.485, 0.456, 0.406])
_DEBUG_STD = np.array([0.229, 0.224, 0.225])


def _tensor_to_pil(t):
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = img * _DEBUG_STD + _DEBUG_MEAN
    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))


def _add_label(img, text, color=(255, 255, 255)):
    """Add label text to image. Supports multi-line with \\n."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    lines = text.split("\n")
    line_heights = []
    max_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        line_heights.append(th)
        max_w = max(max_w, tw)
    total_h = sum(line_heights) + 4 * len(lines) + 2
    draw.rectangle([0, 0, max_w + 10, total_h], fill=(0, 0, 0))
    y = 2
    for i, line in enumerate(lines):
        draw.text((5, y), line, fill=color, font=font)
        y += line_heights[i] + 4
    return img


def save_debug_triplets(images, labels, indices, embeddings, dataset, save_path,
                        max_rows=32, show_correctness=False):
    """
    Mine batch-hard triplets and save visualization.
    Each row: anchor | hardest positive | hardest negative (horizontal concat).
    All rows stacked vertically. Capped at max_rows to stay within JPEG limits.
    When show_correctness=True, each row is labeled CORRECT/WRONG based on d_ap < d_an.
    """
    B = embeddings.size(0)
    # Pairwise euclidean distance
    xx = (embeddings ** 2).sum(1, keepdim=True).expand(B, B)
    dist_mat = (xx + xx.t() - 2 * embeddings.mm(embeddings.t())).clamp(min=1e-12).sqrt()

    labels_row = labels.view(B, 1)
    pos_mask = (labels_row == labels.view(1, B)) & ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
    neg_mask = labels_row != labels.view(1, B)

    # Random positive per anchor
    pos_weights = pos_mask.float()
    pos_weights = pos_weights / pos_weights.sum(dim=1, keepdim=True).clamp(min=1)
    rp_idx = torch.multinomial(pos_weights, 1).squeeze(1)

    # Hardest negative per anchor
    dist_an_mat = dist_mat.clone()
    dist_an_mat[~neg_mask] = float('inf')
    hn_idx = dist_an_mat.argmin(dim=1)

    d_ap = dist_mat[torch.arange(B, device=embeddings.device), rp_idx]
    d_an = dist_mat[torch.arange(B, device=embeddings.device), hn_idx]

    num_rows = min(B, max_rows)
    correct_count = 0
    rows = []
    for i in range(num_rows):
        a_img = _tensor_to_pil(images[i])
        p_img = _tensor_to_pil(images[rp_idx[i]])
        n_img = _tensor_to_pil(images[hn_idx[i]])

        a_name = Path(dataset.samples[indices[i].item()][0]).parent.name
        p_name = Path(dataset.samples[indices[rp_idx[i]].item()][0]).parent.name
        n_name = Path(dataset.samples[indices[hn_idx[i]].item()][0]).parent.name

        is_correct = d_ap[i] < d_an[i]
        if is_correct:
            correct_count += 1

        if show_correctness:
            tag = "OK" if is_correct else "WRONG"
            tag_color = (0, 255, 0) if is_correct else (255, 50, 50)
            a_img = _add_label(a_img, f"{tag} [{a_name}]\nap={d_ap[i]:.2f} an={d_an[i]:.2f} gap={d_an[i]-d_ap[i]:+.2f}", tag_color)
        else:
            a_img = _add_label(a_img, f"Anchor [{a_name}]", (0, 255, 0))
        p_img = _add_label(p_img, f"Pos [{p_name}] d={d_ap[i]:.3f}", (0, 200, 255))
        n_img = _add_label(n_img, f"Neg [{n_name}] d={d_an[i]:.3f}", (255, 80, 80))

        row = Image.new("RGB", (a_img.width * 3, a_img.height))
        row.paste(a_img, (0, 0))
        row.paste(p_img, (a_img.width, 0))
        row.paste(n_img, (a_img.width * 2, 0))
        rows.append(row)

    # Add summary header when showing correctness
    if show_correctness:
        acc = correct_count / num_rows * 100
        header_h = 30
        header = Image.new("RGB", (rows[0].width, header_h), (0, 0, 0))
        draw = ImageDraw.Draw(header)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 4), f"Accuracy: {correct_count}/{num_rows} ({acc:.1f}%)", fill=(255, 255, 0), font=font)
        rows.insert(0, header)

    canvas = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path, quality=90)
    print(f"  [Debug] Saved triplet viz: {save_path} ({num_rows} triplets)")


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, arcface_weight=1.0, rank=0,
                debug_save_path=None, dataset=None):
    model.train()
    total_loss = triplet_total = arcface_total = 0.0
    total_dist_ap = total_dist_an = 0.0

    # Access underlying model through DDP wrapper
    model_module = model.module if hasattr(model, 'module') else model

    for batch_idx, (images, labels, indices) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(images)
        triplet_loss, dist_ap, dist_an = criterion(embeddings, labels)
        loss = triplet_loss

        if model_module.use_arcface:
            logits = model_module.get_arcface_logits(embeddings, labels)
            arc_loss = F.cross_entropy(logits, labels)
            loss = triplet_loss + arcface_weight * arc_loss
            arcface_total += arc_loss.item()

        # Save debug triplet visualization on first batch
        if debug_save_path is not None and batch_idx == 0 and rank == 0 and dataset is not None:
            with torch.no_grad():
                save_debug_triplets(images, labels, indices, embeddings.detach(),
                                    dataset, debug_save_path)

        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        triplet_total += triplet_loss.item()
        total_dist_ap += dist_ap
        total_dist_an += dist_an

        if rank == 0 and (batch_idx + 1) % 10 == 0:
            msg = (f"  Batch {batch_idx + 1}/{len(dataloader)}  "
                   f"Loss: {loss.item():.4f}  "
                   f"Triplet: {triplet_loss.item():.4f}  "
                   f"d_ap: {dist_ap:.3f}  d_an: {dist_an:.3f}")
            if model_module.use_arcface:
                msg += f"  ArcFace: {arc_loss.item():.4f}"
            print(msg)

    n = max(len(dataloader), 1)
    return total_loss/n, triplet_total/n, arcface_total/n, total_dist_ap/n, total_dist_an/n


def evaluate(model, dataloader, criterion, device,
             debug_save_path=None, dataset=None):
    """Validation: batch-hard triplet loss + mean d_ap / d_an for monitoring."""
    model.eval()
    total_loss = total_dist_ap = total_dist_an = 0.0
    debug_saved = False

    with torch.no_grad():
        for images, labels, indices in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            loss, dist_ap, dist_an = criterion(embeddings, labels)
            total_loss    += loss.item()
            total_dist_ap += dist_ap
            total_dist_an += dist_an

            # Save debug viz from first val batch
            if debug_save_path and dataset and not debug_saved:
                save_debug_triplets(images, labels, indices, embeddings,
                                    dataset, debug_save_path, max_rows=32,
                                    show_correctness=True)
                debug_saved = True

    n = max(len(dataloader), 1)
    return total_loss/n, total_dist_ap/n, total_dist_an/n


# =============================================================================
# Main Worker (one per GPU)
# =============================================================================

def main_worker(rank, world_size):
    cfg = Config()
    setup_ddp(rank, world_size)
    gpu_id = rank + cfg.GPU_OFFSET
    device = torch.device(f"cuda:{gpu_id}")

    # Logging — only rank 0 prints and writes log file
    if rank == 0:
        os.makedirs("./logs", exist_ok=True)
        log_path = f"./logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_path, "w")

        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, msg):
                for f in self.files:
                    f.write(msg)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        sys.stdout = Tee(sys.__stdout__, log_file)

        # Wandb — resume existing run if checkpoint has a run id
        _wandb_run_id = None
        if cfg.RESUME and os.path.isfile(cfg.RESUME_CKPT):
            _ckpt_meta = torch.load(cfg.RESUME_CKPT, map_location="cpu", weights_only=False)
            _wandb_run_id = _ckpt_meta.get("wandb_run_id", None)

        wandb.init(
            project=cfg.WANDB_PROJECT,
            name=cfg.WANDB_RUN_NAME or cfg.EXPERIMENT,
            id=_wandb_run_id,
            resume="must" if _wandb_run_id else None,
            config={
                "experiment": cfg.EXPERIMENT,
                "experiment_desc": cfg.EXPERIMENT_DESC,
                "backbone": cfg.BACKBONE,
                "embedding_dim": cfg.EMBEDDING_DIM,
                "num_parts": cfg.NUM_PARTS,
                "part_dim": cfg.PART_DIM,
                "model": "PCB",
                "P": cfg.P,
                "K": cfg.K,
                "num_epochs": cfg.NUM_EPOCHS,
                "learning_rate": cfg.LEARNING_RATE,
                "margin": cfg.MARGIN,
                "use_arcface": cfg.USE_ARCFACE,
                "arcface_s": cfg.ARCFACE_S,
                "arcface_m": cfg.ARCFACE_M,
                "arcface_weight": cfg.ARCFACE_WEIGHT,
                "freeze_backbone_epochs": cfg.FREEZE_BACKBONE_EPOCHS,
                "hard_aug_epoch": cfg.HARD_AUG_EPOCH,
                "warmup_epochs": cfg.WARMUP_EPOCHS,
                "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
                "val_ratio": cfg.VAL_RATIO,
                "mining": "random-pos-hardest-neg",
                "img_size": cfg.IMG_SIZE,
                "world_size": world_size,
                "split_json": cfg.SPLIT_JSON,
            },
        )

        print(f"DDP Training — {world_size} GPUs")
        print(f"Backbone: {cfg.BACKBONE}")
        print(f"P={cfg.P} identities × K={cfg.K} images per GPU")

    # Curriculum augmentation: mild early, moderate later
    mild_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),   # mild out-of-focus blur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    hard_transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.85, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        RandomMotionBlur(p=0.5, kernel_size=15, angle_range=(0, 180)),  # camera pan blur
        transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)),       # out-of-focus blur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    train_transform = hard_transform  # start mild

    val_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets — flat directory, split identities by ratio
    all_ids_dataset = HorseReIDDataset(cfg.DATA_ROOT, cfg.EXCLUDE_FOLDERS, transform=None)
    all_ids = sorted(all_ids_dataset.identity_list)
    random.seed(42)
    shuffled_ids = all_ids.copy()
    random.shuffle(shuffled_ids)
    n_val = max(1, int(len(shuffled_ids) * cfg.VAL_RATIO))
    val_ids = sorted(shuffled_ids[:n_val])
    train_ids = sorted(shuffled_ids[n_val:])

    train_dataset = HorseReIDDataset(cfg.DATA_ROOT, cfg.EXCLUDE_FOLDERS,
                                     identity_list=train_ids, transform=train_transform)
    val_dataset   = HorseReIDDataset(cfg.DATA_ROOT, cfg.EXCLUDE_FOLDERS,
                                     identity_list=val_ids, transform=val_transform)

    if rank == 0:
        print(f"Train: {train_dataset.num_ids} identities, {len(train_dataset)} images")
        print(f"Val:   {val_dataset.num_ids} identities, {len(val_dataset)} images")

    # Distributed PK Samplers
    train_sampler = DistributedPKSampler(
        train_dataset.label_to_indices, P=cfg.P, K=cfg.K,
        rank=rank, world_size=world_size
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.P * cfg.K,
        sampler=train_sampler, num_workers=2, drop_last=True, pin_memory=True
    )

    # Val — uses DistributedPKSampler (shared mode handles small identity counts)
    val_P = min(cfg.P, val_dataset.num_ids)
    val_sampler = DistributedPKSampler(
        val_dataset.label_to_indices, P=val_P, K=cfg.K,
        rank=rank, world_size=world_size
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_P * cfg.K,
        sampler=val_sampler, num_workers=2, drop_last=True, pin_memory=True
    )

    # Model
    model = HorseReIDModelPCB(
        backbone_name=cfg.BACKBONE,
        num_parts=cfg.NUM_PARTS,
        part_dim=cfg.PART_DIM,
        pretrained=cfg.PRETRAINED,
        num_ids=train_dataset.num_ids,
        use_arcface=cfg.USE_ARCFACE,
        arcface_s=cfg.ARCFACE_S,
        arcface_m=cfg.ARCFACE_M,
    ).to(device)

    # Freeze backbone initially
    backbone_frozen = True
    model.freeze_backbone()
    if rank == 0:
        print("Backbone FROZEN")

    # Wrap with DDP
    model = DDP(model, device_ids=[gpu_id])

    # Loss & Optimizer
    criterion = RandomPosHardNegTripletLoss(margin=cfg.MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)

    # LR schedule: linear warmup → constant during frozen → cosine after unfreeze
    def lr_lambda(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return max((epoch + 1) / cfg.WARMUP_EPOCHS, 0.01)
        return 1.0  # constant LR during frozen phase

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = None  # created on unfreeze

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    if cfg.RESUME and os.path.isfile(cfg.RESUME_CKPT):
        map_location = {'cuda:0': f'cuda:{gpu_id}'}
        ckpt = torch.load(cfg.RESUME_CKPT, map_location=map_location, weights_only=False)
        model.module.load_state_dict(ckpt['model_state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        if rank == 0:
            print(f"Resumed from epoch {ckpt['epoch'] + 1}  (best val loss: {best_val_loss:.4f})")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        # Set epoch for proper shuffling
        train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        # Curriculum augmentation: switch to hard transforms
        if epoch == cfg.HARD_AUG_EPOCH:
            train_dataset.transform = hard_transform
            if rank == 0:
                print("Augmentation switched: MILD → HARD")

        # Gradual unfreezing: unfreeze backbone after warmup epochs
        if backbone_frozen and epoch >= cfg.FREEZE_BACKBONE_EPOCHS:
            model.module.unfreeze_backbone()
            backbone_frozen = False
            if rank == 0:
                print("Backbone UNFROZEN")

            # Reset early stopping and best val on unfreeze
            epochs_without_improvement = 0
            best_val_loss = float('inf')

            # Re-create optimizer with differential LR for backbone
            param_groups = [
                {'params': model.module.backbone.parameters(), 'lr': cfg.LEARNING_RATE * 0.1},
                {'params': model.module.part_heads.parameters()},
            ]
            if model.module.use_arcface:
                param_groups.append({'params': model.module.arcface.parameters()})
            optimizer = torch.optim.AdamW(param_groups, lr=cfg.LEARNING_RATE, weight_decay=1e-4)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=30, T_mult=1, eta_min=1e-6
            )

        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{cfg.NUM_EPOCHS}  (lr={current_lr:.2e})")
            print("-" * 40)

        # Save debug triplet visualizations every 10 epochs (10 total across 100 epochs)
        debug_path = None
        val_debug_path = None
        if rank == 0 and epoch % 10 == 0:
            debug_path = f"./debug_samples/triplet_batches/{cfg.EXPERIMENT}/epoch_{epoch:03d}.jpg"
            val_debug_path = f"./debug_samples/triplet_batches/{cfg.EXPERIMENT}/val_epoch_{epoch:03d}.jpg"

        train_loss, t_loss, arc_loss, d_ap, d_an = train_epoch(
            model, train_loader, criterion, optimizer, device, cfg.ARCFACE_WEIGHT, rank,
            debug_save_path=debug_path, dataset=train_dataset
        )

        # Validation
        val_loss, v_d_ap, v_d_an = evaluate(
            model, val_loader, criterion, device,
            debug_save_path=val_debug_path, dataset=val_dataset
        )

        # Step appropriate scheduler
        if epoch < cfg.WARMUP_EPOCHS:
            warmup_scheduler.step()
        elif cosine_scheduler is not None:
            cosine_scheduler.step()

        # Synchronize val loss across ranks for consistent best-model tracking
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss_avg = val_loss_tensor.item()

        if rank == 0:
            print(f"[Train] Loss: {train_loss:.4f}  Triplet: {t_loss:.4f}  "
                  f"d_ap: {d_ap:.3f}  d_an: {d_an:.3f}  gap: {d_an - d_ap:+.3f}")
            print(f"[Val]   Triplet: {val_loss_avg:.4f}  d_ap: {v_d_ap:.3f}  d_an: {v_d_an:.3f}  gap: {v_d_an - v_d_ap:+.3f}")

            log_dict = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train/loss": train_loss,
                "train/triplet_loss": t_loss,
                "train/arcface_loss": arc_loss,
                "train/d_ap": d_ap,
                "train/d_an": d_an,
                "train/d_margin": d_an - d_ap,
                "val/triplet_loss": val_loss_avg,
                "val/d_ap": v_d_ap,
                "val/d_an": v_d_an,
                "val/d_margin": v_d_an - v_d_ap,
                "backbone_frozen": backbone_frozen,
                "hard_aug_active": epoch >= cfg.HARD_AUG_EPOCH,
            }
            if debug_path is not None and os.path.exists(debug_path):
                log_dict["debug/triplets"] = wandb.Image(
                    debug_path, caption=f"Epoch {epoch + 1} — Train: Anchor | Rand Pos | Hard Neg"
                )
            if val_debug_path is not None and os.path.exists(val_debug_path):
                log_dict["debug/val_triplets"] = wandb.Image(
                    val_debug_path, caption=f"Epoch {epoch + 1} — Val: Anchor | Rand Pos | Hard Neg"
                )
            wandb.log(log_dict)

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                epochs_without_improvement = 0
                ckpt_name = f"best_horse_reid_{cfg.EXPERIMENT}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_avg,
                    'experiment': cfg.EXPERIMENT,
                    'experiment_desc': cfg.EXPERIMENT_DESC,
                    'margin': cfg.MARGIN,
                    'arcface_weight': cfg.ARCFACE_WEIGHT,
                    'hard_aug_epoch': cfg.HARD_AUG_EPOCH,
                    'split_json': cfg.SPLIT_JSON,
                    'train_ids': train_dataset.identity_list,
                    'val_ids': val_dataset.identity_list,
                    'wandb_run_id': wandb.run.id,
                }, os.path.join(cfg.SAVE_DIR, ckpt_name))
                print(f"  → Saved best model ({ckpt_name})")
            else:
                # Only count early stopping after backbone is unfrozen
                if not backbone_frozen:
                    epochs_without_improvement += 1
                # if epochs_without_improvement >= cfg.EARLY_STOP_PATIENCE:
                #     print(f"  → Early stopping: no improvement for {cfg.EARLY_STOP_PATIENCE} epochs after unfreeze")

        # Broadcast early stop decision from rank 0
        stop_flag = torch.tensor(
            1 if rank == 0 and epochs_without_improvement >= cfg.EARLY_STOP_PATIENCE else 0,
            device=device
        )
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

        # Wait for rank 0 to finish saving before next epoch
        dist.barrier()

    if rank == 0:
        print("\nTraining complete!")
        print(f"Best val triplet loss: {best_val_loss:.4f}")
        wandb.finish()

    cleanup_ddp()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    world_size = Config.WORLD_SIZE
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
