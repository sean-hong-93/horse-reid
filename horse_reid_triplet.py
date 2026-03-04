"""
Horse Re-ID using Triplet Loss with MobileNetV4 Small backbone
"""

import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
import timm


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Data
    DATA_ROOT = "/home/jovyan/deploy-rnd/dev8/root/storage/datasets/horse_feature_extractor/train"
    VAL_DIR   = "/home/jovyan/deploy-rnd/dev8/root/storage/datasets/horse_feature_extractor/val"
    EXCLUDE_FOLDERS = []

    # Model
    BACKBONE = "mobilenetv4_conv_small"
    EMBEDDING_DIM = 128
    PRETRAINED = True

    # Training — PK sampling: P identities × K images per batch
    P = 12              # Identities per batch  (12//4=3 batches/epoch)
    K = 8              # Images per identity  →  effective batch size = P*K = 32
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4

    # Resume
    RESUME = False
    RESUME_CKPT = "./checkpoints/best_horse_reid.pth"
    MARGIN = 0.3       # Batch-hard triplet margin

    # ArcFace
    USE_ARCFACE = True
    ARCFACE_S = 64.0   # Feature scale
    ARCFACE_M = 0.5    # Angular margin (~28.6°)
    ARCFACE_WEIGHT = 0.1  # Weight of ArcFace loss relative to Triplet loss

    # Gradual unfreezing
    FREEZE_BACKBONE_EPOCHS = 20  # Freeze backbone for first N epochs

    # Image
    IMG_SIZE = (224, 224)

    # Device
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Save
    SAVE_DIR = "./checkpoints"


# =============================================================================
# Dataset
# =============================================================================

class HorseReIDDataset(Dataset):
    """
    Flat (image, label) dataset for Batch Hard Triplet + ArcFace training.

    Returns individual (image, label) pairs — no pre-formed triplets.
    Triplets are mined online inside BatchHardTripletLoss using the full
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
            # images = [p for p in list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            #           if not p.name.startswith("._") and p.stat().st_size >= 4096]
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

        print(f"Dataset: {len(self.identity_list)} identities, {len(self.samples)} images")

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
                # Corrupted file — pick a random valid sample with the same label
                alt_idx = random.choice(self.label_to_indices[label])
                img_path, label = self.samples[alt_idx]
        if img is None:
            raise RuntimeError(f"Failed to load any valid image for label {label} after 20 retries")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def labels(self):
        return [label for _, label in self.samples]

    @property
    def num_ids(self):
        return len(self.identity_list)


# =============================================================================
# PK Sampler
# =============================================================================

class PKSampler(Sampler):
    """
    Samples P identities × K images per batch (PK sampling).

    Yields indices in contiguous PK blocks so that DataLoader(batch_size=P*K)
    produces batches with exactly P distinct identities and K images each.
    This is required for effective Batch Hard mining.

    Identities with fewer than K images are still included — images are
    sampled with replacement to fill K slots.
    """

    def __init__(self, label_to_indices: Dict[int, List[int]], P: int, K: int):
        self.P = P
        self.K = K
        self.label_to_indices = label_to_indices
        self.labels = list(label_to_indices.keys())

        if len(self.labels) < P:
            raise ValueError(
                f"PKSampler needs at least P={P} identities, got {len(self.labels)}"
            )

    def __iter__(self):
        labels = self.labels.copy()
        random.shuffle(labels)

        for start in range(0, len(labels) - self.P + 1, self.P):
            batch_labels = labels[start: start + self.P]
            for label in batch_labels:
                idxs = self.label_to_indices[label].copy()
                random.shuffle(idxs)
                # Sample with replacement if fewer than K images
                selected = idxs[:self.K] if len(idxs) >= self.K else random.choices(idxs, k=self.K)
                yield from selected

    def __len__(self):
        return (len(self.labels) // self.P) * self.P * self.K


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
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )

        # Get backbone output dimension (eval mode needed: BN fails with batch=1 in train mode)
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[-1]
            self.backbone.train()

        print(f"Backbone output dim: {backbone_dim}")

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
            print(f"ArcFace head: num_ids={num_ids}, s={arcface_s}, m={arcface_m}")

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
        print("Backbone FROZEN")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone UNFROZEN")


# =============================================================================
# ArcFace Head
# =============================================================================

class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss head.

    Replaces standard linear classifier by normalizing both features and
    weights, then adding angular margin m to the target class logit:
        cos(θ + m)  instead of  cos(θ)

    This enforces a minimum angular gap between every pair of horse identities,
    producing much tighter and more separable clusters in embedding space.

    Args:
        feat_dim   : dimension of input embeddings (must be L2-normalized)
        num_classes: number of horse identities
        s          : feature scale (default 64.0)
        m          : angular margin in radians (default 0.5 ≈ 28.6°)

    Ref: Deng et al., "ArcFace", CVPR 2019
    """
    def __init__(self, feat_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)  # numerical stability guard
        self.mm = math.sin(math.pi - m) * m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # x is already L2-normalized from the embedding head
        cosine = F.linear(x, F.normalize(self.weight, p=2, dim=1))  # [B, num_classes]

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi  = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        # Numerical guard for large θ
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # Apply margin only to the ground-truth class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine

        return self.s * output


# =============================================================================
# Loss Functions
# =============================================================================

class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss (Hermans et al., "In Defense of the Triplet Loss", 2017).

    For each anchor in the batch:
      - hardest positive : same-identity sample with the LARGEST distance
      - hardest negative : different-identity sample with the SMALLEST distance

    loss = mean( max(0, d_ap - d_an + margin) )

    Requires PK-sampled batches (P identities × K images) so that every anchor
    has valid positives and negatives in the batch. This prevents the easy-triplet
    collapse seen with random triplet sampling.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        dist_mat = self._pairwise_euclidean(embeddings)  # [B, B]

        B = embeddings.size(0)
        labels_row = labels.view(B, 1)

        pos_mask = (labels_row == labels.view(1, B)) & \
                   ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        neg_mask = labels_row != labels.view(1, B)

        # Hardest positive: largest distance among same-identity pairs
        dist_ap = (dist_mat * pos_mask.float()).max(dim=1)[0]

        # Hardest negative: smallest distance among different-identity pairs
        dist_an = dist_mat.clone()
        dist_an[~neg_mask] = float('inf')
        dist_an = dist_an.min(dim=1)[0]

        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        return loss, dist_ap.mean().item(), dist_an.mean().item()

    @staticmethod
    def _pairwise_euclidean(x: torch.Tensor) -> torch.Tensor:
        m = x.size(0)
        xx = (x ** 2).sum(1, keepdim=True).expand(m, m)
        dist = xx + xx.t() - 2 * x.mm(x.t())
        return dist.clamp(min=1e-12).sqrt()


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, arcface_weight=1.0):
    model.train()
    total_loss = triplet_total = arcface_total = 0.0
    total_dist_ap = total_dist_an = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(images)
        triplet_loss, dist_ap, dist_an = criterion(embeddings, labels)
        loss = triplet_loss

        if model.use_arcface:
            logits = model.get_arcface_logits(embeddings, labels)
            arc_loss = F.cross_entropy(logits, labels)
            loss = triplet_loss + arcface_weight * arc_loss
            arcface_total += arc_loss.item()

        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        triplet_total += triplet_loss.item()
        total_dist_ap += dist_ap
        total_dist_an += dist_an

        if (batch_idx + 1) % 10 == 0:
            msg = (f"  Batch {batch_idx + 1}/{len(dataloader)}  "
                   f"Loss: {loss.item():.4f}  "
                   f"Triplet: {triplet_loss.item():.4f}  "
                   f"d_ap: {dist_ap:.3f}  d_an: {dist_an:.3f}")
            if model.use_arcface:
                msg += f"  ArcFace: {arc_loss.item():.4f}"
            print(msg)

    n = len(dataloader)
    return total_loss/n, triplet_total/n, arcface_total/n, total_dist_ap/n, total_dist_an/n


def evaluate(model, dataloader, criterion, device):
    """Validation: batch-hard triplet loss + mean d_ap / d_an for monitoring."""
    model.eval()
    total_loss = total_dist_ap = total_dist_an = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            loss, dist_ap, dist_an = criterion(embeddings, labels)
            total_loss    += loss.item()
            total_dist_ap += dist_ap
            total_dist_an += dist_an

    n = len(dataloader)
    return total_loss/n, total_dist_ap/n, total_dist_an/n


def main():
    cfg = Config()

    # Setup logging to file
    os.makedirs("./logs", exist_ok=True)
    log_path = f"./logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w")

    class Tee:
        """Write to both stdout and log file."""
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

    print(f"Device: {cfg.DEVICE}")
    print(f"Backbone: {cfg.BACKBONE}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Use the pre-split train/val directories from the dataset
    train_dataset = HorseReIDDataset(cfg.DATA_ROOT, cfg.EXCLUDE_FOLDERS,
                                     transform=train_transform)
    val_dataset   = HorseReIDDataset(cfg.VAL_DIR,   cfg.EXCLUDE_FOLDERS,
                                     transform=val_transform)

    # PKSampler: guarantees P identities × K images per batch for hard mining
    pk_sampler = PKSampler(train_dataset.label_to_indices, P=cfg.P, K=cfg.K)
    train_loader = DataLoader(train_dataset, batch_size=cfg.P * cfg.K,
                              sampler=pk_sampler, num_workers=4, drop_last=True)

    # Val uses PKSampler too so the loss is computed on properly structured batches
    val_pk_sampler = PKSampler(val_dataset.label_to_indices, P=min(cfg.P, val_dataset.num_ids), K=cfg.K)
    val_loader = DataLoader(val_dataset, batch_size=min(cfg.P, val_dataset.num_ids) * cfg.K,
                            sampler=val_pk_sampler, num_workers=4, drop_last=True)

    # Model
    model = HorseReIDModel(
        backbone_name=cfg.BACKBONE,
        embedding_dim=cfg.EMBEDDING_DIM,
        pretrained=cfg.PRETRAINED,
        num_ids=train_dataset.num_ids,
        use_arcface=cfg.USE_ARCFACE,
        arcface_s=cfg.ARCFACE_S,
        arcface_m=cfg.ARCFACE_M,
    ).to(cfg.DEVICE)

    # Freeze backbone initially for gradual unfreezing
    backbone_frozen = True
    model.freeze_backbone()

    # Loss & Optimizer
    criterion = BatchHardTripletLoss(margin=cfg.MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    # Warm restarts: LR resets every T_0 epochs — keeps escaping local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=1, eta_min=1e-6
    )

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if cfg.RESUME and os.path.isfile(cfg.RESUME_CKPT):
        ckpt = torch.load(cfg.RESUME_CKPT, map_location=cfg.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        print(f"Resumed from epoch {ckpt['epoch'] + 1}  (best val loss so far: {best_val_loss:.4f})")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        # Gradual unfreezing: unfreeze backbone after warmup epochs
        if backbone_frozen and epoch >= cfg.FREEZE_BACKBONE_EPOCHS:
            model.unfreeze_backbone()
            backbone_frozen = False
            # Re-create optimizer to include backbone params with a lower LR
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': cfg.LEARNING_RATE * 0.1},
                {'params': model.embedding.parameters()},
                {'params': model.arcface.parameters()},
            ], lr=cfg.LEARNING_RATE, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=30, T_mult=1, eta_min=1e-6
            )
        print(f"\nEpoch {epoch + 1}/{cfg.NUM_EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})")
        print("-" * 40)

        train_loss, t_loss, arc_loss, d_ap, d_an = train_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE, cfg.ARCFACE_WEIGHT
        )
        val_loss, v_d_ap, v_d_an = evaluate(model, val_loader, criterion, cfg.DEVICE)

        scheduler.step()

        print(f"[Train] Loss: {train_loss:.4f}  Triplet: {t_loss:.4f}  "
              f"ArcFace: {arc_loss:.4f}  d_ap: {d_ap:.3f}  d_an: {d_an:.3f}")
        print(f"[Val]   Triplet: {val_loss:.4f}  d_ap: {v_d_ap:.3f}  d_an: {v_d_an:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_ids': train_dataset.identity_list,
                'val_ids': val_dataset.identity_list,
            }, os.path.join(cfg.SAVE_DIR, 'best_horse_reid.pth'))
            print(f"  → Saved best model")

    print("\nTraining complete!")
    print(f"Best val triplet loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
