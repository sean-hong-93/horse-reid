"""
Horse Re-ID using Triplet Loss with MobileNetV4 Small backbone
DistributedDataParallel (DDP) version — uses all available GPUs.

Launch:
    torchrun --nproc_per_node=8 horse_reid_triplet_ddp.py
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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
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
    P = 12              # Identities per batch per GPU
    K = 8               # Images per identity  →  effective batch size per GPU = P*K = 96
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

    # DDP
    WORLD_SIZE = 8
    MASTER_PORT = "29500"

    # Save
    SAVE_DIR = "./checkpoints"


# =============================================================================
# DDP Setup
# =============================================================================

def setup_ddp(rank, world_size):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = Config.MASTER_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


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

class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss (Hermans et al., 2017).

    For each anchor in the batch:
      - hardest positive : same-identity sample with the LARGEST distance
      - hardest negative : different-identity sample with the SMALLEST distance

    loss = mean( max(0, d_ap - d_an + margin) )
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        dist_mat = self._pairwise_euclidean(embeddings)

        B = embeddings.size(0)
        labels_row = labels.view(B, 1)

        pos_mask = (labels_row == labels.view(1, B)) & \
                   ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        neg_mask = labels_row != labels.view(1, B)

        dist_ap = (dist_mat * pos_mask.float()).max(dim=1)[0]

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

def train_epoch(model, dataloader, criterion, optimizer, device, arcface_weight=1.0, rank=0):
    model.train()
    total_loss = triplet_total = arcface_total = 0.0
    total_dist_ap = total_dist_an = 0.0

    # Access underlying model through DDP wrapper
    model_module = model.module if hasattr(model, 'module') else model

    for batch_idx, (images, labels) in enumerate(dataloader):
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

    n = max(len(dataloader), 1)
    return total_loss/n, total_dist_ap/n, total_dist_an/n


# =============================================================================
# Main Worker (one per GPU)
# =============================================================================

def main_worker(rank, world_size):
    cfg = Config()
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

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
        print(f"DDP Training — {world_size} GPUs")
        print(f"Backbone: {cfg.BACKBONE}")
        print(f"P={cfg.P} identities × K={cfg.K} images per GPU")

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

    # Datasets
    train_dataset = HorseReIDDataset(cfg.DATA_ROOT, cfg.EXCLUDE_FOLDERS,
                                     transform=train_transform)
    val_dataset   = HorseReIDDataset(cfg.VAL_DIR,   cfg.EXCLUDE_FOLDERS,
                                     transform=val_transform)

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
        sampler=train_sampler, num_workers=4, drop_last=True, pin_memory=True
    )

    # Val — uses DistributedPKSampler (shared mode handles small identity counts)
    val_P = min(cfg.P, val_dataset.num_ids)
    val_sampler = DistributedPKSampler(
        val_dataset.label_to_indices, P=val_P, K=cfg.K,
        rank=rank, world_size=world_size
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_P * cfg.K,
        sampler=val_sampler, num_workers=4, drop_last=True, pin_memory=True
    )

    # Model
    model = HorseReIDModel(
        backbone_name=cfg.BACKBONE,
        embedding_dim=cfg.EMBEDDING_DIM,
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
    model = DDP(model, device_ids=[rank])

    # Loss & Optimizer
    criterion = BatchHardTripletLoss(margin=cfg.MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=1, eta_min=1e-6
    )

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if cfg.RESUME and os.path.isfile(cfg.RESUME_CKPT):
        map_location = {'cuda:0': f'cuda:{rank}'}
        ckpt = torch.load(cfg.RESUME_CKPT, map_location=map_location, weights_only=False)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        if rank == 0:
            print(f"Resumed from epoch {ckpt['epoch'] + 1}  (best val loss: {best_val_loss:.4f})")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        # Set epoch for proper shuffling
        train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        # Gradual unfreezing: unfreeze backbone after warmup epochs
        if backbone_frozen and epoch >= cfg.FREEZE_BACKBONE_EPOCHS:
            model.module.unfreeze_backbone()
            backbone_frozen = False
            if rank == 0:
                print("Backbone UNFROZEN")

            # Re-create optimizer with differential LR for backbone
            optimizer = torch.optim.AdamW([
                {'params': model.module.backbone.parameters(), 'lr': cfg.LEARNING_RATE * 0.1},
                {'params': model.module.embedding.parameters()},
                {'params': model.module.arcface.parameters()},
            ], lr=cfg.LEARNING_RATE, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=30, T_mult=1, eta_min=1e-6
            )

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{cfg.NUM_EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})")
            print("-" * 40)

        train_loss, t_loss, arc_loss, d_ap, d_an = train_epoch(
            model, train_loader, criterion, optimizer, device, cfg.ARCFACE_WEIGHT, rank
        )

        # Validation
        val_loss, v_d_ap, v_d_an = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Synchronize val loss across ranks for consistent best-model tracking
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss_avg = val_loss_tensor.item()

        if rank == 0:
            print(f"[Train] Loss: {train_loss:.4f}  Triplet: {t_loss:.4f}  "
                  f"ArcFace: {arc_loss:.4f}  d_ap: {d_ap:.3f}  d_an: {d_an:.3f}")
            print(f"[Val]   Triplet: {val_loss_avg:.4f}  d_ap: {v_d_ap:.3f}  d_an: {v_d_an:.3f}")

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_avg,
                    'train_ids': train_dataset.identity_list,
                    'val_ids': val_dataset.identity_list,
                }, os.path.join(cfg.SAVE_DIR, 'best_horse_reid.pth'))
                print(f"  → Saved best model")

        # Wait for rank 0 to finish saving before next epoch
        dist.barrier()

    if rank == 0:
        print("\nTraining complete!")
        print(f"Best val triplet loss: {best_val_loss:.4f}")

    cleanup_ddp()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    world_size = Config.WORLD_SIZE
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
