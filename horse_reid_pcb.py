"""
Horse Re-ID with PCB (Part-based Convolutional Baseline).

Instead of a single global embedding, the backbone feature map is split into
horizontal strips (parts), each producing its own embedding. The final
embedding is the L2-normalized concatenation of all part embeddings.

For side-view horses:
  - Top strip:    back, mane, saddle
  - Middle strip: torso, markings, blanket
  - Bottom strip: legs, hooves, boots

Compatible with the same dataset, sampler, loss, and training loop as v4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HorseReIDModelPCB(nn.Module):
    """
    PCB Re-ID model: part-based feature extraction with MobileNetV4 backbone.

    Architecture:
        backbone → [B, C, H, W]
        adaptive_avg_pool2d → [B, C, num_parts, 1]
        split into num_parts × [B, C]
        each part → FC head → [B, part_dim]
        concat → [B, num_parts * part_dim]
        L2 normalize → final embedding

    When use_arcface=True, ArcFace head operates on the final concatenated embedding.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small",
        num_parts: int = 3,
        part_dim: int = 512,
        pretrained: bool = True,
        num_ids: int = None,
        use_arcface: bool = True,
        arcface_s: float = 64.0,
        arcface_m: float = 0.5,
    ):
        super().__init__()
        self.num_parts = num_parts
        self.part_dim = part_dim
        self.embedding_dim = num_parts * part_dim
        self.use_arcface = use_arcface and (num_ids is not None)

        # Backbone — no global pool, keep spatial feature map
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )

        # Get backbone channel dimension
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)
            backbone_channels = feat.shape[1]
            self.backbone.train()

        # Per-part embedding heads
        self.part_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_channels, part_dim),
                nn.BatchNorm1d(part_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(part_dim, part_dim),
            )
            for _ in range(num_parts)
        ])

        # ArcFace head (training only)
        if self.use_arcface:
            from horse_reid_triplet_ddp_v4 import ArcFaceHead
            self.arcface = ArcFaceHead(self.embedding_dim, num_ids, s=arcface_s, m=arcface_m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized concatenated part embeddings."""
        # [B, C, H, W] → [B, C, num_parts, 1]
        feat_map = self.backbone(x)
        parts = F.adaptive_avg_pool2d(feat_map, (self.num_parts, 1))  # [B, C, P, 1]
        parts = parts.squeeze(-1)  # [B, C, P]

        # Each part through its own head
        part_embeddings = []
        for i in range(self.num_parts):
            part_feat = parts[:, :, i]  # [B, C]
            part_emb = self.part_heads[i](part_feat)  # [B, part_dim]
            part_embeddings.append(part_emb)

        # Concat and normalize
        embeddings = torch.cat(part_embeddings, dim=1)  # [B, num_parts * part_dim]
        return F.normalize(embeddings, p=2, dim=1)

    def get_arcface_logits(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.arcface(embeddings, labels)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Quick test
    model = HorseReIDModelPCB(
        backbone_name="mobilenetv4_conv_small",
        num_parts=3,
        part_dim=512,
        pretrained=True,
        num_ids=100,
        use_arcface=False,
    )

    x = torch.randn(4, 3, 224, 224)
    emb = model(x)
    print(f"Input:     {x.shape}")
    print(f"Output:    {emb.shape}")
    print(f"Embedding: {model.embedding_dim} (= {model.num_parts} parts × {model.part_dim} dim)")
    print(f"L2 norm:   {emb.norm(dim=1)}")

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params:    {total:,} total, {trainable:,} trainable")
