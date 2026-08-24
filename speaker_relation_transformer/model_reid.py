"""Face/body metric-learning model for manga character identity."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )


class FaceBodyReID(nn.Module):
    """Create one identity embedding when face, body, or both are available."""

    def __init__(
        self,
        backbone_path: str,
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone_path = backbone_path
        self.backbone = AutoModel.from_pretrained(
            backbone_path, local_files_only=True
        )
        backbone_dim = int(self.backbone.config.hidden_size)
        self.embedding_dim = embedding_dim
        self.face_head = ProjectionHead(backbone_dim, embedding_dim)
        self.body_head = ProjectionHead(backbone_dim, embedding_dim)
        joined_dim = embedding_dim * 2 + 2
        self.fusion = nn.Sequential(
            nn.Linear(joined_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(joined_dim, embedding_dim), nn.Sigmoid()
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def train(self, mode: bool = True) -> "FaceBodyReID":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def _encode(
        self, pixels: torch.Tensor, valid: torch.Tensor, head: nn.Module
    ) -> torch.Tensor:
        result = pixels.new_zeros((pixels.shape[0], self.embedding_dim))
        if valid.any():
            tokens = self.backbone(pixel_values=pixels[valid]).last_hidden_state
            result[valid] = head(tokens[:, 0]).to(result.dtype)
        return result

    def forward(
        self,
        face: torch.Tensor,
        body: torch.Tensor,
        face_valid: torch.Tensor,
        body_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        face_valid = face_valid.bool()
        body_valid = body_valid.bool()
        if (~(face_valid | body_valid)).any():
            raise ValueError("Every ReID instance needs a face or body crop")
        face_feature = self._encode(face, face_valid, self.face_head)
        body_feature = self._encode(body, body_valid, self.body_head)
        masks = torch.stack((face_valid, body_valid), dim=1).to(face.dtype)
        joined = torch.cat((face_feature, body_feature, masks), dim=1)
        gate = self.gate(joined)
        blended = gate * face_feature + (1.0 - gate) * body_feature
        both = (face_valid & body_valid).unsqueeze(1)
        single = torch.where(face_valid.unsqueeze(1), face_feature, body_feature)
        embedding = torch.where(both, self.fusion(joined) + blended, single)
        return {
            "embedding": F.normalize(embedding, dim=1),
            "face_feature": face_feature,
            "body_feature": body_feature,
            "gate": gate,
        }


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss; each identity must occur at least twice."""
    similarities = embeddings @ embeddings.T / temperature
    similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positives.sum(dim=1)
    if not (positive_count > 0).all():
        raise ValueError("Use a P x K sampler with K >= 2")
    denominator = torch.exp(similarities).masked_fill(self_mask, 0.0).sum(dim=1)
    log_probability = similarities - denominator.clamp_min(1e-12).log().unsqueeze(1)
    return -((log_probability * positives).sum(dim=1) / positive_count).mean()


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    distance = 1.0 - embeddings @ embeddings.T
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    hardest_positive = distance.masked_fill(~same, float("-inf")).max(dim=1).values
    hardest_negative = distance.masked_fill(
        labels[:, None].eq(labels[None, :]), float("inf")
    ).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()
