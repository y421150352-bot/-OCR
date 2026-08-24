"""DINOv3 dual-branch character embedding network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(nn.Linear(input_dim, output_dim), nn.GELU(), nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim))


class CharacterReIDModel(nn.Module):
    def __init__(self, backbone_path: str, embedding_dim: int = 512, freeze_backbone: bool = True):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError("Install requirements.txt; transformers with DINOv3 support is required") from error
        self.backbone = AutoModel.from_pretrained(backbone_path, local_files_only=True)
        hidden = int(self.backbone.config.hidden_size)
        self.face_head = ProjectionHead(hidden, embedding_dim)
        self.body_head = ProjectionHead(hidden, embedding_dim)
        self.fusion = nn.Sequential(nn.Linear(embedding_dim * 2 + 2, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim))
        self.gate = nn.Sequential(nn.Linear(embedding_dim * 2 + 2, embedding_dim), nn.Sigmoid())
        self.embedding_dim = embedding_dim
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen backbone must also keep stochastic-depth/dropout disabled.
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def encode_valid(self, pixels: torch.Tensor, valid: torch.Tensor, head: nn.Module) -> torch.Tensor:
        output = pixels.new_zeros((pixels.shape[0], self.embedding_dim))
        if valid.any():
            features = self.backbone(pixel_values=pixels[valid]).last_hidden_state[:, 0]
            # Under CUDA autocast the head returns bfloat16 while the image
            # batch (and this zero-filled container) remains float32. Indexed
            # assignment requires identical dtypes. Keep branch embeddings in
            # float32 for stable metric losses; the cast remains differentiable.
            output[valid] = head(features).to(output.dtype)
        return output

    def forward(self, face: torch.Tensor, body: torch.Tensor, face_valid: torch.Tensor, body_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        face_valid, body_valid = face_valid.bool(), body_valid.bool()
        face_feature = self.encode_valid(face, face_valid, self.face_head)
        body_feature = self.encode_valid(body, body_valid, self.body_head)
        masks = torch.stack((face_valid, body_valid), dim=1).to(face.dtype)
        joined = torch.cat((face_feature, body_feature, masks), dim=1)
        gate = self.gate(joined)
        blended = gate * face_feature + (1 - gate) * body_feature
        both = (face_valid & body_valid).unsqueeze(1)
        single = torch.where(face_valid.unsqueeze(1), face_feature, body_feature)
        embedding = torch.where(both, self.fusion(joined) + blended, single)
        return {"embedding": F.normalize(embedding, dim=1), "face_feature": face_feature, "body_feature": body_feature, "gate": gate}
