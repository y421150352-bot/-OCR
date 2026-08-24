"""DINOv3 ROI fusion and listwise candidate relation Transformer."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import roi_align


class SpeakerRelationTransformer(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        geometry_dim: int = 45,
        hidden_dim: int = 512,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
        context_grid: int = 0,
        roi_size: int = 2,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.geometry_dim = geometry_dim
        self.hidden_dim = hidden_dim
        self.context_grid = context_grid
        self.roi_size = roi_size
        roi_dim = visual_dim * roi_size * roi_size
        self.context_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim)
        )
        self.dialogue_projection = nn.Sequential(
            nn.LayerNorm(roi_dim), nn.Linear(roi_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.character_projection = nn.Sequential(
            nn.LayerNorm(roi_dim), nn.Linear(roi_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)
        self.register_buffer("geometry_mean", torch.zeros(geometry_dim), persistent=True)
        self.register_buffer("geometry_std", torch.ones(geometry_dim), persistent=True)

    def set_geometry_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.geometry_dim or std.numel() != self.geometry_dim:
            raise ValueError("Geometry normalization dimension mismatch")
        self.geometry_mean.copy_(mean)
        self.geometry_std.copy_(std.clamp_min(1e-4))

    @staticmethod
    def _scale_boxes(
        boxes: torch.Tensor,
        original_hw: torch.Tensor,
        resized_hw: torch.Tensor,
        padded_hw: torch.Tensor,
        feature_hw: tuple[int, int],
    ) -> torch.Tensor:
        original_h, original_w = original_hw
        resized_h, resized_w = resized_hw
        padded_h, padded_w = padded_hw
        feature_h, feature_w = feature_hw
        scaled = boxes.clone()
        scaled[:, (0, 2)] *= (resized_w / original_w) * (feature_w / padded_w)
        scaled[:, (1, 3)] *= (resized_h / original_h) * (feature_h / padded_h)
        scaled[:, (0, 2)] = scaled[:, (0, 2)].clamp(0, feature_w)
        scaled[:, (1, 3)] = scaled[:, (1, 3)].clamp(0, feature_h)
        return scaled

    def _roi_features(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        original_hw: torch.Tensor,
        resized_hw: torch.Tensor,
        padded_hw: torch.Tensor,
    ) -> torch.Tensor:
        feature_h, feature_w = feature_map.shape[-2:]
        scaled = self._scale_boxes(boxes, original_hw, resized_hw, padded_hw, (feature_h, feature_w))
        batch_column = torch.zeros((len(scaled), 1), device=scaled.device, dtype=scaled.dtype)
        rois = torch.cat([batch_column, scaled], dim=1)
        pooled = roi_align(
            feature_map,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        return pooled.flatten(1)

    def forward_page(
        self,
        page_features: torch.Tensor,
        geometry: torch.Tensor,
        text_boxes: torch.Tensor,
        body_boxes: torch.Tensor,
        original_hw: torch.Tensor,
        resized_hw: torch.Tensor,
        padded_hw: torch.Tensor,
    ) -> list[torch.Tensor]:
        # page_features: [patch_h, patch_w, visual_dim]
        feature_map = page_features.permute(2, 0, 1).unsqueeze(0)
        # Preserve every dense DINOv3 patch by default. Spatial pooling is an
        # explicit fallback only (for example --context-grid 28 or 14) when a
        # particular page shape exceeds the available memory/time budget.
        if self.context_grid > 0 and max(feature_map.shape[-2:]) > self.context_grid:
            native_h, native_w = feature_map.shape[-2:]
            scale = self.context_grid / max(native_h, native_w)
            target_h = max(1, int(round(native_h * scale)))
            target_w = max(1, int(round(native_w * scale)))
            context_map = torch.nn.functional.adaptive_avg_pool2d(
                feature_map, (target_h, target_w)
            )
        else:
            context_map = feature_map
        context = context_map.flatten(2).transpose(1, 2)
        memory = self.context_projection(context)
        character_roi = self._roi_features(feature_map, body_boxes, original_hw, resized_hw, padded_hw)
        character_tokens = self.character_projection(character_roi)
        dialogue_roi = self._roi_features(feature_map, text_boxes, original_hw, resized_hw, padded_hw)
        dialogue_tokens = self.dialogue_projection(dialogue_roi)
        normalized_geometry = (geometry - self.geometry_mean) / self.geometry_std
        geometry_tokens = self.geometry_projection(normalized_geometry)

        logits: list[torch.Tensor] = []
        for query_index in range(geometry.shape[0]):
            candidates = (
                character_tokens
                + dialogue_tokens[query_index].unsqueeze(0)
                + geometry_tokens[query_index]
            )
            related = self.relation_decoder(candidates.unsqueeze(0), memory)
            logits.append(self.scorer(self.output_norm(related[0])).squeeze(-1))
        return logits


def multi_positive_listwise_loss(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    if not torch.any(positives):
        raise ValueError("Every training query must contain at least one positive candidate")
    return torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[positives], dim=0)
