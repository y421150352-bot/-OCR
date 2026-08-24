"""Edge-aware bipartite Graph Transformer for dialogue-to-character ranking."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import roi_align


class EdgeAwareGraphLayer(nn.Module):
    """Update D-C edge tokens through candidate competition and page context."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.candidate_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        edge_tokens: torch.Tensor,
        page_memory: torch.Tensor | None,
        geometry_attention_bias: torch.Tensor,
        dialogue_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        patch_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # edge_tokens: [batch, dialogues, characters, hidden]
        # geometry_attention_bias: [batch, dialogues, heads, characters, characters]
        batch_size, dialogues, characters, _ = edge_tokens.shape
        normalized = self.candidate_norm(edge_tokens)[dialogue_mask]
        attention_mask = geometry_attention_bias[dialogue_mask].reshape(
            -1, characters, characters
        )
        candidate_valid = candidate_mask.unsqueeze(1).expand(
            batch_size, dialogues, characters
        )[dialogue_mask]
        candidate_padding_mask = torch.zeros(
            candidate_valid.shape,
            device=edge_tokens.device,
            dtype=attention_mask.dtype,
        ).masked_fill(~candidate_valid, float("-inf"))
        attended, _ = self.candidate_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=candidate_padding_mask,
            need_weights=False,
        )
        attended_full = torch.zeros_like(edge_tokens)
        attended_full[dialogue_mask] = attended.to(edge_tokens.dtype)
        edge_tokens = edge_tokens + self.residual_dropout(attended_full)

        if page_memory is not None:
            if patch_mask is None:
                raise ValueError("patch_mask is required with page_memory")
            # Every relation edge independently reads the same dense page memory.
            flattened = self.context_norm(edge_tokens).reshape(
                batch_size, dialogues * characters, self.hidden_dim
            )
            patch_padding_mask = torch.zeros(
                patch_mask.shape, device=edge_tokens.device, dtype=edge_tokens.dtype
            ).masked_fill(~patch_mask, float("-inf"))
            contextualized, _ = self.context_attention(
                flattened,
                page_memory,
                page_memory,
                key_padding_mask=patch_padding_mask,
                need_weights=False,
            )
            edge_tokens = edge_tokens + self.residual_dropout(
                contextualized.reshape(
                    batch_size, dialogues, characters, self.hidden_dim
                )
            )
        edge_tokens = edge_tokens + self.residual_dropout(
            self.ffn(self.ffn_norm(edge_tokens))
        )
        edge_valid = dialogue_mask.unsqueeze(-1) & candidate_mask.unsqueeze(1)
        return edge_tokens.masked_fill(~edge_valid.unsqueeze(-1), 0.0)


class SpeakerBipartiteGraphTransformer(nn.Module):
    """Rank all D-C edges while using 45D geometry as tokens and attention bias."""

    def __init__(
        self,
        visual_dim: int,
        geometry_dim: int = 45,
        hidden_dim: int = 384,
        heads: int = 8,
        layers: int = 2,
        dropout: float = 0.15,
        attention_dropout: float = 0.1,
        geometry_bias_hidden: int = 128,
        geometry_bias_scale_init: float = 0.1,
        ablation: str = "full",
        context_grid: int = 0,
        roi_size: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if layers < 1:
            raise ValueError("layers must be at least 1")
        if geometry_bias_scale_init < 0.0:
            raise ValueError("geometry_bias_scale_init must be non-negative")
        if ablation not in {"full", "geometry_only", "visual_only", "no_geometry_bias"}:
            raise ValueError(f"Unknown V2 ablation: {ablation}")

        self.visual_dim = visual_dim
        self.geometry_dim = geometry_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.context_grid = context_grid
        self.roi_size = roi_size
        self.ablation = ablation
        self.use_visual = ablation != "geometry_only"
        self.use_geometry = ablation != "visual_only"
        self.use_geometry_bias = ablation in {"full", "geometry_only"}
        roi_dim = visual_dim * roi_size * roi_size

        self.context_projection = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
        )
        self.dialogue_projection = nn.Sequential(
            nn.LayerNorm(roi_dim),
            nn.Linear(roi_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.character_projection = nn.Sequential(
            nn.LayerNorm(roi_dim),
            nn.Linear(roi_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.geometry_attention_bias = nn.Sequential(
            nn.Linear(geometry_dim, geometry_bias_hidden),
            nn.GELU(),
            nn.Linear(geometry_bias_hidden, heads),
        )
        self.geometry_bias_scale = nn.Parameter(
            torch.tensor(float(geometry_bias_scale_init))
        )
        self.edge_input_norm = nn.LayerNorm(hidden_dim)
        self.graph_layers = nn.ModuleList(
            EdgeAwareGraphLayer(
                hidden_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.register_buffer("geometry_mean", torch.zeros(geometry_dim), persistent=True)
        self.register_buffer("geometry_std", torch.ones(geometry_dim), persistent=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Begin with no structural bias and let training introduce it gradually.
        final_bias_layer = self.geometry_attention_bias[-1]
        nn.init.zeros_(final_bias_layer.weight)
        nn.init.zeros_(final_bias_layer.bias)
        nn.init.normal_(self.scorer[-1].weight, std=0.02)

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
        scaled = self._scale_boxes(
            boxes, original_hw, resized_hw, padded_hw, (feature_h, feature_w)
        )
        batch_column = torch.zeros(
            (len(scaled), 1), device=scaled.device, dtype=scaled.dtype
        )
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

    def _page_memory(
        self, feature_map: torch.Tensor, patch_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.context_grid > 0 and max(feature_map.shape[-2:]) > self.context_grid:
            native_h, native_w = feature_map.shape[-2:]
            scale = self.context_grid / max(native_h, native_w)
            target_h = max(1, int(round(native_h * scale)))
            target_w = max(1, int(round(native_w * scale)))
            float_mask = patch_mask.unsqueeze(1).to(feature_map.dtype)
            pooled_weight = torch.nn.functional.adaptive_avg_pool2d(
                float_mask, (target_h, target_w)
            )
            context_map = torch.nn.functional.adaptive_avg_pool2d(
                feature_map * float_mask, (target_h, target_w)
            ) / pooled_weight.clamp_min(1e-6)
            context_mask = pooled_weight[:, 0] > 0
        else:
            context_map = feature_map
            context_mask = patch_mask
        context = context_map.flatten(2).transpose(1, 2)
        return self.context_projection(context), context_mask.flatten(1)

    def _geometry_bias(self, normalized_geometry: torch.Tensor) -> torch.Tensor:
        # Pairwise relative D-C edge attributes produce a true query-key bias:
        # [B, D, C(query), C(key), 45] -> [B, D, heads, C, C].
        pairwise_delta = (
            normalized_geometry.unsqueeze(3) - normalized_geometry.unsqueeze(2)
        )
        raw_bias = self.geometry_attention_bias(pairwise_delta)
        scaled_bias = torch.tanh(raw_bias) * self.geometry_bias_scale
        return scaled_bias.permute(0, 1, 4, 2, 3).contiguous()

    def _batched_roi_features(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        box_mask: torch.Tensor,
        original_hw: torch.Tensor,
        resized_hw: torch.Tensor,
        padded_hw: torch.Tensor,
        feature_hw: torch.Tensor,
    ) -> torch.Tensor:
        rois: list[torch.Tensor] = []
        for batch_index in range(len(feature_map)):
            valid_boxes = boxes[batch_index, box_mask[batch_index]].clone()
            original_h, original_w = original_hw[batch_index]
            resized_h, resized_w = resized_hw[batch_index]
            padded_h, padded_w = padded_hw[batch_index]
            native_feature_h, native_feature_w = feature_hw[batch_index]
            valid_boxes[:, (0, 2)] *= (
                (resized_w / original_w) * (native_feature_w / padded_w)
            )
            valid_boxes[:, (1, 3)] *= (
                (resized_h / original_h) * (native_feature_h / padded_h)
            )
            valid_boxes[:, (0, 2)] = valid_boxes[:, (0, 2)].clamp(
                0, native_feature_w
            )
            valid_boxes[:, (1, 3)] = valid_boxes[:, (1, 3)].clamp(
                0, native_feature_h
            )
            batch_column = torch.full(
                (len(valid_boxes), 1),
                float(batch_index),
                device=boxes.device,
                dtype=boxes.dtype,
            )
            rois.append(torch.cat([batch_column, valid_boxes], dim=1))
        pooled = roi_align(
            feature_map,
            torch.cat(rois, dim=0),
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        ).flatten(1)
        output = pooled.new_zeros(
            boxes.shape[0], boxes.shape[1], pooled.shape[-1]
        )
        output[box_mask] = pooled
        return output

    def forward_batch(
        self,
        page_features: torch.Tensor,
        patch_mask: torch.Tensor,
        feature_hw: torch.Tensor,
        geometry: torch.Tensor,
        text_boxes: torch.Tensor,
        body_boxes: torch.Tensor,
        dialogue_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        original_hw: torch.Tensor,
        resized_hw: torch.Tensor,
        padded_hw: torch.Tensor,
    ) -> torch.Tensor:
        # page_features: [batch, patch_h, patch_w, visual_dim]
        edge_valid = dialogue_mask.unsqueeze(-1) & candidate_mask.unsqueeze(1)
        normalized_geometry = (geometry - self.geometry_mean) / self.geometry_std
        normalized_geometry = normalized_geometry.masked_fill(
            ~edge_valid.unsqueeze(-1), 0.0
        )

        edge_components: list[torch.Tensor] = []
        page_memory: torch.Tensor | None = None
        context_mask: torch.Tensor | None = None
        if self.use_visual:
            feature_map = page_features.permute(0, 3, 1, 2)
            page_memory, context_mask = self._page_memory(feature_map, patch_mask)
            dialogue_roi = self._batched_roi_features(
                feature_map,
                text_boxes,
                dialogue_mask,
                original_hw,
                resized_hw,
                padded_hw,
                feature_hw,
            )
            character_roi = self._batched_roi_features(
                feature_map,
                body_boxes,
                candidate_mask,
                original_hw,
                resized_hw,
                padded_hw,
                feature_hw,
            )
            dialogue_tokens = self.dialogue_projection(dialogue_roi)
            character_tokens = self.character_projection(character_roi)
            edge_components.append(
                dialogue_tokens.unsqueeze(2) + character_tokens.unsqueeze(1)
            )
        if self.use_geometry:
            edge_components.append(self.geometry_projection(normalized_geometry))
        edge_tokens = self.edge_input_norm(sum(edge_components)).masked_fill(
            ~edge_valid.unsqueeze(-1), 0.0
        )
        if self.use_geometry_bias:
            geometry_bias = self._geometry_bias(normalized_geometry)
        else:
            batch_size, dialogues, candidates = geometry.shape[:3]
            geometry_bias = edge_tokens.new_zeros(
                batch_size, dialogues, self.heads, candidates, candidates
            )
        for layer in self.graph_layers:
            edge_tokens = layer(
                edge_tokens,
                page_memory,
                geometry_bias,
                dialogue_mask,
                candidate_mask,
                context_mask,
            )

        return self.scorer(self.output_norm(edge_tokens)).squeeze(-1)

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
        patch_h, patch_w = page_features.shape[:2]
        dialogues, candidates = geometry.shape[:2]
        score_matrix = self.forward_batch(
            page_features=page_features.unsqueeze(0),
            patch_mask=torch.ones(
                1, patch_h, patch_w, dtype=torch.bool, device=page_features.device
            ),
            feature_hw=torch.tensor(
                [[patch_h, patch_w]], dtype=torch.long, device=page_features.device
            ),
            geometry=geometry.unsqueeze(0),
            text_boxes=text_boxes.unsqueeze(0),
            body_boxes=body_boxes.unsqueeze(0),
            dialogue_mask=torch.ones(
                1, dialogues, dtype=torch.bool, device=geometry.device
            ),
            candidate_mask=torch.ones(
                1, candidates, dtype=torch.bool, device=geometry.device
            ),
            original_hw=original_hw.unsqueeze(0),
            resized_hw=resized_hw.unsqueeze(0),
            padded_hw=padded_hw.unsqueeze(0),
        )[0]
        return list(score_matrix.unbind(dim=0))
