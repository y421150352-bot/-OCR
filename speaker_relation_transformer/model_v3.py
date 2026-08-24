"""Geometry + dialogue-context text bipartite Graph Transformer V3."""

from __future__ import annotations

import torch
from torch import nn


class GeometryTextGraphLayer(nn.Module):
    """Exchange information across candidates and across page dialogues."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dropout: float,
        attention_dropout: float,
        use_dialogue_graph: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.use_dialogue_graph = use_dialogue_graph
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.candidate_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=attention_dropout, batch_first=True
        )
        self.dialogue_norm = nn.LayerNorm(hidden_dim)
        self.dialogue_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=attention_dropout, batch_first=True
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
        geometry_attention_bias: torch.Tensor,
        dialogue_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, dialogues, candidates, _ = edge_tokens.shape
        edge_valid = dialogue_mask.unsqueeze(-1) & candidate_mask.unsqueeze(1)

        # For each dialogue, all candidate body instances compete.
        candidate_input = self.candidate_norm(edge_tokens)[dialogue_mask]
        attention_mask = geometry_attention_bias[dialogue_mask].reshape(
            -1, candidates, candidates
        )
        candidate_valid = candidate_mask.unsqueeze(1).expand(
            batch_size, dialogues, candidates
        )[dialogue_mask]
        candidate_padding_mask = torch.zeros(
            candidate_valid.shape,
            device=edge_tokens.device,
            dtype=attention_mask.dtype,
        ).masked_fill(~candidate_valid, float("-inf"))
        candidate_output, _ = self.candidate_attention(
            candidate_input,
            candidate_input,
            candidate_input,
            attn_mask=attention_mask,
            key_padding_mask=candidate_padding_mask,
            need_weights=False,
        )
        candidate_full = torch.zeros_like(edge_tokens)
        candidate_full[dialogue_mask] = candidate_output.to(edge_tokens.dtype)
        edge_tokens = edge_tokens + self.residual_dropout(candidate_full)

        if self.use_dialogue_graph:
            # For each candidate instance, connect its edges across all page
            # dialogues. candidate_only disables exactly this added path.
            dialogue_input = self.dialogue_norm(edge_tokens).permute(0, 2, 1, 3)
            dialogue_input = dialogue_input[candidate_mask]
            dialogue_valid = dialogue_mask.unsqueeze(1).expand(
                batch_size, candidates, dialogues
            )[candidate_mask]
            dialogue_padding_mask = torch.zeros(
                dialogue_valid.shape,
                device=edge_tokens.device,
                dtype=edge_tokens.dtype,
            ).masked_fill(~dialogue_valid, float("-inf"))
            dialogue_output, _ = self.dialogue_attention(
                dialogue_input,
                dialogue_input,
                dialogue_input,
                key_padding_mask=dialogue_padding_mask,
                need_weights=False,
            )
            dialogue_full = torch.zeros(
                batch_size,
                candidates,
                dialogues,
                self.hidden_dim,
                device=edge_tokens.device,
                dtype=edge_tokens.dtype,
            )
            dialogue_full[candidate_mask] = dialogue_output.to(edge_tokens.dtype)
            edge_tokens = edge_tokens + self.residual_dropout(
                dialogue_full.permute(0, 2, 1, 3)
            )

        edge_tokens = edge_tokens + self.residual_dropout(
            self.ffn(self.ffn_norm(edge_tokens))
        )
        return edge_tokens.masked_fill(~edge_valid.unsqueeze(-1), 0.0)


class SpeakerGeometryTextGraphTransformer(nn.Module):
    """Rank D-C edges using geometry and frozen prev/current/next text embeddings."""

    model_version = "v3_geometry_text_graph"

    def __init__(
        self,
        text_dim: int,
        geometry_dim: int = 45,
        hidden_dim: int = 384,
        heads: int = 8,
        layers: int = 2,
        dropout: float = 0.15,
        attention_dropout: float = 0.1,
        geometry_bias_hidden: int = 128,
        geometry_bias_scale_init: float = 0.1,
        use_text: bool = True,
        use_dialogue_graph: bool = True,
        reid_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if layers < 1:
            raise ValueError("layers must be at least 1")
        if text_dim < 1:
            raise ValueError("text_dim must be positive")
        if geometry_bias_scale_init < 0:
            raise ValueError("geometry_bias_scale_init must be non-negative")

        self.text_dim = text_dim
        self.geometry_dim = geometry_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.use_text = use_text
        self.use_dialogue_graph = use_dialogue_graph
        self.reid_dim = reid_dim

        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.text_input_norm = nn.LayerNorm(text_dim)
        self.text_slot_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_slot_position = nn.Parameter(torch.empty(3, hidden_dim))
        self.text_context_mixer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        if reid_dim > 0:
            self.reid_projection: nn.Module | None = nn.Sequential(
                nn.LayerNorm(reid_dim),
                nn.Linear(reid_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.reid_gate: nn.Module | None = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
            )
        else:
            self.reid_projection = None
            self.reid_gate = None
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
            GeometryTextGraphLayer(
                hidden_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_dialogue_graph=use_dialogue_graph,
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
        nn.init.normal_(self.text_slot_position, std=0.02)
        final_bias_layer = self.geometry_attention_bias[-1]
        nn.init.zeros_(final_bias_layer.weight)
        nn.init.zeros_(final_bias_layer.bias)
        nn.init.normal_(self.scorer[-1].weight, std=0.02)

    def set_geometry_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.geometry_dim or std.numel() != self.geometry_dim:
            raise ValueError("Geometry normalization dimension mismatch")
        self.geometry_mean.copy_(mean)
        self.geometry_std.copy_(std.clamp_min(1e-4))

    def _geometry_bias(self, normalized_geometry: torch.Tensor) -> torch.Tensor:
        pairwise_delta = (
            normalized_geometry.unsqueeze(3) - normalized_geometry.unsqueeze(2)
        )
        raw_bias = self.geometry_attention_bias(pairwise_delta)
        scaled_bias = torch.tanh(raw_bias) * self.geometry_bias_scale
        return scaled_bias.permute(0, 1, 4, 2, 3).contiguous()

    def _encode_text_context(
        self,
        text_context: torch.Tensor,
        text_context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if text_context.shape[-2] != 3 or text_context_mask.shape[-1] != 3:
            raise ValueError("Text context slots must be [previous, current, next]")
        normalized = self.text_input_norm(text_context)
        slots = self.text_slot_projection(normalized)
        slots = slots + self.text_slot_position.view(1, 1, 3, self.hidden_dim)
        slots = slots.masked_fill(~text_context_mask.unsqueeze(-1), 0.0)
        return self.text_context_mixer(slots.flatten(-2))

    def forward_batch(
        self,
        geometry: torch.Tensor,
        text_context: torch.Tensor,
        text_context_mask: torch.Tensor,
        dialogue_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_reid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_valid = dialogue_mask.unsqueeze(-1) & candidate_mask.unsqueeze(1)
        normalized_geometry = (geometry - self.geometry_mean) / self.geometry_std
        normalized_geometry = normalized_geometry.masked_fill(
            ~edge_valid.unsqueeze(-1), 0.0
        )
        geometry_token = self.geometry_projection(normalized_geometry)
        if self.use_text:
            text_token = self._encode_text_context(text_context, text_context_mask)
            # The learned gate lets dialogue semantics modulate which geometry
            # cues matter while keeping candidate identity grounded in geometry.
            edge_tokens = geometry_token + text_token.unsqueeze(2) * self.text_gate(
                geometry_token
            )
        else:
            # Strict no-text control: retain the identical geometry projection,
            # two-axis graph, scorer, initialization order, and training path.
            edge_tokens = geometry_token
        if self.reid_projection is not None:
            if candidate_reid is None:
                raise ValueError("candidate_reid is required when reid_dim > 0")
            if candidate_reid.shape[:2] != candidate_mask.shape:
                raise ValueError("candidate_reid shape does not match candidate_mask")
            if candidate_reid.shape[-1] != self.reid_dim:
                raise ValueError("candidate_reid embedding dimension mismatch")
            identity_token = self.reid_projection(candidate_reid).unsqueeze(1)
            assert self.reid_gate is not None
            # The frozen ReID vector describes candidate identity, while the
            # geometry-conditioned gate decides how much each D-C edge uses it.
            edge_tokens = edge_tokens + identity_token * self.reid_gate(geometry_token)
        edge_tokens = self.edge_input_norm(edge_tokens).masked_fill(
            ~edge_valid.unsqueeze(-1), 0.0
        )
        geometry_bias = self._geometry_bias(normalized_geometry)
        for layer in self.graph_layers:
            edge_tokens = layer(
                edge_tokens,
                geometry_bias,
                dialogue_mask,
                candidate_mask,
            )
        return self.scorer(self.output_norm(edge_tokens)).squeeze(-1)

    def forward_page(
        self,
        geometry: torch.Tensor,
        text_context: torch.Tensor,
        text_context_mask: torch.Tensor,
        candidate_reid: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        dialogues, candidates = geometry.shape[:2]
        scores = self.forward_batch(
            geometry=geometry.unsqueeze(0),
            text_context=text_context.unsqueeze(0),
            text_context_mask=text_context_mask.unsqueeze(0),
            dialogue_mask=torch.ones(
                1, dialogues, dtype=torch.bool, device=geometry.device
            ),
            candidate_mask=torch.ones(
                1, candidates, dtype=torch.bool, device=geometry.device
            ),
            candidate_reid=(
                candidate_reid.unsqueeze(0) if candidate_reid is not None else None
            ),
        )[0]
        return list(scores.unbind(dim=0))
