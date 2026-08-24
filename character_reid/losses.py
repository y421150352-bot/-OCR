"""Metric-learning losses used by the ReID trainer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    count = positive.sum(dim=1)
    if not (count > 0).all():
        raise ValueError("SupCon batch contains an identity without a positive; use P x K with K >= 2")
    return -((log_prob * positive).sum(dim=1) / count).mean()


def batch_hard_triplet_loss(features: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    distance = 1 - features @ features.T
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    hardest_positive = distance.masked_fill(~same, float("-inf")).max(dim=1).values
    hardest_negative = distance.masked_fill(labels[:, None].eq(labels[None, :]), float("inf")).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, classes: int, scale: float = 32.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin = scale, margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cosine.gather(1, labels[:, None])) + self.margin)
        logits = cosine.scatter(1, labels[:, None], target) * self.scale
        return F.cross_entropy(logits, labels)
