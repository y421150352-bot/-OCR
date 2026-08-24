#!/usr/bin/env python3
"""Train and evaluate the cached-DINOv3 speaker relation Transformer."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    BucketedPageBatchSampler,
    GeometryGraphPageDataset,
    GeometryTextPageDataset,
    PageDataset,
    ShuffledPageSampler,
    compute_geometry_stats,
    geometry_text_page_batch_collate,
    load_text_cache_dim,
    load_reid_cache_dim,
    page_batch_collate,
    single_page_collate,
)
BASELINE = {"top1": 0.752261, "top3": 0.959397, "mrr": 0.856472}


def multi_positive_listwise_loss(
    logits: torch.Tensor, positive: torch.Tensor
) -> torch.Tensor:
    """Negative log probability assigned to any valid speaker instance."""
    if not positive.any():
        raise ValueError("Every dialogue must have at least one positive candidate")
    return torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[positive], dim=0)


def parse_args(default_model_version: str = "v1") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-version", choices=("v1", "v2", "v3"), default=default_model_version
    )
    parser.add_argument(
        "--ablation",
        choices=("full", "geometry_only", "visual_only", "no_geometry_bias"),
        default="full",
        help="V2 modality/bias ablation; full is the production Graph Transformer",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/dinov3_vitb16_l896"))
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=Path("cache/text_multilingual_e5_base"),
        help="V3 frozen dialogue embedding cache",
    )
    parser.add_argument(
        "--v3-text-ablation",
        choices=("full", "no_text"),
        default="full",
        help="V3 only: disable text while retaining the identical two-axis graph",
    )
    parser.add_argument(
        "--v3-graph-mode",
        choices=("two_axis", "candidate_only"),
        default="two_axis",
        help="V3 only: candidate_only removes cross-dialogue graph attention",
    )
    parser.add_argument(
        "--reid-cache-dir",
        type=Path,
        help="Optional frozen Face+Body candidate embeddings; V3 no-text only",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument(
        "--page-batch-size", type=int,
        help="True pages per GPU forward pass (V2 default: 4; V1 must remain 1)",
    )
    parser.add_argument(
        "--eval-page-batch-size", type=int,
        help="Pages per validation/test forward pass; defaults to --page-batch-size",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument(
        "--cosine-epochs", type=int, default=40,
        help="Number of post-warmup epochs used to decay from --lr to --min-lr",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="Stop after this many epochs without a val Top-1 improvement; 0 disables",
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=0.0,
        help="Minimum absolute val Top-1 gain that resets early stopping",
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--attention-dropout", type=float, default=0.1,
        help="V2 graph self/cross-attention dropout",
    )
    parser.add_argument(
        "--geometry-bias-hidden", type=int, default=128,
        help="V2 hidden dimension for 45D geometry-to-head attention bias",
    )
    parser.add_argument(
        "--geometry-bias-scale-init", type=float, default=0.1,
        help="V2 initial learnable scale applied to tanh geometry attention bias",
    )
    parser.add_argument(
        "--context-grid", type=int, default=0,
        help="0 keeps all patches; a positive value sets the pooled grid's long side",
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--bucket-multiplier", type=int, default=32)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-pages", type=int, default=0, help="Debug only")
    args = parser.parse_args()
    architecture_defaults = {
        "v1": {
            "hidden_dim": 512, "layers": 4, "dropout": 0.1,
            "page_batch_size": 1, "grad_accum": 8, "num_workers": 4,
        },
        "v2": {
            "hidden_dim": 384, "layers": 2, "dropout": 0.15,
            "page_batch_size": 4, "grad_accum": 2, "num_workers": 8,
        },
        "v3": {
            "hidden_dim": 384, "layers": 2, "dropout": 0.15,
            "page_batch_size": 8, "grad_accum": 1, "num_workers": 8,
        },
    }[args.model_version]
    for name, value in architecture_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.output_dir is None:
        if args.model_version == "v1":
            args.output_dir = Path("runs/vitb16_relation_lr1e4")
        elif args.model_version == "v2":
            suffix = "full" if args.ablation == "full" else args.ablation
            args.output_dir = Path(f"runs/vitb16_bipartite_graph_v2_{suffix}_b4")
        else:
            if (
                args.v3_text_ablation == "full"
                and args.v3_graph_mode == "candidate_only"
            ):
                suffix = "pure_context_candidate_graph"
            elif args.v3_text_ablation == "full":
                suffix = "prev_current_next"
            elif args.reid_cache_dir is not None:
                suffix = "no_text_face_body_reid"
            elif args.v3_graph_mode == "two_axis":
                suffix = "no_text"
            else:
                suffix = "no_text_candidate_graph"
            args.output_dir = Path(f"runs/geometry_text_graph_v3_{suffix}")
    if args.eval_page_batch_size is None:
        args.eval_page_batch_size = args.page_batch_size
    return args


def learning_rate_for_epoch(
    epoch: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
    cosine_epochs: int,
) -> float:
    """Return the deterministic LR for a one-based epoch index."""
    if epoch < 1:
        raise ValueError("epoch must be one-based and positive")
    if not 0.0 <= min_lr <= base_lr:
        raise ValueError("Require 0 <= min_lr <= lr")
    if warmup_epochs < 0 or cosine_epochs < 1:
        raise ValueError("warmup_epochs must be >= 0 and cosine_epochs must be >= 1")
    if warmup_epochs and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs

    decay_epoch = epoch - warmup_epochs
    progress = min(max(decay_epoch / cosine_epochs, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_learning_rate(optimizer: AdamW, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def epochs_since_best(history: list[dict[str, object]], best_top1: float) -> int:
    """Recover the early-stopping counter from old or resumed checkpoints."""
    if not history:
        return 0
    for index in range(len(history) - 1, -1, -1):
        validation = history[index].get("val", {})
        if isinstance(validation, dict) and float(validation.get("top1", -1.0)) >= best_top1 - 1e-12:
            return len(history) - index - 1
    return len(history)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_page(page: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in page.items()}


def forward_logits_and_targets(
    model: torch.nn.Module,
    page: dict[str, object],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if page.get("batched"):
        if "text_context" in page:
            score_matrix = model.forward_batch(
                page["geometry"],
                page["text_context"],
                page["text_context_mask"],
                page["dialogue_mask"],
                page["candidate_mask"],
                page.get("candidate_reid"),
            )
        else:
            score_matrix = model.forward_batch(
                page["page_features"],
                page["patch_mask"],
                page["feature_hw"],
                page["geometry"],
                page["text_boxes"],
                page["body_boxes"],
                page["dialogue_mask"],
                page["candidate_mask"],
                page["original_hw"],
                page["resized_hw"],
                page["padded_hw"],
            )
        logits: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for batch_index in range(score_matrix.shape[0]):
            dialogue_mask = page["dialogue_mask"][batch_index]
            candidate_mask = page["candidate_mask"][batch_index]
            page_scores = score_matrix[batch_index, dialogue_mask][:, candidate_mask]
            page_targets = page["labels"][batch_index, dialogue_mask][:, candidate_mask]
            logits.extend(page_scores.unbind(dim=0))
            targets.extend(page_targets.unbind(dim=0))
        return logits, targets

    if "text_context" in page:
        logits = model.forward_page(
            page["geometry"],
            page["text_context"],
            page["text_context_mask"],
            page.get("candidate_reid"),
        )
    else:
        logits = model.forward_page(
            page["page_features"], page["geometry"], page["text_boxes"], page["body_boxes"],
            page["original_hw"], page["resized_hw"], page["padded_hw"],
        )
    return logits, list(page["labels"].unbind(dim=0))


def rank_metrics(logits: list[torch.Tensor], labels: list[torch.Tensor]) -> dict[str, float]:
    metrics = {
        "queries": 0.0,
        "top1": 0.0,
        "top3": 0.0,
        "mrr": 0.0,
        "recall_at_1_sum": 0.0,
        "f1_at_1_sum": 0.0,
        "recall_at_3_sum": 0.0,
        "map_sum": 0.0,
        "exact_match_at_1_sum": 0.0,
        "positive_total": 0.0,
    }
    for scores, positive in zip(logits, labels):
        order = torch.argsort(scores, descending=True)
        ranked_positive = positive[order]
        positive_count = int(positive.sum())
        hit_at_1 = float(ranked_positive[0])
        hit_at_3 = float(ranked_positive[:3].any())
        positives_at_3 = float(ranked_positive[:3].sum())
        first = int(torch.nonzero(ranked_positive, as_tuple=False)[0, 0])
        cumulative_hits = ranked_positive.float().cumsum(dim=0)
        ranks = torch.arange(1, len(ranked_positive) + 1, device=scores.device, dtype=torch.float32)
        average_precision = float(((cumulative_hits / ranks) * ranked_positive).sum() / positive_count)

        metrics["queries"] += 1.0
        metrics["top1"] += hit_at_1
        metrics["top3"] += hit_at_3
        metrics["mrr"] += 1.0 / (first + 1)
        metrics["recall_at_1_sum"] += hit_at_1 / positive_count
        metrics["f1_at_1_sum"] += (2.0 / (1.0 + positive_count)) if hit_at_1 else 0.0
        metrics["recall_at_3_sum"] += positives_at_3 / positive_count
        metrics["map_sum"] += average_precision
        metrics["exact_match_at_1_sum"] += float(hit_at_1 and positive_count == 1)
        metrics["positive_total"] += positive_count
    return metrics


def merge_metrics(total: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + value


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype | None) -> dict[str, float]:
    model.eval()
    total: dict[str, float] = {"loss_sum": 0.0}
    for page in tqdm(loader, desc="evaluate", leave=False):
        page = move_page(page, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, targets = forward_logits_and_targets(model, page)
            losses = [
                multi_positive_listwise_loss(score, target)
                for score, target in zip(logits, targets)
            ]
        query_count = len(losses)
        total["loss_sum"] += float(torch.stack(losses).sum())
        merge_metrics(total, rank_metrics(logits, targets))
    queries = max(total["queries"], 1.0)
    true_positives_at_1 = total["top1"]
    micro_precision_at_1 = true_positives_at_1 / queries
    micro_recall_at_1 = true_positives_at_1 / max(total["positive_total"], 1.0)
    micro_f1_at_1 = (
        2.0 * micro_precision_at_1 * micro_recall_at_1
        / max(micro_precision_at_1 + micro_recall_at_1, 1e-12)
    )
    return {
        "loss": total["loss_sum"] / queries,
        "accuracy": total["top1"] / queries,
        "top1": total["top1"] / queries,
        "top3": total["top3"] / queries,
        "mrr": total["mrr"] / queries,
        "precision_at_1": micro_precision_at_1,
        "recall_at_1": total["recall_at_1_sum"] / queries,
        "f1_at_1": total["f1_at_1_sum"] / queries,
        "micro_recall_at_1": micro_recall_at_1,
        "micro_f1_at_1": micro_f1_at_1,
        "recall_at_3": total["recall_at_3_sum"] / queries,
        "map": total["map_sum"] / queries,
        "exact_match_at_1": total["exact_match_at_1_sum"] / queries,
        "positive_labels": int(total["positive_total"]),
        "queries": int(total["queries"]),
    }


def load_visual_dim(cache_dir: Path, dataset: PageDataset) -> int:
    from data import cache_filename
    path = cache_dir / cache_filename(str(dataset.records[0]["key"]))
    with np.load(path) as cached:
        return int(cached["features"].shape[-1])


def main(default_model_version: str = "v1") -> None:
    args = parse_args(default_model_version)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be non-negative")
    if args.page_batch_size < 1 or args.eval_page_batch_size < 1:
        raise ValueError("Page batch sizes must be at least 1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be at least 1")
    if args.prefetch_factor < 1 or args.bucket_multiplier < 1:
        raise ValueError("Prefetch and bucket multipliers must be at least 1")
    if args.model_version == "v1" and (
        args.page_batch_size != 1 or args.eval_page_batch_size != 1
    ):
        raise ValueError("V1 supports only --page-batch-size 1; use train_v2.py for batching")
    if args.model_version != "v2" and args.ablation != "full":
        raise ValueError("--ablation applies only to V2")
    if args.model_version != "v3" and args.v3_text_ablation != "full":
        raise ValueError("--v3-text-ablation applies only to V3")
    if args.model_version != "v3" and args.v3_graph_mode != "two_axis":
        raise ValueError("--v3-graph-mode applies only to V3")
    if args.reid_cache_dir is not None and not (
        args.model_version == "v3" and args.v3_text_ablation == "no_text"
    ):
        raise ValueError("--reid-cache-dir currently requires V3 --v3-text-ablation no_text")
    # Validate the complete schedule before allocating GPU memory.
    learning_rate_for_epoch(
        1, args.lr, args.min_lr, args.warmup_epochs, args.cosine_epochs
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires the RTX 5090 CUDA environment")
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "geometry_stats.npz"
    if stats_path.exists():
        with np.load(stats_path) as stats:
            geometry_mean, geometry_std = stats["mean"], stats["std"]
    else:
        geometry_mean, geometry_std = compute_geometry_stats(args.data_dir, stats_path)

    if args.model_version == "v3" and args.v3_text_ablation == "no_text":
        datasets = {
            split: GeometryGraphPageDataset(
                args.data_dir, split, reid_cache_dir=args.reid_cache_dir
            )
            for split in ("train", "val", "test")
        }
    elif args.model_version == "v3":
        datasets = {
            split: GeometryTextPageDataset(args.data_dir, args.text_cache_dir, split)
            for split in ("train", "val", "test")
        }
    else:
        datasets = {
            split: PageDataset(args.data_dir, args.cache_dir, split)
            for split in ("train", "val", "test")
        }
    if args.max_pages:
        for dataset in datasets.values():
            dataset.records = dataset.records[:args.max_pages]
    worker_options = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        worker_options["prefetch_factor"] = args.prefetch_factor
    if args.model_version in {"v2", "v3"}:
        sampler = BucketedPageBatchSampler(
            datasets["train"].records,
            batch_size=args.page_batch_size,
            seed=args.seed,
            bucket_multiplier=args.bucket_multiplier,
        )
        batch_collate = (
            geometry_text_page_batch_collate
            if args.model_version == "v3"
            else page_batch_collate
        )
        loaders = {
            "train": DataLoader(
                datasets["train"], batch_sampler=sampler, collate_fn=batch_collate,
                **worker_options,
            ),
            "val": DataLoader(
                datasets["val"], batch_size=args.eval_page_batch_size, shuffle=False,
                collate_fn=batch_collate, **worker_options,
            ),
            "test": DataLoader(
                datasets["test"], batch_size=args.eval_page_batch_size, shuffle=False,
                collate_fn=batch_collate, **worker_options,
            ),
        }
    else:
        sampler = ShuffledPageSampler(len(datasets["train"]), args.seed)
        loaders = {
            "train": DataLoader(
                datasets["train"], batch_size=1, sampler=sampler,
                collate_fn=single_page_collate, **worker_options,
            ),
            "val": DataLoader(
                datasets["val"], batch_size=1, shuffle=False,
                collate_fn=single_page_collate, **worker_options,
            ),
            "test": DataLoader(
                datasets["test"], batch_size=1, shuffle=False,
                collate_fn=single_page_collate, **worker_options,
            ),
        }
    visual_dim: int | None = None
    text_dim: int | None = None
    reid_dim = 0
    text_cache_metadata: dict[str, object] | None = None
    if args.model_version == "v3" and args.v3_text_ablation == "no_text":
        # Keep compatibility with existing multilingual-e5-base no-text
        # checkpoints. The text branch is disabled and these dimensions never
        # participate in forward computation.
        text_dim = 768
        if args.reid_cache_dir is not None:
            reid_dim = load_reid_cache_dim(args.reid_cache_dir, datasets["train"])
    elif args.model_version == "v3":
        text_dim = load_text_cache_dim(args.text_cache_dir, datasets["train"])
        text_config_path = args.text_cache_dir / "config.json"
        if not text_config_path.is_file():
            raise FileNotFoundError(f"Missing text cache config: {text_config_path}")
        text_config = json.loads(text_config_path.read_text(encoding="utf-8"))
        if int(text_config.get("embedding_dim", -1)) != text_dim:
            raise ValueError("Text cache config embedding_dim does not match cache files")
        text_cache_metadata = {
            key: text_config.get(key)
            for key in (
                "model_name",
                "pooling",
                "l2_normalized",
                "prefix",
                "max_length",
                "context_order",
                "context_slots",
                "embedding_dim",
            )
        }
    else:
        visual_dim = load_visual_dim(args.cache_dir, datasets["train"])
    if args.model_version == "v1":
        from model import SpeakerRelationTransformer

        model = SpeakerRelationTransformer(
            visual_dim=visual_dim,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
            context_grid=args.context_grid,
        ).to(device)
    elif args.model_version == "v2":
        from model_v2 import SpeakerBipartiteGraphTransformer

        model = SpeakerBipartiteGraphTransformer(
            visual_dim=visual_dim,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            geometry_bias_hidden=args.geometry_bias_hidden,
            geometry_bias_scale_init=args.geometry_bias_scale_init,
            ablation=args.ablation,
            context_grid=args.context_grid,
        ).to(device)
    else:
        from model_v3 import SpeakerGeometryTextGraphTransformer

        if text_dim is None:
            raise AssertionError("V3 text dimension was not loaded")
        model = SpeakerGeometryTextGraphTransformer(
            text_dim=text_dim,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            geometry_bias_hidden=args.geometry_bias_hidden,
            geometry_bias_scale_init=args.geometry_bias_scale_init,
            use_text=args.v3_text_ablation == "full",
            use_dialogue_graph=args.v3_graph_mode == "two_axis",
            reid_dim=reid_dim,
        ).to(device)
    model.set_geometry_stats(torch.from_numpy(geometry_mean).to(device), torch.from_numpy(geometry_std).to(device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    start_epoch, best_top1 = 1, -1.0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_version = str(checkpoint.get("config", {}).get("model_version", "v1"))
        if checkpoint_version != args.model_version:
            raise ValueError(
                f"Checkpoint model_version={checkpoint_version} cannot resume "
                f"model_version={args.model_version}"
            )
        checkpoint_ablation = str(checkpoint.get("config", {}).get("ablation", "full"))
        if checkpoint_ablation != args.ablation:
            raise ValueError(
                f"Checkpoint ablation={checkpoint_ablation} cannot resume "
                f"ablation={args.ablation}"
            )
        if args.model_version == "v3":
            checkpoint_text_ablation = str(
                checkpoint.get("config", {}).get("v3_text_ablation", "full")
            )
            if checkpoint_text_ablation != args.v3_text_ablation:
                raise ValueError(
                    f"Checkpoint v3_text_ablation={checkpoint_text_ablation} cannot "
                    f"resume v3_text_ablation={args.v3_text_ablation}"
                )
            checkpoint_graph_mode = str(
                checkpoint.get("config", {}).get("v3_graph_mode", "two_axis")
            )
            if checkpoint_graph_mode != args.v3_graph_mode:
                raise ValueError(
                    f"Checkpoint v3_graph_mode={checkpoint_graph_mode} cannot "
                    f"resume v3_graph_mode={args.v3_graph_mode}"
                )
            checkpoint_text_cache = checkpoint.get("config", {}).get("text_cache")
            if (
                args.v3_text_ablation == "full"
                and checkpoint_text_cache != text_cache_metadata
            ):
                raise ValueError(
                    "Checkpoint text cache metadata differs from --text-cache-dir"
                )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        best_top1 = float(checkpoint["best_top1"])
        history = list(checkpoint.get("history", []))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", epochs_since_best(history, best_top1))
        )
        checkpoint_config = checkpoint.get("config", {})
        if "warmup_epochs" not in checkpoint_config:
            print(
                "Warning: resuming a legacy checkpoint with the new deterministic "
                "warmup/cosine schedule. For a clean comparison, start a new output directory.",
                flush=True,
            )

    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update({
        "visual_dim": visual_dim,
        "text_dim": text_dim,
        "text_cache": text_cache_metadata,
        "reid_dim": reid_dim,
        "effective_page_batch": args.page_batch_size * args.grad_accum,
        "baseline": BASELINE,
    })
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_lr = learning_rate_for_epoch(
            epoch, args.lr, args.min_lr, args.warmup_epochs, args.cosine_epochs
        )
        set_learning_rate(optimizer, epoch_lr)
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_queries = 0
        running_pages = 0
        started = time.time()
        progress = tqdm(loaders["train"], desc=f"train {epoch}/{args.epochs}")
        for step, page in enumerate(progress, 1):
            page = move_page(page, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits, targets = forward_logits_and_targets(model, page)
                losses = [
                    multi_positive_listwise_loss(score, target)
                    for score, target in zip(logits, targets)
                ]
                page_loss = torch.stack(losses).mean() / args.grad_accum
            scaler.scale(page_loss).backward()
            if step % args.grad_accum == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(page_loss.detach()) * args.grad_accum * len(losses)
            running_queries += len(losses)
            running_pages += len(page["key"]) if page.get("batched") else 1
            progress.set_postfix(
                loss=f"{running_loss / max(running_queries, 1):.4f}",
                lr=f"{epoch_lr:.2e}",
                pages_s=f"{running_pages / max(time.time() - started, 1e-6):.2f}",
            )
        validation = evaluate(model, loaders["val"], device, amp_dtype)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_queries, 1),
            "val": validation,
            "lr": epoch_lr,
            "seconds": time.time() - started,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))
        improved = validation["top1"] > best_top1 + args.early_stopping_min_delta
        if improved:
            best_top1 = validation["top1"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schedule": {
                "type": "linear_warmup_cosine",
                "warmup_epochs": args.warmup_epochs,
                "cosine_epochs": args.cosine_epochs,
                "base_lr": args.lr,
                "min_lr": args.min_lr,
            },
            "best_top1": best_top1,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "config": config,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: val Top-1 has not improved "
                f"for {epochs_without_improvement} epochs; best={best_top1:.6f}.",
                flush=True,
            )
            break

    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test = evaluate(model, loaders["test"], device, amp_dtype)
    best_history_record = next(
        record for record in history if int(record["epoch"]) == int(best["epoch"])
    )
    result = {
        "best_epoch": int(best["epoch"]),
        "completed_epochs": int(history[-1]["epoch"]),
        "stopped_early": stopped_early,
        "validation": best_history_record["val"],
        "test": test,
        "geometry_baseline": BASELINE,
        "test_delta": {key: test[key] - BASELINE[key] for key in ("top1", "top3", "mrr")},
    }
    (args.output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
