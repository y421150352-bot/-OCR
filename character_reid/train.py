#!/usr/bin/env python3
"""Train the supervised DINOv3 face/body ReID model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import BookPKBatchSampler, MangaReIDDataset
from evaluate import book_metrics
from losses import ArcFaceHead, batch_hard_triplet_loss, supervised_contrastive_loss
from model import CharacterReIDModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backbone", default="../speaker_relation_transformer/pretrained/dinov3-vitb16")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline_supcon"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--p", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--books-per-batch", type=int, default=2)
    parser.add_argument("--modality-dropout", type=float, default=0.15, help="For paired samples: probability to drop face and separately body")
    parser.add_argument("--include-other-train", action="store_true")
    parser.add_argument("--include-native-face-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=0, help="Limit batches for a smoke test; 0 uses the full epoch")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--arcface-weight", type=float, default=0.0)
    parser.add_argument("--triplet-weight", type=float, default=0.0)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--gallery-per-id", type=int, default=3)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Stop after this many validation checks without mAP improvement; 0 disables")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    return parser.parse_args()


def evaluate_retrieval(
    model: CharacterReIDModel,
    loader: DataLoader,
    dataset: MangaReIDDataset,
    device: torch.device,
    gallery_per_id: int,
) -> dict[str, float | int]:
    """Query-weighted within-book validation, excluding ``Other`` identities."""
    model.eval()
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    batch["face"].to(device, non_blocking=True),
                    batch["body"].to(device, non_blocking=True),
                    batch["face_valid"].to(device, non_blocking=True),
                    batch["body_valid"].to(device, non_blocking=True),
                )
            feature_parts.append(output["embedding"].float().cpu().numpy())
            label_parts.append(batch["label"].numpy())
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    input_types = np.asarray([str(record["input_type"]) for record in dataset.records])
    by_book: dict[str, list[int]] = defaultdict(list)
    excluded_other = 0
    for index, record in enumerate(dataset.records):
        if str(record.get("character_name", "")).strip().casefold() == "other":
            excluded_other += 1
            continue
        by_book[str(record["book"])].append(index)
    per_book = {
        book: book_metrics(features, labels, input_types, indexes, gallery_per_id, 3407 + offset)
        for offset, (book, indexes) in enumerate(sorted(by_book.items()))
    }
    queries = sum(int(row["queries"]) for row in per_book.values())
    def weighted(metric: str) -> float:
        return sum(float(row[metric]) * int(row["queries"]) for row in per_book.values()) / queries if queries else 0.0
    return {
        "rank_1": weighted("rank_1"),
        "rank_5": weighted("rank_5"),
        "mAP": weighted("mAP"),
        "queries": queries,
        "gallery": sum(int(row["gallery"]) for row in per_book.values()),
        "books": len(per_book),
        "excluded_other_instances": excluded_other,
    }


def main() -> None:
    args = parse_args()
    args.backbone = str(Path(args.backbone).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MangaReIDDataset(
        args.data_dir / "train.jsonl",
        training=True,
        exclude_other=not args.include_other_train,
        exclude_face_only=not args.include_native_face_only,
        modality_dropout=args.modality_dropout,
    )
    val_dataset = MangaReIDDataset(args.data_dir / "val.jsonl", training=False)
    sampler = BookPKBatchSampler(
        dataset.records, dataset.labels, args.p, args.k,
        books_per_batch=args.books_per_batch,
        batches_per_epoch=args.batches_per_epoch or None,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = CharacterReIDModel(args.backbone, freeze_backbone=not args.unfreeze_backbone).to(device)
    arcface = ArcFaceHead(model.embedding_dim, len(dataset.identity_to_label)).to(device) if args.arcface_weight else None
    parameters = list(filter(lambda p: p.requires_grad, model.parameters())) + (list(arcface.parameters()) if arcface else [])
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, len(loader) * args.epochs)
    warmup_steps = max(0, round(len(loader) * args.warmup_epochs))
    min_lr_ratio = args.min_lr / args.lr
    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"device": str(device), "identities": len(dataset.identity_to_label)}
    (args.output_dir / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    history: list[dict] = []
    best_map = -1.0
    best_epoch = 0
    checks_without_improvement = 0
    interactive = sys.stderr.isatty()
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        running = 0.0
        batch_count = 0
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", disable=not interactive, dynamic_ncols=True)
        for batch in progress:
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch["face"].to(device), batch["body"].to(device), batch["face_valid"].to(device), batch["body_valid"].to(device))
                embedding = output["embedding"]
                loss = supervised_contrastive_loss(embedding, labels)
                if arcface:
                    loss = loss + args.arcface_weight * arcface(embedding, labels)
                if args.triplet_weight:
                    loss = loss + args.triplet_weight * batch_hard_triplet_loss(embedding, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            batch_count += 1
            progress.set_postfix(loss=f"{running / batch_count:.4f}")
        checkpoint = {
            "model": model.state_dict(), "epoch": epoch, "config": config,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        row: dict = {
            "epoch": epoch, "train_loss": running / max(1, batch_count),
            "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started,
        }
        if args.validate_every > 0 and (epoch % args.validate_every == 0 or epoch == args.epochs):
            validation = evaluate_retrieval(model, val_loader, val_dataset, device, args.gallery_per_id)
            row.update({f"val_{key}": value for key, value in validation.items()})
            if float(validation["mAP"]) > best_map + args.early_stop_min_delta:
                best_map = float(validation["mAP"])
                best_epoch = epoch
                checks_without_improvement = 0
                torch.save(checkpoint, args.output_dir / "best.pt")
            else:
                checks_without_improvement += 1
            row["best_val_mAP"] = best_map
            row["best_epoch"] = best_epoch
            row["checks_without_improvement"] = checks_without_improvement
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        message = f"epoch {epoch:03d}/{args.epochs:03d} train_loss={row['train_loss']:.4f} lr={row['lr']:.2e} time={row['seconds']:.1f}s"
        if "val_mAP" in row:
            message += (
                f" val_R1={row['val_rank_1']:.4f} val_R5={row['val_rank_5']:.4f}"
                f" val_mAP={row['val_mAP']:.4f} best_mAP={best_map:.4f}@{best_epoch}"
                f" patience={checks_without_improvement}/{args.early_stop_patience}"
            )
        print(message, flush=True)
        if args.early_stop_patience > 0 and checks_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: validation mAP did not improve by "
                f"{args.early_stop_min_delta:g} for {checks_without_improvement} checks. "
                f"Best epoch={best_epoch}, best val mAP={best_map:.4f}.",
                flush=True,
            )
            break
    print(
        f"Saved last checkpoint to {args.output_dir / 'last.pt'}; "
        f"best checkpoint is {args.output_dir / 'best.pt'} (epoch={best_epoch}, val_mAP={best_map:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
