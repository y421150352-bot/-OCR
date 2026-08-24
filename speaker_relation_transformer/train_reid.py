#!/usr/bin/env python3
"""Train Face+Body character ReID with book-disjoint evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_reid import (
    FaceBodyReID,
    batch_hard_triplet_loss,
    supervised_contrastive_loss,
)
from reid_data import BookPKBatchSampler, MangaReIDDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/reid"))
    parser.add_argument("--backbone", default="pretrained/dinov3-vitb16")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/face_body_reid"))
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--p", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--books-per-batch", type=int, default=2)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size-eval", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--triplet-weight", type=float, default=0.5)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--gallery-per-id", type=int, default=3)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def embed_dataset(
    model: FaceBodyReID,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    embeddings: list[np.ndarray] = []
    keys: list[str] = []
    for batch in tqdm(loader, desc="ReID evaluate", leave=False):
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            output = model(
                batch["face"].to(device, non_blocking=True),
                batch["body"].to(device, non_blocking=True),
                batch["face_valid"].to(device, non_blocking=True),
                batch["body_valid"].to(device, non_blocking=True),
            )
        embeddings.append(output["embedding"].float().cpu().numpy())
        keys.extend(str(key) for key in batch["key"])
    return np.concatenate(embeddings), keys


def retrieval_metrics(
    embeddings: np.ndarray,
    records: list[dict[str, object]],
    gallery_per_id: int,
    seed: int,
) -> dict[str, float | int]:
    """Few-shot identity retrieval inside each unseen validation book."""
    rng = random.Random(seed)
    by_book_identity: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(records):
        if str(record.get("character_name", "")).strip().casefold() == "other":
            continue
        by_book_identity[str(record["book"])][str(record["identity"])].append(index)
    rank1 = rank5 = reciprocal_rank = queries = gallery_count = 0.0
    for identities in by_book_identity.values():
        gallery: dict[str, np.ndarray] = {}
        query_indexes: list[tuple[int, str]] = []
        for identity, indexes in identities.items():
            shuffled = indexes.copy()
            rng.shuffle(shuffled)
            take = min(gallery_per_id, max(0, len(shuffled) - 1))
            if take == 0:
                continue
            gallery_indexes = shuffled[:take]
            prototype = embeddings[gallery_indexes].mean(axis=0)
            prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
            gallery[identity] = prototype
            query_indexes.extend((index, identity) for index in shuffled[take:])
            gallery_count += take
        if not gallery:
            continue
        identity_names = list(gallery)
        prototypes = np.stack([gallery[name] for name in identity_names])
        for index, target in query_indexes:
            order = np.argsort(-(prototypes @ embeddings[index]))
            target_rank = next(
                rank
                for rank, position in enumerate(order, 1)
                if identity_names[int(position)] == target
            )
            queries += 1
            rank1 += target_rank == 1
            rank5 += target_rank <= 5
            reciprocal_rank += 1.0 / target_rank
    denominator = max(queries, 1.0)
    return {
        "rank1": rank1 / denominator,
        "rank5": rank5 / denominator,
        "mAP": reciprocal_rank / denominator,
        "queries": int(queries),
        "gallery": int(gallery_count),
        "books": len(by_book_identity),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ReID training requires CUDA")
    if args.epochs < 1 or args.p < 2 or args.k < 2:
        raise ValueError("Require epochs >= 1, P >= 2 and K >= 2")
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    train_dataset = MangaReIDDataset(
        args.data_dir / "train.jsonl",
        training=True,
        modality_dropout=args.modality_dropout,
    )
    val_dataset = MangaReIDDataset(
        args.data_dir / "val.jsonl", training=False
    )
    sampler = BookPKBatchSampler(
        train_dataset.records,
        train_dataset.labels,
        p=args.p,
        k=args.k,
        books_per_batch=args.books_per_batch,
        batches_per_epoch=args.batches_per_epoch or None,
    )
    worker_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **worker_options)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
        **worker_options,
    )
    backbone = str(Path(args.backbone).resolve())
    model = FaceBodyReID(
        backbone,
        embedding_dim=args.embedding_dim,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr / args.lr + (1.0 - args.min_lr / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "backbone": backbone,
        "train_instances": len(train_dataset),
        "val_instances": len(val_dataset),
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best_map = -1.0
    bad_epochs = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"ReID {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, 1):
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                output = model(
                    batch["face"].to(device, non_blocking=True),
                    batch["body"].to(device, non_blocking=True),
                    batch["face_valid"].to(device, non_blocking=True),
                    batch["body_valid"].to(device, non_blocking=True),
                )
                embedding = output["embedding"]
                supcon = supervised_contrastive_loss(embedding, labels)
                triplet = batch_hard_triplet_loss(
                    embedding, labels, margin=args.triplet_margin
                )
                loss = supcon + args.triplet_weight * triplet
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach())
            progress.set_postfix(loss=f"{running_loss / step:.4f}")
        embeddings, _ = embed_dataset(model, val_loader, device, amp_dtype)
        metrics = retrieval_metrics(
            embeddings, val_dataset.records, args.gallery_per_id, args.seed + epoch
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "validation": metrics,
        }
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "validation": metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        improved = float(metrics["mAP"]) > best_map
        if improved:
            best_map = float(metrics["mAP"])
            bad_epochs = 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            bad_epochs += 1
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.early_stopping_patience and bad_epochs >= args.early_stopping_patience:
            print(f"Early stopping: best validation mAP={best_map:.6f}", flush=True)
            break


if __name__ == "__main__":
    main()
