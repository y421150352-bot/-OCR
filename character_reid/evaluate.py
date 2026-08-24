#!/usr/bin/env python3
"""Evaluate character retrieval inside each book, excluding Manga109 ``Other``."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MangaReIDDataset
from model import CharacterReIDModel


def book_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    input_types: np.ndarray,
    indexes: list[int],
    gallery_per_id: int,
    seed: int,
) -> dict[str, float | int]:
    """Build a gallery and rank queries using candidates from one book only."""
    groups: dict[int, list[int]] = defaultdict(list)
    for index in indexes:
        groups[int(labels[index])].append(index)

    rng = random.Random(seed)
    gallery: list[int] = []
    query: list[int] = []
    eligible_identities = 0
    singleton_identities = 0
    for label in sorted(groups):
        identity_indexes = groups[label].copy()
        rng.shuffle(identity_indexes)
        if len(identity_indexes) < 2:
            singleton_identities += 1
            continue
        eligible_identities += 1
        count = min(gallery_per_id, len(identity_indexes) - 1)
        gallery.extend(identity_indexes[:count])
        query.extend(identity_indexes[count:])

    if not query or not gallery:
        return {
            "rank_1": 0.0, "rank_5": 0.0, "mAP": 0.0,
            "queries": 0, "gallery": len(gallery),
            "eligible_identities": eligible_identities,
            "singleton_identities": singleton_identities,
            "by_query_type": {
                input_type: {"rank_1": 0.0, "rank_5": 0.0, "mAP": 0.0, "queries": 0}
                for input_type in ("face-only", "body-only", "face+body")
            },
        }

    gallery_array = np.asarray(gallery, dtype=np.int64)
    query_array = np.asarray(query, dtype=np.int64)
    similarity = features[query_array] @ features[gallery_array].T
    ranked = labels[gallery_array][np.argsort(-similarity, axis=1)]
    relevant = ranked == labels[query_array, None]
    average_precisions: list[float] = []
    for row in relevant:
        positions = np.flatnonzero(row)
        average_precisions.append(
            float(np.mean(np.arange(1, len(positions) + 1) / (positions + 1)))
            if len(positions) else 0.0
        )
    result: dict[str, object] = {
        "rank_1": float(relevant[:, :1].any(axis=1).mean()),
        "rank_5": float(relevant[:, :5].any(axis=1).mean()),
        "mAP": float(np.mean(average_precisions)),
        "queries": int(len(query_array)),
        "gallery": int(len(gallery_array)),
        "eligible_identities": eligible_identities,
        "singleton_identities": singleton_identities,
    }
    by_query_type: dict[str, dict[str, float | int]] = {}
    query_types = input_types[query_array]
    for input_type in ("face-only", "body-only", "face+body"):
        mask = query_types == input_type
        count = int(mask.sum())
        if not count:
            by_query_type[input_type] = {"rank_1": 0.0, "rank_5": 0.0, "mAP": 0.0, "queries": 0}
            continue
        selected = relevant[mask]
        selected_aps = np.asarray(average_precisions)[mask]
        by_query_type[input_type] = {
            "rank_1": float(selected[:, :1].any(axis=1).mean()),
            "rank_5": float(selected[:, :5].any(axis=1).mean()),
            "mAP": float(selected_aps.mean()),
            "queries": count,
        }
    result["by_query_type"] = by_query_type
    return result  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gallery-per-id", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--include-other", action="store_true", help="Include character_name=Other (excluded by default)")
    parser.add_argument("--output", type=Path, default=Path("runs/test_metrics_per_book.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CharacterReIDModel(str(checkpoint["config"]["backbone"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = MangaReIDDataset(args.manifest)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    features: list[np.ndarray] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    batch["face"].to(device, non_blocking=True),
                    batch["body"].to(device, non_blocking=True),
                    batch["face_valid"].to(device, non_blocking=True),
                    batch["body_valid"].to(device, non_blocking=True),
                )
            features.append(output["embedding"].float().cpu().numpy())
            labels.extend(batch["label"].numpy().tolist())
    features_np = np.concatenate(features)
    labels_np = np.asarray(labels)
    input_types_np = np.asarray([str(record["input_type"]) for record in dataset.records])

    by_book: dict[str, list[int]] = defaultdict(list)
    excluded_other = 0
    for index, record in enumerate(dataset.records):
        if not args.include_other and str(record.get("character_name", "")).strip().casefold() == "other":
            excluded_other += 1
            continue
        by_book[str(record["book"])].append(index)

    per_book: dict[str, dict[str, float | int]] = {}
    for offset, book in enumerate(sorted(by_book)):
        per_book[book] = book_metrics(
            features_np, labels_np, input_types_np, by_book[book], args.gallery_per_id, args.seed + offset,
        )

    total_queries = sum(int(row["queries"]) for row in per_book.values())
    def weighted(metric: str) -> float:
        return (
            sum(float(row[metric]) * int(row["queries"]) for row in per_book.values()) / total_queries
            if total_queries else 0.0
        )

    overall = {
        "rank_1": weighted("rank_1"),
        "rank_5": weighted("rank_5"),
        "mAP": weighted("mAP"),
        "queries": total_queries,
        "gallery": sum(int(row["gallery"]) for row in per_book.values()),
        "books": len(per_book),
        "eligible_identities": sum(int(row["eligible_identities"]) for row in per_book.values()),
        "singleton_identities": sum(int(row["singleton_identities"]) for row in per_book.values()),
        "excluded_other_instances": excluded_other,
    }
    overall_by_query_type: dict[str, dict[str, float | int]] = {}
    for input_type in ("face-only", "body-only", "face+body"):
        type_queries = sum(int(row["by_query_type"][input_type]["queries"]) for row in per_book.values())  # type: ignore[index]
        type_result: dict[str, float | int] = {"queries": type_queries}
        for metric in ("rank_1", "rank_5", "mAP"):
            type_result[metric] = (
                sum(
                    float(row["by_query_type"][input_type][metric]) * int(row["by_query_type"][input_type]["queries"])  # type: ignore[index]
                    for row in per_book.values()
                ) / type_queries if type_queries else 0.0
            )
        overall_by_query_type[input_type] = type_result
    overall["by_query_type"] = overall_by_query_type
    result = {
        "protocol": "within_book_gallery_query",
        "exclude_other": not args.include_other,
        "gallery_per_identity": args.gallery_per_id,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "overall_query_weighted": overall,
        "per_book": per_book,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
