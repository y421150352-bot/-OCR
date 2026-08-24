#!/usr/bin/env python3
"""Evaluate a named, prototype-based character bank on unseen Manga109 books.

Unlike ``evaluate.py``, gallery images are not ranked independently.  Their
embeddings are aggregated into one or more prototypes for each character, and
the query ranks character identities directly.  This matches deployment after
clusters have been reviewed and named by a human.
"""

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


INPUT_TYPES = ("face-only", "body-only", "face+body")


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.clip(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12, None)


def extract_embeddings(
    model: CharacterReIDModel,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    batch["face"].to(device, non_blocking=True),
                    batch["body"].to(device, non_blocking=True),
                    batch["face_valid"].to(device, non_blocking=True),
                    batch["body_valid"].to(device, non_blocking=True),
                )
            parts.append(output["embedding"].float().cpu().numpy())
    return normalize(np.concatenate(parts))


def rank_metrics(ranked_labels: np.ndarray, true_labels: np.ndarray) -> dict[str, float | int]:
    correct = ranked_labels == true_labels[:, None]
    positions = np.argmax(correct, axis=1)
    found = correct.any(axis=1)
    reciprocal_ranks = np.where(found, 1.0 / (positions + 1), 0.0)
    return {
        "rank_1": float(correct[:, :1].any(axis=1).mean()),
        "rank_5": float(correct[:, :5].any(axis=1).mean()),
        # With one ranked entry per identity, AP for one relevant identity is RR.
        "mAP": float(reciprocal_ranks.mean()),
        "queries": int(len(true_labels)),
    }


def choose_gallery_and_query(
    indexes: list[int],
    bank_size: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    shuffled = indexes.copy()
    rng.shuffle(shuffled)
    # Always retain at least one query. For rare identities the bank therefore
    # contains fewer than the requested maximum, matching practical cold start.
    count = min(bank_size, len(shuffled) - 1)
    return shuffled[:count], shuffled[count:]


def build_scores(
    features: np.ndarray,
    input_types: np.ndarray,
    galleries: dict[int, list[int]],
    queries: list[int],
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    identity_labels = np.asarray(sorted(galleries), dtype=np.int64)
    global_prototypes: list[np.ndarray] = []
    typed_prototypes: dict[str, list[np.ndarray | None]] = {kind: [] for kind in INPUT_TYPES}

    for label in identity_labels.tolist():
        gallery = galleries[label]
        global_prototypes.append(normalize(features[gallery].mean(axis=0)))
        for kind in INPUT_TYPES:
            selected = [index for index in gallery if input_types[index] == kind]
            typed_prototypes[kind].append(
                normalize(features[selected].mean(axis=0)) if selected else None
            )

    global_matrix = np.stack(global_prototypes)
    query_matrix = features[np.asarray(queries, dtype=np.int64)]
    scores = query_matrix @ global_matrix.T

    if strategy == "modality_aware":
        # Prefer the same-modality prototype when the named bank has one;
        # otherwise fall back to the character's global prototype.
        for row, query_index in enumerate(queries):
            kind = str(input_types[query_index])
            for column, prototype in enumerate(typed_prototypes[kind]):
                if prototype is not None:
                    scores[row, column] = float(features[query_index] @ prototype)
    elif strategy != "global_mean":
        raise ValueError(f"Unknown strategy: {strategy}")
    return scores, identity_labels


def evaluate_book_trial(
    features: np.ndarray,
    labels: np.ndarray,
    input_types: np.ndarray,
    indexes: list[int],
    bank_size: int,
    seed: int,
    strategy: str,
) -> dict:
    groups: dict[int, list[int]] = defaultdict(list)
    for index in indexes:
        groups[int(labels[index])].append(index)

    rng = random.Random(seed)
    galleries: dict[int, list[int]] = {}
    queries: list[int] = []
    actual_gallery_sizes: list[int] = []
    singleton_identities = 0
    for label in sorted(groups):
        if len(groups[label]) < 2:
            singleton_identities += 1
            continue
        gallery, identity_queries = choose_gallery_and_query(groups[label], bank_size, rng)
        galleries[label] = gallery
        queries.extend(identity_queries)
        actual_gallery_sizes.append(len(gallery))

    if not queries:
        empty = {"rank_1": 0.0, "rank_5": 0.0, "mAP": 0.0, "queries": 0}
        return {
            **empty,
            "identities": len(galleries),
            "singleton_identities": singleton_identities,
            "mean_gallery_per_identity": 0.0,
            "by_query_type": {kind: dict(empty) for kind in INPUT_TYPES},
        }

    scores, identity_labels = build_scores(
        features, input_types, galleries, queries, strategy
    )
    order = np.argsort(-scores, axis=1)
    ranked_labels = identity_labels[order]
    true_labels = labels[np.asarray(queries, dtype=np.int64)]
    result = rank_metrics(ranked_labels, true_labels)
    result.update(
        {
            "identities": len(galleries),
            "singleton_identities": singleton_identities,
            "mean_gallery_per_identity": float(np.mean(actual_gallery_sizes)),
        }
    )
    query_types = input_types[np.asarray(queries, dtype=np.int64)]
    by_type: dict[str, dict[str, float | int]] = {}
    for kind in INPUT_TYPES:
        mask = query_types == kind
        by_type[kind] = (
            rank_metrics(ranked_labels[mask], true_labels[mask])
            if mask.any()
            else {"rank_1": 0.0, "rank_5": 0.0, "mAP": 0.0, "queries": 0}
        )
    result["by_query_type"] = by_type
    return result


def weighted_books(per_book: dict[str, dict]) -> dict:
    queries = sum(int(row["queries"]) for row in per_book.values())

    def weighted(metric: str) -> float:
        return (
            sum(float(row[metric]) * int(row["queries"]) for row in per_book.values()) / queries
            if queries
            else 0.0
        )

    overall: dict = {
        "rank_1": weighted("rank_1"),
        "rank_5": weighted("rank_5"),
        "mAP": weighted("mAP"),
        "queries": queries,
        "books": len(per_book),
        "identities": sum(int(row["identities"]) for row in per_book.values()),
        "mean_gallery_per_identity": float(
            np.mean([row["mean_gallery_per_identity"] for row in per_book.values()])
        ),
    }
    by_type: dict[str, dict] = {}
    for kind in INPUT_TYPES:
        kind_queries = sum(int(row["by_query_type"][kind]["queries"]) for row in per_book.values())
        by_type[kind] = {"queries": kind_queries}
        for metric in ("rank_1", "rank_5", "mAP"):
            by_type[kind][metric] = (
                sum(
                    float(row["by_query_type"][kind][metric])
                    * int(row["by_query_type"][kind]["queries"])
                    for row in per_book.values()
                )
                / kind_queries
                if kind_queries
                else 0.0
            )
    overall["by_query_type"] = by_type
    return overall


def summarize_trials(trials: list[dict]) -> dict:
    summary: dict = {"trials": len(trials)}
    for metric in ("rank_1", "rank_5", "mAP", "queries", "mean_gallery_per_identity"):
        values = np.asarray([trial[metric] for trial in trials], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    summary["by_query_type"] = {}
    for kind in INPUT_TYPES:
        summary["by_query_type"][kind] = {}
        for metric in ("rank_1", "rank_5", "mAP", "queries"):
            values = np.asarray(
                [trial["by_query_type"][kind][metric] for trial in trials],
                dtype=np.float64,
            )
            summary["by_query_type"][kind][metric] = {
                "mean": float(values.mean()), "std": float(values.std())
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_strict/test.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-sizes", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--strategies", nargs="+", choices=["global_mean", "modality_aware"], default=["global_mean", "modality_aware"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/character_bank_metrics.json"))
    args = parser.parse_args()
    if args.trials < 1 or any(size < 1 for size in args.bank_sizes):
        raise SystemExit("trials and all bank sizes must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CharacterReIDModel(str(checkpoint["config"]["backbone"])).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = MangaReIDDataset(args.manifest, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    features = extract_embeddings(model, loader, device)
    labels = np.asarray(dataset.labels, dtype=np.int64)
    input_types = np.asarray([str(record["input_type"]) for record in dataset.records])

    by_book: dict[str, list[int]] = defaultdict(list)
    excluded_other = 0
    for index, record in enumerate(dataset.records):
        if not args.include_other and str(record.get("character_name", "")).strip().casefold() == "other":
            excluded_other += 1
            continue
        by_book[str(record["book"])].append(index)

    results: dict = {}
    for strategy in args.strategies:
        results[strategy] = {}
        for bank_size in sorted(set(args.bank_sizes)):
            trial_results: list[dict] = []
            detailed_trials: list[dict] = []
            for trial in range(args.trials):
                per_book = {
                    book: evaluate_book_trial(
                        features,
                        labels,
                        input_types,
                        indexes,
                        bank_size,
                        args.seed + trial * 1000 + book_offset,
                        strategy,
                    )
                    for book_offset, (book, indexes) in enumerate(sorted(by_book.items()))
                }
                overall = weighted_books(per_book)
                trial_results.append(overall)
                detailed_trials.append({"trial": trial, "overall": overall, "per_book": per_book})
            results[strategy][str(bank_size)] = {
                "summary": summarize_trials(trial_results),
                "trial_details": detailed_trials,
            }

    payload = {
        "protocol": "within_book_named_character_prototype_bank",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "exclude_other": not args.include_other,
        "excluded_other_instances": excluded_other,
        "bank_sizes_are_maximum": True,
        "note": "Each identity retains at least one query; rare identities use fewer gallery examples than requested.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {
        strategy: {
            bank_size: {
                "rank_1": round(row["summary"]["rank_1"]["mean"], 6),
                "rank_5": round(row["summary"]["rank_5"]["mean"], 6),
                "mAP": round(row["summary"]["mAP"]["mean"], 6),
                "rank_1_std": round(row["summary"]["rank_1"]["std"], 6),
                "queries_mean": round(row["summary"]["queries"]["mean"], 1),
                "gallery_per_id_mean": round(row["summary"]["mean_gallery_per_identity"]["mean"], 2),
            }
            for bank_size, row in sizes.items()
        }
        for strategy, sizes in results.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Saved full metrics to {args.output}")


if __name__ == "__main__":
    main()
