#!/usr/bin/env python3
"""Evaluate zero-example, within-book character clustering.

The validation split is used only to select one global similarity threshold.
Each test book is then clustered independently without exposing character IDs to
the clustering algorithm. Ground-truth IDs are read only after clustering to
compute B-cubed, pairwise, ARI, and NMI metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb, log
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

from data import MangaReIDDataset
from model import CharacterReIDModel


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1

    def labels(self) -> np.ndarray:
        roots = [self.find(index) for index in range(len(self.parent))]
        mapping = {root: label for label, root in enumerate(sorted(set(roots)))}
        return np.asarray([mapping[root] for root in roots], dtype=np.int64)


def normalize(array: np.ndarray) -> np.ndarray:
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)


def extract_embeddings(model, dataset, batch_size, workers, device, description) -> np.ndarray:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, dynamic_ncols=True):
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
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


def mutual_knn_edges(features: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mutual-kNN undirected edges and their cosine similarities."""
    count = len(features)
    if count < 2:
        return (
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    k = min(top_k, count - 1)
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    neighbor_sets = [set(row.tolist()) for row in neighbors]
    left, right, scores = [], [], []
    for source in range(count):
        for target in neighbors[source].tolist():
            if source < target and source in neighbor_sets[target]:
                left.append(source)
                right.append(target)
                scores.append(float(similarity[source, target]))
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
    )


def cluster_from_edges(
    count: int,
    left: np.ndarray,
    right: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    union_find = UnionFind(count)
    selected = np.flatnonzero(scores >= threshold)
    # Strongest links first makes the operation deterministic and easier to
    # extend later with cannot-link constraints.
    for edge in selected[np.argsort(-scores[selected])]:
        union_find.union(int(left[edge]), int(right[edge]))
    return union_find.labels()


def hdbscan_cluster(
    features: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Cluster one book; convert HDBSCAN noise into separate singleton clusters.

    DINO embeddings are L2-normalized, so Euclidean distance is monotonic with
    cosine distance and is substantially better supported by HDBSCAN's fast
    implementations than metric="cosine".
    """
    if hdbscan is None:
        raise SystemExit(
            "Missing dependency 'hdbscan'. Install it with: pip install hdbscan>=0.8.40"
        )
    count = len(features)
    if count < min_cluster_size:
        raw = np.full(count, -1, dtype=np.int64)
    else:
        raw = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=cluster_selection_method,
            # A single cluster for an entire book is not a useful character
            # partition. Keeping this False also prevents validation B-cubed
            # from selecting the degenerate "one giant cluster + noise" case.
            allow_single_cluster=False,
            core_dist_n_jobs=1,
        ).fit_predict(np.asarray(features, dtype=np.float64))

    noise_mask = raw < 0
    completed = raw.astype(np.int64, copy=True)
    next_label = int(completed[~noise_mask].max()) + 1 if (~noise_mask).any() else 0
    completed[noise_mask] = np.arange(next_label, next_label + int(noise_mask.sum()))
    return completed, {
        "noise_instances": int(noise_mask.sum()),
        "noise_rate": float(noise_mask.mean()) if count else 0.0,
        "hdbscan_clusters": int(len(set(raw[~noise_mask].tolist()))),
    }


def pca_features(features: np.ndarray, dimensions: int) -> np.ndarray:
    """Fit PCA independently inside one unseen book, whiten, then normalize."""
    if PCA is None:
        raise SystemExit(
            "Missing dependency 'scikit-learn'. Install it with: "
            "pip install scikit-learn>=1.3"
        )
    count, width = features.shape
    components = min(dimensions, width, max(1, count - 1))
    if count < 2:
        return features.copy()
    reduced = PCA(
        n_components=components, whiten=True, svd_solver="randomized",
        random_state=3407,
    ).fit_transform(features)
    return normalize(np.asarray(reduced, dtype=np.float32))


def raw_hdbscan_labels(
    features: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> np.ndarray:
    if hdbscan is None:
        raise SystemExit(
            "Missing dependency 'hdbscan'. Install it with: pip install hdbscan>=0.8.40"
        )
    if len(features) < min_cluster_size:
        return np.full(len(features), -1, dtype=np.int64)
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=False,
        core_dist_n_jobs=1,
    ).fit_predict(np.asarray(features, dtype=np.float64))


def merge_seed_clusters(
    original_features: np.ndarray,
    raw_labels: np.ndarray,
    merge_threshold: float,
    assignment_threshold: float,
    assignment_margin: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Merge mutual-nearest seed prototypes, then conservatively attach noise.

    Mutual-nearest merging is iterative. This is deliberately less permissive
    than connected components and prevents one chain of medium similarities
    from collapsing many characters into a giant cluster.
    """
    seed_ids = sorted(int(label) for label in np.unique(raw_labels) if label >= 0)
    groups = [np.flatnonzero(raw_labels == label).tolist() for label in seed_ids]
    seed_count = len(groups)

    while len(groups) > 1:
        prototypes = normalize(np.stack([
            original_features[np.asarray(group, dtype=np.int64)].mean(axis=0)
            for group in groups
        ]))
        similarity = prototypes @ prototypes.T
        np.fill_diagonal(similarity, -np.inf)
        nearest = similarity.argmax(axis=1)
        candidates = []
        for left, right in enumerate(nearest.tolist()):
            if left < right and nearest[right] == left and similarity[left, right] >= merge_threshold:
                candidates.append((float(similarity[left, right]), left, right))
        if not candidates:
            break
        # Disjoint strongest mutual pairs can safely merge in the same round.
        used: set[int] = set()
        pairs: dict[int, int] = {}
        for _, left, right in sorted(candidates, reverse=True):
            if left not in used and right not in used:
                pairs[left] = right
                used.update((left, right))
        new_groups = []
        for index, group in enumerate(groups):
            if index in pairs:
                new_groups.append(group + groups[pairs[index]])
            elif index not in used:
                new_groups.append(group)
        groups = new_groups

    predicted = np.full(len(raw_labels), -1, dtype=np.int64)
    for label, group in enumerate(groups):
        predicted[np.asarray(group, dtype=np.int64)] = label

    noise_indexes = np.flatnonzero(raw_labels < 0)
    attached = 0
    if groups and len(noise_indexes):
        prototypes = normalize(np.stack([
            original_features[np.asarray(group, dtype=np.int64)].mean(axis=0)
            for group in groups
        ]))
        similarity = original_features[noise_indexes] @ prototypes.T
        nearest = similarity.argmax(axis=1)
        scores = similarity[np.arange(len(noise_indexes)), nearest]
        if similarity.shape[1] >= 2:
            top_two = np.partition(similarity, kth=-2, axis=1)[:, -2:]
            margins = top_two[:, 1] - top_two[:, 0]
        else:
            # With one candidate prototype there is no ambiguous runner-up.
            margins = np.full(len(noise_indexes), np.inf, dtype=np.float32)
        accepted = (scores >= assignment_threshold) & (margins >= assignment_margin)
        predicted[noise_indexes[accepted]] = nearest[accepted]
        attached = int(accepted.sum())

    remaining = np.flatnonzero(predicted < 0)
    next_label = len(groups)
    predicted[remaining] = np.arange(next_label, next_label + len(remaining))
    largest = int(max((len(group) for group in groups), default=0))
    return predicted, {
        "seed_clusters": seed_count,
        "merged_prototype_clusters": len(groups),
        "raw_noise_instances": int(len(noise_indexes)),
        "attached_noise_instances": attached,
        "assignment_margin": float(assignment_margin),
        "remaining_noise_instances": int(len(remaining)),
        "remaining_noise_rate": float(len(remaining) / len(raw_labels)) if len(raw_labels) else 0.0,
        "largest_seed_or_merged_cluster_before_assignment": largest,
    }


def contingency(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    true_values, true_inverse = np.unique(true, return_inverse=True)
    pred_values, pred_inverse = np.unique(predicted, return_inverse=True)
    matrix = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    np.add.at(matrix, (true_inverse, pred_inverse), 1)
    return matrix


def clustering_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    table = contingency(true, predicted)
    total = int(table.sum())
    true_sizes = table.sum(axis=1)
    pred_sizes = table.sum(axis=0)

    # B-cubed: per-instance precision/recall, efficiently aggregated by cells.
    b_precision = float(
        sum((cell * cell) / pred_sizes[column] for row in range(table.shape[0])
            for column, cell in enumerate(table[row]) if cell)
        / max(1, total)
    )
    b_recall = float(
        sum((cell * cell) / true_sizes[row] for row in range(table.shape[0])
            for cell in table[row] if cell)
        / max(1, total)
    )
    b_f1 = 2 * b_precision * b_recall / max(1e-12, b_precision + b_recall)

    true_pairs = sum(comb(int(size), 2) for size in true_sizes)
    pred_pairs = sum(comb(int(size), 2) for size in pred_sizes)
    correct_pairs = sum(comb(int(cell), 2) for cell in table.ravel())
    pair_precision = correct_pairs / pred_pairs if pred_pairs else 1.0
    pair_recall = correct_pairs / true_pairs if true_pairs else 1.0
    pair_f1 = 2 * pair_precision * pair_recall / max(1e-12, pair_precision + pair_recall)

    all_pairs = comb(total, 2) if total >= 2 else 0
    expected = true_pairs * pred_pairs / all_pairs if all_pairs else 0.0
    maximum = 0.5 * (true_pairs + pred_pairs)
    ari = (correct_pairs - expected) / (maximum - expected) if maximum != expected else 1.0

    mutual_information = 0.0
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            cell = int(table[row, column])
            if cell:
                mutual_information += (cell / total) * log(
                    (cell * total) / (true_sizes[row] * pred_sizes[column])
                )
    true_entropy = -sum((size / total) * log(size / total) for size in true_sizes if size)
    pred_entropy = -sum((size / total) * log(size / total) for size in pred_sizes if size)
    nmi = mutual_information / max(1e-12, (true_entropy + pred_entropy) / 2)

    return {
        "b_cubed_precision": b_precision,
        "b_cubed_recall": b_recall,
        "b_cubed_f1": b_f1,
        "pairwise_precision": float(pair_precision),
        "pairwise_recall": float(pair_recall),
        "pairwise_f1": float(pair_f1),
        "ari": float(ari),
        "nmi": float(nmi),
        "instances": total,
        "true_clusters": int(len(true_sizes)),
        "predicted_clusters": int(len(pred_sizes)),
        "cluster_count_error": int(len(pred_sizes) - len(true_sizes)),
        "singleton_predicted_clusters": int((pred_sizes == 1).sum()),
        "largest_predicted_cluster": int(pred_sizes.max()) if len(pred_sizes) else 0,
    }


def filtered_books(dataset: MangaReIDDataset, include_other: bool) -> tuple[dict[str, list[int]], int]:
    books: dict[str, list[int]] = defaultdict(list)
    excluded = 0
    for index, record in enumerate(dataset.records):
        if not include_other and str(record.get("character_name", "")).strip().casefold() == "other":
            excluded += 1
            continue
        books[str(record["book"])].append(index)
    return dict(books), excluded


def prepare_books(dataset, features, top_k, include_other):
    books, excluded = filtered_books(dataset, include_other)
    prepared = {}
    for book, indexes in sorted(books.items()):
        local_features = features[np.asarray(indexes, dtype=np.int64)]
        left, right, scores = mutual_knn_edges(local_features, top_k)
        true = np.asarray([dataset.labels[index] for index in indexes], dtype=np.int64)
        prepared[book] = {
            "indexes": indexes, "features": local_features, "true": true,
            "left": left, "right": right, "scores": scores,
        }
    return prepared, excluded


def evaluate_threshold(prepared: dict, threshold: float) -> tuple[dict, dict]:
    per_book = {}
    for book, data in prepared.items():
        predicted = cluster_from_edges(
            len(data["indexes"]), data["left"], data["right"],
            data["scores"], threshold,
        )
        per_book[book] = clustering_metrics(data["true"], predicted)

    total = sum(int(row["instances"]) for row in per_book.values())
    weighted_keys = (
        "b_cubed_precision", "b_cubed_recall", "b_cubed_f1",
        "pairwise_precision", "pairwise_recall", "pairwise_f1", "ari", "nmi",
    )
    overall = {
        key: (
            sum(float(row[key]) * int(row["instances"]) for row in per_book.values()) / total
            if total else 0.0
        )
        for key in weighted_keys
    }
    overall.update(
        {
            "instances": total,
            "books": len(per_book),
            "true_clusters": sum(int(row["true_clusters"]) for row in per_book.values()),
            "predicted_clusters": sum(int(row["predicted_clusters"]) for row in per_book.values()),
            "cluster_count_error": sum(int(row["cluster_count_error"]) for row in per_book.values()),
        }
    )
    precision, recall = overall["pairwise_precision"], overall["pairwise_recall"]
    overall["pairwise_f0_5"] = 1.25 * precision * recall / max(1e-12, 0.25 * precision + recall)
    return overall, per_book


def evaluate_hdbscan(
    prepared: dict,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> tuple[dict, dict]:
    per_book = {}
    for book, data in prepared.items():
        predicted, diagnostics = hdbscan_cluster(
            data["features"], min_cluster_size, min_samples,
            cluster_selection_method,
        )
        per_book[book] = {
            **clustering_metrics(data["true"], predicted),
            **diagnostics,
        }

    total = sum(int(row["instances"]) for row in per_book.values())
    weighted_keys = (
        "b_cubed_precision", "b_cubed_recall", "b_cubed_f1",
        "pairwise_precision", "pairwise_recall", "pairwise_f1", "ari", "nmi",
        "noise_rate",
    )
    overall = {
        key: (
            sum(float(row[key]) * int(row["instances"]) for row in per_book.values()) / total
            if total else 0.0
        )
        for key in weighted_keys
    }
    overall.update({
        "instances": total,
        "books": len(per_book),
        "true_clusters": sum(int(row["true_clusters"]) for row in per_book.values()),
        "predicted_clusters": sum(int(row["predicted_clusters"]) for row in per_book.values()),
        "cluster_count_error": sum(int(row["cluster_count_error"]) for row in per_book.values()),
        "noise_instances": sum(int(row["noise_instances"]) for row in per_book.values()),
        "hdbscan_clusters": sum(int(row["hdbscan_clusters"]) for row in per_book.values()),
    })
    precision, recall = overall["pairwise_precision"], overall["pairwise_recall"]
    overall["pairwise_f0_5"] = 1.25 * precision * recall / max(1e-12, 0.25 * precision + recall)
    return overall, per_book


def evaluate_pca_hdbscan(
    prepared: dict,
    pca_dimensions: int,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    merge_threshold: float,
    assignment_threshold: float,
    assignment_margin: float,
    pca_cache: dict,
    seed_cache: dict,
) -> tuple[dict, dict]:
    per_book = {}
    for book, data in prepared.items():
        pca_key = (book, pca_dimensions)
        if pca_key not in pca_cache:
            pca_cache[pca_key] = pca_features(data["features"], pca_dimensions)
        seed_key = (
            book, pca_dimensions, min_cluster_size, min_samples,
            cluster_selection_method,
        )
        if seed_key not in seed_cache:
            seed_cache[seed_key] = raw_hdbscan_labels(
                pca_cache[pca_key], min_cluster_size, min_samples,
                cluster_selection_method,
            )
        predicted, diagnostics = merge_seed_clusters(
            data["features"], seed_cache[seed_key], merge_threshold,
            assignment_threshold, assignment_margin,
        )
        metrics = clustering_metrics(data["true"], predicted)
        per_book[book] = {
            **metrics,
            **diagnostics,
            "largest_predicted_cluster_ratio": (
                float(metrics["largest_predicted_cluster"]) / int(metrics["instances"])
                if metrics["instances"] else 0.0
            ),
        }

    total = sum(int(row["instances"]) for row in per_book.values())
    weighted_keys = (
        "b_cubed_precision", "b_cubed_recall", "b_cubed_f1",
        "pairwise_precision", "pairwise_recall", "pairwise_f1", "ari", "nmi",
        "remaining_noise_rate", "largest_predicted_cluster_ratio",
    )
    overall = {
        key: (
            sum(float(row[key]) * int(row["instances"]) for row in per_book.values()) / total
            if total else 0.0
        )
        for key in weighted_keys
    }
    precision = overall["pairwise_precision"]
    recall = overall["pairwise_recall"]
    beta_squared = 0.25
    overall["pairwise_f0_5"] = (
        (1 + beta_squared) * precision * recall /
        max(1e-12, beta_squared * precision + recall)
    )
    overall.update({
        "instances": total,
        "books": len(per_book),
        "true_clusters": sum(int(row["true_clusters"]) for row in per_book.values()),
        "predicted_clusters": sum(int(row["predicted_clusters"]) for row in per_book.values()),
        "cluster_count_error": sum(int(row["cluster_count_error"]) for row in per_book.values()),
        "seed_clusters": sum(int(row["seed_clusters"]) for row in per_book.values()),
        "merged_prototype_clusters": sum(int(row["merged_prototype_clusters"]) for row in per_book.values()),
        "raw_noise_instances": sum(int(row["raw_noise_instances"]) for row in per_book.values()),
        "attached_noise_instances": sum(int(row["attached_noise_instances"]) for row in per_book.values()),
        "remaining_noise_instances": sum(int(row["remaining_noise_instances"]) for row in per_book.values()),
        "max_book_largest_cluster_ratio": max(
            (float(row["largest_predicted_cluster_ratio"]) for row in per_book.values()),
            default=0.0,
        ),
    })
    return overall, per_book


def adaptive_pca_hdbscan_test(
    prepared: dict,
    ranked_parameters: list[dict],
    maximum_cluster_ratio: float,
    maximum_noise_rate: float,
) -> tuple[dict, dict]:
    """Choose a safe validation-ranked configuration independently per book.

    The choice uses only predicted cluster size and remaining-noise diagnostics.
    Ground-truth IDs are accessed strictly after a configuration is selected.
    """
    per_book = {}
    pca_cache: dict = {}
    seed_cache: dict = {}
    for book, data in prepared.items():
        attempts = []
        selected = None
        safest = None
        for rank, parameters in enumerate(ranked_parameters, start=1):
            pca_key = (book, parameters["pca_dimensions"])
            if pca_key not in pca_cache:
                pca_cache[pca_key] = pca_features(
                    data["features"], parameters["pca_dimensions"],
                )
            seed_key = (
                book, parameters["pca_dimensions"], parameters["min_cluster_size"],
                parameters["min_samples"], parameters["cluster_selection_method"],
            )
            if seed_key not in seed_cache:
                seed_cache[seed_key] = raw_hdbscan_labels(
                    pca_cache[pca_key], parameters["min_cluster_size"],
                    parameters["min_samples"], parameters["cluster_selection_method"],
                )
            predicted, diagnostics = merge_seed_clusters(
                data["features"], seed_cache[seed_key],
                parameters["merge_threshold"], parameters["assignment_threshold"],
                parameters["assignment_margin"],
            )
            counts = np.bincount(predicted)
            largest_ratio = float(counts.max() / len(predicted)) if len(predicted) else 0.0
            noise_rate = float(diagnostics["remaining_noise_rate"])
            candidate = {
                "rank": rank, "parameters": parameters, "predicted": predicted,
                "diagnostics": diagnostics,
                "largest_predicted_cluster_ratio": largest_ratio,
            }
            attempts.append({
                "rank": rank,
                "parameters": parameters,
                "largest_predicted_cluster_ratio": largest_ratio,
                "remaining_noise_rate": noise_rate,
            })
            risk = (
                max(0.0, largest_ratio - maximum_cluster_ratio)
                + max(0.0, noise_rate - maximum_noise_rate),
                largest_ratio,
                noise_rate,
                rank,
            )
            if safest is None or risk < safest[0]:
                safest = (risk, candidate)
            if largest_ratio <= maximum_cluster_ratio and noise_rate <= maximum_noise_rate:
                selected = candidate
                break
        if selected is None:
            selected = safest[1]

        metrics = clustering_metrics(data["true"], selected["predicted"])
        per_book[book] = {
            **metrics,
            **selected["diagnostics"],
            "largest_predicted_cluster_ratio": selected["largest_predicted_cluster_ratio"],
            "adaptive_configuration_rank": selected["rank"],
            "adaptive_fallback_used": selected["rank"] > 1,
            "selected_parameters": selected["parameters"],
            "adaptive_attempts": attempts,
        }
        print(
            f"test adaptive book={book} rank={selected['rank']} "
            f"largest={selected['largest_predicted_cluster_ratio']:.4f} "
            f"noise={selected['diagnostics']['remaining_noise_rate']:.4f}",
            flush=True,
        )

    total = sum(int(row["instances"]) for row in per_book.values())
    weighted_keys = (
        "b_cubed_precision", "b_cubed_recall", "b_cubed_f1",
        "pairwise_precision", "pairwise_recall", "pairwise_f1", "ari", "nmi",
        "remaining_noise_rate", "largest_predicted_cluster_ratio",
    )
    overall = {
        key: sum(float(row[key]) * int(row["instances"]) for row in per_book.values()) / total
        if total else 0.0
        for key in weighted_keys
    }
    precision, recall = overall["pairwise_precision"], overall["pairwise_recall"]
    overall["pairwise_f0_5"] = 1.25 * precision * recall / max(
        1e-12, 0.25 * precision + recall,
    )
    overall.update({
        "instances": total,
        "books": len(per_book),
        "true_clusters": sum(int(row["true_clusters"]) for row in per_book.values()),
        "predicted_clusters": sum(int(row["predicted_clusters"]) for row in per_book.values()),
        "cluster_count_error": sum(int(row["cluster_count_error"]) for row in per_book.values()),
        "seed_clusters": sum(int(row["seed_clusters"]) for row in per_book.values()),
        "merged_prototype_clusters": sum(int(row["merged_prototype_clusters"]) for row in per_book.values()),
        "raw_noise_instances": sum(int(row["raw_noise_instances"]) for row in per_book.values()),
        "attached_noise_instances": sum(int(row["attached_noise_instances"]) for row in per_book.values()),
        "remaining_noise_instances": sum(int(row["remaining_noise_instances"]) for row in per_book.values()),
        "adaptive_fallback_books": sum(bool(row["adaptive_fallback_used"]) for row in per_book.values()),
        "max_book_largest_cluster_ratio": max(
            (float(row["largest_predicted_cluster_ratio"]) for row in per_book.values()),
            default=0.0,
        ),
    })
    return overall, per_book


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-manifest", type=Path, default=Path("data_strict/val.jsonl"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data_strict/test.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--algorithm", choices=["pca_hdbscan", "hdbscan", "mutual_knn"],
        default="pca_hdbscan",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-cluster-sizes", type=int, nargs="+", default=[2, 3, 5, 8, 10])
    parser.add_argument("--min-samples", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument(
        "--cluster-selection-methods", nargs="+", choices=["eom", "leaf"],
        default=["eom", "leaf"],
    )
    parser.add_argument("--pca-dimensions", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--merge-thresholds", type=float, nargs="+", default=[0.78, 0.84])
    parser.add_argument("--assignment-thresholds", type=float, nargs="+", default=[0.72, 0.78])
    parser.add_argument("--assignment-margins", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--adaptive-max-cluster-ratio", type=float, default=0.50)
    parser.add_argument("--adaptive-max-noise-rate", type=float, default=0.60)
    parser.add_argument("--adaptive-max-candidates", type=int, default=40)
    parser.add_argument(
        "--selection-metric",
        choices=["pairwise_f0_5", "b_cubed_f1", "pairwise_f1", "ari", "nmi"],
        default="pairwise_f0_5",
        help="Validation selection metric; pairwise F0.5 prioritizes cluster purity over recall",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/clustering_metrics.json"))
    args = parser.parse_args()
    if args.top_k < 1 or not args.thresholds:
        raise SystemExit("top-k and threshold list must be non-empty and positive")
    if any(value < 2 for value in args.min_cluster_sizes):
        raise SystemExit("Every min-cluster-size must be >= 2")
    if any(value < 1 for value in args.min_samples):
        raise SystemExit("Every min-samples value must be >= 1")
    if any(value < 2 for value in args.pca_dimensions):
        raise SystemExit("Every PCA dimension must be >= 2")
    if any(not 0.0 <= value <= 1.0 for value in args.merge_thresholds + args.assignment_thresholds + args.assignment_margins):
        raise SystemExit("Merge, assignment, and margin values must be within [0, 1]")
    if not 0.0 < args.adaptive_max_cluster_ratio <= 1.0:
        raise SystemExit("adaptive-max-cluster-ratio must be within (0, 1]")
    if not 0.0 <= args.adaptive_max_noise_rate <= 1.0:
        raise SystemExit("adaptive-max-noise-rate must be within [0, 1]")
    if args.adaptive_max_candidates < 1:
        raise SystemExit("adaptive-max-candidates must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CharacterReIDModel(str(checkpoint["config"]["backbone"])).to(device)
    model.load_state_dict(checkpoint["model"])

    val_dataset = MangaReIDDataset(args.val_manifest, training=False)
    test_dataset = MangaReIDDataset(args.test_manifest, training=False)
    val_features = extract_embeddings(
        model, val_dataset, args.batch_size, args.workers, device, "extract val embeddings",
    )
    test_features = extract_embeddings(
        model, test_dataset, args.batch_size, args.workers, device, "extract test embeddings",
    )
    val_prepared, val_excluded = prepare_books(val_dataset, val_features, args.top_k, args.include_other)
    test_prepared, test_excluded = prepare_books(test_dataset, test_features, args.top_k, args.include_other)

    calibration = []
    if args.algorithm == "pca_hdbscan":
        pca_cache: dict = {}
        seed_cache: dict = {}
        for pca_dimensions in sorted(set(args.pca_dimensions)):
            for min_cluster_size in sorted(set(args.min_cluster_sizes)):
                for min_samples in sorted(set(args.min_samples)):
                    for cluster_selection_method in args.cluster_selection_methods:
                        for merge_threshold in sorted(set(args.merge_thresholds)):
                            for assignment_threshold in sorted(set(args.assignment_thresholds)):
                                for assignment_margin in sorted(set(args.assignment_margins)):
                                    overall, _ = evaluate_pca_hdbscan(
                                        val_prepared, pca_dimensions, min_cluster_size,
                                        min_samples, cluster_selection_method,
                                        merge_threshold, assignment_threshold,
                                        assignment_margin, pca_cache, seed_cache,
                                    )
                                    calibration.append({
                                        "pca_dimensions": pca_dimensions,
                                        "min_cluster_size": min_cluster_size,
                                        "min_samples": min_samples,
                                        "cluster_selection_method": cluster_selection_method,
                                        "merge_threshold": merge_threshold,
                                        "assignment_threshold": assignment_threshold,
                                        "assignment_margin": assignment_margin,
                                        **overall,
                                    })
                                    print(
                                        f"val PCA-HDBSCAN dim={pca_dimensions} "
                                        f"method={cluster_selection_method} mcs={min_cluster_size} "
                                        f"ms={min_samples} merge={merge_threshold:.2f} "
                                        f"assign={assignment_threshold:.2f} "
                                        f"margin={assignment_margin:.2f} "
                                        f"pair_P={overall['pairwise_precision']:.4f} "
                                        f"pair_R={overall['pairwise_recall']:.4f} "
                                        f"F0.5={overall['pairwise_f0_5']:.4f} "
                                        f"ARI={overall['ari']:.4f} "
                                        f"remain_noise={overall['remaining_noise_rate']:.4f} "
                                        f"clusters={overall['merged_prototype_clusters']}",
                                        flush=True,
                                    )
        # Reject clearly unusable giant-cluster solutions when alternatives
        # exist. Test labels remain untouched; constraints are validation-only.
        feasible = [
            row for row in calibration
            if row["max_book_largest_cluster_ratio"] <= 0.60
            and row["remaining_noise_rate"] <= 0.70
        ]
        candidates = feasible or calibration
        best = max(
            candidates,
            key=lambda row: (
                row[args.selection_metric], row["pairwise_precision"],
                row["ari"], -row["remaining_noise_rate"],
            ),
        )
        selected_parameters = {
            "pca_dimensions": int(best["pca_dimensions"]),
            "min_cluster_size": int(best["min_cluster_size"]),
            "min_samples": int(best["min_samples"]),
            "cluster_selection_method": str(best["cluster_selection_method"]),
            "merge_threshold": float(best["merge_threshold"]),
            "assignment_threshold": float(best["assignment_threshold"]),
            "assignment_margin": float(best["assignment_margin"]),
        }
        ranked_rows = sorted(
            candidates,
            key=lambda row: (
                row[args.selection_metric], row["pairwise_precision"], row["ari"],
                -row["remaining_noise_rate"],
            ),
            reverse=True,
        )
        parameter_keys = (
            "pca_dimensions", "min_cluster_size", "min_samples",
            "cluster_selection_method", "merge_threshold",
            "assignment_threshold", "assignment_margin",
        )
        # First cover diverse PCA/HDBSCAN structures, then fill remaining
        # slots by validation rank. This gives a difficult new book genuinely
        # different fallbacks instead of many near-identical thresholds.
        diverse, seen_structures = [], set()
        for row in ranked_rows:
            structure = (
                row["pca_dimensions"], row["min_cluster_size"],
                row["min_samples"], row["cluster_selection_method"],
            )
            if structure not in seen_structures:
                diverse.append(row)
                seen_structures.add(structure)
        ordered_rows = diverse + [row for row in ranked_rows if row not in diverse]
        ranked_parameters = [
            {key: row[key] for key in parameter_keys}
            for row in ordered_rows[:args.adaptive_max_candidates]
        ]
        # The global best must always be the first attempted configuration.
        if selected_parameters in ranked_parameters:
            ranked_parameters.remove(selected_parameters)
        ranked_parameters.insert(0, selected_parameters)
        ranked_parameters = ranked_parameters[:args.adaptive_max_candidates]
        test_overall, test_per_book = adaptive_pca_hdbscan_test(
            test_prepared, ranked_parameters,
            args.adaptive_max_cluster_ratio, args.adaptive_max_noise_rate,
        )
    elif args.algorithm == "hdbscan":
        for min_cluster_size in sorted(set(args.min_cluster_sizes)):
            for min_samples in sorted(set(args.min_samples)):
                for cluster_selection_method in args.cluster_selection_methods:
                    overall, _ = evaluate_hdbscan(
                        val_prepared, min_cluster_size, min_samples,
                        cluster_selection_method,
                    )
                    calibration.append({
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        "cluster_selection_method": cluster_selection_method,
                        **overall,
                    })
                    print(
                        f"val HDBSCAN method={cluster_selection_method} "
                        f"min_cluster_size={min_cluster_size} "
                        f"min_samples={min_samples} "
                        f"B3_F1={overall['b_cubed_f1']:.4f} "
                        f"pair_F1={overall['pairwise_f1']:.4f} "
                        f"ARI={overall['ari']:.4f} noise={overall['noise_rate']:.4f} "
                        f"clusters={overall['hdbscan_clusters']}",
                        flush=True,
                    )
        best = max(
            calibration,
            key=lambda row: (
                row[args.selection_metric], row["b_cubed_precision"],
                -row["noise_rate"], -row["min_cluster_size"], -row["min_samples"],
            ),
        )
        selected_parameters = {
            "min_cluster_size": int(best["min_cluster_size"]),
            "min_samples": int(best["min_samples"]),
            "cluster_selection_method": str(best["cluster_selection_method"]),
        }
        test_overall, test_per_book = evaluate_hdbscan(
            test_prepared, **selected_parameters,
        )
    else:
        for threshold in sorted(set(args.thresholds)):
            overall, _ = evaluate_threshold(val_prepared, threshold)
            calibration.append({"threshold": threshold, **overall})
        best = max(
            calibration,
            key=lambda row: (
                row[args.selection_metric], row["b_cubed_precision"], row["threshold"],
            ),
        )
        selected_parameters = {"threshold": float(best["threshold"]), "top_k": args.top_k}
        test_overall, test_per_book = evaluate_threshold(
            test_prepared, selected_parameters["threshold"],
        )

    payload = {
        "protocol": f"zero_example_within_book_{args.algorithm}_clustering",
        "algorithm": args.algorithm,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "exclude_other": not args.include_other,
        "selected_parameters": selected_parameters,
        "parameters_selected_on": "validation_books",
        "selection_metric": args.selection_metric,
        "adaptive_safety": (
            {
                "enabled": True,
                "maximum_cluster_ratio": args.adaptive_max_cluster_ratio,
                "maximum_remaining_noise_rate": args.adaptive_max_noise_rate,
                "maximum_validation_ranked_candidates": args.adaptive_max_candidates,
                "selection_uses_test_labels": False,
            }
            if args.algorithm == "pca_hdbscan" else {"enabled": False}
        ),
        "validation_excluded_other_instances": val_excluded,
        "test_excluded_other_instances": test_excluded,
        "validation_parameter_sweep": calibration,
        "test_overall_instance_weighted": test_overall,
        "test_per_book": test_per_book,
        "important_note": "Character IDs are not used by clustering or adaptive fallback. Validation IDs rank parameter sets; each test book chooses among them only by predicted maximum-cluster ratio and remaining-noise rate. Test IDs are read afterward for metrics. Remaining noise becomes unique singleton clusters.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "algorithm": args.algorithm,
        "selected_parameters": selected_parameters,
        "selection_metric": args.selection_metric,
        "validation_best": best,
        "test_overall": test_overall,
        "test_per_book": test_per_book,
    }, ensure_ascii=False, indent=2))
    print(f"Saved full metrics to {args.output}")


if __name__ == "__main__":
    main()
