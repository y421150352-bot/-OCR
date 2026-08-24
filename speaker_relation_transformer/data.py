"""Page-level data loading for cached DINOv3 speaker ranking."""

from __future__ import annotations

import json
from pathlib import Path
import math
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def load_page_index(data_dir: Path, split: str) -> list[dict[str, object]]:
    path = data_dir / f"{split}_pages.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cache_filename(key: str) -> str:
    return key.replace("/", "__").replace("'", "_") + ".npz"


class PageDataset(Dataset[dict[str, object]]):
    def __init__(self, data_dir: Path, cache_dir: Path, split: str) -> None:
        self.data_dir = data_dir.resolve()
        self.cache_dir = cache_dir.resolve()
        self.split = split
        self.records = load_page_index(self.data_dir, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with np.load(self.data_dir / str(record["pack"])) as pack:
            geometry = torch.from_numpy(pack["geometry"].astype(np.float32))
            labels = torch.from_numpy(pack["labels"].astype(np.bool_))
            text_boxes = torch.from_numpy(pack["text_boxes"].astype(np.float32))
            body_boxes = torch.from_numpy(pack["body_boxes"].astype(np.float32))
        cache_path = self.cache_dir / cache_filename(str(record["key"]))
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing DINOv3 cache: {cache_path}")
        with np.load(cache_path) as cached:
            page_features = torch.from_numpy(cached["features"].astype(np.float32))
            resized_hw = torch.from_numpy(cached["resized_hw"].astype(np.float32))
            padded_hw = torch.from_numpy(cached["padded_hw"].astype(np.float32))
        return {
            "key": str(record["key"]),
            "page_features": page_features,
            "geometry": geometry,
            "labels": labels,
            "text_boxes": text_boxes,
            "body_boxes": body_boxes,
            "original_hw": torch.tensor([float(record["height"]), float(record["width"])]),
            "resized_hw": resized_hw,
            "padded_hw": padded_hw,
        }


class GeometryTextPageDataset(Dataset[dict[str, object]]):
    """Page packs plus frozen dialogue embeddings, without visual features."""

    def __init__(self, data_dir: Path, text_cache_dir: Path, split: str) -> None:
        self.data_dir = data_dir.resolve()
        self.text_cache_dir = text_cache_dir.resolve()
        self.split = split
        self.records = load_page_index(self.data_dir, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with np.load(self.data_dir / str(record["pack"])) as pack:
            geometry = torch.from_numpy(pack["geometry"].astype(np.float32))
            labels = torch.from_numpy(pack["labels"].astype(np.bool_))

        cache_path = self.text_cache_dir / self.split / cache_filename(str(record["key"]))
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing text cache: {cache_path}")
        with np.load(cache_path) as cached:
            embeddings = torch.from_numpy(cached["embeddings"].astype(np.float32))
            cached_ids = [str(value) for value in cached["text_ids"].tolist()]

        expected_ids = [str(value) for value in record["text_ids"]]
        dialogues = int(geometry.shape[0])
        if cached_ids != expected_ids:
            raise ValueError(f"{record['key']}: cached text_ids do not match page index")
        if embeddings.ndim != 2 or embeddings.shape[0] != dialogues:
            raise ValueError(
                f"{record['key']}: embeddings {tuple(embeddings.shape)} do not match "
                f"{dialogues} dialogues"
            )

        # Slot order is [previous, current, next]. Missing boundary slots remain
        # zero and are explicitly identified by text_context_mask.
        text_context = embeddings.new_zeros(dialogues, 3, embeddings.shape[-1])
        text_context_mask = torch.zeros(dialogues, 3, dtype=torch.bool)
        text_context[:, 1] = embeddings
        text_context_mask[:, 1] = True
        if dialogues > 1:
            text_context[1:, 0] = embeddings[:-1]
            text_context[:-1, 2] = embeddings[1:]
            text_context_mask[1:, 0] = True
            text_context_mask[:-1, 2] = True

        return {
            "key": str(record["key"]),
            "geometry": geometry,
            "labels": labels,
            "text_context": text_context,
            "text_context_mask": text_context_mask,
        }


class GeometryGraphPageDataset(Dataset[dict[str, object]]):
    """Geometry-only page data for the no-text two-axis Graph control."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        text_dim: int = 768,
        reid_cache_dir: Path | None = None,
    ) -> None:
        if text_dim < 1:
            raise ValueError("text_dim must be positive")
        self.data_dir = data_dir.resolve()
        self.split = split
        self.text_dim = text_dim
        self.reid_cache_dir = reid_cache_dir.resolve() if reid_cache_dir else None
        self.records = load_page_index(self.data_dir, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with np.load(self.data_dir / str(record["pack"])) as pack:
            geometry = torch.from_numpy(pack["geometry"].astype(np.float32))
            labels = torch.from_numpy(pack["labels"].astype(np.bool_))
        dialogues = int(geometry.shape[0])
        # The no-text model ignores these interface placeholders. Keeping a
        # singleton dimension lets it share the exact batching/training path
        # without requiring text JSON, E5 weights, or embedding cache files.
        result: dict[str, object] = {
            "key": str(record["key"]),
            "geometry": geometry,
            "labels": labels,
            "text_context": torch.zeros(
                dialogues, 3, self.text_dim, dtype=torch.float32
            ),
            "text_context_mask": torch.zeros(dialogues, 3, dtype=torch.bool),
        }
        if self.reid_cache_dir is not None:
            cache_path = self.reid_cache_dir / self.split / cache_filename(
                str(record["key"])
            )
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing ReID cache: {cache_path}")
            with np.load(cache_path) as cached:
                candidate_reid = torch.from_numpy(
                    cached["embeddings"].astype(np.float32)
                )
                cached_ids = [str(value) for value in cached["candidate_ids"].tolist()]
            expected_ids = [str(value) for value in record["candidate_ids"]]
            if cached_ids != expected_ids:
                raise ValueError(f"{record['key']}: ReID candidate_ids do not match")
            if candidate_reid.shape[0] != geometry.shape[1]:
                raise ValueError(f"{record['key']}: ReID candidate count mismatch")
            result["candidate_reid"] = candidate_reid
        return result


class ShuffledPageSampler(Sampler[int]):
    """Deterministic epoch-aware page sampler used for resumable training."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.size, generator=generator).tolist()

    def __len__(self) -> int:
        return self.size


class BucketedPageBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled batches with locally similar D/C page workloads."""

    def __init__(
        self,
        records: list[dict[str, object]],
        batch_size: int,
        seed: int,
        bucket_multiplier: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if bucket_multiplier < 1:
            raise ValueError("bucket_multiplier must be at least 1")
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_multiplier = bucket_multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _size_key(record: dict[str, object]) -> tuple[int, int, int]:
        queries = int(record.get("queries", 1))
        candidates = int(record.get("candidates", 1))
        width = max(int(record.get("width", 1)), 1)
        height = max(int(record.get("height", 1)), 1)
        aspect_bucket = round(8 * width / height)
        return aspect_bucket, queries * candidates, candidates

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shuffled = torch.randperm(len(self.records), generator=generator).tolist()
        pool_size = self.batch_size * self.bucket_multiplier
        batches: list[list[int]] = []
        for start in range(0, len(shuffled), pool_size):
            pool = shuffled[start : start + pool_size]
            pool.sort(key=lambda index: self._size_key(self.records[index]))
            batches.extend(
                pool[offset : offset + self.batch_size]
                for offset in range(0, len(pool), self.batch_size)
            )
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        for index in batch_order:
            yield batches[index]

    def __len__(self) -> int:
        return math.ceil(len(self.records) / self.batch_size)


def single_page_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    if len(batch) != 1:
        raise ValueError("Variable page/query shapes require batch_size=1; use gradient accumulation")
    return batch[0]


def page_batch_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Pad variable page grids and D/C axes for true multi-page GPU batches."""
    if not batch:
        raise ValueError("Cannot collate an empty page batch")
    batch_size = len(batch)
    max_patch_h = max(int(page["page_features"].shape[0]) for page in batch)
    max_patch_w = max(int(page["page_features"].shape[1]) for page in batch)
    visual_dim = int(batch[0]["page_features"].shape[-1])
    geometry_dim = int(batch[0]["geometry"].shape[-1])
    max_dialogues = max(int(page["geometry"].shape[0]) for page in batch)
    max_candidates = max(int(page["geometry"].shape[1]) for page in batch)

    page_features = torch.zeros(
        batch_size, max_patch_h, max_patch_w, visual_dim, dtype=torch.float32
    )
    patch_mask = torch.zeros(batch_size, max_patch_h, max_patch_w, dtype=torch.bool)
    feature_hw = torch.zeros(batch_size, 2, dtype=torch.long)
    geometry = torch.zeros(
        batch_size, max_dialogues, max_candidates, geometry_dim, dtype=torch.float32
    )
    labels = torch.zeros(batch_size, max_dialogues, max_candidates, dtype=torch.bool)
    text_boxes = torch.zeros(batch_size, max_dialogues, 4, dtype=torch.float32)
    body_boxes = torch.zeros(batch_size, max_candidates, 4, dtype=torch.float32)
    dialogue_mask = torch.zeros(batch_size, max_dialogues, dtype=torch.bool)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)

    for batch_index, page in enumerate(batch):
        patch_h, patch_w = page["page_features"].shape[:2]
        dialogues, candidates = page["geometry"].shape[:2]
        page_features[batch_index, :patch_h, :patch_w] = page["page_features"]
        patch_mask[batch_index, :patch_h, :patch_w] = True
        feature_hw[batch_index] = torch.tensor([patch_h, patch_w])
        geometry[batch_index, :dialogues, :candidates] = page["geometry"]
        labels[batch_index, :dialogues, :candidates] = page["labels"]
        text_boxes[batch_index, :dialogues] = page["text_boxes"]
        body_boxes[batch_index, :candidates] = page["body_boxes"]
        dialogue_mask[batch_index, :dialogues] = True
        candidate_mask[batch_index, :candidates] = True

    return {
        "batched": True,
        "key": [str(page["key"]) for page in batch],
        "page_features": page_features,
        "patch_mask": patch_mask,
        "feature_hw": feature_hw,
        "geometry": geometry,
        "labels": labels,
        "text_boxes": text_boxes,
        "body_boxes": body_boxes,
        "dialogue_mask": dialogue_mask,
        "candidate_mask": candidate_mask,
        "original_hw": torch.stack([page["original_hw"] for page in batch]),
        "resized_hw": torch.stack([page["resized_hw"] for page in batch]),
        "padded_hw": torch.stack([page["padded_hw"] for page in batch]),
    }


def geometry_text_page_batch_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Pad D/C axes for Geometry + Graph + Text training."""
    if not batch:
        raise ValueError("Cannot collate an empty page batch")
    batch_size = len(batch)
    geometry_dim = int(batch[0]["geometry"].shape[-1])
    text_dim = int(batch[0]["text_context"].shape[-1])
    max_dialogues = max(int(page["geometry"].shape[0]) for page in batch)
    max_candidates = max(int(page["geometry"].shape[1]) for page in batch)

    geometry = torch.zeros(
        batch_size, max_dialogues, max_candidates, geometry_dim, dtype=torch.float32
    )
    labels = torch.zeros(batch_size, max_dialogues, max_candidates, dtype=torch.bool)
    text_context = torch.zeros(
        batch_size, max_dialogues, 3, text_dim, dtype=torch.float32
    )
    text_context_mask = torch.zeros(batch_size, max_dialogues, 3, dtype=torch.bool)
    dialogue_mask = torch.zeros(batch_size, max_dialogues, dtype=torch.bool)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    has_reid = "candidate_reid" in batch[0]
    if any(("candidate_reid" in page) != has_reid for page in batch):
        raise ValueError("A batch cannot mix pages with and without ReID cache")
    candidate_reid = None
    if has_reid:
        reid_dim = int(batch[0]["candidate_reid"].shape[-1])
        candidate_reid = torch.zeros(
            batch_size, max_candidates, reid_dim, dtype=torch.float32
        )

    for batch_index, page in enumerate(batch):
        dialogues, candidates = page["geometry"].shape[:2]
        if int(page["text_context"].shape[-1]) != text_dim:
            raise ValueError("Text embedding dimensions differ within batch")
        geometry[batch_index, :dialogues, :candidates] = page["geometry"]
        labels[batch_index, :dialogues, :candidates] = page["labels"]
        text_context[batch_index, :dialogues] = page["text_context"]
        text_context_mask[batch_index, :dialogues] = page["text_context_mask"]
        dialogue_mask[batch_index, :dialogues] = True
        candidate_mask[batch_index, :candidates] = True
        if candidate_reid is not None:
            if int(page["candidate_reid"].shape[-1]) != candidate_reid.shape[-1]:
                raise ValueError("ReID embedding dimensions differ within batch")
            candidate_reid[batch_index, :candidates] = page["candidate_reid"]

    result: dict[str, object] = {
        "batched": True,
        "key": [str(page["key"]) for page in batch],
        "geometry": geometry,
        "labels": labels,
        "text_context": text_context,
        "text_context_mask": text_context_mask,
        "dialogue_mask": dialogue_mask,
        "candidate_mask": candidate_mask,
    }
    if candidate_reid is not None:
        result["candidate_reid"] = candidate_reid
    return result


def load_reid_cache_dim(
    reid_cache_dir: Path, dataset: GeometryGraphPageDataset
) -> int:
    if not dataset.records:
        raise ValueError("Empty dataset")
    record = dataset.records[0]
    path = reid_cache_dir / dataset.split / cache_filename(str(record["key"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing ReID cache: {path}")
    with np.load(path) as cached:
        embeddings = cached["embeddings"]
        if embeddings.ndim != 2 or embeddings.shape[-1] < 1:
            raise ValueError(f"Invalid ReID embeddings in {path}: {embeddings.shape}")
        return int(embeddings.shape[-1])


def load_text_cache_dim(text_cache_dir: Path, dataset: GeometryTextPageDataset) -> int:
    if not dataset.records:
        raise ValueError("Empty dataset")
    record = dataset.records[0]
    path = text_cache_dir / dataset.split / cache_filename(str(record["key"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing text cache: {path}")
    with np.load(path) as cached:
        embeddings = cached["embeddings"]
        if embeddings.ndim != 2 or embeddings.shape[-1] < 1:
            raise ValueError(f"Invalid text embeddings in {path}: {embeddings.shape}")
        return int(embeddings.shape[-1])


def compute_geometry_stats(data_dir: Path, output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute stable train-only normalization without loading all rows at once."""
    records = load_page_index(data_dir, "train")
    count = 0
    total: np.ndarray | None = None
    total_sq: np.ndarray | None = None
    for record in records:
        with np.load(data_dir / str(record["pack"])) as pack:
            values = pack["geometry"].astype(np.float64).reshape(-1, pack["geometry"].shape[-1])
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            total_sq = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    if not count or total is None or total_sq is None:
        raise ValueError("Empty training data")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    np.savez(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32), count=np.int64(count))
    return mean.astype(np.float32), std.astype(np.float32)
