"""Datasets, crop preprocessing, and P x K identity sampling."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def letterbox(image: Image.Image, size: int, training: bool) -> torch.Tensor:
    image = image.convert("RGB")
    if training:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.2)))
        if random.random() < 0.25:
            image = image.rotate(random.uniform(-5, 5), fillcolor="white")
    image.thumbnail((size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    tensor = TF.to_tensor(canvas)
    if training and random.random() < 0.25:
        h = random.randint(max(1, size // 20), max(2, size // 4))
        w = random.randint(max(1, size // 20), max(2, size // 4))
        y, x = random.randint(0, size - h), random.randint(0, size - w)
        tensor[:, y:y + h, x:x + w] = 1.0
    return TF.normalize(tensor, MEAN, STD)


def jitter_box(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    scale = random.uniform(-0.1, 0.1)
    dx, dy = random.uniform(-0.05, 0.05) * bw, random.uniform(-0.05, 0.05) * bh
    pad_x, pad_y = bw * scale / 2, bh * scale / 2
    return (max(0, round(x1 - pad_x + dx)), max(0, round(y1 - pad_y + dy)), min(width, round(x2 + pad_x + dx)), min(height, round(y2 + pad_y + dy)))


class MangaReIDDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        dataset_root: Path | None = None,
        image_size: int = 224,
        training: bool = False,
        exclude_other: bool = False,
        exclude_face_only: bool = False,
        modality_dropout: float = 0.0,
    ):
        records = read_jsonl(manifest)
        if exclude_other:
            records = [r for r in records if str(r.get("character_name", "")).strip().casefold() != "other"]
        if exclude_face_only:
            records = [r for r in records if r.get("input_type") != "face-only"]
        self.records = records
        if dataset_root is None:
            metadata = json.loads((manifest.parent / "metadata.json").read_text(encoding="utf-8"))
            dataset_root = Path(metadata["dataset_root"])
        if not 0.0 <= modality_dropout <= 0.5:
            raise ValueError("modality_dropout must be between 0 and 0.5")
        self.dataset_root, self.image_size, self.training = dataset_root, image_size, training
        self.modality_dropout = modality_dropout
        identities = sorted({r["identity"] for r in self.records})
        self.identity_to_label = {value: index for index, value in enumerate(identities)}
        self.labels = [self.identity_to_label[r["identity"]] for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(self.dataset_root / record["image"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        # Keep batches limited to collatable tensors/scalars/strings.  The raw
        # record contains optional None boxes for face-only/body-only samples,
        # which PyTorch's default_collate cannot handle and is not used by the
        # trainer or evaluator.
        result: dict = {"label": self.labels[index], "key": record["key"]}
        for kind in ("face", "body"):
            box = record[f"{kind}_box"]
            valid = box is not None
            if valid:
                crop_box = jitter_box(box, *image.size) if self.training else tuple(box)
                crop = image.crop(crop_box)
                result[kind] = letterbox(crop, self.image_size, self.training)
            else:
                result[kind] = torch.zeros(3, self.image_size, self.image_size)
            result[f"{kind}_valid"] = valid
        # A paired crop is sometimes presented as one modality. This simulates
        # detector misses without relying on the very small native face-only set.
        if self.training and result["face_valid"] and result["body_valid"]:
            draw = random.random()
            if draw < self.modality_dropout:
                result["face"].zero_()
                result["face_valid"] = False
            elif draw < 2 * self.modality_dropout:
                result["body"].zero_()
                result["body_valid"] = False
        return result


class PKBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: list[int], p: int = 16, k: int = 4, batches_per_epoch: int | None = None):
        self.p, self.k = p, k
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_label[label].append(index)
        self.identities = list(self.by_label)
        if len(self.identities) < p:
            raise ValueError(f"P={p}, but the dataset only has {len(self.identities)} identities")
        self.batches_per_epoch = batches_per_epoch or max(1, len(labels) // (p * k))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen = random.sample(self.identities, self.p)
            yield [index for label in chosen for index in (random.sample(self.by_label[label], self.k) if len(self.by_label[label]) >= self.k else random.choices(self.by_label[label], k=self.k))]


class BookPKBatchSampler(Sampler[list[int]]):
    """P x K sampler whose negative identities come from few selected books."""

    def __init__(
        self,
        records: list[dict],
        labels: list[int],
        p: int = 16,
        k: int = 4,
        books_per_batch: int = 2,
        batches_per_epoch: int | None = None,
    ):
        if books_per_batch < 1 or p < books_per_batch:
            raise ValueError("Require 1 <= books_per_batch <= P")
        self.p, self.k, self.books_per_batch = p, k, books_per_batch
        self.by_label: dict[int, list[int]] = defaultdict(list)
        self.labels_by_book: dict[str, list[int]] = defaultdict(list)
        for index, (record, label) in enumerate(zip(records, labels)):
            self.by_label[label].append(index)
            self.labels_by_book[str(record["book"])].append(label)
        self.labels_by_book = {
            book: sorted(set(book_labels)) for book, book_labels in self.labels_by_book.items()
        }
        self.books = sorted(self.labels_by_book)
        if len(self.books) < books_per_batch:
            raise ValueError(f"Need {books_per_batch} books, found {len(self.books)}")
        if sum(sorted((len(v) for v in self.labels_by_book.values()), reverse=True)[:books_per_batch]) < p:
            raise ValueError(f"No {books_per_batch} books contain P={p} identities in total")
        self.batches_per_epoch = batches_per_epoch or max(1, len(labels) // (p * k))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def choose_identities(self) -> list[int]:
        # Prefer an even identity allocation, then fill shortages from another
        # selected book. Retry only when the selected books cannot provide P.
        target = math.ceil(self.p / self.books_per_batch)
        for _ in range(100):
            books = random.sample(self.books, self.books_per_batch)
            chosen: list[int] = []
            leftovers: list[int] = []
            for book in books:
                identities = self.labels_by_book[book].copy()
                random.shuffle(identities)
                chosen.extend(identities[:target])
                leftovers.extend(identities[target:])
            if len(chosen) < self.p:
                random.shuffle(leftovers)
                chosen.extend(leftovers[: self.p - len(chosen)])
            if len(chosen) >= self.p:
                return random.sample(chosen, self.p)
        # Deterministic fallback using the most identity-rich books.
        books = sorted(self.books, key=lambda b: len(self.labels_by_book[b]), reverse=True)[:self.books_per_batch]
        pool = [label for book in books for label in self.labels_by_book[book]]
        return random.sample(pool, self.p)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen = self.choose_identities()
            yield [
                index
                for label in chosen
                for index in (
                    random.sample(self.by_label[label], self.k)
                    if len(self.by_label[label]) >= self.k
                    else random.choices(self.by_label[label], k=self.k)
                )
            ]
