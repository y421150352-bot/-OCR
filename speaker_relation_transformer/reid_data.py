"""Manga109 face/body crop dataset and within-book P x K sampler."""

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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def letterbox(image: Image.Image, size: int, training: bool) -> torch.Tensor:
    image = image.convert("RGB")
    if training:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.2)))
        if random.random() < 0.25:
            image = image.rotate(random.uniform(-5.0, 5.0), fillcolor="white")
    image.thumbnail((size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    tensor = TF.normalize(TF.to_tensor(canvas), MEAN, STD)
    if training and random.random() < 0.25:
        height = random.randint(max(1, size // 20), max(2, size // 4))
        width = random.randint(max(1, size // 20), max(2, size // 4))
        top = random.randint(0, size - height)
        left = random.randint(0, size - width)
        tensor[:, top : top + height, left : left + width] = 1.0
    return tensor


def jitter_box(
    box: list[int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    scale = random.uniform(-0.1, 0.1)
    dx = random.uniform(-0.05, 0.05) * width
    dy = random.uniform(-0.05, 0.05) * height
    pad_x, pad_y = width * scale / 2, height * scale / 2
    return (
        max(0, round(x1 - pad_x + dx)),
        max(0, round(y1 - pad_y + dy)),
        min(image_width, round(x2 + pad_x + dx)),
        min(image_height, round(y2 + pad_y + dy)),
    )


class MangaReIDDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest: Path,
        dataset_root: Path | None = None,
        image_size: int = 224,
        training: bool = False,
        modality_dropout: float = 0.15,
        exclude_other: bool = True,
    ) -> None:
        records = read_jsonl(manifest)
        if exclude_other:
            records = [
                record
                for record in records
                if str(record.get("character_name", "")).strip().casefold() != "other"
            ]
        metadata = json.loads(
            (manifest.parent / "metadata.json").read_text(encoding="utf-8")
        )
        self.dataset_root = (
            dataset_root.resolve()
            if dataset_root is not None
            else Path(str(metadata["dataset_root"]))
        )
        self.records = records
        self.image_size = image_size
        self.training = training
        self.modality_dropout = modality_dropout
        identities = sorted(str(record["identity"]) for record in records)
        identity_to_label = {
            identity: index for index, identity in enumerate(sorted(set(identities)))
        }
        self.labels = [identity_to_label[str(record["identity"])] for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(self.dataset_root / str(record["image"])) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        result: dict[str, object] = {
            "label": self.labels[index],
            "key": str(record["key"]),
        }
        for kind in ("face", "body"):
            box = record.get(f"{kind}_box")
            valid = box is not None
            if valid:
                crop_box = (
                    jitter_box(box, *image.size)  # type: ignore[arg-type]
                    if self.training
                    else tuple(box)  # type: ignore[arg-type]
                )
                result[kind] = letterbox(
                    image.crop(crop_box), self.image_size, self.training
                )
            else:
                result[kind] = torch.zeros(3, self.image_size, self.image_size)
            result[f"{kind}_valid"] = valid
        if self.training and result["face_valid"] and result["body_valid"]:
            draw = random.random()
            if draw < self.modality_dropout:
                result["face"].zero_()  # type: ignore[union-attr]
                result["face_valid"] = False
            elif draw < 2.0 * self.modality_dropout:
                result["body"].zero_()  # type: ignore[union-attr]
                result["body_valid"] = False
        return result


class BookPKBatchSampler(Sampler[list[int]]):
    """Sample difficult negatives from a small number of manga books."""

    def __init__(
        self,
        records: list[dict[str, object]],
        labels: list[int],
        p: int = 16,
        k: int = 4,
        books_per_batch: int = 2,
        batches_per_epoch: int | None = None,
    ) -> None:
        if k < 2:
            raise ValueError("K must be at least 2 for contrastive learning")
        self.p, self.k, self.books_per_batch = p, k, books_per_batch
        self.by_label: dict[int, list[int]] = defaultdict(list)
        labels_by_book: dict[str, set[int]] = defaultdict(set)
        for index, (record, label) in enumerate(zip(records, labels)):
            self.by_label[label].append(index)
            labels_by_book[str(record["book"])].add(label)
        self.labels_by_book = {
            book: sorted(book_labels) for book, book_labels in labels_by_book.items()
        }
        self.books = sorted(self.labels_by_book)
        self.batches_per_epoch = batches_per_epoch or max(1, len(labels) // (p * k))
        if len(self.books) < books_per_batch:
            raise ValueError("Not enough books for --books-per-batch")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _choose_identities(self) -> list[int]:
        target = math.ceil(self.p / self.books_per_batch)
        for _ in range(100):
            selected_books = random.sample(self.books, self.books_per_batch)
            selected: list[int] = []
            leftovers: list[int] = []
            for book in selected_books:
                identities = self.labels_by_book[book].copy()
                random.shuffle(identities)
                selected.extend(identities[:target])
                leftovers.extend(identities[target:])
            random.shuffle(leftovers)
            selected.extend(leftovers[: max(0, self.p - len(selected))])
            if len(selected) >= self.p:
                return random.sample(selected, self.p)
        richest = sorted(
            self.books, key=lambda book: len(self.labels_by_book[book]), reverse=True
        )[: self.books_per_batch]
        return random.sample(
            [label for book in richest for label in self.labels_by_book[book]], self.p
        )

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            identities = self._choose_identities()
            yield [
                index
                for identity in identities
                for index in (
                    random.sample(self.by_label[identity], self.k)
                    if len(self.by_label[identity]) >= self.k
                    else random.choices(self.by_label[identity], k=self.k)
                )
            ]
