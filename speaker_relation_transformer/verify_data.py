#!/usr/bin/env python3
"""Verify every page pack before upload to the GPU server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("speaker_relation_transformer/data"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/Manga109s_released_2023_12_07"))
    args = parser.parse_args()
    split_books: dict[str, set[str]] = {}
    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        records = read_jsonl(args.data_dir / f"{split}_pages.jsonl")
        books: set[str] = set()
        queries = rows = positives = 0
        for record in records:
            books.add(str(record["book"]))
            image_path = args.dataset_root / str(record["image"])
            pack_path = args.data_dir / str(record["pack"])
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            if not pack_path.exists():
                raise FileNotFoundError(pack_path)
            with np.load(pack_path) as pack:
                geometry = pack["geometry"]
                labels = pack["labels"]
                text_boxes = pack["text_boxes"]
                body_boxes = pack["body_boxes"]
            q = int(record["queries"])
            c = int(record["candidates"])
            if geometry.shape != (q, c, 45):
                raise ValueError(f"{record['key']}: geometry {geometry.shape} != {(q, c, 45)}")
            if labels.shape != (q, c) or text_boxes.shape != (q, 4) or body_boxes.shape != (c, 4):
                raise ValueError(f"{record['key']}: inconsistent pack shapes")
            if not np.all(labels.sum(axis=1) >= 1):
                raise ValueError(f"{record['key']}: query without positive speaker")
            if not np.isfinite(geometry).all():
                raise ValueError(f"{record['key']}: non-finite geometry")
            queries += q
            rows += q * c
            positives += int(labels.sum())
        split_books[split] = books
        summary[split] = {
            "books": len(books), "pages": len(records), "queries": queries,
            "rows": rows, "positives": positives,
        }
    if split_books["train"] & split_books["val"] or split_books["train"] & split_books["test"] or split_books["val"] & split_books["test"]:
        raise ValueError("Book-disjoint split was violated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
