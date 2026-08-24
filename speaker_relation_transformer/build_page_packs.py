#!/usr/bin/env python3
"""Convert the geometry baseline into page-level visual training packs."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

import numpy as np
def box_from_node(node: ET.Element) -> list[float]:
    return [float(node.get(k, 0)) for k in ("xmin", "ymin", "xmax", "ymax")]


def load_book_pages(annotation_path: Path) -> dict[str, dict[str, object]]:
    root = ET.parse(annotation_path).getroot()
    pages: dict[str, dict[str, object]] = {}
    for page in root.iter("page"):
        page_index = str(page.get("index"))
        pages[page_index] = {
            "width": int(float(page.get("width", 1))),
            "height": int(float(page.get("height", 1))),
            "texts": {n.get("id"): box_from_node(n) for n in page.findall("text") if n.get("id")},
            "bodies": {n.get("id"): box_from_node(n) for n in page.findall("body") if n.get("id")},
        }
    return pages


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_relative_path(book: str, page_index: str) -> str:
    return f"images/{book}/{int(page_index):03d}.jpg"


def build_split(split: str, baseline_dir: Path, dataset_root: Path, output_dir: Path) -> dict[str, int]:
    source = np.load(baseline_dir / f"{split}.npz")
    all_x = source["X"]
    all_y = source["y"]
    groups = source["groups"]
    queries = read_jsonl(baseline_dir / f"{split}_queries.jsonl")
    if len(groups) != len(queries):
        raise ValueError(f"{split}: group/query count mismatch")

    by_page: OrderedDict[tuple[str, str], list[tuple[dict[str, object], np.ndarray, np.ndarray]]] = OrderedDict()
    offset = 0
    for query, group_size_raw in zip(queries, groups):
        group_size = int(group_size_raw)
        x = all_x[offset:offset + group_size]
        y = all_y[offset:offset + group_size]
        offset += group_size
        key = (str(query["book"]), str(query["page_index"]))
        by_page.setdefault(key, []).append((query, x, y))
    if offset != len(all_x):
        raise ValueError(f"{split}: did not consume every feature row")

    pack_dir = output_dir / "packs" / split
    pack_dir.mkdir(parents=True, exist_ok=True)
    annotation_cache: dict[str, dict[str, dict[str, object]]] = {}
    index_path = output_dir / f"{split}_pages.jsonl"
    query_total = 0
    with index_path.open("w", encoding="utf-8") as writer:
        page_total = len(by_page)
        for page_number, ((book, page_index), items) in enumerate(by_page.items(), 1):
            if book not in annotation_cache:
                annotation_cache[book] = load_book_pages(dataset_root / "annotations" / f"{book}.xml")
            page = annotation_cache[book][page_index]
            first_query = items[0][0]
            candidate_ids = [str(v) for v in first_query["candidate_ids"]]
            if any([str(v) for v in q["candidate_ids"]] != candidate_ids for q, _, _ in items):
                raise ValueError(f"{book}/{page_index}: candidates differ within page")
            text_ids = [str(q["text_id"]) for q, _, _ in items]
            texts = page["texts"]
            bodies = page["bodies"]
            geometry = np.stack([x for _, x, _ in items]).astype(np.float32)
            labels = np.stack([y for _, _, y in items]).astype(np.uint8)
            text_boxes = np.asarray([texts[text_id] for text_id in text_ids], dtype=np.float32)
            body_boxes = np.asarray([bodies[body_id] for body_id in candidate_ids], dtype=np.float32)
            safe_book = book.replace("'", "_")
            pack_name = f"{safe_book}__{int(page_index):03d}.npz"
            np.savez_compressed(
                pack_dir / pack_name,
                geometry=geometry,
                labels=labels,
                text_boxes=text_boxes,
                body_boxes=body_boxes,
            )
            record = {
                "key": f"{book}/{int(page_index):03d}",
                "book": book,
                "page_index": page_index,
                "image": image_relative_path(book, page_index),
                "pack": f"packs/{split}/{pack_name}",
                "width": page["width"],
                "height": page["height"],
                "queries": len(items),
                "candidates": len(candidate_ids),
                "text_ids": text_ids,
                "candidate_ids": candidate_ids,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            query_total += len(items)
            if page_number % 500 == 0 or page_number == page_total:
                print(f"pack {split}: {page_number}/{page_total}", flush=True)
    return {"pages": len(by_page), "queries": query_total, "rows": int(len(all_x))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("speaker_geometry_baseline/data"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/Manga109s_released_2023_12_07"))
    parser.add_argument("--output-dir", type=Path, default=Path("speaker_relation_transformer/data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("splits.json", "feature_schema.json", "dataset_summary.json"):
        shutil.copy2(args.baseline_dir / name, output_dir / name)
    summary = {
        split: build_split(split, args.baseline_dir, args.dataset_root, output_dir)
        for split in ("train", "val", "test")
    }
    (output_dir / "visual_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
