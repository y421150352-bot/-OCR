#!/usr/bin/env python3
"""Export zero-example character clusters as browsable crop directories.

This reproduces the final per-book adaptive PCA-HDBSCAN configuration saved by
``evaluate_clustering.py``.  Character IDs are never used for clustering and
are hidden from exported names by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from data import MangaReIDDataset
from evaluate_clustering import (
    extract_embeddings,
    merge_seed_clusters,
    normalize,
    pca_features,
    raw_hdbscan_labels,
)
from model import CharacterReIDModel


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return value or "item"


def union_crop_box(record: dict, padding: float) -> tuple[int, int, int, int]:
    boxes = [record.get("face_box"), record.get("body_box")]
    boxes = [box for box in boxes if box is not None]
    x1 = min(int(box[0]) for box in boxes)
    y1 = min(int(box[1]) for box in boxes)
    x2 = max(int(box[2]) for box in boxes)
    y2 = max(int(box[3]) for box in boxes)
    pad_x = round((x2 - x1) * padding)
    pad_y = round((y2 - y1) * padding)
    return (
        max(0, x1 - pad_x), max(0, y1 - pad_y),
        min(int(record["width"]), x2 + pad_x),
        min(int(record["height"]), y2 + pad_y),
    )


def export_crop(
    dataset_root: Path,
    record: dict,
    destination: Path,
    crop_size: int,
    padding: float,
) -> None:
    with Image.open(dataset_root / record["image"]) as raw:
        page = ImageOps.exif_transpose(raw).convert("RGB")
        crop = page.crop(union_crop_box(record, padding))
    crop.thumbnail((crop_size, crop_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (crop_size, crop_size), "white")
    canvas.paste(crop, ((crop_size - crop.width) // 2, (crop_size - crop.height) // 2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92, subsampling=0)


def contact_sheet(
    image_paths: list[Path],
    destination: Path,
    title: str,
    columns: int,
    tile_size: int,
    maximum_images: int,
) -> None:
    selected = image_paths[:maximum_images]
    if not selected:
        return
    rows = (len(selected) + columns - 1) // columns
    header = 54
    canvas = Image.new("RGB", (columns * tile_size, header + rows * tile_size), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), title, fill="black", font=ImageFont.load_default())
    for position, path in enumerate(selected):
        with Image.open(path) as image:
            tile = image.convert("RGB")
        tile.thumbnail((tile_size - 4, tile_size - 4), Image.Resampling.LANCZOS)
        x = (position % columns) * tile_size + (tile_size - tile.width) // 2
        y = header + (position // columns) * tile_size + (tile_size - tile.height) // 2
        canvas.paste(tile, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90, subsampling=0)


def selected_parameters(metrics: dict, book: str) -> dict:
    row = metrics.get("test_per_book", {}).get(book, {})
    parameters = row.get("selected_parameters") or metrics.get("selected_parameters")
    if not parameters:
        raise KeyError(f"No selected_parameters found for book {book!r}")
    required = {
        "pca_dimensions", "min_cluster_size", "min_samples",
        "cluster_selection_method", "merge_threshold",
        "assignment_threshold", "assignment_margin",
    }
    missing = required - set(parameters)
    if missing:
        raise KeyError(f"Missing clustering parameters for {book}: {sorted(missing)}")
    return {key: parameters[key] for key in required}


def cluster_book(features: np.ndarray, parameters: dict) -> tuple[np.ndarray, dict]:
    reduced = pca_features(features, int(parameters["pca_dimensions"]))
    raw = raw_hdbscan_labels(
        reduced,
        int(parameters["min_cluster_size"]),
        int(parameters["min_samples"]),
        str(parameters["cluster_selection_method"]),
    )
    return merge_seed_clusters(
        features,
        raw,
        float(parameters["merge_threshold"]),
        float(parameters["assignment_threshold"]),
        float(parameters["assignment_margin"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_strict/test.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True, help="Adaptive clustering metrics JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cluster_export"))
    parser.add_argument("--books", nargs="*", help="Only export these books; default exports all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--sheet-columns", type=int, default=8)
    parser.add_argument("--sheet-max-images", type=int, default=64)
    parser.add_argument("--overview-max-clusters", type=int, default=120)
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--show-ground-truth", action="store_true")
    args = parser.parse_args()
    if args.crop_size < 64 or not 0 <= args.padding <= 0.5:
        raise SystemExit("Require crop-size >= 64 and padding within [0, 0.5]")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CharacterReIDModel(str(checkpoint["config"]["backbone"])).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = MangaReIDDataset(args.manifest, training=False)
    features = extract_embeddings(
        model, dataset, args.batch_size, args.workers, device, "extract cluster-export embeddings",
    )

    by_book: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        if not args.include_other and str(record.get("character_name", "")).strip().casefold() == "other":
            continue
        by_book[str(record["book"])].append(index)
    if args.books:
        requested = set(args.books)
        unknown = requested - set(by_book)
        if unknown:
            raise SystemExit(f"Unknown books: {sorted(unknown)}")
        by_book = {book: indexes for book, indexes in by_book.items() if book in requested}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for book in sorted(by_book):
        indexes = by_book[book]
        local_features = features[np.asarray(indexes, dtype=np.int64)]
        parameters = selected_parameters(metrics, book)
        predicted, diagnostics = cluster_book(local_features, parameters)
        groups: dict[int, list[int]] = defaultdict(list)
        for local_index, label in enumerate(predicted.tolist()):
            groups[int(label)].append(local_index)

        # Stable cluster order: large clusters first. Singleton labels are kept
        # separate from candidate character clusters.
        non_singletons = sorted(
            (members for members in groups.values() if len(members) >= 2),
            key=lambda members: (-len(members), min(members)),
        )
        singletons = sorted(members[0] for members in groups.values() if len(members) == 1)
        book_dir = args.output_dir / safe_name(book)
        assignments_path = book_dir / "assignments.csv"
        book_dir.mkdir(parents=True, exist_ok=True)
        assignment_rows = []
        representatives: list[Path] = []

        for cluster_number, members in enumerate(non_singletons, start=1):
            member_array = np.asarray(members, dtype=np.int64)
            prototype = normalize(local_features[member_array].mean(axis=0, keepdims=True))[0]
            scores = local_features[member_array] @ prototype
            order = member_array[np.argsort(-scores)].tolist()
            cluster_name = f"cluster_{cluster_number:04d}_n{len(members):04d}"
            cluster_dir = book_dir / "clusters" / cluster_name
            exported = []
            for rank, local_index in enumerate(order, start=1):
                global_index = indexes[local_index]
                record = dataset.records[global_index]
                suffix = ""
                if args.show_ground_truth:
                    suffix = f"_gt-{safe_name(str(record.get('character_name') or record['character_id']))}"
                filename = (
                    f"{rank:04d}_page-{int(record['page']):03d}_"
                    f"{safe_name(str(record['input_type']))}{suffix}.jpg"
                )
                destination = cluster_dir / filename
                export_crop(dataset.dataset_root, record, destination, args.crop_size, args.padding)
                exported.append(destination)
                assignment_rows.append({
                    "book": book, "key": record["key"], "page": record["page"],
                    "input_type": record["input_type"], "predicted_cluster": cluster_name,
                    "cluster_size": len(members), "prototype_similarity": f"{float(scores[np.where(member_array == local_index)[0][0]]):.6f}",
                    "is_singleton": False,
                    "ground_truth": (record["identity"] if args.show_ground_truth else ""),
                    "crop_file": destination.relative_to(book_dir).as_posix(),
                })
            contact_sheet(
                exported, cluster_dir / "contact_sheet.jpg",
                f"{book} / {cluster_name}", args.sheet_columns,
                args.crop_size, args.sheet_max_images,
            )
            representatives.append(exported[0])

        singleton_files = []
        for number, local_index in enumerate(singletons, start=1):
            global_index = indexes[local_index]
            record = dataset.records[global_index]
            suffix = ""
            if args.show_ground_truth:
                suffix = f"_gt-{safe_name(str(record.get('character_name') or record['character_id']))}"
            destination = book_dir / "singletons" / (
                f"singleton_{number:04d}_page-{int(record['page']):03d}_"
                f"{safe_name(str(record['input_type']))}{suffix}.jpg"
            )
            export_crop(dataset.dataset_root, record, destination, args.crop_size, args.padding)
            singleton_files.append(destination)
            assignment_rows.append({
                "book": book, "key": record["key"], "page": record["page"],
                "input_type": record["input_type"], "predicted_cluster": f"singleton_{number:04d}",
                "cluster_size": 1, "prototype_similarity": "1.000000",
                "is_singleton": True,
                "ground_truth": (record["identity"] if args.show_ground_truth else ""),
                "crop_file": destination.relative_to(book_dir).as_posix(),
            })

        contact_sheet(
            representatives, book_dir / "overview.jpg",
            f"{book}: one representative per non-singleton cluster",
            args.sheet_columns, args.crop_size, args.overview_max_clusters,
        )
        contact_sheet(
            singleton_files, book_dir / "singletons_contact_sheet.jpg",
            f"{book}: singleton / unresolved instances",
            args.sheet_columns, args.crop_size, args.sheet_max_images,
        )
        with assignments_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
            writer.writeheader()
            writer.writerows(assignment_rows)
        (book_dir / "clustering_info.json").write_text(json.dumps({
            "book": book,
            "instances": len(indexes),
            "candidate_clusters": len(non_singletons),
            "singletons": len(singletons),
            "parameters": parameters,
            "diagnostics": diagnostics,
            "ground_truth_visible": args.show_ground_truth,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append({
            "book": book, "instances": len(indexes),
            "candidate_clusters": len(non_singletons), "singletons": len(singletons),
        })
        print(
            f"Exported {book}: clusters={len(non_singletons)} "
            f"singletons={len(singletons)} instances={len(indexes)}",
            flush=True,
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved browsable character clusters to {args.output_dir}")


if __name__ == "__main__":
    main()
