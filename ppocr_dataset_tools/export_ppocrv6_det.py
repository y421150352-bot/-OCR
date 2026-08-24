#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("MANGA_OCR_WORKSPACE", "manga_ppocrv6_dataset"))
UNIFIED = ROOT / "unified"
OUT = ROOT / "det"

IMAGE_ROOT = Path(os.environ.get("MANGA109_ROOT", "datasets/Manga109s_released_2023_12_07")) / "images"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clean_points(points, width, height):
    result = []

    for p in points:
        if len(p) != 2:
            continue

        x = int(round(float(p[0])))
        y = int(round(float(p[1])))

        x = clamp(x, 0, width - 1)
        y = clamp(y, 0, height - 1)

        result.append([x, y])

    # remove consecutive duplicate points
    cleaned = []

    for p in result:
        if not cleaned or p != cleaned[-1]:
            cleaned.append(p)

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    return cleaned


def polygon_area(points):
    if len(points) < 3:
        return 0.0

    area = 0.0

    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]

        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_annotations = 0
    skipped_invalid = 0
    missing_images = 0

    for split in ("train", "val", "test"):
        src = UNIFIED / f"{split}.json"

        data = json.loads(
            src.read_text(encoding="utf-8")
        )

        output_lines = []

        split_annotations = 0
        split_skipped = 0

        for page in data["pages"]:
            image_rel = page["image"]
            image_path = IMAGE_ROOT / image_rel

            if not image_path.is_file():
                print("Missing image:", image_path)
                missing_images += 1
                continue

            width = int(page["width"])
            height = int(page["height"])

            labels = []

            for ann in page["annotations"]:
                text = str(ann.get("text", ""))

                # Detection does not need text content for learning,
                # but DetLabelEncode expects a transcription.
                text = (
                    text
                    .replace("\r", "")
                    .replace("\n", "")
                    .replace("\t", " ")
                )

                points = clean_points(
                    ann["polygon"],
                    width,
                    height,
                )

                if (
                    len(points) < 3
                    or polygon_area(points) < 1.0
                ):
                    skipped_invalid += 1
                    split_skipped += 1
                    continue

                labels.append({
                    "transcription": text,
                    "points": points,
                })

            # Keep pages even if labels becomes empty.
            # This is useful as a negative/background page if such
            # pages ever exist in the dataset.
            line = (
                image_rel
                + "\t"
                + json.dumps(
                    labels,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

            output_lines.append(line)

            total_pages += 1
            total_annotations += len(labels)
            split_annotations += len(labels)

        output_path = OUT / f"{split}.txt"

        output_path.write_text(
            "\n".join(output_lines) + "\n",
            encoding="utf-8",
        )

        print(
            f"{split:5s}: "
            f"pages={len(output_lines):5d} "
            f"annotations={split_annotations:6d} "
            f"skipped={split_skipped}"
        )

    print()
    print("=" * 65)
    print("PP-OCRv6 Detection export")
    print("=" * 65)
    print(f"Pages               : {total_pages}")
    print(f"Annotations         : {total_annotations}")
    print(f"Invalid skipped     : {skipped_invalid}")
    print(f"Missing images      : {missing_images}")
    print(f"Output              : {OUT}")


if __name__ == "__main__":
    main()
