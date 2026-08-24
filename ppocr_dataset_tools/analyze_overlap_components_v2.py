#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(os.environ.get("MANGA109_ROOT", "datasets/Manga109s_released_2023_12_07"))

MANGA_DIR = ROOT / "annotations"
COO_DIR = ROOT / "annotations_COO"

OUTPUT = Path(os.environ.get("MANGA_OCR_WORKSPACE", "manga_ppocrv6_dataset")) / "component_overlap_analysis_v2.json"

# 真 polygon IoM
OVERLAP_THRESHOLD = 0.70

# near duplicate 字符串相似度
NEAR_SIM_THRESHOLD = 0.85


def normalize(text: str | None) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    return re.sub(r"\s+", "", text)


def core_text(text: str | None) -> str:
    """
    匹配时忽略多数标点和符号，
    但保留日文小假名、促音、长音等真正字符。
    """
    text = normalize(text)

    result = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith("P"):
            continue

        if category.startswith("S"):
            continue

        result.append(char)

    return "".join(result)


def bbox_overlap(a, b):
    """
    仅用于快速预筛。
    不作为最终 overlap。
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    return x2 > x1 and y2 > y1


def polygon_area(points):
    if len(points) < 3:
        return 0.0

    area = 0.0

    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]

        area += x1 * y2 - x2 * y1

    return abs(area) * 0.5


def rect_area(box):
    return max(
        0.0,
        box[2] - box[0],
    ) * max(
        0.0,
        box[3] - box[1],
    )


def polygon_rect_iom(rect, polygon):
    """
    真正计算：
      Manga109 rectangle
          ∩
      COO polygon

    为避免整页 rasterize，只在二者共同 bbox 的局部 ROI 中画 mask。
    """

    if len(polygon) < 3:
        return 0.0

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    poly_bbox = [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]

    if not bbox_overlap(rect, poly_bbox):
        return 0.0

    left = int(
        max(
            0,
            np.floor(
                min(rect[0], poly_bbox[0])
            ),
        )
    )

    top = int(
        max(
            0,
            np.floor(
                min(rect[1], poly_bbox[1])
            ),
        )
    )

    right = int(
        np.ceil(
            max(rect[2], poly_bbox[2])
        )
    )

    bottom = int(
        np.ceil(
            max(rect[3], poly_bbox[3])
        )
    )

    width = max(1, right - left + 1)
    height = max(1, bottom - top + 1)

    # 防止异常标注导致巨型 mask
    if width * height > 10_000_000:
        raise ValueError(
            f"Unexpectedly large overlap ROI: "
            f"{width}x{height}"
        )

    manga_mask = Image.new(
        "1",
        (width, height),
        0,
    )

    coo_mask = Image.new(
        "1",
        (width, height),
        0,
    )

    manga_draw = ImageDraw.Draw(manga_mask)
    coo_draw = ImageDraw.Draw(coo_mask)

    manga_draw.rectangle(
        [
            rect[0] - left,
            rect[1] - top,
            rect[2] - left,
            rect[3] - top,
        ],
        fill=1,
    )

    coo_draw.polygon(
        [
            (
                x - left,
                y - top,
            )
            for x, y in polygon
        ],
        fill=1,
    )

    manga_array = np.asarray(
        manga_mask,
        dtype=np.uint8,
    )

    coo_array = np.asarray(
        coo_mask,
        dtype=np.uint8,
    )

    intersection = np.count_nonzero(
        manga_array & coo_array
    )

    manga_pixels = np.count_nonzero(
        manga_array
    )

    coo_pixels = np.count_nonzero(
        coo_array
    )

    denominator = min(
        manga_pixels,
        coo_pixels,
    )

    if denominator <= 0:
        return 0.0

    return float(
        intersection / denominator
    )


def parse_manga(path: Path):
    root = ET.parse(path).getroot()

    pages = defaultdict(list)

    for page in root.findall(".//page"):
        page_index = int(
            page.attrib["index"]
        )

        for node in page.findall(".//text"):
            text = "".join(
                node.itertext()
            ).strip()

            pages[page_index].append({
                "kind": "manga",
                "id": node.attrib.get("id"),
                "text": text,
                "core": core_text(text),
                "bbox": [
                    float(node.attrib["xmin"]),
                    float(node.attrib["ymin"]),
                    float(node.attrib["xmax"]),
                    float(node.attrib["ymax"]),
                ],
            })

    return pages


def parse_coo(path: Path):
    root = ET.parse(path).getroot()

    pages = defaultdict(list)

    for page in root.findall(".//page"):
        page_index = int(
            page.attrib["index"]
        )

        for node in page.findall(
            ".//onomatopoeia"
        ):
            points = []

            index = 0

            while (
                f"x{index}" in node.attrib
                and
                f"y{index}" in node.attrib
            ):
                points.append([
                    float(
                        node.attrib[f"x{index}"]
                    ),
                    float(
                        node.attrib[f"y{index}"]
                    ),
                ])

                index += 1

            if len(points) < 3:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            text = "".join(
                node.itertext()
            ).strip()

            pages[page_index].append({
                "kind": "coo",
                "id": node.attrib.get("id"),
                "text": text,
                "core": core_text(text),
                "polygon": points,
                "bbox": [
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                ],
                "polygon_area": polygon_area(
                    points
                ),
            })

    return pages


def best_concat_match(
    target_text,
    piece_texts,
):
    target = core_text(target_text)

    pieces = [
        core_text(text)
        for text in piece_texts
        if core_text(text)
    ]

    if not pieces:
        return "", 0.0

    if len(pieces) <= 6:
        best_joined = ""
        best_score = -1.0

        for permutation in itertools.permutations(
            pieces
        ):
            joined = "".join(permutation)

            score = SequenceMatcher(
                None,
                target,
                joined,
            ).ratio()

            if score > best_score:
                best_score = score
                best_joined = joined

        return best_joined, best_score

    joined = "".join(pieces)

    score = SequenceMatcher(
        None,
        target,
        joined,
    ).ratio()

    return joined, score


def build_components(
    manga_rows,
    coo_rows,
):
    adjacency = defaultdict(set)
    overlaps = {}

    for mi, manga in enumerate(
        manga_rows
    ):
        for ci, coo in enumerate(
            coo_rows
        ):

            # 第一层：bbox 完全没交集就不算 polygon
            if not bbox_overlap(
                manga["bbox"],
                coo["bbox"],
            ):
                continue

            overlap = polygon_rect_iom(
                manga["bbox"],
                coo["polygon"],
            )

            if overlap < OVERLAP_THRESHOLD:
                continue

            m_node = ("m", mi)
            c_node = ("c", ci)

            adjacency[m_node].add(c_node)
            adjacency[c_node].add(m_node)

            overlaps[(mi, ci)] = overlap

    visited = set()
    components = []

    for node in adjacency:

        if node in visited:
            continue

        visited.add(node)
        stack = [node]
        nodes = []

        while stack:
            current = stack.pop()
            nodes.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        manga_indices = sorted(
            index
            for kind, index in nodes
            if kind == "m"
        )

        coo_indices = sorted(
            index
            for kind, index in nodes
            if kind == "c"
        )

        components.append({
            "manga_indices": manga_indices,
            "coo_indices": coo_indices,
            "overlaps": overlaps,
        })

    return components


def classify_component(
    manga_rows,
    coo_rows,
    manga_indices,
    coo_indices,
):
    manga = [
        manga_rows[i]
        for i in manga_indices
    ]

    coo = [
        coo_rows[i]
        for i in coo_indices
    ]

    if len(manga) == 1 and len(coo) == 1:
        m = manga[0]["core"]
        c = coo[0]["core"]

        if m and m == c:
            return (
                "exact_duplicate",
                1.0,
                c,
            )

        similarity = SequenceMatcher(
            None,
            m,
            c,
        ).ratio()

        if similarity >= NEAR_SIM_THRESHOLD:
            return (
                "near_duplicate",
                similarity,
                c,
            )

        if m and c and (
            m in c
            or
            c in m
        ):
            return (
                "partial_overlap",
                similarity,
                c,
            )

        return (
            "different_overlap",
            similarity,
            c,
        )

    if len(manga) == 1 and len(coo) > 1:
        joined, similarity = (
            best_concat_match(
                manga[0]["text"],
                [
                    row["text"]
                    for row in coo
                ],
            )
        )

        if (
            joined
            and joined == manga[0]["core"]
        ):
            return (
                "group_duplicate",
                1.0,
                joined,
            )

        if similarity >= NEAR_SIM_THRESHOLD:
            return (
                "near_group_duplicate",
                similarity,
                joined,
            )

        if (
            joined
            and manga[0]["core"]
            and (
                joined in manga[0]["core"]
                or
                manga[0]["core"] in joined
            )
        ):
            return (
                "partial_overlap",
                similarity,
                joined,
            )

        return (
            "complex_overlap",
            similarity,
            joined,
        )

    if len(manga) > 1 and len(coo) == 1:
        joined, similarity = (
            best_concat_match(
                coo[0]["text"],
                [
                    row["text"]
                    for row in manga
                ],
            )
        )

        if (
            joined
            and joined == coo[0]["core"]
        ):
            return (
                "group_duplicate",
                1.0,
                joined,
            )

        if similarity >= NEAR_SIM_THRESHOLD:
            return (
                "near_group_duplicate",
                similarity,
                joined,
            )

        if (
            joined
            and coo[0]["core"]
            and (
                joined in coo[0]["core"]
                or
                coo[0]["core"] in joined
            )
        ):
            return (
                "partial_overlap",
                similarity,
                joined,
            )

        return (
            "complex_overlap",
            similarity,
            joined,
        )

    return (
        "many_to_many",
        0.0,
        "",
    )


def main():
    manga_books = {
        path.stem: path
        for path in MANGA_DIR.glob("*.xml")
    }

    coo_books = {
        path.stem: path
        for path in COO_DIR.glob("*.xml")
    }

    books = sorted(
        set(manga_books)
        & set(coo_books)
    )

    counts = Counter()
    records = []

    for book_index, book in enumerate(
        books,
        1,
    ):
        manga_pages = parse_manga(
            manga_books[book]
        )

        coo_pages = parse_coo(
            coo_books[book]
        )

        page_indexes = sorted(
            set(manga_pages)
            | set(coo_pages)
        )

        for page_index in page_indexes:
            manga_rows = manga_pages.get(
                page_index,
                [],
            )

            coo_rows = coo_pages.get(
                page_index,
                [],
            )

            components = build_components(
                manga_rows,
                coo_rows,
            )

            for component in components:
                manga_indices = component[
                    "manga_indices"
                ]

                coo_indices = component[
                    "coo_indices"
                ]

                category, similarity, joined = (
                    classify_component(
                        manga_rows,
                        coo_rows,
                        manga_indices,
                        coo_indices,
                    )
                )

                counts[category] += 1

                overlap_values = []

                for mi in manga_indices:
                    for ci in coo_indices:
                        value = component[
                            "overlaps"
                        ].get((mi, ci))

                        if value is not None:
                            overlap_values.append(
                                value
                            )

                records.append({
                    "book": book,
                    "page": page_index,
                    "category": category,
                    "similarity": round(
                        float(similarity),
                        4,
                    ),
                    "max_true_iom": round(
                        max(overlap_values)
                        if overlap_values
                        else 0.0,
                        4,
                    ),
                    "manga": [
                        manga_rows[i]
                        for i in manga_indices
                    ],
                    "coo": [
                        coo_rows[i]
                        for i in coo_indices
                    ],
                    "best_joined_core": joined,
                })

        if (
            book_index % 10 == 0
            or
            book_index == len(books)
        ):
            print(
                f"Processed books: "
                f"{book_index}/{len(books)}"
            )

    result = {
        "method": (
            "true COO polygon vs "
            "Manga109 rectangle IoM"
        ),
        "parameters": {
            "overlap_threshold":
                OVERLAP_THRESHOLD,
            "near_similarity_threshold":
                NEAR_SIM_THRESHOLD,
        },
        "summary": dict(counts),
        "components": records,
    }

    OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 68)
    print(
        "TRUE polygon overlap "
        "component analysis"
    )
    print("=" * 68)

    for name, count in counts.most_common():
        print(
            f"{name:24s}: {count}"
        )

    safe = (
        counts["exact_duplicate"]
        + counts["group_duplicate"]
        + counts["near_duplicate"]
        + counts["near_group_duplicate"]
    )

    uncertain = (
        counts["partial_overlap"]
        + counts["different_overlap"]
        + counts["complex_overlap"]
        + counts["many_to_many"]
    )

    print()
    print(
        f"Total components          : "
        f"{len(records)}"
    )
    print(
        f"Auto-merge candidates     : "
        f"{safe}"
    )
    print(
        f"Still ambiguous components: "
        f"{uncertain}"
    )
    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
