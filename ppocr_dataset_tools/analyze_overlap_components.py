#!/usr/bin/env python3

from __future__ import annotations

import json
import itertools
import os
import re
import unicodedata
import xml.etree.ElementTree as ET

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(os.environ.get("MANGA109_ROOT", "datasets/Manga109s_released_2023_12_07"))

MANGA_DIR = ROOT / "annotations"
COO_DIR = ROOT / "annotations_COO"

OUTPUT = Path(os.environ.get("MANGA_OCR_WORKSPACE", "manga_ppocrv6_dataset")) / "component_overlap_analysis.json"

OVERLAP_THRESHOLD = 0.70
NEAR_SIM_THRESHOLD = 0.85


def raw_normalize(text: str | None) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    return re.sub(r"\s+", "", text)


def core_text(text: str | None) -> str:
    """
    用于判断两个标注是不是同一段文字。

    去掉：
    空格
    标点
    波浪号
    省略号
    感叹号等

    保留真正的日文字符，比如：
    っ / ッ / ー
    """
    text = raw_normalize(text)

    result = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith("P"):
            continue

        if category.startswith("S"):
            continue

        result.append(char)

    return "".join(result)


def bbox_area(box):
    x1, y1, x2, y2 = box

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def intersection_area(a, b):
    return (
        max(
            0.0,
            min(a[2], b[2])
            - max(a[0], b[0]),
        )
        *
        max(
            0.0,
            min(a[3], b[3])
            - max(a[1], b[1]),
        )
    )


def iom(a, b):
    inter = intersection_area(a, b)

    denominator = min(
        bbox_area(a),
        bbox_area(b),
    )

    if denominator <= 0:
        return 0.0

    return inter / denominator


def parse_manga(path: Path):
    root = ET.parse(path).getroot()

    pages = defaultdict(list)

    for page in root.findall(".//page"):
        page_index = int(page.attrib["index"])

        for node in page.findall(".//text"):
            text = "".join(node.itertext()).strip()

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
        page_index = int(page.attrib["index"])

        for node in page.findall(".//onomatopoeia"):
            points = []

            index = 0

            while (
                f"x{index}" in node.attrib
                and f"y{index}" in node.attrib
            ):
                points.append([
                    float(node.attrib[f"x{index}"]),
                    float(node.attrib[f"y{index}"]),
                ])

                index += 1

            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            text = "".join(node.itertext()).strip()

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
            })

    return pages


def best_concat_match(single: str, pieces: list[str]):
    """
    Manga 一个框对应多个 COO 时，
    尝试不同 COO 顺序，看能不能拼回 Manga 文本。
    """

    single = core_text(single)

    pieces = [
        core_text(x)
        for x in pieces
        if core_text(x)
    ]

    if not pieces:
        return "", 0.0

    # 通常一个 component 不会很多块。
    # <= 6 时直接枚举顺序，最可靠。
    if len(pieces) <= 6:

        best_text = ""
        best_score = -1.0

        for order in itertools.permutations(pieces):

            joined = "".join(order)

            score = SequenceMatcher(
                None,
                single,
                joined,
            ).ratio()

            if score > best_score:
                best_score = score
                best_text = joined

        return best_text, best_score

    # 太多时避免排列组合爆炸
    joined = "".join(pieces)

    score = SequenceMatcher(
        None,
        single,
        joined,
    ).ratio()

    return joined, score


def build_components(manga_rows, coo_rows):
    """
    构造 Manga <-> COO overlap 图，
    返回连接组件。
    """

    adjacency = defaultdict(set)

    edge_overlap = {}

    for mi, manga in enumerate(manga_rows):

        for ci, coo in enumerate(coo_rows):

            overlap = iom(
                manga["bbox"],
                coo["bbox"],
            )

            if overlap < OVERLAP_THRESHOLD:
                continue

            m_node = ("m", mi)
            c_node = ("c", ci)

            adjacency[m_node].add(c_node)
            adjacency[c_node].add(m_node)

            edge_overlap[(mi, ci)] = overlap

    visited = set()
    components = []

    for node in list(adjacency):

        if node in visited:
            continue

        stack = [node]
        visited.add(node)

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

        components.append(
            (
                manga_indices,
                coo_indices,
                edge_overlap,
            )
        )

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

    manga_cores = [
        row["core"]
        for row in manga
        if row["core"]
    ]

    coo_cores = [
        row["core"]
        for row in coo
        if row["core"]
    ]

    # 1 Manga -> 1 COO
    if len(manga) == 1 and len(coo) == 1:

        m = manga[0]["core"]
        c = coo[0]["core"]

        if m == c and m:
            return "exact_duplicate", 1.0, c

        similarity = SequenceMatcher(
            None, m, c
        ).ratio()

        if similarity >= NEAR_SIM_THRESHOLD:
            return "near_duplicate", similarity, c

        if m and c and (
            m in c or c in m
        ):
            return "partial_overlap", similarity, c

        return "different_overlap", similarity, c

    # 1 Manga -> N COO
    if len(manga) == 1 and len(coo) > 1:

        joined, similarity = best_concat_match(
            manga[0]["text"],
            [row["text"] for row in coo],
        )

        if joined == manga[0]["core"]:
            return "group_duplicate", 1.0, joined

        if similarity >= NEAR_SIM_THRESHOLD:
            return "near_group_duplicate", similarity, joined

        if joined and manga[0]["core"] and (
            joined in manga[0]["core"]
            or manga[0]["core"] in joined
        ):
            return "partial_overlap", similarity, joined

        return "complex_overlap", similarity, joined

    # N Manga -> 1 COO
    if len(manga) > 1 and len(coo) == 1:

        joined, similarity = best_concat_match(
            coo[0]["text"],
            [row["text"] for row in manga],
        )

        if joined == coo[0]["core"]:
            return "group_duplicate", 1.0, joined

        if similarity >= NEAR_SIM_THRESHOLD:
            return "near_group_duplicate", similarity, joined

        if joined and coo[0]["core"] and (
            joined in coo[0]["core"]
            or coo[0]["core"] in joined
        ):
            return "partial_overlap", similarity, joined

        return "complex_overlap", similarity, joined

    # 多对多先不自动解决
    return "many_to_many", 0.0, ""


def main():

    manga_books = {
        p.stem: p
        for p in MANGA_DIR.glob("*.xml")
    }

    coo_books = {
        p.stem: p
        for p in COO_DIR.glob("*.xml")
    }

    books = sorted(
        set(manga_books)
        & set(coo_books)
    )

    category_counts = Counter()

    records = []

    for book_index, book in enumerate(books, 1):

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
                page_index, []
            )

            coo_rows = coo_pages.get(
                page_index, []
            )

            components = build_components(
                manga_rows,
                coo_rows,
            )

            for (
                manga_indices,
                coo_indices,
                edge_overlap,
            ) in components:

                category, similarity, joined = (
                    classify_component(
                        manga_rows,
                        coo_rows,
                        manga_indices,
                        coo_indices,
                    )
                )

                category_counts[category] += 1

                overlaps = []

                for mi in manga_indices:
                    for ci in coo_indices:

                        value = iom(
                            manga_rows[mi]["bbox"],
                            coo_rows[ci]["bbox"],
                        )

                        if value >= OVERLAP_THRESHOLD:
                            overlaps.append(value)

                records.append({
                    "book": book,
                    "page": page_index,
                    "category": category,
                    "similarity": round(
                        float(similarity), 4
                    ),
                    "max_iom": round(
                        max(overlaps)
                        if overlaps
                        else 0.0,
                        4,
                    ),
                    "manga": [
                        {
                            "id": manga_rows[i]["id"],
                            "text": manga_rows[i]["text"],
                            "bbox": manga_rows[i]["bbox"],
                        }
                        for i in manga_indices
                    ],
                    "coo": [
                        {
                            "id": coo_rows[i]["id"],
                            "text": coo_rows[i]["text"],
                            "bbox": coo_rows[i]["bbox"],
                        }
                        for i in coo_indices
                    ],
                    "best_joined_core": joined,
                })

        if (
            book_index % 10 == 0
            or book_index == len(books)
        ):
            print(
                f"Processed books: "
                f"{book_index}/{len(books)}"
            )

    result = {
        "parameters": {
            "overlap_threshold":
                OVERLAP_THRESHOLD,
            "near_similarity_threshold":
                NEAR_SIM_THRESHOLD,
        },
        "summary": dict(category_counts),
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
    print("=" * 64)
    print("Overlap component analysis")
    print("=" * 64)

    for name, count in (
        category_counts.most_common()
    ):
        print(
            f"{name:24s}: {count}"
        )

    print()
    print(
        f"Total components          : "
        f"{len(records)}"
    )

    safe = (
        category_counts["exact_duplicate"]
        +
        category_counts["group_duplicate"]
    )

    near = (
        category_counts["near_duplicate"]
        +
        category_counts[
            "near_group_duplicate"
        ]
    )

    print(
        f"Safe auto-merge components: {safe}"
    )

    print(
        f"Near/review components     : {near}"
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
