#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


ROOT = Path(os.environ.get("MANGA109_ROOT", "datasets/Manga109s_released_2023_12_07"))
MANGA_DIR = ROOT / "annotations"
COO_DIR = ROOT / "annotations_COO"

OUTPUT = Path(os.environ.get("MANGA_OCR_WORKSPACE", "manga_ppocrv6_dataset")) / "overlap_analysis.json"

DUPLICATE_IOM_THRESHOLD = 0.70
CONFLICT_IOM_THRESHOLD = 0.70


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", text)


def rect_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iom(a, b):
    inter = intersection_area(a, b)
    denom = min(rect_area(a), rect_area(b))
    if denom <= 0:
        return 0.0
    return inter / denom


def parse_manga(path: Path):
    root = ET.parse(path).getroot()
    pages = {}

    for page in root.findall(".//page"):
        page_index = int(page.attrib["index"])
        rows = []

        for node in page.findall(".//text"):
            box = [
                float(node.attrib["xmin"]),
                float(node.attrib["ymin"]),
                float(node.attrib["xmax"]),
                float(node.attrib["ymax"]),
            ]

            raw_text = "".join(node.itertext()).strip()

            rows.append({
                "id": node.attrib.get("id"),
                "text": raw_text,
                "norm": normalize_text(raw_text),
                "bbox": box,
            })

        pages[page_index] = rows

    return pages


def parse_coo(path: Path):
    root = ET.parse(path).getroot()
    pages = {}

    for page in root.findall(".//page"):
        page_index = int(page.attrib["index"])
        rows = []

        for node in page.findall(".//onomatopoeia"):
            points = []
            index = 0

            while f"x{index}" in node.attrib and f"y{index}" in node.attrib:
                points.append([
                    float(node.attrib[f"x{index}"]),
                    float(node.attrib[f"y{index}"]),
                ])
                index += 1

            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            bbox = [min(xs), min(ys), max(xs), max(ys)]
            raw_text = "".join(node.itertext()).strip()

            rows.append({
                "id": node.attrib.get("id"),
                "text": raw_text,
                "norm": normalize_text(raw_text),
                "polygon": points,
                "bbox": bbox,
            })

        pages[page_index] = rows

    return pages


def main():
    manga_books = {p.stem: p for p in MANGA_DIR.glob("*.xml")}
    coo_books = {p.stem: p for p in COO_DIR.glob("*.xml")}

    common_books = sorted(set(manga_books) & set(coo_books))

    totals = defaultdict(int)
    duplicates = []
    conflicts = []
    book_stats = {}

    for book_idx, book in enumerate(common_books, 1):
        manga_pages = parse_manga(manga_books[book])
        coo_pages = parse_coo(coo_books[book])

        stats = defaultdict(int)

        all_pages = sorted(set(manga_pages) | set(coo_pages))

        for page_index in all_pages:
            manga_rows = manga_pages.get(page_index, [])
            coo_rows = coo_pages.get(page_index, [])

            stats["pages"] += 1
            stats["manga_text"] += len(manga_rows)
            stats["coo"] += len(coo_rows)

            totals["pages"] += 1
            totals["manga_text"] += len(manga_rows)
            totals["coo"] += len(coo_rows)

            used_manga = set()
            used_coo = set()

            # 先找“文字相同 + 空间高度重合”的重复标注
            candidates = []

            for mi, m in enumerate(manga_rows):
                for ci, c in enumerate(coo_rows):
                    overlap = iom(m["bbox"], c["bbox"])

                    if (
                        m["norm"]
                        and m["norm"] == c["norm"]
                        and overlap >= DUPLICATE_IOM_THRESHOLD
                    ):
                        candidates.append((overlap, mi, ci))

            # 一对一贪心匹配，优先 overlap 最大
            for overlap, mi, ci in sorted(candidates, reverse=True):
                if mi in used_manga or ci in used_coo:
                    continue

                used_manga.add(mi)
                used_coo.add(ci)

                m = manga_rows[mi]
                c = coo_rows[ci]

                duplicates.append({
                    "book": book,
                    "page": page_index,
                    "iom": round(overlap, 4),
                    "manga_id": m["id"],
                    "manga_text": m["text"],
                    "manga_bbox": m["bbox"],
                    "coo_id": c["id"],
                    "coo_text": c["text"],
                    "coo_bbox": c["bbox"],
                })

            duplicate_count = len(used_coo)
            stats["duplicates"] += duplicate_count
            totals["duplicates"] += duplicate_count

            # 再找空间高度重合但文字不同的疑似冲突
            for mi, m in enumerate(manga_rows):
                for ci, c in enumerate(coo_rows):
                    if mi in used_manga and ci in used_coo:
                        continue

                    overlap = iom(m["bbox"], c["bbox"])

                    if overlap >= CONFLICT_IOM_THRESHOLD and m["norm"] != c["norm"]:
                        conflicts.append({
                            "book": book,
                            "page": page_index,
                            "iom": round(overlap, 4),
                            "manga_id": m["id"],
                            "manga_text": m["text"],
                            "coo_id": c["id"],
                            "coo_text": c["text"],
                        })

            coo_extra = len(coo_rows) - len(used_coo)
            manga_extra = len(manga_rows) - len(used_manga)

            stats["coo_extra"] += coo_extra
            stats["manga_extra"] += manga_extra

            totals["coo_extra"] += coo_extra
            totals["manga_extra"] += manga_extra

        book_stats[book] = dict(stats)

        if book_idx % 10 == 0 or book_idx == len(common_books):
            print(f"Processed books: {book_idx}/{len(common_books)}")

    totals["common_books"] = len(common_books)
    totals["conflicts"] = len(conflicts)

    # 如果做 union 并去掉 duplicate，理论最终标注数
    totals["union_after_dedup"] = (
        totals["manga_text"]
        + totals["coo"]
        - totals["duplicates"]
    )

    result = {
        "parameters": {
            "duplicate_iom_threshold": DUPLICATE_IOM_THRESHOLD,
            "conflict_iom_threshold": CONFLICT_IOM_THRESHOLD,
        },
        "summary": dict(totals),
        "duplicates_sample": duplicates[:100],
        "conflicts_sample": conflicts[:100],
        "book_stats": book_stats,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("Manga109 + COO overlap analysis")
    print("=" * 60)
    print(f"Common books       : {totals['common_books']}")
    print(f"Pages              : {totals['pages']}")
    print(f"Manga109 text      : {totals['manga_text']}")
    print(f"COO onomatopoeia   : {totals['coo']}")
    print(f"Duplicates         : {totals['duplicates']}")
    print(f"COO extra          : {totals['coo_extra']}")
    print(f"Manga109 non-COO   : {totals['manga_extra']}")
    print(f"Possible conflicts : {totals['conflicts']}")
    print(f"Union after dedup   : {totals['union_after_dedup']}")
    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
