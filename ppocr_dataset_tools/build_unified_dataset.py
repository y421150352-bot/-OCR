#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ============================================================
# Paths
# ============================================================

MANGA_ROOT = Path(os.environ.get("MANGA109_ROOT", "datasets/Manga109s_released_2023_12_07"))
IMAGE_ROOT = MANGA_ROOT / "images"
MANGA_ANN_DIR = MANGA_ROOT / "annotations"
COO_ANN_DIR = MANGA_ROOT / "annotations_COO"

COO_DATA_DIR = Path(os.environ.get("COO_DATA_ROOT", "datasets/COO-Comic-Onomatopoeia/COO-data"))

OUTPUT_ROOT = Path(os.environ.get("MANGA_OCR_WORKSPACE", "manga_ppocrv6_dataset"))
OVERLAP_ANALYSIS = OUTPUT_ROOT / "component_overlap_analysis_v2.json"

UNIFIED_DIR = OUTPUT_ROOT / "unified"
AUDIT_DIR = OUTPUT_ROOT / "audit"


AUTO_MERGE_CATEGORIES = {
    "exact_duplicate",
    "group_duplicate",
    "near_duplicate",
    "near_group_duplicate",
}

AMBIGUOUS_CATEGORIES = {
    "partial_overlap",
    "different_overlap",
    "complex_overlap",
    "many_to_many",
}


# ============================================================
# Utilities
# ============================================================

def read_book_list(path: Path) -> list[str]:
    books = []

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        books.append(line)

    return books


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]


def manga_rect_to_polygon(
    bbox: list[float],
) -> list[list[float]]:
    xmin, ymin, xmax, ymax = bbox

    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]


def find_page_image(book: str, page_index: int) -> Path:
    book_dir = IMAGE_ROOT / book
    stem = f"{page_index:03d}"

    candidates = [
        book_dir / f"{stem}.jpg",
        book_dir / f"{stem}.jpeg",
        book_dir / f"{stem}.png",
        book_dir / f"{stem}.JPG",
        book_dir / f"{stem}.JPEG",
        book_dir / f"{stem}.PNG",
    ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Page image not found: book={book} page={page_index} "
        f"under {book_dir}"
    )


# ============================================================
# XML parsing
# ============================================================

def parse_manga_book(path: Path) -> dict[int, dict[str, Any]]:
    root = ET.parse(path).getroot()

    pages: dict[int, dict[str, Any]] = {}

    for page in root.findall(".//page"):
        page_index = int(page.attrib["index"])

        rows = []

        for node in page.findall("./text"):
            text = "".join(node.itertext()).strip()

            bbox = [
                float(node.attrib["xmin"]),
                float(node.attrib["ymin"]),
                float(node.attrib["xmax"]),
                float(node.attrib["ymax"]),
            ]

            rows.append({
                "id": str(node.attrib["id"]),
                "text": text,
                "bbox": bbox,
                "polygon": manga_rect_to_polygon(bbox),
            })

        pages[page_index] = {
            "width": int(page.attrib["width"]),
            "height": int(page.attrib["height"]),
            "annotations": rows,
        }

    return pages


def parse_coo_book(path: Path) -> dict[int, dict[str, Any]]:
    root = ET.parse(path).getroot()

    pages: dict[int, dict[str, Any]] = {}

    for page in root.findall(".//page"):
        page_index = int(page.attrib["index"])

        rows = []

        for node in page.findall("./onomatopoeia"):
            points = []
            i = 0

            while (
                f"x{i}" in node.attrib
                and f"y{i}" in node.attrib
            ):
                points.append([
                    float(node.attrib[f"x{i}"]),
                    float(node.attrib[f"y{i}"]),
                ])
                i += 1

            if len(points) < 3:
                continue

            text = "".join(node.itertext()).strip()

            rows.append({
                "id": str(node.attrib["id"]),
                "text": text,
                "polygon": points,
                "bbox": polygon_bbox(points),
            })

        pages[page_index] = {
            "width": int(page.attrib["width"]),
            "height": int(page.attrib["height"]),
            "annotations": rows,
        }

    return pages


# ============================================================
# Overlap-component index
# ============================================================

def load_component_index():
    payload = json.loads(
        OVERLAP_ANALYSIS.read_text(encoding="utf-8")
    )

    index = defaultdict(list)

    for component in payload["components"]:
        key = (
            str(component["book"]),
            int(component["page"]),
        )

        index[key].append(component)

    return index


# ============================================================
# Build final annotations
# ============================================================

def build_page_annotations(
    book: str,
    page_index: int,
    manga_rows: list[dict[str, Any]],
    coo_rows: list[dict[str, Any]],
    components: list[dict[str, Any]],
    split: str,
    merged_audit: list[dict[str, Any]],
    ambiguous_audit: list[dict[str, Any]],
    stats: Counter,
) -> list[dict[str, Any]]:

    manga_by_id = {
        str(row["id"]): row
        for row in manga_rows
    }

    coo_by_id = {
        str(row["id"]): row
        for row in coo_rows
    }

    if len(manga_by_id) != len(manga_rows):
        raise ValueError(
            f"Duplicate Manga109 text id: "
            f"{book} page {page_index}"
        )

    if len(coo_by_id) != len(coo_rows):
        raise ValueError(
            f"Duplicate COO id: "
            f"{book} page {page_index}"
        )

    consumed_manga: set[str] = set()
    consumed_coo: set[str] = set()

    final_rows: list[dict[str, Any]] = []

    for component_index, component in enumerate(
        components,
        1,
    ):
        category = str(component["category"])

        manga_ids = [
            str(row["id"])
            for row in component.get("manga", [])
        ]

        coo_ids = [
            str(row["id"])
            for row in component.get("coo", [])
        ]

        for manga_id in manga_ids:
            if manga_id not in manga_by_id:
                raise KeyError(
                    f"Overlap JSON Manga id not found: "
                    f"{book} page={page_index} id={manga_id}"
                )

        for coo_id in coo_ids:
            if coo_id not in coo_by_id:
                raise KeyError(
                    f"Overlap JSON COO id not found: "
                    f"{book} page={page_index} id={coo_id}"
                )

        # One raw annotation may only belong to one connected component.
        overlap_m = consumed_manga.intersection(manga_ids)
        overlap_c = consumed_coo.intersection(coo_ids)

        if overlap_m or overlap_c:
            raise RuntimeError(
                f"Annotation appears in multiple components: "
                f"{book} page={page_index} "
                f"Manga={sorted(overlap_m)} COO={sorted(overlap_c)}"
            )

        consumed_manga.update(manga_ids)
        consumed_coo.update(coo_ids)

        component_id = (
            f"{book}:p{page_index:03d}:component{component_index:03d}"
        )

        if category in AUTO_MERGE_CATEGORIES:
            stats["auto_merge_components"] += 1
            stats[f"merge_{category}"] += 1

            manga_texts = [
                manga_by_id[x]["text"]
                for x in manga_ids
            ]

            coo_texts = [
                coo_by_id[x]["text"]
                for x in coo_ids
            ]

            audit_record = {
                "component_id": component_id,
                "split": split,
                "book": book,
                "page_index": page_index,
                "category": category,
                "similarity": component.get("similarity"),
                "max_true_iom": component.get("max_true_iom"),
                "manga109_text_ids": manga_ids,
                "manga109_texts": manga_texts,
                "coo_ids": coo_ids,
                "coo_texts": coo_texts,
            }

            merged_audit.append(audit_record)

            # IMPORTANT:
            # For a duplicate component, COO geometry wins.
            #
            # 1 Manga -> N COO:
            # keep all N COO annotations separately.
            #
            # N Manga -> 1 COO:
            # keep the single COO annotation.
            for coo_id in coo_ids:
                coo = coo_by_id[coo_id]

                final_rows.append({
                    "annotation_id": f"coo:{coo_id}",
                    "source": "merged_coo",
                    "sources": [
                        "manga109",
                        "coo",
                    ],
                    "type": "onomatopoeia",
                    "text": coo["text"],
                    "polygon": coo["polygon"],
                    "bbox": coo["bbox"],

                    "coo_id": coo_id,
                    "manga109_text_ids": manga_ids,

                    "manga109_texts": manga_texts,
                    "coo_component_texts": coo_texts,

                    "merge_reason": category,
                    "component_id": component_id,
                })

                stats["final_merged_coo"] += 1

        elif category in AMBIGUOUS_CATEGORIES:
            stats["ambiguous_components"] += 1
            stats[f"ambiguous_{category}"] += 1

            stats[
                "excluded_ambiguous_manga_annotations"
            ] += len(manga_ids)

            stats[
                "excluded_ambiguous_coo_annotations"
            ] += len(coo_ids)

            ambiguous_audit.append({
                "component_id": component_id,
                "split": split,
                "book": book,
                "page_index": page_index,
                "category": category,
                "similarity": component.get("similarity"),
                "max_true_iom": component.get("max_true_iom"),

                "manga109": [
                    manga_by_id[x]
                    for x in manga_ids
                ],

                "coo": [
                    coo_by_id[x]
                    for x in coo_ids
                ],
            })

        else:
            raise ValueError(
                f"Unknown overlap category: {category}"
            )

    # --------------------------------------------------------
    # Manga109 annotations not involved in any overlap component
    # --------------------------------------------------------

    for manga_id, row in manga_by_id.items():
        if manga_id in consumed_manga:
            continue

        final_rows.append({
            "annotation_id": f"manga109:{manga_id}",
            "source": "manga109",
            "sources": ["manga109"],
            "type": "text",
            "text": row["text"],
            "polygon": row["polygon"],
            "bbox": row["bbox"],

            "manga109_text_ids": [manga_id],
            "coo_id": None,
            "merge_reason": None,
            "component_id": None,
        })

        stats["final_manga109_only"] += 1

    # --------------------------------------------------------
    # COO annotations not involved in any overlap component
    # --------------------------------------------------------

    for coo_id, row in coo_by_id.items():
        if coo_id in consumed_coo:
            continue

        final_rows.append({
            "annotation_id": f"coo:{coo_id}",
            "source": "coo",
            "sources": ["coo"],
            "type": "onomatopoeia",
            "text": row["text"],
            "polygon": row["polygon"],
            "bbox": row["bbox"],

            "coo_id": coo_id,
            "manga109_text_ids": [],
            "merge_reason": None,
            "component_id": None,
        })

        stats["final_coo_only"] += 1

    # Deterministic order.
    final_rows.sort(
        key=lambda row: (
            float(row["bbox"][1]),
            float(row["bbox"][0]),
            str(row["annotation_id"]),
        )
    )

    annotation_ids = [
        row["annotation_id"]
        for row in final_rows
    ]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise RuntimeError(
            f"Duplicate final annotation ids: "
            f"{book} page={page_index}"
        )

    return final_rows


# ============================================================
# Main
# ============================================================

def main():
    UNIFIED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_paths = {
        "train": COO_DATA_DIR / "books_train.txt",
        "val": COO_DATA_DIR / "books_val.txt",
        "test": COO_DATA_DIR / "books_test.txt",
    }

    splits = {
        name: read_book_list(path)
        for name, path in split_paths.items()
    }

    # --------------------------------------------------------
    # Validate split overlap
    # --------------------------------------------------------

    all_assignments: dict[str, str] = {}

    for split, books in splits.items():
        for book in books:
            if book in all_assignments:
                raise RuntimeError(
                    f"Book appears in multiple splits: "
                    f"{book} -> "
                    f"{all_assignments[book]} and {split}"
                )

            all_assignments[book] = split

    component_index = load_component_index()

    stats = Counter()
    merged_audit = []
    ambiguous_audit = []

    split_summaries = {}

    for split, configured_books in splits.items():
        pages_output = []

        processed_books = []
        unavailable_books = []

        for book_number, book in enumerate(
            configured_books,
            1,
        ):
            manga_xml = MANGA_ANN_DIR / f"{book}.xml"
            coo_xml = COO_ANN_DIR / f"{book}.xml"
            image_book_dir = IMAGE_ROOT / book

            # Dataset splits may contain books outside the locally
            # released Manga109 subset. Those are reported, not fatal.
            if (
                not manga_xml.is_file()
                or not image_book_dir.is_dir()
            ):
                unavailable_books.append(book)
                continue

            manga_pages = parse_manga_book(
                manga_xml
            )

            if coo_xml.is_file():
                coo_pages = parse_coo_book(
                    coo_xml
                )
            else:
                coo_pages = {}

            processed_books.append(book)
            stats[f"{split}_books"] += 1

            page_indexes = sorted(
                set(manga_pages)
                | set(coo_pages)
            )

            for page_index in page_indexes:
                manga_page = manga_pages.get(
                    page_index
                )

                coo_page = coo_pages.get(
                    page_index
                )

                # --------------------------------------------------------
                # Resolve page dimensions.
                #
                # The real image is authoritative.
                # Some COO XML pages contain width=0 / height=0, which
                # means the dimensions are unavailable rather than 0x0.
                # --------------------------------------------------------

                image_path = find_page_image(
                    book,
                    page_index,
                )

                from PIL import Image

                with Image.open(image_path) as image:
                    image_width, image_height = image.size

                width = int(image_width)
                height = int(image_height)

                # Validate Manga109 dimensions when available.
                if manga_page is not None:
                    manga_width = int(manga_page["width"])
                    manga_height = int(manga_page["height"])

                    if (
                        manga_width > 0
                        and manga_height > 0
                        and (manga_width, manga_height) != (width, height)
                    ):
                        raise ValueError(
                            f"Manga109/image dimension mismatch: "
                            f"{book} page={page_index} "
                            f"Manga109={(manga_width, manga_height)} "
                            f"image={(width, height)}"
                        )

                # Validate COO dimensions only when COO actually
                # provides a non-zero size.
                if coo_page is not None:
                    coo_width = int(coo_page["width"])
                    coo_height = int(coo_page["height"])

                    if (
                        coo_width > 0
                        and coo_height > 0
                        and (coo_width, coo_height) != (width, height)
                    ):
                        raise ValueError(
                            f"COO/image dimension mismatch: "
                            f"{book} page={page_index} "
                            f"COO={(coo_width, coo_height)} "
                            f"image={(width, height)}"
                        )

                manga_rows = (
                    manga_page["annotations"]
                    if manga_page
                    else []
                )

                coo_rows = (
                    coo_page["annotations"]
                    if coo_page
                    else []
                )

                stats["raw_manga109_annotations"] += len(
                    manga_rows
                )

                stats["raw_coo_annotations"] += len(
                    coo_rows
                )

                final_annotations = (
                    build_page_annotations(
                        book=book,
                        page_index=page_index,
                        manga_rows=manga_rows,
                        coo_rows=coo_rows,
                        components=component_index.get(
                            (book, page_index),
                            [],
                        ),
                        split=split,
                        merged_audit=merged_audit,
                        ambiguous_audit=ambiguous_audit,
                        stats=stats,
                    )
                )

                image_path = find_page_image(
                    book,
                    page_index,
                )

                image_rel = image_path.relative_to(
                    IMAGE_ROOT
                ).as_posix()

                pages_output.append({
                    "book": book,
                    "page_index": page_index,
                    "image": image_rel,
                    "width": width,
                    "height": height,
                    "annotations": final_annotations,
                })

                stats[f"{split}_pages"] += 1
                stats[
                    f"{split}_final_annotations"
                ] += len(final_annotations)

        split_payload = {
            "version": 1,
            "split": split,
            "image_root": str(IMAGE_ROOT),
            "configured_books": configured_books,
            "processed_books": processed_books,
            "unavailable_books": unavailable_books,
            "pages": pages_output,
        }

        output_path = (
            UNIFIED_DIR / f"{split}.json"
        )

        output_path.write_text(
            json.dumps(
                split_payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        split_summaries[split] = {
            "configured_books": len(
                configured_books
            ),
            "processed_books": len(
                processed_books
            ),
            "unavailable_books": len(
                unavailable_books
            ),
            "pages": len(pages_output),
            "annotations": sum(
                len(page["annotations"])
                for page in pages_output
            ),
            "output": str(output_path),
        }

        print(
            f"{split}: "
            f"books={len(processed_books)} "
            f"pages={len(pages_output)} "
            f"annotations="
            f"{split_summaries[split]['annotations']}"
        )

        if unavailable_books:
            print(
                f"  unavailable books: "
                f"{len(unavailable_books)}"
            )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    stats["final_annotations"] = (
        stats["final_manga109_only"]
        + stats["final_coo_only"]
        + stats["final_merged_coo"]
    )

    merged_path = (
        AUDIT_DIR / "merged_components.json"
    )

    ambiguous_path = (
        AUDIT_DIR / "ambiguous_overlap.json"
    )

    statistics_path = (
        AUDIT_DIR / "statistics.json"
    )

    merged_path.write_text(
        json.dumps(
            merged_audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ambiguous_path.write_text(
        json.dumps(
            ambiguous_audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    statistics = {
        "version": 1,

        "policy": {
            "auto_merge_categories": sorted(
                AUTO_MERGE_CATEGORIES
            ),
            "ambiguous_categories": sorted(
                AMBIGUOUS_CATEGORIES
            ),
            "duplicate_geometry_policy":
                "keep COO polygon",
            "ambiguous_policy":
                "exclude all annotations in ambiguous component",
        },

        "splits": split_summaries,

        "counts": dict(stats),
    }

    statistics_path.write_text(
        json.dumps(
            statistics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 68)
    print("Unified Manga109 + COO dataset")
    print("=" * 68)

    print(
        f"Raw Manga109 annotations       : "
        f"{stats['raw_manga109_annotations']}"
    )

    print(
        f"Raw COO annotations            : "
        f"{stats['raw_coo_annotations']}"
    )

    print()

    print(
        f"Auto-merge components          : "
        f"{stats['auto_merge_components']}"
    )

    for category in sorted(
        AUTO_MERGE_CATEGORIES
    ):
        print(
            f"  {category:25s}: "
            f"{stats[f'merge_{category}']}"
        )

    print()

    print(
        f"Ambiguous components excluded  : "
        f"{stats['ambiguous_components']}"
    )

    for category in sorted(
        AMBIGUOUS_CATEGORIES
    ):
        print(
            f"  {category:25s}: "
            f"{stats[f'ambiguous_{category}']}"
        )

    print()

    print(
        f"Excluded ambiguous Manga109    : "
        f"{stats['excluded_ambiguous_manga_annotations']}"
    )

    print(
        f"Excluded ambiguous COO         : "
        f"{stats['excluded_ambiguous_coo_annotations']}"
    )

    print()

    print(
        f"Final Manga109-only            : "
        f"{stats['final_manga109_only']}"
    )

    print(
        f"Final COO-only                 : "
        f"{stats['final_coo_only']}"
    )

    print(
        f"Final merged COO               : "
        f"{stats['final_merged_coo']}"
    )

    print(
        f"FINAL ANNOTATIONS              : "
        f"{stats['final_annotations']}"
    )

    print()

    print(f"Unified : {UNIFIED_DIR}")
    print(f"Merged  : {merged_path}")
    print(f"Ambig   : {ambiguous_path}")
    print(f"Stats   : {statistics_path}")


if __name__ == "__main__":
    main()
