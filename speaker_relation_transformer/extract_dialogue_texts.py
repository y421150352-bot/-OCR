#!/usr/bin/env python3
"""Export Manga109 dialogue transcriptions aligned to existing page packs.

The page indexes already contain the original Manga109 ``text_id`` values, so
alignment is exact: XML text nodes are looked up by ID and their boxes are
checked against the corresponding ``text_boxes`` rows in each NPZ pack.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


BOX_KEYS = ("xmin", "ymin", "xmax", "ymax")
SPLITS = ("train", "val", "test")
MAX_ERROR_EXAMPLES = 100


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_xml_texts(annotation_path: Path) -> dict[str, dict[str, tuple[str, np.ndarray]]]:
    """Return page -> text ID -> (transcription, xyxy box)."""
    root = ET.parse(annotation_path).getroot()
    pages: dict[str, dict[str, tuple[str, np.ndarray]]] = {}
    for page in root.iter("page"):
        page_index = str(page.get("index"))
        page_texts: dict[str, tuple[str, np.ndarray]] = {}
        for node in page.findall("text"):
            text_id = node.get("id")
            if not text_id:
                continue
            if text_id in page_texts:
                raise ValueError(f"{annotation_path}: duplicate text id {text_id}")
            box = np.asarray([float(node.get(key, 0)) for key in BOX_KEYS], dtype=np.float32)
            # Preserve the official transcription exactly. ElementTree already
            # decodes XML entities and the source annotations are UTF-8.
            page_texts[text_id] = (node.text or "", box)
        pages[page_index] = page_texts
    return pages


def add_error(report: dict[str, Any], error: dict[str, Any]) -> None:
    report["error_counts"][error["type"]] += 1
    if len(report["error_examples"]) < MAX_ERROR_EXAMPLES:
        report["error_examples"].append(error)


def export_split(
    split: str,
    data_dir: Path,
    annotations_dir: Path,
    output_root: Path,
    bbox_atol: float,
) -> dict[str, Any]:
    records = read_jsonl(data_dir / f"{split}_pages.jsonl")
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_cache: dict[str, dict[str, dict[str, tuple[str, np.ndarray]]]] = {}
    report: dict[str, Any] = {
        "pages": len(records),
        "pages_written": 0,
        "dialogues": 0,
        "matched_dialogues": 0,
        "empty_transcriptions": 0,
        "error_counts": Counter(),
        "error_examples": [],
    }

    for page_number, record in enumerate(records, 1):
        book = str(record["book"])
        page_index = str(record["page_index"])
        key = str(record["key"])
        text_ids = [str(value) for value in record["text_ids"]]
        report["dialogues"] += len(text_ids)

        if book not in annotation_cache:
            annotation_path = annotations_dir / f"{book}.xml"
            if not annotation_path.is_file():
                add_error(report, {"type": "missing_annotation", "key": key, "path": str(annotation_path)})
                continue
            try:
                annotation_cache[book] = load_xml_texts(annotation_path)
            except (ET.ParseError, OSError, ValueError) as exc:
                add_error(report, {"type": "invalid_annotation", "key": key, "error": str(exc)})
                continue

        xml_pages = annotation_cache[book]
        if page_index not in xml_pages:
            add_error(report, {"type": "missing_xml_page", "key": key})
            continue
        xml_texts = xml_pages[page_index]

        pack_path = data_dir / str(record["pack"])
        if not pack_path.is_file():
            add_error(report, {"type": "missing_pack", "key": key, "path": str(pack_path)})
            continue
        with np.load(pack_path) as pack:
            if "text_boxes" not in pack:
                add_error(report, {"type": "missing_text_boxes", "key": key})
                continue
            pack_boxes = pack["text_boxes"].astype(np.float32, copy=False)

        if pack_boxes.shape != (len(text_ids), 4):
            add_error(
                report,
                {
                    "type": "shape_mismatch",
                    "key": key,
                    "text_ids": len(text_ids),
                    "text_boxes_shape": list(pack_boxes.shape),
                },
            )
            continue

        texts: list[str] = []
        xml_boxes: list[np.ndarray] = []
        missing_ids: list[str] = []
        for text_id in text_ids:
            item = xml_texts.get(text_id)
            if item is None:
                missing_ids.append(text_id)
                continue
            transcription, box = item
            texts.append(transcription)
            xml_boxes.append(box)

        if missing_ids:
            add_error(report, {"type": "missing_text_ids", "key": key, "text_ids": missing_ids})
            continue

        expected_boxes = np.stack(xml_boxes) if xml_boxes else np.empty((0, 4), dtype=np.float32)
        mismatched_rows = np.flatnonzero(
            ~np.isclose(pack_boxes, expected_boxes, rtol=0.0, atol=bbox_atol).all(axis=1)
        )
        if len(mismatched_rows):
            rows = mismatched_rows[:10].tolist()
            add_error(
                report,
                {
                    "type": "bbox_mismatch",
                    "key": key,
                    "row_indices": rows,
                    "text_ids": [text_ids[index] for index in rows],
                    "npz_boxes": pack_boxes[rows].tolist(),
                    "xml_boxes": expected_boxes[rows].tolist(),
                },
            )
            continue

        payload = {
            "schema_version": 1,
            "key": key,
            "book": book,
            "page_index": page_index,
            "text_ids": text_ids,
            # texts[i] is aligned with text_ids[i], geometry[i], labels[i], and
            # text_boxes[i] in the corresponding page pack.
            "texts": texts,
        }
        output_path = output_dir / f"{Path(str(record['pack'])).stem}.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["pages_written"] += 1
        report["matched_dialogues"] += len(texts)
        report["empty_transcriptions"] += sum(not text for text in texts)

        if page_number % 1000 == 0 or page_number == len(records):
            print(f"texts {split}: {page_number}/{len(records)}", flush=True)

    report["error_counts"] = dict(sorted(report["error_counts"].items()))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing *_pages.jsonl and packs/ (default: project data directory)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/Manga109s_released_2023_12_07"),
        help="Manga109-s root containing annotations/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Text JSON root (default: <data-dir>/texts)",
    )
    parser.add_argument(
        "--bbox-atol",
        type=float,
        default=0.0,
        help="Absolute bbox comparison tolerance (default: exact equality)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    annotations_dir = args.dataset_root.resolve() / "annotations"
    output_root = (args.output_dir or (data_dir / "texts")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    split_reports = {
        split: export_split(split, data_dir, annotations_dir, output_root, args.bbox_atol)
        for split in SPLITS
    }
    totals = {
        field: sum(int(report[field]) for report in split_reports.values())
        for field in ("pages", "pages_written", "dialogues", "matched_dialogues", "empty_transcriptions")
    }
    totals["errors"] = sum(
        sum(int(count) for count in report["error_counts"].values())
        for report in split_reports.values()
    )
    final_report = {
        "schema_version": 1,
        "data_dir": str(data_dir),
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_root),
        "bbox_atol": args.bbox_atol,
        "totals": totals,
        "splits": split_reports,
    }
    report_path = data_dir / "text_alignment_report.json"
    report_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), **totals}, ensure_ascii=False, indent=2))
    if totals["errors"]:
        print("Text export completed with alignment errors; see the report.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
