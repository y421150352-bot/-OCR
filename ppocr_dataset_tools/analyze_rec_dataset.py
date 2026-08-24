#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DATA_ROOT = Path(os.environ.get("MANGA_OCR_WORKSPACE", "manga_ppocrv6_dataset"))
UNIFIED_DIR = DATA_ROOT / "unified"

DICT_PATH = Path(os.environ.get("PADDLEOCR_ROOT", "PaddleOCR")) / "ppocr/utils/dict/ppocrv6_dict.txt"

OUTPUT_PATH = (
    DATA_ROOT / "audit" / "rec_dataset_statistics.json"
)


def clean_for_rec(text: str) -> str:
    """
    不做 Unicode 归一化，不修改原始字符。
    这里只处理无法直接放进 TSV label 的换行/Tab。
    """
    text = text.replace("\r", "")
    text = text.replace("\n", "")
    text = text.replace("\t", " ")
    return text


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    pos = (len(values) - 1) * p / 100.0

    left = math.floor(pos)
    right = math.ceil(pos)

    if left == right:
        return float(values[left])

    weight = pos - left

    return (
        values[left] * (1.0 - weight)
        + values[right] * weight
    )


def char_desc(ch):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "UNKNOWN"

    return {
        "char": ch,
        "repr": repr(ch),
        "codepoint": f"U+{ord(ch):04X}",
        "name": name,
    }


def main():
    if not DICT_PATH.is_file():
        raise FileNotFoundError(
            f"Dictionary not found: {DICT_PATH}"
        )

    # PaddleOCR dictionary is effectively character-level.
    dict_tokens = [
        line.rstrip("\r\n")
        for line in DICT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.rstrip("\r\n")
    ]

    single_char_dict = {
        token
        for token in dict_tokens
        if len(token) == 1
    }

    multi_char_tokens = [
        token
        for token in dict_tokens
        if len(token) != 1
    ]

    # Config has use_space_char: true
    allowed_chars = set(single_char_dict)
    allowed_chars.add(" ")

    total = 0
    split_counts = Counter()
    source_counts = Counter()

    raw_multiline = 0
    raw_tab = 0
    empty_after_clean = 0

    lengths = []
    split_lengths = defaultdict(list)

    length_threshold_counts = Counter()

    charset = Counter()
    missing_charset = Counter()
    missing_examples = defaultdict(list)

    vertical_count = 0
    horizontal_count = 0
    square_count = 0
    invalid_bbox_count = 0

    longest = []

    for split in ("train", "val", "test"):
        path = UNIFIED_DIR / f"{split}.json"

        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        for page in payload["pages"]:
            book = page["book"]
            page_index = page["page_index"]

            for ann in page["annotations"]:
                total += 1
                split_counts[split] += 1

                source = ann.get(
                    "source",
                    "unknown",
                )
                source_counts[source] += 1

                raw_text = str(
                    ann.get("text", "")
                )

                if "\n" in raw_text or "\r" in raw_text:
                    raw_multiline += 1

                if "\t" in raw_text:
                    raw_tab += 1

                text = clean_for_rec(raw_text)

                if not text:
                    empty_after_clean += 1

                length = len(text)

                lengths.append(length)
                split_lengths[split].append(length)

                for threshold in (
                    25,
                    32,
                    40,
                    48,
                    64,
                    80,
                    100,
                ):
                    if length > threshold:
                        length_threshold_counts[
                            threshold
                        ] += 1

                longest.append({
                    "length": length,
                    "split": split,
                    "book": book,
                    "page": page_index,
                    "annotation_id": ann.get(
                        "annotation_id"
                    ),
                    "source": source,
                    "text": text,
                })

                for ch in text:
                    charset[ch] += 1

                    if ch not in allowed_chars:
                        missing_charset[ch] += 1

                        if (
                            len(missing_examples[ch])
                            < 5
                        ):
                            missing_examples[ch].append({
                                "split": split,
                                "book": book,
                                "page": page_index,
                                "text": text,
                            })

                bbox = ann.get("bbox")

                if (
                    not bbox
                    or len(bbox) != 4
                ):
                    invalid_bbox_count += 1
                    continue

                x1, y1, x2, y2 = map(
                    float,
                    bbox,
                )

                width = x2 - x1
                height = y2 - y1

                if width <= 0 or height <= 0:
                    invalid_bbox_count += 1
                    continue

                ratio = height / width

                if ratio >= 1.5:
                    vertical_count += 1
                elif ratio <= 1 / 1.5:
                    horizontal_count += 1
                else:
                    square_count += 1

    longest.sort(
        key=lambda row: row["length"],
        reverse=True,
    )

    missing_rows = []

    for ch, count in missing_charset.most_common():
        row = char_desc(ch)
        row["count"] = count
        row["examples"] = missing_examples[ch]
        missing_rows.append(row)

    length_stats = {
        "count": len(lengths),
        "min": min(lengths) if lengths else 0,
        "p50": percentile(lengths, 50),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "max": max(lengths) if lengths else 0,
    }

    split_length_stats = {}

    for split, values in split_lengths.items():
        split_length_stats[split] = {
            "count": len(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "max": max(values) if values else 0,
        }

    result = {
        "total_annotations": total,
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),

        "raw_multiline_labels": raw_multiline,
        "raw_tab_labels": raw_tab,
        "empty_after_clean": empty_after_clean,

        "length_stats": length_stats,
        "split_length_stats":
            split_length_stats,

        "over_length": {
            str(k): v
            for k, v
            in length_threshold_counts.items()
        },

        "orientation_by_bbox": {
            "vertical_h_over_w_ge_1.5":
                vertical_count,
            "horizontal_w_over_h_ge_1.5":
                horizontal_count,
            "roughly_square":
                square_count,
            "invalid_bbox":
                invalid_bbox_count,
        },

        "dictionary": {
            "path": str(DICT_PATH),
            "tokens":
                len(dict_tokens),
            "single_character_tokens":
                len(single_char_dict),
            "multi_character_tokens":
                len(multi_char_tokens),
            "multi_character_examples":
                multi_char_tokens[:50],
        },

        "dataset_charset": {
            "unique_characters":
                len(charset),
            "missing_unique_characters":
                len(missing_charset),
            "missing_character_occurrences":
                sum(missing_charset.values()),
            "missing_characters":
                missing_rows,
        },

        "longest_labels":
            longest[:50],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("PP-OCRv6 Recognition Dataset Analysis")
    print("=" * 70)

    print()
    print(f"Total annotations : {total}")
    print(
        "Split             : "
        f"train={split_counts['train']} "
        f"val={split_counts['val']} "
        f"test={split_counts['test']}"
    )

    print()
    print(
        f"Multiline labels  : "
        f"{raw_multiline}"
    )
    print(
        f"Tab labels        : "
        f"{raw_tab}"
    )
    print(
        f"Empty labels      : "
        f"{empty_after_clean}"
    )

    print()
    print("Text length:")
    print(
        f"  P50 = "
        f"{length_stats['p50']:.1f}"
    )
    print(
        f"  P90 = "
        f"{length_stats['p90']:.1f}"
    )
    print(
        f"  P95 = "
        f"{length_stats['p95']:.1f}"
    )
    print(
        f"  P99 = "
        f"{length_stats['p99']:.1f}"
    )
    print(
        f"  MAX = "
        f"{length_stats['max']}"
    )

    print()
    print("Long labels:")

    for threshold in (
        25,
        32,
        40,
        48,
        64,
        80,
        100,
    ):
        count = length_threshold_counts[
            threshold
        ]

        ratio = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"  > {threshold:3d}: "
            f"{count:6d} "
            f"({ratio:6.3f}%)"
        )

    print()
    print("BBox orientation:")
    print(
        f"  vertical-ish   : "
        f"{vertical_count}"
    )
    print(
        f"  horizontal-ish : "
        f"{horizontal_count}"
    )
    print(
        f"  square-ish     : "
        f"{square_count}"
    )

    print()
    print("Dictionary:")
    print(
        f"  tokens         : "
        f"{len(dict_tokens)}"
    )
    print(
        f"  unique dataset chars : "
        f"{len(charset)}"
    )
    print(
        f"  missing chars        : "
        f"{len(missing_charset)}"
    )
    print(
        f"  missing occurrences  : "
        f"{sum(missing_charset.values())}"
    )

    if missing_charset:
        print()
        print("Top 30 missing characters:")

        for ch, count in (
            missing_charset.most_common(30)
        ):
            desc = char_desc(ch)

            print(
                f"  {repr(ch):8s} "
                f"{desc['codepoint']:10s} "
                f"{count:7d}  "
                f"{desc['name']}"
            )

    print()
    print("Top 10 longest labels:")

    for row in longest[:10]:
        print(
            f"  len={row['length']:3d} "
            f"{row['book']} "
            f"p{row['page']} "
            f"{row['source']} "
            f"{row['text']!r}"
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
