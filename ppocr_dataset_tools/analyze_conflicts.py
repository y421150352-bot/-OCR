#!/usr/bin/env python3

import csv
import json
import os
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


WORKSPACE_ROOT = Path(os.environ.get("MANGA_OCR_WORKSPACE", "manga_ppocrv6_dataset"))
INPUT = WORKSPACE_ROOT / "overlap_analysis.json"
OUTPUT = WORKSPACE_ROOT / "conflict_review.csv"


def norm(text: str) -> str:
    return "".join(
        unicodedata.normalize("NFKC", text or "").split()
    )


def similarity(a: str, b: str) -> float:
    a = norm(a)
    b = norm(b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


data = json.loads(INPUT.read_text(encoding="utf-8"))

rows = []

for item in data.get("conflicts_sample", []):
    sim = similarity(
        item.get("manga_text", ""),
        item.get("coo_text", ""),
    )

    rows.append({
        "book": item.get("book"),
        "page": item.get("page"),
        "iom": item.get("iom"),
        "text_similarity": round(sim, 4),
        "manga_id": item.get("manga_id"),
        "manga_text": item.get("manga_text"),
        "coo_id": item.get("coo_id"),
        "coo_text": item.get("coo_text"),
    })

rows.sort(
    key=lambda row: (
        -float(row["text_similarity"]),
        -float(row["iom"]),
    )
)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "book",
            "page",
            "iom",
            "text_similarity",
            "manga_id",
            "manga_text",
            "coo_id",
            "coo_text",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"conflicts written: {len(rows)}")
print(f"saved: {OUTPUT}")

print()
print("Top 30 high-similarity conflicts:")
print("=" * 80)

for row in rows[:30]:
    print(
        f"{row['book']} p{row['page']} "
        f"IoM={row['iom']:.3f} "
        f"sim={row['text_similarity']:.3f} | "
        f"Manga109={row['manga_text']!r} | "
        f"COO={row['coo_text']!r}"
    )
