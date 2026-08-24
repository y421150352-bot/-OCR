#!/usr/bin/env python3
"""Assign character IDs/names to every speaker candidate using a ReID bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import cache_filename, load_page_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reid-cache-dir", type=Path, default=Path("cache/reid"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--character-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unknown-threshold", type=float, default=-1.0)
    args = parser.parse_args()
    with np.load(args.character_bank) as bank:
        bank_books = np.asarray(bank["books"]).astype(str)
        bank_ids = np.asarray(bank["character_ids"]).astype(str)
        bank_names = np.asarray(bank["character_names"]).astype(str)
        prototypes = bank["prototypes"].astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pages_written = candidates_written = 0
    with args.output.open("w", encoding="utf-8") as writer:
        for page in load_page_index(args.data_dir, args.split):
            book = str(page["book"])
            bank_mask = bank_books == book
            if not bank_mask.any():
                raise ValueError(f"Character bank has no prototypes for {book}")
            book_ids = bank_ids[bank_mask]
            book_names = bank_names[bank_mask]
            book_prototypes = prototypes[bank_mask]
            cache_path = (
                args.reid_cache_dir
                / args.split
                / cache_filename(str(page["key"]))
            )
            with np.load(cache_path) as cached:
                embeddings = cached["embeddings"].astype(np.float32)
                candidate_ids = [str(value) for value in cached["candidate_ids"].tolist()]
            similarities = embeddings @ book_prototypes.T
            predictions = []
            for candidate_id, scores in zip(candidate_ids, similarities):
                best = int(np.argmax(scores))
                confidence = float(scores[best])
                unknown = confidence < args.unknown_threshold
                predictions.append(
                    {
                        "candidate_id": candidate_id,
                        "character_id": None if unknown else str(book_ids[best]),
                        "character_name": "UNKNOWN" if unknown else str(book_names[best]),
                        "similarity": confidence,
                    }
                )
            writer.write(
                json.dumps(
                    {
                        "key": str(page["key"]),
                        "book": book,
                        "page_index": int(page["page_index"]),
                        "candidates": predictions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            pages_written += 1
            candidates_written += len(predictions)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pages": pages_written,
                "candidates": candidates_written,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
