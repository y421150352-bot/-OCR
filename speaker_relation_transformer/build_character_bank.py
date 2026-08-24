#!/usr/bin/env python3
"""Build per-book named character prototypes from aligned ReID caches."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import cache_filename, load_page_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reid-cache-dir", type=Path, default=Path("cache/reid"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--examples-per-character", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.examples_per_character < 1:
        raise ValueError("--examples-per-character must be positive")
    grouped: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    example_keys: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for page in load_page_index(args.data_dir, args.split):
        cache_path = (
            args.reid_cache_dir
            / args.split
            / cache_filename(str(page["key"]))
        )
        with np.load(cache_path) as cached:
            embeddings = cached["embeddings"].astype(np.float32)
            candidate_ids = [str(value) for value in cached["candidate_ids"].tolist()]
            character_ids = [str(value) for value in cached["character_ids"].tolist()]
            character_names = [str(value) for value in cached["character_names"].tolist()]
        for embedding, candidate_id, character_id, character_name in zip(
            embeddings, candidate_ids, character_ids, character_names
        ):
            if character_name.strip().casefold() == "other":
                continue
            identity = (str(page["book"]), character_id, character_name)
            if len(grouped[identity]) < args.examples_per_character:
                grouped[identity].append(embedding)
                example_keys[identity].append(f"{page['key']}/{candidate_id}")
    books: list[str] = []
    character_ids: list[str] = []
    character_names: list[str] = []
    prototypes: list[np.ndarray] = []
    examples: list[str] = []
    for identity in sorted(grouped):
        prototype = np.stack(grouped[identity]).mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        books.append(identity[0])
        character_ids.append(identity[1])
        character_names.append(identity[2])
        prototypes.append(prototype)
        examples.append(json.dumps(example_keys[identity], ensure_ascii=False))
    if not prototypes:
        raise ValueError("No character prototypes were created")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        books=np.asarray(books, dtype=np.str_),
        character_ids=np.asarray(character_ids, dtype=np.str_),
        character_names=np.asarray(character_names, dtype=np.str_),
        prototypes=np.stack(prototypes).astype(np.float32),
        example_keys=np.asarray(examples, dtype=np.str_),
        examples_per_character=np.int64(args.examples_per_character),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "books": len(set(books)),
                "characters": len(prototypes),
                "examples": sum(len(values) for values in grouped.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
