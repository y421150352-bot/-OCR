#!/usr/bin/env python3
"""Build named character prototypes from a JSONL set of annotated examples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inference_utils import embed_instance, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True, help="JSONL: name, image, optional face_box/body_box")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("character_bank.npz"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    with args.examples.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            with Image.open(record["image"]) as raw:
                image = raw.convert("RGB")
            grouped[record["name"]].append(embed_instance(model, image, record.get("face_box"), record.get("body_box"), device))
    names, prototypes = [], []
    for name, vectors in sorted(grouped.items()):
        prototype = np.mean(vectors, axis=0); prototype /= np.linalg.norm(prototype).clip(1e-12)
        names.append(name); prototypes.append(prototype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, names=np.asarray(names), prototypes=np.stack(prototypes).astype(np.float32))
    print(f"Saved {len(names)} identities to {args.output}")


if __name__ == "__main__":
    main()
