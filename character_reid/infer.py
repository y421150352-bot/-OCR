#!/usr/bin/env python3
"""Identify RT-DETR body/face detections against a named character bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from build_reid_dataset import Box, pair_boxes
from inference_utils import embed_instance, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True, help="detections.json from rtdetr_manga_test/run_inference.py")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=Path("reid_predictions.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    with np.load(args.bank) as bank:
        names, prototypes = bank["names"], bank["prototypes"]
    payload = json.loads(args.detections.read_text(encoding="utf-8"))
    results = []
    for page in payload["images"]:
        detections = page["detections"]
        faces = [Box(f"face_{i}", "face", "predicted", tuple(d["box"])) for i, d in enumerate(detections) if d["class_name"] == "face"]
        bodies = [Box(f"body_{i}", "body", "predicted", tuple(d["box"])) for i, d in enumerate(detections) if d["class_name"] == "body"]
        frames = [tuple(d["box"]) for d in detections if d["class_name"] == "frame"]
        pairs, face_only, body_only = pair_boxes(faces, bodies, frames)
        instances = [(f, b) for f, b in pairs] + [(f, None) for f in face_only] + [(None, b) for b in body_only]
        with Image.open(args.image_dir / page["image"]) as raw:
            image = raw.convert("RGB")
        page_results = []
        for face, body in instances:
            vector = embed_instance(model, image, list(face.xyxy) if face else None, list(body.xyxy) if body else None, device)
            scores = prototypes @ vector
            best = int(np.argmax(scores)); score = float(scores[best])
            page_results.append({"face_box": list(face.xyxy) if face else None, "body_box": list(body.xyxy) if body else None, "name": str(names[best]) if score >= args.threshold else "unknown", "score": round(score, 6)})
        results.append({"image": page["image"], "characters": page_results})
    args.output.write_text(json.dumps({"threshold": args.threshold, "images": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
