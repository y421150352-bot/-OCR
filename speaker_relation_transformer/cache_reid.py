#!/usr/bin/env python3
"""Cache Face+Body identity embeddings aligned to speaker candidate_ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data import cache_filename, load_page_index
from model_reid import FaceBodyReID
from reid_data import letterbox, read_jsonl


def load_checkpoint(path: Path, device: torch.device) -> tuple[FaceBodyReID, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = FaceBodyReID(
        str(config["backbone"]),
        embedding_dim=int(config.get("embedding_dim", 256)),
        freeze_backbone=True,
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), int(checkpoint["epoch"])


def crop_tensor(
    image: Image.Image, box: list[int] | None, image_size: int
) -> tuple[torch.Tensor, bool]:
    if box is None:
        return torch.zeros(3, image_size, image_size), False
    return letterbox(image.crop(tuple(box)), image_size, training=False), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reid-data-dir", type=Path, default=Path("data/reid"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/reid"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ReID caching requires CUDA")
    device = torch.device("cuda")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    metadata = json.loads(
        (args.reid_data_dir / "metadata.json").read_text(encoding="utf-8")
    )
    dataset_root = Path(str(metadata["dataset_root"]))
    model, checkpoint_epoch = load_checkpoint(args.checkpoint, device)
    cache_metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "embedding_dim": model.embedding_dim,
        "image_size": args.image_size,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache_config.json").write_text(
        json.dumps(cache_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for split in ("train", "val", "test"):
        instances = read_jsonl(args.reid_data_dir / f"{split}.jsonl")
        by_body = {
            (str(record["book"]), int(record["page"]), str(record["body_annotation_id"])): record
            for record in instances
            if record.get("body_annotation_id") is not None
        }
        output_split = args.output_dir / split
        output_split.mkdir(parents=True, exist_ok=True)
        pages = load_page_index(args.data_dir, split)
        for page in tqdm(pages, desc=f"ReID cache {split}"):
            output_path = output_split / cache_filename(str(page["key"]))
            if output_path.is_file() and not args.overwrite:
                continue
            book = str(page["book"])
            page_index = int(page["page_index"])
            candidate_ids = [str(value) for value in page["candidate_ids"]]
            records = []
            missing = []
            for candidate_id in candidate_ids:
                record = by_body.get((book, page_index, candidate_id))
                if record is None:
                    missing.append(candidate_id)
                else:
                    records.append(record)
            if missing:
                raise ValueError(
                    f"{page['key']}: {len(missing)} speaker candidates missing from ReID manifest: "
                    f"{missing[:5]}"
                )
            with Image.open(dataset_root / str(page["image"])) as raw:
                image = raw.convert("RGB")
            faces: list[torch.Tensor] = []
            bodies: list[torch.Tensor] = []
            face_valid: list[bool] = []
            body_valid: list[bool] = []
            for record in records:
                face, has_face = crop_tensor(
                    image, record.get("face_box"), args.image_size  # type: ignore[arg-type]
                )
                body, has_body = crop_tensor(
                    image, record.get("body_box"), args.image_size  # type: ignore[arg-type]
                )
                faces.append(face)
                bodies.append(body)
                face_valid.append(has_face)
                body_valid.append(has_body)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                output = model(
                    torch.stack(faces).to(device),
                    torch.stack(bodies).to(device),
                    torch.tensor(face_valid, device=device),
                    torch.tensor(body_valid, device=device),
                )
            temporary = output_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                embeddings=output["embedding"].float().cpu().numpy().astype(np.float16),
                candidate_ids=np.asarray(candidate_ids, dtype=np.str_),
                character_ids=np.asarray(
                    [str(record["character_id"]) for record in records], dtype=np.str_
                ),
                character_names=np.asarray(
                    [str(record["character_name"]) for record in records], dtype=np.str_
                ),
                face_valid=np.asarray(face_valid, dtype=np.bool_),
                body_valid=np.asarray(body_valid, dtype=np.bool_),
            )
            temporary.replace(output_path)


if __name__ == "__main__":
    main()
