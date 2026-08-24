#!/usr/bin/env python3
"""Cache frozen DINOv3 patch grids, one resumable file per Manga109 page."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def read_records(data_dir: Path, max_pages_per_split: int = 0) -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        with (data_dir / f"{split}_pages.jsonl").open("r", encoding="utf-8") as handle:
            split_records = [json.loads(line) for line in handle if line.strip()]
            if max_pages_per_split:
                split_records = split_records[:max_pages_per_split]
            for record in split_records:
                records[str(record["key"])] = record
    return list(records.values())


def preprocess(image: Image.Image, long_side: int, mean: list[float], std: list[float], patch: int) -> tuple[torch.Tensor, int, int]:
    width, height = image.size
    scale = long_side / max(width, height)
    resized_w = max(patch, int(round(width * scale)))
    resized_h = max(patch, int(round(height * scale)))
    image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_t = torch.tensor(mean, dtype=tensor.dtype)[:, None, None]
    std_t = torch.tensor(std, dtype=tensor.dtype)[:, None, None]
    tensor = (tensor - mean_t) / std_t
    padded_h = math.ceil(resized_h / patch) * patch
    padded_w = math.ceil(resized_w / patch) * patch
    tensor = F.pad(tensor, (0, padded_w - resized_w, 0, padded_h - resized_h))
    return tensor.unsqueeze(0), resized_h, resized_w


def resolve_patch_size(config: object) -> int:
    patch = getattr(config, "patch_size", 16)
    return int(patch[0] if isinstance(patch, (tuple, list)) else patch)


def extract_patch_grid(outputs: object, config: object, grid_h: int, grid_w: int) -> torch.Tensor:
    tokens = outputs.last_hidden_state
    patch_count = grid_h * grid_w
    if tokens.shape[1] < patch_count:
        raise ValueError(f"Model returned {tokens.shape[1]} tokens for {patch_count} patches")
    # DINOv3 places patch tokens after CLS and register tokens. Taking the tail is
    # robust to the exact number of register tokens in the selected checkpoint.
    patches = tokens[:, -patch_count:, :]
    return patches.reshape(1, grid_h, grid_w, patches.shape[-1])[0]


def cache_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as cached:
            return (
                cached["features"].ndim == 3
                and cached["resized_hw"].shape == (2,)
                and cached["padded_hw"].shape == (2,)
            )
    except (OSError, ValueError, KeyError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/dinov3_vitb16_l896"))
    parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--long-side", type=int, default=896)
    parser.add_argument("--max-pages", type=int, default=0, help="Debug pages per split; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DINOv3 caching is intended to run on the RTX 5090 server")
    torch.backends.cuda.matmul.allow_tf32 = True
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    patch = resolve_patch_size(model.config)
    mean = list(processor.image_mean)
    std = list(processor.image_std)
    records = read_records(args.data_dir, args.max_pages)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "long_side": args.long_side,
        "patch_size": patch,
        "mean": mean,
        "std": std,
        "pages": len(records),
    }
    (args.cache_dir / "cache_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for record in tqdm(records, desc="DINOv3 cache"):
        output_path = args.cache_dir / (str(record["key"]).replace("/", "__").replace("'", "_") + ".npz")
        if cache_is_valid(output_path) and not args.overwrite:
            continue
        image_path = args.dataset_root / str(record["image"])
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        pixel_values, resized_h, resized_w = preprocess(image, args.long_side, mean, std, patch)
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values=pixel_values)
        grid = extract_patch_grid(outputs, model.config, pixel_values.shape[-2] // patch, pixel_values.shape[-1] // patch)
        temporary_path = output_path.with_suffix(".tmp.npz")
        np.savez(
            temporary_path,
            features=grid.float().cpu().numpy().astype(np.float16),
            resized_hw=np.asarray([resized_h, resized_w], dtype=np.int32),
            padded_hw=np.asarray(pixel_values.shape[-2:], dtype=np.int32),
        )
        temporary_path.replace(output_path)


if __name__ == "__main__":
    main()
