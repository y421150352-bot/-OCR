#!/usr/bin/env python3
"""Export Top-1 speaker errors with metadata and annotated Manga109 pages."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PageDataset, page_batch_collate
from model_v2 import SpeakerBipartiteGraphTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 exports an annotated image for every error; CSV/JSONL always contain all errors",
    )
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"))
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_") or "sample"


def move_to_device(page: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in page.items()
    }


def build_model(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device):
    if str(config.get("model_version", "v2")) != "v2":
        raise ValueError("export_errors.py currently supports Graph Transformer V2 checkpoints")
    model = SpeakerBipartiteGraphTransformer(
        visual_dim=int(config["visual_dim"]),
        hidden_dim=int(config.get("hidden_dim", 384)),
        layers=int(config.get("layers", 2)),
        heads=int(config.get("heads", 8)),
        dropout=float(config.get("dropout", 0.15)),
        attention_dropout=float(config.get("attention_dropout", 0.1)),
        geometry_bias_hidden=int(config.get("geometry_bias_hidden", 128)),
        geometry_bias_scale_init=float(config.get("geometry_bias_scale_init", 0.1)),
        ablation=str(config.get("ablation", "full")),
        context_grid=int(config.get("context_grid", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def forward_batch(model, page: dict[str, object]) -> torch.Tensor:
    return model.forward_batch(
        page["page_features"],
        page["patch_mask"],
        page["feature_hw"],
        page["geometry"],
        page["text_boxes"],
        page["body_boxes"],
        page["dialogue_mask"],
        page["candidate_mask"],
        page["original_hw"],
        page["resized_hw"],
        page["padded_hw"],
    )


def collect_errors(
    model,
    loader: DataLoader,
    records_by_key: dict[str, dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    errors: list[dict[str, Any]] = []
    query_total = 0
    with torch.inference_mode():
        for cpu_page in tqdm(loader, desc="export errors"):
            page = move_to_device(cpu_page, device)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                batch_scores = forward_batch(model, page)
            batch_scores = batch_scores.float().cpu()

            for batch_index, key in enumerate(cpu_page["key"]):
                record = records_by_key[str(key)]
                dialogue_count = int(cpu_page["dialogue_mask"][batch_index].sum())
                candidate_count = int(cpu_page["candidate_mask"][batch_index].sum())
                scores = batch_scores[batch_index, :dialogue_count, :candidate_count]
                labels = cpu_page["labels"][batch_index, :dialogue_count, :candidate_count]
                text_boxes = cpu_page["text_boxes"][batch_index, :dialogue_count]
                body_boxes = cpu_page["body_boxes"][batch_index, :candidate_count]
                candidate_ids = [str(value) for value in record["candidate_ids"]]
                text_ids = [str(value) for value in record["text_ids"]]

                for query_index in range(dialogue_count):
                    query_total += 1
                    query_scores = scores[query_index]
                    correct_indices = labels[query_index].nonzero(as_tuple=False).flatten().tolist()
                    if not correct_indices:
                        continue
                    order = torch.argsort(query_scores, descending=True)
                    predicted_index = int(order[0])
                    if bool(labels[query_index, predicted_index]):
                        continue

                    probabilities = torch.softmax(query_scores, dim=0)
                    correct_set = set(int(index) for index in correct_indices)
                    best_correct_rank = min(
                        rank
                        for rank, index in enumerate(order.tolist(), start=1)
                        if int(index) in correct_set
                    )
                    top_candidates = []
                    for rank, index_raw in enumerate(order[: min(top_k, candidate_count)].tolist(), 1):
                        index = int(index_raw)
                        top_candidates.append(
                            {
                                "rank": rank,
                                "candidate_index": index,
                                "candidate_id": candidate_ids[index],
                                "score": float(query_scores[index]),
                                "probability": float(probabilities[index]),
                                "is_correct": index in correct_set,
                            }
                        )
                    best_correct_score = max(float(query_scores[index]) for index in correct_set)
                    errors.append(
                        {
                            "error_index": len(errors),
                            "key": str(key),
                            "book": str(record["book"]),
                            "page_index": str(record["page_index"]),
                            "image": str(record["image"]),
                            "query_index": query_index,
                            "text_id": text_ids[query_index],
                            "text_box": [float(value) for value in text_boxes[query_index].tolist()],
                            "predicted_index": predicted_index,
                            "predicted_id": candidate_ids[predicted_index],
                            "predicted_box": [float(value) for value in body_boxes[predicted_index].tolist()],
                            "correct_indices": correct_indices,
                            "correct_ids": [candidate_ids[index] for index in correct_indices],
                            "correct_boxes": [
                                [float(value) for value in body_boxes[index].tolist()]
                                for index in correct_indices
                            ],
                            "candidate_count": candidate_count,
                            "best_correct_rank": best_correct_rank,
                            "score_margin_pred_minus_best_correct": (
                                float(query_scores[predicted_index]) - best_correct_score
                            ),
                            "top_candidates": top_candidates,
                        }
                    )
    return errors, query_total


def draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    color: tuple[int, int, int],
    label: str,
    width: int,
) -> None:
    xy = tuple(round(value) for value in box)
    draw.rectangle(xy, outline=color, width=width)
    left, top, _, _ = xy
    text_box = draw.textbbox((left, top), label)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_top = max(0, top - text_height - 4)
    draw.rectangle(
        (left, label_top, left + text_width + 6, label_top + text_height + 4), fill=color
    )
    draw.text((left + 3, label_top + 2), label, fill=(255, 255, 255))


def export_images(
    errors: list[dict[str, Any]],
    dataset_root: Path,
    image_dir: Path,
    max_images: int,
) -> int:
    image_dir.mkdir(parents=True, exist_ok=True)
    selected = errors if max_images == 0 else errors[:max_images]
    for error in tqdm(selected, desc="draw errors"):
        source = dataset_root / error["image"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing source page: {source}")
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        width = max(2, round(min(image.size) / 350))
        draw_labeled_box(
            draw, error["text_box"], (30, 110, 255), f"TEXT {error['text_id']}", width
        )
        draw_labeled_box(
            draw,
            error["predicted_box"],
            (230, 40, 40),
            f"PRED {error['predicted_id']}",
            width,
        )
        for correct_id, correct_box in zip(error["correct_ids"], error["correct_boxes"]):
            draw_labeled_box(draw, correct_box, (30, 180, 70), f"TRUE {correct_id}", width)
        filename = (
            f"{int(error['error_index']):05d}__{safe_name(error['book'])}"
            f"__p{safe_name(error['page_index'])}__text_{safe_name(error['text_id'])}.jpg"
        )
        image.save(image_dir / filename, quality=92)
        error["annotated_image"] = f"images/{filename}"
    return len(selected)


def write_reports(
    errors: list[dict[str, Any]],
    query_total: int,
    output_dir: Path,
    config: dict[str, Any],
    split: str,
    exported_images: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    fields = [
        "error_index", "key", "book", "page_index", "query_index", "text_id",
        "predicted_id", "correct_ids", "candidate_count", "best_correct_rank",
        "score_margin_pred_minus_best_correct", "annotated_image",
    ]
    with (output_dir / "errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for error in errors:
            row = {field: error.get(field, "") for field in fields}
            row["correct_ids"] = "|".join(error["correct_ids"])
            writer.writerow(row)

    summary = {
        "split": split,
        "ablation": str(config.get("ablation", "full")),
        "queries": query_total,
        "top1_errors": len(errors),
        "top1_error_rate": len(errors) / query_total if query_total else 0.0,
        "top1_accuracy": 1.0 - len(errors) / query_total if query_total else 0.0,
        "annotated_images": exported_images,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved error analysis to: {output_dir}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.top_k < 1 or args.max_images < 0:
        raise ValueError("Invalid batch/worker/top-k/max-images setting")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config", {}))
    cache_dir = args.cache_dir or Path(str(config.get("cache_dir", "cache/dinov3_vitb16_l896")))
    output_dir = args.output_dir or run_dir / f"{args.split}_errors"
    amp_name = args.amp or str(config.get("amp", "bf16"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp_name]

    device = torch.device("cuda")
    model = build_model(config, checkpoint, device)
    dataset = PageDataset(args.data_dir, cache_dir, args.split)
    records_by_key = {str(record["key"]): record for record in dataset.records}
    worker_options: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        worker_options["prefetch_factor"] = 4
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=page_batch_collate,
        **worker_options,
    )
    errors, query_total = collect_errors(
        model, loader, records_by_key, device, amp_dtype, args.top_k
    )
    exported_images = export_images(
        errors, args.dataset_root.resolve(), output_dir / "images", args.max_images
    )
    write_reports(
        errors, query_total, output_dir, config, args.split, exported_images
    )


if __name__ == "__main__":
    main()
