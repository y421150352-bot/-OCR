#!/usr/bin/env python3
"""Export per-query V3 vs LightGBM wins/regressions as annotated pages."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GeometryTextPageDataset, geometry_text_page_batch_collate
from model_v3 import SpeakerGeometryTextGraphTransformer


COLORS = {
    "text": (30, 110, 255),
    "gt": (25, 175, 70),
    "lightgbm": (225, 45, 45),
    "v3": (155, 45, 210),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lightgbm-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--text-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"))
    parser.add_argument(
        "--max-images-per-category",
        type=int,
        default=0,
        help="0 exports every corrected/regressed example",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_") or "sample"


def build_v3(
    config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device
) -> SpeakerGeometryTextGraphTransformer:
    if str(config.get("model_version")) != "v3":
        raise ValueError("Expected a V3 checkpoint")
    model = SpeakerGeometryTextGraphTransformer(
        text_dim=int(config["text_dim"]),
        hidden_dim=int(config.get("hidden_dim", 384)),
        layers=int(config.get("layers", 2)),
        heads=int(config.get("heads", 8)),
        dropout=float(config.get("dropout", 0.15)),
        attention_dropout=float(config.get("attention_dropout", 0.1)),
        geometry_bias_hidden=int(config.get("geometry_bias_hidden", 128)),
        geometry_bias_scale_init=float(config.get("geometry_bias_scale_init", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def move_to_device(page: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in page.items()
    }


def load_page_text(data_dir: Path, split: str, record: dict[str, Any]) -> list[str]:
    path = data_dir / "texts" / split / f"{Path(str(record['pack'])).stem}.json"
    if not path.is_file():
        return [""] * len(record["text_ids"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if [str(value) for value in payload["text_ids"]] != [
        str(value) for value in record["text_ids"]
    ]:
        raise ValueError(f"{record['key']}: text JSON IDs differ from page index")
    return [str(value) for value in payload["texts"]]


def collect_comparisons(
    v3: SpeakerGeometryTextGraphTransformer,
    lightgbm: Any,
    loader: DataLoader,
    records_by_key: dict[str, dict[str, Any]],
    data_dir: Path,
    split: str,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {
        "queries": 0,
        "both_correct": 0,
        "corrected": 0,
        "regressed": 0,
        "both_wrong": 0,
        "same_prediction": 0,
        "different_prediction": 0,
    }
    with torch.inference_mode():
        for cpu_page in tqdm(loader, desc="compare V3 vs LightGBM"):
            gpu_page = move_to_device(cpu_page, device)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                v3_scores = v3.forward_batch(
                    gpu_page["geometry"],
                    gpu_page["text_context"],
                    gpu_page["text_context_mask"],
                    gpu_page["dialogue_mask"],
                    gpu_page["candidate_mask"],
                )
            v3_scores = v3_scores.float().cpu()

            for batch_index, key_raw in enumerate(cpu_page["key"]):
                key = str(key_raw)
                record = records_by_key[key]
                dialogues = int(cpu_page["dialogue_mask"][batch_index].sum())
                candidates = int(cpu_page["candidate_mask"][batch_index].sum())
                geometry = cpu_page["geometry"][batch_index, :dialogues, :candidates]
                labels = cpu_page["labels"][batch_index, :dialogues, :candidates]
                page_v3_scores = v3_scores[batch_index, :dialogues, :candidates]
                flat_geometry = geometry.numpy().reshape(-1, geometry.shape[-1])
                page_lgb_scores = np.asarray(lightgbm.predict(flat_geometry)).reshape(
                    dialogues, candidates
                )

                pack_path = data_dir / str(record["pack"])
                with np.load(pack_path) as pack:
                    text_boxes = pack["text_boxes"].astype(np.float32)
                    body_boxes = pack["body_boxes"].astype(np.float32)
                candidate_ids = [str(value) for value in record["candidate_ids"]]
                text_ids = [str(value) for value in record["text_ids"]]
                texts = load_page_text(data_dir, split, record)

                for query_index in range(dialogues):
                    label = labels[query_index]
                    gt_indices = label.nonzero(as_tuple=False).flatten().tolist()
                    if not gt_indices:
                        raise ValueError(f"{key} query {query_index}: no positive label")
                    v3_index = int(torch.argmax(page_v3_scores[query_index]))
                    lgb_index = int(np.argmax(page_lgb_scores[query_index]))
                    v3_correct = bool(label[v3_index])
                    lgb_correct = bool(label[lgb_index])
                    if v3_correct and lgb_correct:
                        category = "both_correct"
                    elif v3_correct:
                        category = "corrected"
                    elif lgb_correct:
                        category = "regressed"
                    else:
                        category = "both_wrong"
                    counts["queries"] += 1
                    counts[category] += 1
                    counts[
                        "same_prediction" if v3_index == lgb_index else "different_prediction"
                    ] += 1
                    rows.append(
                        {
                            "comparison_index": len(rows),
                            "category": category,
                            "key": key,
                            "book": str(record["book"]),
                            "page_index": str(record["page_index"]),
                            "image": str(record["image"]),
                            "query_index": query_index,
                            "text_id": text_ids[query_index],
                            "text": texts[query_index],
                            "text_box": text_boxes[query_index].tolist(),
                            "gt_indices": gt_indices,
                            "gt_ids": [candidate_ids[index] for index in gt_indices],
                            "gt_boxes": [body_boxes[index].tolist() for index in gt_indices],
                            "lightgbm_index": lgb_index,
                            "lightgbm_id": candidate_ids[lgb_index],
                            "lightgbm_box": body_boxes[lgb_index].tolist(),
                            "lightgbm_score": float(page_lgb_scores[query_index, lgb_index]),
                            "lightgbm_correct": lgb_correct,
                            "v3_index": v3_index,
                            "v3_id": candidate_ids[v3_index],
                            "v3_box": body_boxes[v3_index].tolist(),
                            "v3_score": float(page_v3_scores[query_index, v3_index]),
                            "v3_correct": v3_correct,
                            "candidate_count": candidates,
                        }
                    )
    return rows, counts


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
    bounds = draw.textbbox((left, top), label)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    label_top = max(0, top - text_height - 5)
    draw.rectangle(
        (left, label_top, left + text_width + 7, label_top + text_height + 5),
        fill=color,
    )
    draw.text((left + 3, label_top + 2), label, fill=(255, 255, 255))


def draw_comparison(row: dict[str, Any], dataset_root: Path, output: Path) -> None:
    source = dataset_root / row["image"]
    if not source.is_file():
        raise FileNotFoundError(f"Missing source page: {source}")
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(3, round(min(image.size) / 300))
    draw_labeled_box(draw, row["text_box"], COLORS["text"], "TEXT", width)
    for gt_id, gt_box in zip(row["gt_ids"], row["gt_boxes"]):
        draw_labeled_box(draw, gt_box, COLORS["gt"], f"GT {gt_id}", width)
    draw_labeled_box(
        draw,
        row["lightgbm_box"],
        COLORS["lightgbm"],
        f"LGB {row['lightgbm_id']}",
        width,
    )
    draw_labeled_box(
        draw, row["v3_box"], COLORS["v3"], f"V3 {row['v3_id']}", width
    )
    image.save(output, quality=92)


def export_images(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    max_per_category: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for category in ("corrected", "regressed"):
        selected = [row for row in rows if row["category"] == category]
        if max_per_category:
            selected = selected[:max_per_category]
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for local_index, row in enumerate(tqdm(selected, desc=f"draw {category}")):
            filename = (
                f"{local_index:05d}__{safe_name(row['book'])}"
                f"__p{safe_name(row['page_index'])}__text_{safe_name(row['text_id'])}.jpg"
            )
            draw_comparison(row, dataset_root, category_dir / filename)
            row["annotated_image"] = f"{category}/{filename}"
        counts[category] = len(selected)
    return counts


def write_reports(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    exported: dict[str, int],
    output_dir: Path,
    split: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparisons.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = [
        "category", "book", "page_index", "query_index", "text_id", "text",
        "gt_ids", "lightgbm_id", "v3_id", "candidate_count", "annotated_image",
    ]
    with (output_dir / "comparisons.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field, "") for field in fields}
            output["gt_ids"] = "|".join(row["gt_ids"])
            writer.writerow(output)

    summary = {
        "split": split,
        **counts,
        "lightgbm_correct": counts["both_correct"] + counts["regressed"],
        "v3_correct": counts["both_correct"] + counts["corrected"],
        "net_correct_gain": counts["corrected"] - counts["regressed"],
        "lightgbm_top1": (counts["both_correct"] + counts["regressed"]) / counts["queries"],
        "v3_top1": (counts["both_correct"] + counts["corrected"]) / counts["queries"],
        "exported_images": exported,
        "colors": COLORS,
    }
    expected_v3 = 0.7645226130653267 if split == "test" else None
    expected_lightgbm = 0.7522613065326633 if split == "test" else None
    if expected_v3 is not None:
        if abs(summary["v3_top1"] - expected_v3) > 1e-10:
            raise ValueError(
                f"Recomputed V3 Top-1 {summary['v3_top1']:.12f} does not match "
                f"expected {expected_v3:.12f}; check run/data/cache alignment"
            )
        if abs(summary["lightgbm_top1"] - expected_lightgbm) > 1e-10:
            raise ValueError(
                f"Recomputed LightGBM Top-1 {summary['lightgbm_top1']:.12f} does "
                f"not match expected {expected_lightgbm:.12f}; check model/data alignment"
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cards = []
    for row in rows:
        image = row.get("annotated_image")
        if not image:
            continue
        cards.append(
            '<article><a href="{image}"><img loading="lazy" src="{image}"></a>'
            '<p><b>{category}</b> · {book} p{page} · {text_id}</p>'
            '<p>{text}</p><p>GT: {gt} · LGB: {lgb} · V3: {v3}</p></article>'.format(
                image=html.escape(str(image)),
                category=html.escape(str(row["category"])),
                book=html.escape(str(row["book"])),
                page=html.escape(str(row["page_index"])),
                text_id=html.escape(str(row["text_id"])),
                text=html.escape(str(row["text"])),
                gt=html.escape("|".join(row["gt_ids"])),
                lgb=html.escape(str(row["lightgbm_id"])),
                v3=html.escape(str(row["v3_id"])),
            )
        )
    report = """<!doctype html><meta charset="utf-8"><title>V3 vs LightGBM</title>
<style>body{font-family:sans-serif;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}article{border:1px solid #ccc;padding:10px}img{width:100%;height:auto}p{margin:.4em 0}</style>
<h1>V3 vs LightGBM</h1><p>Blue=TEXT, green=GT, red=LightGBM, purple=V3.</p><main>""" + "\n".join(cards) + "</main>"
    (output_dir / "index.html").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved comparison to: {output_dir}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_images_per_category < 0:
        raise ValueError("Invalid batch/worker/image limit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("Install LightGBM: .venv/bin/python -m pip install lightgbm") from exc

    run_dir = args.run_dir.resolve()
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config", {}))
    text_cache_dir = args.text_cache_dir or Path(
        str(config.get("text_cache_dir", "cache/text_multilingual_e5_base"))
    )
    output_dir = args.output_dir or run_dir / f"{args.split}_vs_lightgbm"
    amp_name = args.amp or str(config.get("amp", "bf16"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp_name]
    device = torch.device("cuda")
    v3 = build_v3(config, checkpoint, device)
    lightgbm = lgb.Booster(model_file=str(args.lightgbm_model.resolve()))
    dataset = GeometryTextPageDataset(args.data_dir, text_cache_dir, args.split)
    records_by_key = {str(record["key"]): record for record in dataset.records}
    worker_options: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers:
        worker_options["prefetch_factor"] = 4
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=geometry_text_page_batch_collate,
        **worker_options,
    )
    rows, counts = collect_comparisons(
        v3,
        lightgbm,
        loader,
        records_by_key,
        args.data_dir.resolve(),
        args.split,
        device,
        amp_dtype,
    )
    exported = export_images(
        rows,
        args.dataset_root.resolve(),
        output_dir.resolve(),
        args.max_images_per_category,
    )
    write_reports(rows, counts, exported, output_dir.resolve(), args.split)


if __name__ == "__main__":
    main()
