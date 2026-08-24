from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps


CLASS_NAMES = {0: "body", 1: "text", 2: "frame", 3: "face"}
COLORS = {
    0: (30, 144, 255),
    1: (255, 60, 60),
    2: (50, 205, 50),
    3: (255, 20, 147),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Manga109 RT-DETRv4 ONNX inference.")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.onnx"))
    parser.add_argument("--input", type=Path, default=Path(__file__).parents[1] / "test")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, default=1280)
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively scan chapter subdirectories and preserve them in the output",
    )
    return parser.parse_args()


def prepare_image(image: Image.Image, input_size: int) -> np.ndarray:
    resized = image.resize((input_size, input_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=max(14, round(min(image.size) / 55)))
    line_width = max(2, round(min(image.size) / 350))

    # Draw large boxes first so faces and labels remain visible.
    for item in sorted(detections, key=lambda row: row["area"], reverse=True):
        class_id = item["class_id"]
        x1, y1, x2, y2 = item["box"]
        color = COLORS.get(class_id, (255, 255, 0))
        label = f'{item["class_name"]} {item["score"]:.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text_box = draw.textbbox((x1, y1), label, font=font, stroke_width=1)
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 4)
        text_box = draw.textbbox((x1, label_y), label, font=font, stroke_width=1)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x1, label_y), label, fill=color, font=font, stroke_width=1)
    return result


def main() -> None:
    args = parse_args()
    if args.input.is_file():
        images = [args.input] if args.input.suffix.lower() in IMAGE_SUFFIXES else []
    else:
        iterator = args.input.rglob("*") if args.recursive else args.input.iterdir()
        images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {args.input}")
    if not args.model.is_file() or args.model.stat().st_size == 0:
        raise SystemExit(f"Model is missing or empty: {args.model}")

    args.output.mkdir(parents=True, exist_ok=True)
    available_providers = ort.get_available_providers()
    providers = [
        provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available_providers
    ]
    if not providers:
        raise RuntimeError(f"No supported ONNX Runtime provider. Available: {available_providers}")
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_names = {value.name for value in session.get_inputs()}
    if not {"images", "orig_target_sizes"}.issubset(input_names):
        raise RuntimeError(f"Unexpected model inputs: {sorted(input_names)}")

    all_results: list[dict] = []
    total_counts: Counter[str] = Counter()
    started = time.perf_counter()

    for index, path in enumerate(images, start=1):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        tensor = prepare_image(image, args.input_size)
        original_size = np.array([[width, height]], dtype=np.int64)

        inference_started = time.perf_counter()
        raw_outputs = session.run(None, {"images": tensor, "orig_target_sizes": original_size})
        output_map = {meta.name: value for meta, value in zip(session.get_outputs(), raw_outputs)}
        labels = output_map["labels"][0]
        boxes = output_map["boxes"][0]
        scores = output_map["scores"][0]

        detections: list[dict] = []
        for label, box, score in zip(labels, boxes, scores):
            score_value = float(score)
            if score_value < args.threshold:
                continue
            class_id = int(label)
            x1, y1, x2, y2 = (float(value) for value in box)
            x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
            y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
            detection = {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "score": round(score_value, 6),
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "area": round(max(0.0, x2 - x1) * max(0.0, y2 - y1), 2),
            }
            detections.append(detection)
            total_counts[detection["class_name"]] += 1

        elapsed = time.perf_counter() - inference_started
        visualized = draw_detections(image, detections)
        relative_path = path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
        output_image = args.output / relative_path.parent / f"{path.stem}_detected.jpg"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        visualized.save(output_image, quality=95)

        counts = Counter(item["class_name"] for item in detections)
        all_results.append(
            {
                "image": relative_path.as_posix(),
                "width": width,
                "height": height,
                "seconds": round(elapsed, 3),
                "counts": dict(sorted(counts.items())),
                "detections": detections,
            }
        )
        print(
            f"[{index:02d}/{len(images):02d}] {path.name}: "
            f"body={counts['body']} face={counts['face']} text={counts['text']} "
            f"frame={counts['frame']} ({elapsed:.2f}s)",
            flush=True,
        )

    summary = {
        "model": str(args.model.resolve()),
        "input": str(args.input.resolve()),
        "threshold": args.threshold,
        "providers": session.get_providers(),
        "image_count": len(images),
        "total_seconds": round(time.perf_counter() - started, 3),
        "total_counts": dict(sorted(total_counts.items())),
        "images": all_results,
    }
    with (args.output / "detections.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps({key: summary[key] for key in ("image_count", "total_seconds", "total_counts")}, indent=2))


if __name__ == "__main__":
    main()
