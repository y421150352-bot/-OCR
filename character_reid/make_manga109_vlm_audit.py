#!/usr/bin/env python3
"""Create image cards showing correct and incorrect Manga109 VLM decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def intersection_over_union(first: list[float], second: list[float]) -> float:
    xmin = max(first[0], second[0])
    ymin = max(first[1], second[1])
    xmax = min(first[2], second[2])
    ymax = min(first[3], second[3])
    intersection = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0, first_area + second_area - intersection)


def best_match(
    box: list[float], rows: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    if not rows:
        return None, 0.0
    row = max(rows, key=lambda item: intersection_over_union(box, item["box"]))
    return row, intersection_over_union(box, row["box"])


def crop_bounds(
    image: Image.Image,
    boxes: list[list[float]],
    padding: int = 80,
) -> tuple[int, int, int, int]:
    xmin = max(0, int(min(box[0] for box in boxes)) - padding)
    ymin = max(0, int(min(box[1] for box in boxes)) - padding)
    xmax = min(image.width, int(max(box[2] for box in boxes)) + padding)
    ymax = min(image.height, int(max(box[3] for box in boxes)) + padding)
    return xmin, ymin, max(xmin + 1, xmax), max(ymin + 1, ymax)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    offset: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top = offset
    adjusted = [box[0] - left, box[1] - top, box[2] - left, box[3] - top]
    draw.rectangle(adjusted, outline=color, width=4)
    draw.text(
        (adjusted[0] + 3, max(0, adjusted[1] - 22)),
        label,
        fill=color,
        font=font,
        stroke_width=1,
        stroke_fill="white",
    )


def make_card(
    image_path: Path,
    dialogue: dict[str, Any],
    gt_text: dict[str, Any],
    speaker_correct: bool,
    identity_correct: bool | None,
) -> Image.Image:
    with Image.open(image_path) as raw:
        page = ImageOps.exif_transpose(raw).convert("RGB")

    boxes = [list(dialogue["text_box"])]
    if gt_text.get("speaker_body_box") is not None:
        boxes.append(list(gt_text["speaker_body_box"]))
    predicted_body = dialogue.get("speaker_body")
    if isinstance(predicted_body, list):
        boxes.append(list(predicted_body))
    bounds = crop_bounds(page, boxes)
    crop = page.crop(bounds)
    draw = ImageDraw.Draw(crop)
    font = ImageFont.load_default(size=20)

    draw_box(draw, list(dialogue["text_box"]), bounds[:2], (220, 35, 35), "TEXT", font)
    if gt_text.get("speaker_body_box") is not None:
        draw_box(
            draw,
            list(gt_text["speaker_body_box"]),
            bounds[:2],
            (30, 175, 55),
            "GT",
            font,
        )
    if isinstance(predicted_body, list):
        draw_box(draw, predicted_body, bounds[:2], (35, 95, 235), "PRED", font)

    identity_status = (
        "N/A" if identity_correct is None else "OK" if identity_correct else "WRONG"
    )
    banner = (
        f"{dialogue.get('dialogue_id')} | speaker={'OK' if speaker_correct else 'WRONG'} "
        f"| identity={identity_status}"
    )
    banner_height = 34
    canvas = Image.new("RGB", (crop.width, crop.height + banner_height), "white")
    canvas.paste(crop, (0, banner_height))
    ImageDraw.Draw(canvas).text((8, 8), banner, fill="black", font=font)
    return canvas


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    gt_pages = {str(page["image"]): page for page in ground_truth["pages"]}

    cluster_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for instance in result.get("character_instances", []):
        gt_page = gt_pages.get(str(instance["image"]))
        if gt_page is None:
            continue
        body, overlap = best_match(
            list(instance["body_box"]), gt_page.get("bodies", [])
        )
        if body is not None and overlap >= 0.5:
            cluster_votes[str(instance["character_cluster_id"])][
                str(body["character_id"])
            ] += 1
    cluster_characters = {
        cluster_id: votes.most_common(1)[0][0]
        for cluster_id, votes in cluster_votes.items()
        if votes
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    totals = Counter()
    for page in result.get("pages", []):
        image_name = str(page["image"])
        gt_page = gt_pages.get(image_name)
        if gt_page is None:
            continue
        for dialogue in page.get("dialogues", []):
            gt_text, text_overlap = best_match(
                list(dialogue["text_box"]), gt_page.get("texts", [])
            )
            if gt_text is None or text_overlap < 0.5:
                continue
            is_dialogue = bool(gt_text.get("is_dialogue"))
            predicted_body = dialogue.get("speaker_body")
            speaker_iou = (
                intersection_over_union(predicted_body, gt_text["speaker_body_box"])
                if is_dialogue
                and isinstance(predicted_body, list)
                and gt_text.get("speaker_body_box") is not None
                else 0.0
            )
            speaker_correct = is_dialogue and speaker_iou >= 0.5
            predicted_character = cluster_characters.get(
                str(dialogue.get("character_cluster_id", "unknown"))
            )
            identity_correct = (
                is_dialogue
                and predicted_character is not None
                and predicted_character == gt_text.get("speaker_character_id")
            )
            if not is_dialogue:
                category = "not_dialogue"
                identity_result: bool | None = None
            elif dialogue.get("character_name") == "unknown":
                category = "unknown"
                identity_result = identity_correct
            elif speaker_correct and identity_correct:
                category = "correct"
                identity_result = identity_correct
            else:
                category = "wrong"
                identity_result = identity_correct

            card = make_card(
                args.image_dir / image_name,
                dialogue,
                gt_text,
                speaker_correct,
                identity_result,
            )
            filename = f"{dialogue['dialogue_id']}_{gt_text['text_id']}.jpg"
            card_path = args.output_dir / category / filename
            card_path.parent.mkdir(parents=True, exist_ok=True)
            card.save(card_path, quality=94)
            totals[category] += 1
            manifest.append(
                {
                    "card": str(card_path.relative_to(args.output_dir)),
                    "category": category,
                    "image": image_name,
                    "dialogue_id": dialogue["dialogue_id"],
                    "gt_text_id": gt_text["text_id"],
                    "speaker_correct": speaker_correct,
                    "identity_correct": identity_result,
                    "speaker_iou": round(speaker_iou, 4),
                    "ground_truth_character": gt_text.get("speaker_character_name"),
                    "predicted_cluster": dialogue.get("character_cluster_id"),
                }
            )

    (args.output_dir / "manifest.json").write_text(
        json.dumps({"summary": totals, "rows": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(f"Audit cards: {args.output_dir}")


if __name__ == "__main__":
    main()
