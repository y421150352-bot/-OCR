#!/usr/bin/env python3
"""Compare two-pass VLM pipeline output with Manga109 speaker annotations."""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path)
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


def normalize_text(value: Any) -> str:
    return "".join(str(value or "").split())


def best_match(
    box: list[float], rows: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    if not rows:
        return None, 0.0
    row = max(rows, key=lambda item: intersection_over_union(box, item["box"]))
    return row, intersection_over_union(box, row["box"])


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    gt_pages = {str(page["image"]): page for page in ground_truth["pages"]}

    cluster_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for instance in result.get("character_instances", []):
        page = gt_pages.get(str(instance["image"]))
        if page is None:
            continue
        body, overlap = best_match(list(instance["body_box"]), page.get("bodies", []))
        if body is None or overlap < 0.5:
            continue
        cluster_id = str(instance["character_cluster_id"])
        character_id = str(body["character_id"])
        cluster_votes[cluster_id][character_id] += 1
    cluster_characters = {
        cluster_id: votes.most_common(1)[0][0]
        for cluster_id, votes in cluster_votes.items()
        if votes
    }

    rows = []
    for page in result.get("pages", []):
        gt_page = gt_pages.get(str(page["image"]))
        if gt_page is None:
            continue
        for dialogue in page.get("dialogues", []):
            gt_text, text_overlap = best_match(
                list(dialogue["text_box"]), gt_page.get("texts", [])
            )
            if gt_text is None or text_overlap < 0.5:
                continue
            predicted_body = dialogue.get("speaker_body")
            speaker_overlap = (
                intersection_over_union(predicted_body, gt_text["speaker_body_box"])
                if isinstance(predicted_body, list)
                and gt_text.get("speaker_body_box") is not None
                else 0.0
            )
            candidate_overlaps = (
                [
                    intersection_over_union(
                        list(candidate["body_box"]), gt_text["speaker_body_box"]
                    )
                    for candidate in dialogue.get("top_candidates", [])
                ]
                if gt_text.get("speaker_body_box") is not None
                else []
            )
            predicted_cluster = str(dialogue.get("character_cluster_id", "unknown"))
            predicted_character = cluster_characters.get(predicted_cluster)
            is_dialogue = bool(gt_text["is_dialogue"])
            predicted_type = str(dialogue.get("text_type", "unknown"))
            requires_character_link = bool(
                dialogue.get("requires_character_link", predicted_type == "dialogue")
            )
            normalized_ground_truth = normalize_text(gt_text["text"])
            normalized_prediction = normalize_text(dialogue.get("recognized_text"))
            rows.append(
                {
                    "image": page["image"],
                    "dialogue_id": dialogue.get("dialogue_id"),
                    "text_id": gt_text["text_id"],
                    "gt_text": gt_text["text"],
                    "predicted_text": dialogue.get("recognized_text", "unknown"),
                    "text_exact": normalized_prediction == normalized_ground_truth,
                    "text_similarity": round(
                        difflib.SequenceMatcher(
                            None, normalized_prediction, normalized_ground_truth
                        ).ratio(),
                        4,
                    ),
                    "gt_is_dialogue": is_dialogue,
                    "predicted_type": predicted_type,
                    "requires_character_link": requires_character_link,
                    "dialogue_type_correct": requires_character_link == is_dialogue,
                    "gt_speaker_body_id": gt_text.get("speaker_body_id"),
                    "predicted_speaker_instance_id": dialogue.get(
                        "speaker_instance_id"
                    ),
                    "speaker_body_iou": round(speaker_overlap, 4),
                    "speaker_correct": is_dialogue and speaker_overlap >= 0.5,
                    "speaker_top1_contains_gt": is_dialogue
                    and bool(candidate_overlaps)
                    and candidate_overlaps[0] >= 0.5,
                    "speaker_top5_contains_gt": is_dialogue
                    and bool(candidate_overlaps)
                    and max(candidate_overlaps[:5]) >= 0.5,
                    "speaker_vlm_accepted": dialogue.get("speaker_vlm_top5", {}).get(
                        "status"
                    )
                    == "accepted",
                    "gt_character_id": gt_text.get("speaker_character_id"),
                    "gt_character_name": gt_text.get("speaker_character_name"),
                    "predicted_cluster_id": predicted_cluster,
                    "predicted_character_id_by_cluster_vote": predicted_character,
                    "identity_correct": is_dialogue
                    and predicted_character is not None
                    and predicted_character == gt_text.get("speaker_character_id"),
                    "identity_vlm_accepted": dialogue.get("identity_vlm_top5", {}).get(
                        "status"
                    )
                    == "accepted",
                    "unknown": dialogue.get("character_name") == "unknown",
                }
            )

    dialogue_rows = [row for row in rows if row["gt_is_dialogue"]]
    accepted_speaker_rows = [
        row for row in dialogue_rows if row["speaker_vlm_accepted"]
    ]
    accepted_identity_rows = [
        row for row in dialogue_rows if row["identity_vlm_accepted"]
    ]
    metrics = {
        "pages": len(gt_pages),
        "ground_truth_texts": sum(
            len(page.get("texts", [])) for page in gt_pages.values()
        ),
        "matched_texts": len(rows),
        "text_exact_accuracy": sum(row["text_exact"] for row in rows)
        / max(1, len(rows)),
        "mean_text_similarity": sum(row["text_similarity"] for row in rows)
        / max(1, len(rows)),
        "dialogue_filter_accuracy": sum(row["dialogue_type_correct"] for row in rows)
        / max(1, len(rows)),
        "character_link_accuracy": sum(row["dialogue_type_correct"] for row in rows)
        / max(1, len(rows)),
        "ground_truth_dialogues": len(dialogue_rows),
        "speaker_candidate_top1_recall": sum(
            row["speaker_top1_contains_gt"] for row in dialogue_rows
        )
        / max(1, len(dialogue_rows)),
        "speaker_candidate_top5_recall": sum(
            row["speaker_top5_contains_gt"] for row in dialogue_rows
        )
        / max(1, len(dialogue_rows)),
        "speaker_body_accuracy": sum(row["speaker_correct"] for row in dialogue_rows)
        / max(1, len(dialogue_rows)),
        "speaker_accepted_precision": sum(
            row["speaker_correct"] for row in accepted_speaker_rows
        )
        / max(1, len(accepted_speaker_rows)),
        "speaker_accepted_count": len(accepted_speaker_rows),
        "identity_accuracy": sum(row["identity_correct"] for row in dialogue_rows)
        / max(1, len(dialogue_rows)),
        "identity_accepted_precision": sum(
            row["identity_correct"] for row in accepted_identity_rows
        )
        / max(1, len(accepted_identity_rows)),
        "identity_accepted_count": len(accepted_identity_rows),
        "unknown_rate_on_dialogues": sum(row["unknown"] for row in dialogue_rows)
        / max(1, len(dialogue_rows)),
    }
    output = {"metrics": metrics, "rows": rows}
    output_path = args.output or args.result.with_name("manga109_evaluation.json")
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Evaluation: {output_path.resolve()}")


if __name__ == "__main__":
    main()
