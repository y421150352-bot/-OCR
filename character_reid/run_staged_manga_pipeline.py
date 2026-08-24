#!/usr/bin/env python3
"""Two-stage manga pipeline: review character clusters before speaker inference."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from run_final_tail_ray_panel_pipeline import (
    FINAL_SPEAKER_PROTOCOL,
    V3SpeakerRanker,
    build_instances,
    cluster_characters,
    draw_pages,
    extract_embeddings,
    inject_magi_tails,
    match_dialogues,
    verify_panel_pages_with_vlm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cluster", "finalize"), required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reid-checkpoint", type=Path)
    parser.add_argument("--v3-checkpoint", type=Path)
    parser.add_argument("--text-model", type=str)
    parser.add_argument("--reviewed-instances", type=Path)
    parser.add_argument("--ocr-bundles-dir", type=Path)
    parser.add_argument("--magi-dir", type=Path)
    parser.add_argument("--tail-weight", type=float, default=6.0)
    parser.add_argument("--tail-text-max-distance", type=float, default=0.12)
    parser.add_argument("--tail-ray-width", type=float, default=0.035)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--vlm-top5", action="store_true")
    parser.add_argument("--vlm-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--vlm-endpoint", default="")
    parser.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--vlm-timeout", type=int, default=300)
    parser.add_argument("--vlm-retries", type=int, default=3)
    parser.add_argument("--vlm-panel-batch-size", type=int, default=1)
    parser.add_argument("--vlm-identity-batch-size", type=int, default=5)
    parser.add_argument(
        "--vlm-first-pass-confidence-threshold", type=float, default=0.80
    )
    parser.add_argument("--vlm-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--vlm-speaker-top-k", type=int, default=5)
    parser.add_argument("--vlm-max-pages", type=int, default=0)
    parser.add_argument("--vlm-max-dialogues", type=int, default=0)
    parser.add_argument("--vlm-image-size", type=int, default=224)
    parser.add_argument("--vlm-save-boards", action="store_true")
    parser.add_argument("--largest-cluster-limit", type=float, default=0.55)
    parser.add_argument("--small-book-similarity", type=float, default=0.72)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    for name in (
        "vlm_first_pass_confidence_threshold",
        "vlm_confidence_threshold",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if (
        args.vlm_timeout < 1
        or args.vlm_retries < 1
        or args.vlm_panel_batch_size < 1
        or args.vlm_identity_batch_size < 1
        or not 1 <= args.vlm_speaker_top_k <= 5
    ):
        parser.error("Invalid VLM timeout, retry, batch-size, or speaker Top-K value")
    return args


def stable_labels(labels: np.ndarray) -> np.ndarray:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        groups[int(label)].append(index)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    stable = np.empty(len(labels), dtype=np.int64)
    for number, members in enumerate(ordered, 1):
        stable[np.asarray(members)] = number
    return stable


def cosine_average_linkage_labels(
    similarity: np.ndarray, threshold: float
) -> np.ndarray:
    """Small-N average-linkage agglomeration without an extra dependency."""
    groups = [[index] for index in range(len(similarity))]
    while len(groups) > 1:
        best_pair = None
        best_score = float(threshold)
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                score = float(similarity[np.ix_(groups[left], groups[right])].mean())
                if score >= best_score:
                    best_score = score
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        groups[left] = groups[left] + groups[right]
        del groups[right]
    labels = np.empty(len(similarity), dtype=np.int64)
    for label, members in enumerate(groups):
        labels[np.asarray(members, dtype=np.int64)] = label
    return labels


def cluster_for_book(
    features: np.ndarray,
    largest_limit: float,
    small_book_similarity: float = 0.72,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a clustering protocol by instance count.

    HDBSCAN needs enough density to form a core. On one-page uploads it often
    marks every point as noise, so pair-capable cosine agglomeration is used
    instead. Both protocols operate on the trained ReID embeddings.
    """
    if len(features) <= 40:
        if len(features) == 1:
            return np.asarray([1], dtype=np.int64), {
                "mode": "small_book_cosine_agglomerative",
                "similarity_threshold": small_book_similarity,
                "clusters": 1,
                "singleton_clusters": 1,
            }
        similarity_matrix = (
            np.asarray(features, dtype=np.float32)
            @ np.asarray(features, dtype=np.float32).T
        )
        np.fill_diagonal(similarity_matrix, -np.inf)
        nearest_similarities = similarity_matrix.max(axis=1)
        adaptive_similarity = float(np.percentile(nearest_similarities, 60))
        effective_similarity = min(
            float(small_book_similarity), max(0.64, adaptive_similarity)
        )
        labels = stable_labels(
            cosine_average_linkage_labels(similarity_matrix, effective_similarity)
        )
        counts = np.bincount(labels)[1:]
        return labels, {
            "mode": "small_book_cosine_agglomerative",
            "requested_similarity_threshold": float(small_book_similarity),
            "effective_similarity_threshold": effective_similarity,
            "distance_threshold": float(1.0 - effective_similarity),
            "nearest_similarity_p60": adaptive_similarity,
            "clusters": int(len(counts)),
            "singleton_clusters": int((counts == 1).sum()),
            "largest_cluster_ratio": (
                float(counts.max() / len(features)) if len(counts) else 0.0
            ),
        }
    labels, diagnostics = cluster_characters(features, largest_limit)
    return labels, {"mode": "validated_book_pca_hdbscan", **diagnostics}


def save_review_crops(
    instances: list[dict[str, Any]], image_dir: Path, output_dir: Path, crop_size: int
) -> None:
    destination_dir = output_dir / "review_crops"
    destination_dir.mkdir(parents=True, exist_ok=True)
    current_name = None
    image = None
    try:
        for row in instances:
            if row["image"] != current_name:
                if image is not None:
                    image.close()
                image = ImageOps.exif_transpose(
                    Image.open(image_dir / row["image"])
                ).convert("RGB")
                current_name = row["image"]
            crop = image.crop(tuple(row["body_box"]))
            crop.thumbnail((crop_size, crop_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (crop_size, crop_size), "white")
            canvas.paste(
                crop, ((crop_size - crop.width) // 2, (crop_size - crop.height) // 2)
            )
            canvas.save(destination_dir / f"{row['instance_id']}.jpg", quality=92)
    finally:
        if image is not None:
            image.close()


def run_cluster(
    args: argparse.Namespace, payload: dict[str, Any], injected_tail_count: int
) -> None:
    if args.reid_checkpoint is None:
        raise SystemExit("cluster stage requires --reid-checkpoint")
    instances, _ = build_instances(payload)
    features = extract_embeddings(
        instances, args.image_dir, args.reid_checkpoint, torch.device(args.device)
    )
    np.save(args.output_dir / "reid_embeddings.npy", features)
    labels, clustering = cluster_for_book(
        features, args.largest_cluster_limit, args.small_book_similarity
    )
    for embedding_index, (row, label) in enumerate(zip(instances, labels.tolist())):
        row["character_cluster_id"] = f"cluster_{label:03d}"
        row["character_name"] = f"角色簇 {label:03d}"
        row["embedding_index"] = embedding_index
        row["library_member"] = True
        row["excluded"] = False
    save_review_crops(instances, args.image_dir, args.output_dir, args.crop_size)
    pages = [{"image": page["image"], "dialogues": []} for page in payload["images"]]
    result = {
        "stage": "cluster_review",
        "protocol": "book_level_reid_clustering_review_before_speaker_inference",
        "clustering": clustering,
        "summary": {
            "pages": len(payload["images"]),
            "character_instances": len(instances),
            "character_clusters": len(set(labels.tolist())),
            "dialogues": 0,
            "magiv3_tails_injected": injected_tail_count,
        },
        "character_instances": instances,
        "pages": pages,
    }
    (args.output_dir / "pipeline_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print("Waiting for character-cluster review.", flush=True)


def run_finalize(
    args: argparse.Namespace, payload: dict[str, Any], injected_tail_count: int
) -> None:
    if args.reviewed_instances is None or not args.reviewed_instances.is_file():
        raise SystemExit("finalize stage requires --reviewed-instances")
    if args.v3_checkpoint is None:
        raise SystemExit("finalize stage requires --v3-checkpoint")
    instances = [
        row
        for row in json.loads(args.reviewed_instances.read_text(encoding="utf-8"))
        if not row.get("excluded", False)
    ]
    if not instances:
        raise SystemExit("No reviewed character instances remain")
    embedding_path = args.output_dir / "reid_embeddings.npy"
    if not embedding_path.is_file():
        raise SystemExit(f"Missing ReID embeddings: {embedding_path}")
    features = np.asarray(np.load(embedding_path), dtype=np.float32)
    library_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        if (
            row.get("library_member", True)
            and row.get("character_cluster_id") != "unassigned"
        ):
            library_groups[row["character_cluster_id"]].append(row)
    if not library_groups:
        raise SystemExit("The reviewed character library has no reference samples")
    cluster_ids = sorted(library_groups)
    prototypes = []
    names = []
    for cluster_id in cluster_ids:
        members = library_groups[cluster_id]
        indexes = np.asarray(
            [int(row["embedding_index"]) for row in members], dtype=np.int64
        )
        prototype = features[indexes].mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        prototypes.append(prototype)
        names.append(str(members[0]["character_name"]))
    prototype_matrix = np.asarray(prototypes, dtype=np.float32)
    for row in instances:
        vector = features[int(row["embedding_index"])]
        similarities = vector @ prototype_matrix.T
        order = np.argsort(-similarities, kind="stable")[
            : min(args.top_k, len(cluster_ids))
        ]
        row["retrieval_top_k"] = [
            {
                "rank": rank,
                "character_name": names[index],
                "similarity": round(float(similarities[index]), 6),
            }
            for rank, index in enumerate(order.tolist(), 1)
        ]
        if (
            not row.get("library_member", True)
            or row.get("character_cluster_id") == "unassigned"
        ):
            best = int(order[0])
            row["character_cluster_id"] = cluster_ids[best]
            row["character_name"] = names[best]
            row["retrieval_similarity"] = round(float(similarities[best]), 6)
            row["assignment_source"] = "reid_character_bank_top1"
        else:
            row["retrieval_similarity"] = round(
                float(similarities[cluster_ids.index(row["character_cluster_id"])]), 6
            )
            row["assignment_source"] = "manual_character_bank_reference"
    bank = {
        "protocol": "manual_reference_library_then_reid_topk_assignment",
        "characters": [
            {
                "character_cluster_id": cluster_id,
                "character_name": name,
                "reference_instance_ids": [
                    row["instance_id"] for row in library_groups[cluster_id]
                ],
            }
            for cluster_id, name in zip(cluster_ids, names)
        ],
    }
    (args.output_dir / "character_bank.json").write_text(
        json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "character_bank_prototypes.npz",
        prototypes=prototype_matrix,
        cluster_ids=np.asarray(cluster_ids),
        names=np.asarray(names),
    )
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        by_image[row["image"]].append(row)
    device = torch.device(args.device)
    ranker = V3SpeakerRanker(args.v3_checkpoint, args.text_model, device)
    speaker_candidate_count = max(
        args.top_k,
        args.vlm_speaker_top_k if args.vlm_top5 else args.top_k,
    )
    pages = match_dialogues(
        payload,
        by_image,
        args.ocr_bundles_dir,
        ranker,
        speaker_candidate_count,
        args.tail_text_max_distance,
        args.tail_weight,
        args.tail_ray_width,
    )
    vlm_top5: dict[str, Any] = {"enabled": False}
    if args.vlm_top5:
        vlm_features = np.asarray(
            [features[int(row["embedding_index"])] for row in instances],
            dtype=np.float32,
        )
        print(
            f"Gemini web verification: model={args.vlm_model} "
            f"page_limit={args.vlm_max_pages or 'all'}",
            flush=True,
        )
        vlm_top5 = {
            "enabled": True,
            "model": args.vlm_model,
            "first_pass_confidence_threshold": args.vlm_first_pass_confidence_threshold,
            "identity_confidence_threshold": args.vlm_confidence_threshold,
            "raw_responses": str(args.output_dir / "vlm_raw_responses.jsonl"),
            **verify_panel_pages_with_vlm(
                pages,
                instances,
                vlm_features,
                args.image_dir,
                args.output_dir,
                args,
            ),
        }
    result = {
        "stage": "final",
        "protocol": "reviewed_character_library_then_dialogue_speaker_mapping",
        "speaker_protocol": FINAL_SPEAKER_PROTOCOL,
        "speaker_model": "v3",
        "vlm_top5": vlm_top5,
        "summary": {
            "pages": len(payload["images"]),
            "character_instances": len(instances),
            "character_clusters": len(
                {row["character_cluster_id"] for row in instances}
            ),
            "dialogues": sum(len(page["dialogues"]) for page in pages),
            "magiv3_tails_injected": injected_tail_count,
            "tail_fused_dialogues": sum(
                dialogue.get("speaker_source") == "v3_tail_fusion"
                for page in pages
                for dialogue in page["dialogues"]
            ),
            "vlm_top5_reviewed": int(vlm_top5.get("reviewed", 0)),
            "vlm_top5_unknown": int(vlm_top5.get("unknown", 0)),
            "vlm_top5_changed": int(vlm_top5.get("changed", 0)),
        },
        "character_instances": instances,
        "pages": pages,
    }
    (args.output_dir / "pipeline_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draw_pages(pages, args.image_dir, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.detections.read_text(encoding="utf-8"))
    injected_tail_count = inject_magi_tails(payload, args.magi_dir)
    (args.output_dir / "detections_with_magiv3_tails.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.stage == "cluster":
        run_cluster(args, payload, injected_tail_count)
    else:
        run_finalize(args, payload, injected_tail_count)


if __name__ == "__main__":
    main()
