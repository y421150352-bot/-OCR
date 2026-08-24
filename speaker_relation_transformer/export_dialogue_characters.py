#!/usr/bin/env python3
"""Export Dialogue -> speaker body -> predicted character ID/name."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data import GeometryGraphPageDataset, cache_filename
from model_v3 import SpeakerGeometryTextGraphTransformer


def load_character_predictions(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    pages: dict[str, dict[str, dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            pages[str(record["key"])] = {
                str(candidate["candidate_id"]): candidate
                for candidate in record["candidates"]
            }
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reid-cache-dir", type=Path, default=Path("cache/reid"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--character-predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Speaker inference requires CUDA")
    device = torch.device("cuda")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("v3_text_ablation")) != "no_text":
        raise ValueError("This exporter currently expects a no-text V3 checkpoint")
    reid_dim = int(config.get("reid_dim", 0))
    if reid_dim < 1:
        raise ValueError("Checkpoint was not trained with ReID embeddings")
    model = SpeakerGeometryTextGraphTransformer(
        text_dim=int(config.get("text_dim", 768)),
        hidden_dim=int(config["hidden_dim"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        attention_dropout=float(config["attention_dropout"]),
        geometry_bias_hidden=int(config["geometry_bias_hidden"]),
        geometry_bias_scale_init=float(config["geometry_bias_scale_init"]),
        use_text=False,
        use_dialogue_graph=str(config.get("v3_graph_mode")) == "two_axis",
        reid_dim=reid_dim,
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    dataset = GeometryGraphPageDataset(
        args.data_dir, args.split, reid_cache_dir=args.reid_cache_dir
    )
    character_pages = load_character_predictions(args.character_predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queries = speaker_correct = character_correct = 0
    with args.output.open("w", encoding="utf-8") as writer:
        for index in tqdm(range(len(dataset)), desc="Export dialogue characters"):
            page = dataset[index]
            record = dataset.records[index]
            key = str(record["key"])
            if key not in character_pages:
                raise ValueError(f"Missing character predictions for {key}")
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                score_rows = model.forward_page(
                    page["geometry"].to(device),
                    page["text_context"].to(device),
                    page["text_context_mask"].to(device),
                    page["candidate_reid"].to(device),
                )
            scores = torch.stack(score_rows).float().cpu()
            labels = page["labels"]
            candidate_ids = [str(value) for value in record["candidate_ids"]]
            text_ids = [str(value) for value in record["text_ids"]]
            cache_path = (
                args.reid_cache_dir / args.split / cache_filename(key)
            )
            with np.load(cache_path) as cached:
                gt_character_ids = [str(value) for value in cached["character_ids"].tolist()]
                gt_character_names = [str(value) for value in cached["character_names"].tolist()]
            for dialogue_index, (text_id, row_scores) in enumerate(zip(text_ids, scores)):
                candidate_index = int(row_scores.argmax())
                candidate_id = candidate_ids[candidate_index]
                identity_prediction = character_pages[key][candidate_id]
                positive_indexes = torch.nonzero(
                    labels[dialogue_index], as_tuple=False
                ).flatten().tolist()
                valid_character_ids = sorted(
                    {gt_character_ids[position] for position in positive_indexes}
                )
                valid_character_names = sorted(
                    {gt_character_names[position] for position in positive_indexes}
                )
                is_speaker_correct = bool(labels[dialogue_index, candidate_index])
                predicted_character_id = identity_prediction["character_id"]
                is_character_correct = (
                    predicted_character_id is not None
                    and str(predicted_character_id) in valid_character_ids
                )
                queries += 1
                speaker_correct += is_speaker_correct
                character_correct += is_character_correct
                writer.write(
                    json.dumps(
                        {
                            "key": key,
                            "book": str(record["book"]),
                            "page_index": int(record["page_index"]),
                            "text_id": text_id,
                            "predicted_body_id": candidate_id,
                            "predicted_character_id": predicted_character_id,
                            "predicted_character_name": identity_prediction["character_name"],
                            "character_similarity": identity_prediction["similarity"],
                            "speaker_score": float(row_scores[candidate_index]),
                            "speaker_correct": is_speaker_correct,
                            "character_correct": is_character_correct,
                            "gt_character_ids": valid_character_ids,
                            "gt_character_names": valid_character_names,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    metrics = {
        "queries": queries,
        "speaker_body_top1": speaker_correct / max(queries, 1),
        "dialogue_character_top1": character_correct / max(queries, 1),
        "output": str(args.output.resolve()),
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
