#!/usr/bin/env python3
"""Cache frozen Japanese-capable embeddings for every Manga109 dialogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import cache_filename, load_page_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--text-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/text_multilingual_e5_base"),
    )
    parser.add_argument(
        "--model-name",
        default="intfloat/multilingual-e5-base",
        help="Hugging Face model name or local model directory",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--page-chunk-dialogues",
        type=int,
        default=4096,
        help="Encode this many dialogues before splitting results back into page files",
    )
    parser.add_argument("--prefix", default="query: ")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-pages", type=int, default=0, help="Debug only, per split")
    return parser.parse_args()


def cache_is_valid(path: Path, expected_ids: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as cached:
            embeddings = cached["embeddings"]
            text_ids = [str(value) for value in cached["text_ids"].tolist()]
        return (
            embeddings.ndim == 2
            and embeddings.shape[0] == len(expected_ids)
            and embeddings.shape[1] > 0
            and text_ids == expected_ids
            and np.isfinite(embeddings).all()
        )
    except (OSError, ValueError, KeyError):
        return False


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def encode_texts(
    texts: list[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    max_length: int,
    prefix: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start : start + batch_size]]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(**tokens)
            embeddings = mean_pool(output.last_hidden_state, tokens["attention_mask"])
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
        chunks.append(embeddings.cpu().numpy().astype(np.float16))
    return np.concatenate(chunks, axis=0)


def atomic_save(path: Path, embeddings: np.ndarray, text_ids: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=embeddings,
            text_ids=np.asarray(text_ids, dtype=np.str_),
        )
    temporary.replace(path)


def load_page_text(
    text_dir: Path, split: str, record: dict[str, object]
) -> tuple[list[str], list[str]]:
    path = text_dir / split / f"{Path(str(record['pack'])).stem}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text_ids = [str(value) for value in payload["text_ids"]]
    texts = [str(value) for value in payload["texts"]]
    expected_ids = [str(value) for value in record["text_ids"]]
    if text_ids != expected_ids:
        raise ValueError(f"{record['key']}: text JSON IDs differ from page index")
    if len(texts) != len(text_ids):
        raise ValueError(f"{record['key']}: text/text_id count mismatch")
    return texts, text_ids


def write_page_chunk(
    pages: list[tuple[Path, list[str], list[str]]],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    flat_texts = [text for _, texts, _ in pages for text in texts]
    flat_embeddings = encode_texts(
        flat_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
        args.prefix,
    )
    offset = 0
    for path, texts, text_ids in pages:
        end = offset + len(texts)
        atomic_save(path, flat_embeddings[offset:end], text_ids)
        offset = end
    if offset != len(flat_embeddings):
        raise AssertionError("Did not consume every encoded dialogue")
    return offset


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Text caching requires transformers. Install the project requirements first."
        ) from exc
    if args.batch_size < 1 or args.max_length < 1 or args.page_chunk_dialogues < 1:
        raise ValueError("Batch size, max length, and page chunk size must be positive")
    data_dir = args.data_dir.resolve()
    text_dir = (args.text_dir or (data_dir / "texts")).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "schema_version": 1,
        "model_name": args.model_name,
        "pooling": "attention_mask_mean",
        "l2_normalized": True,
        "prefix": args.prefix,
        "max_length": args.max_length,
        "context_order": "Manga109Dialog speaker_to_text relation order within page",
        "context_slots": ["previous", "current", "next"],
        "source_text_dir": str(text_dir),
    }
    config_path = output_dir / "config.json"
    if config_path.is_file() and not args.overwrite:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in config}
        if comparable != config:
            raise ValueError(
                f"Text cache configuration differs: {config_path}. "
                "Use another --output-dir or pass --overwrite."
            )

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).eval().to(device)
    config["embedding_dim"] = int(model.config.hidden_size)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    total_pages = total_dialogues = skipped_pages = 0
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        records = load_page_index(data_dir, split)
        if args.max_pages:
            records = records[: args.max_pages]
        pending: list[tuple[Path, list[str], list[str]]] = []
        pending_dialogues = 0
        progress = tqdm(records, desc=f"text cache {split}")
        for record in progress:
            texts, text_ids = load_page_text(text_dir, split, record)
            cache_path = split_dir / cache_filename(str(record["key"]))
            if not args.overwrite and cache_is_valid(cache_path, text_ids):
                skipped_pages += 1
                total_pages += 1
                total_dialogues += len(texts)
                continue
            pending.append((cache_path, texts, text_ids))
            pending_dialogues += len(texts)
            if pending_dialogues >= args.page_chunk_dialogues:
                total_dialogues += write_page_chunk(
                    pending, tokenizer, model, device, args
                )
                total_pages += len(pending)
                pending = []
                pending_dialogues = 0
        if pending:
            total_dialogues += write_page_chunk(pending, tokenizer, model, device, args)
            total_pages += len(pending)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "device": str(device),
                "pages": total_pages,
                "dialogues": total_dialogues,
                "skipped_pages": skipped_pages,
                "embedding_dim": int(model.config.hidden_size),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
