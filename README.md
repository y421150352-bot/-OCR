# Manga OCR, Character ReID, and Speaker Attribution

This repository contains the source code developed for an end-to-end manga
analysis pipeline:

- `character_reid/`: supervised contrastive character ReID, character-bank
  retrieval, unsupervised clustering, staged inference, and the review web UI.
- `speaker_relation_transformer/`: geometry baselines, Relation Transformer
  variants, text/ReID feature caching, training, evaluation, and ablation tools.
- `rtdetr_manga/`: ONNX inference entry point for manga body, face, frame, and
  text detection.
- `ppocr_dataset_tools/`: scripts for auditing and preparing unified manga OCR
  annotations for PaddleOCR.

## Data and model files

No manga pages, COO/Manga109 annotations, user uploads, model checkpoints,
embeddings, or generated experiment outputs are included. Obtain each dataset
from its official source and comply with its license and access terms.

Large checkpoints are intentionally excluded from Git. Configure their paths
through the command-line arguments documented by each script.

## Environment

Each component has its own dependency list where available. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r character_reid/requirements.txt
pip install -r speaker_relation_transformer/requirements.txt
```

Run each entry point with `--help` before training or inference to review its
dataset, checkpoint, and output-path arguments.

## Security

API credentials such as `GEMINI_API_KEY` must be supplied through environment
variables. Never commit `.env` files, private keys, datasets, or task outputs.

## Repository status

This is a research code release. Dataset preparation and model checkpoints are
kept separate from the source repository.

