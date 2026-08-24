# DINOv3 Speaker Relation Transformer

This project is intentionally isolated from `speaker_geometry_baseline` and the
production OCR pipeline. It trains a page-level dialogue-to-visible-character
ranker on Manga109Dialog.

## Model

- Frozen `facebook/dinov3-vitb16-pretrain-lvd1689m` page patch features.
- ROI pooling for every dialogue box and candidate body box.
- MLP projection of the existing 45 geometry features.
- Candidate self-attention plus cross-attention to the complete dense DINOv3
  page patch grid (no pooling by default).
- Multi-positive listwise ranking loss.

The first stage does not fine-tune DINOv3. This makes it possible to cache page
features once and train the relation head quickly on a single RTX 5090.

## 1. Build page packs locally

```powershell
python speaker_relation_transformer\build_page_packs.py
```

The script consumes the exact book-disjoint split and 45-dimensional features
from the geometry baseline, but writes all new files under this project.

Verify all packs and source image paths before upload:

```powershell
python speaker_relation_transformer\verify_data.py
```

Export the official Manga109 Japanese transcriptions without running OCR:

```powershell
python speaker_relation_transformer\extract_dialogue_texts.py
```

This writes one UTF-8 JSON file per page under `data/texts/{train,val,test}`.
`texts[i]` is aligned with `text_ids[i]` in the page index and with row `i` of
`geometry`, `labels`, and `text_boxes` in the NPZ pack. The exporter resolves
texts by their original XML IDs and checks every XML bbox against the NPZ before
writing. A complete validation summary is written to
`data/text_alignment_report.json`.

## 2. Copy to the GPU server

Copy both the complete Manga109-s dataset and this project directory. A proposed
server layout is:

```text
/home/USER/speaker_relation_transformer/
/home/USER/datasets/Manga109s_released_2023_12_07/
```

Recommended: use WinSCP with SFTP so a large dataset transfer can resume. Map:

```text
C:\path\to\speaker_relation_transformer
  -> /home/USER/speaker_relation_transformer

C:\path\to\datasets\Manga109s_released_2023_12_07
  -> /home/USER/datasets/Manga109s_released_2023_12_07
```

Alternatively, from Windows PowerShell (enter the SSH password interactively):

```powershell
ssh -p PORT USER@SERVER "mkdir -p /home/USER/datasets"
scp -P PORT -r "C:\path\to\speaker_relation_transformer" `
  USER@SERVER:/home/USER/
scp -P PORT -r "C:\path\to\datasets\Manga109s_released_2023_12_07" `
  USER@SERVER:/home/USER/datasets/
```

The old `speaker_geometry_baseline` directory does not need to be uploaded.
Its split and 45-dimensional features have already been copied into the new
project's `data/` directory.

Do not publish or redistribute Manga109-s or trained weights without complying
with its license and acknowledgement requirements.

## 3. Create the server environment

Use Python 3.11 or 3.12, not the server's system Python 3.14.

```bash
cd /home/USER/speaker_relation_transformer
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
```

If Hugging Face requests DINOv3 license acceptance, accept it on the model page
and set `HF_TOKEN` on the server. Never paste the token into source code.

## 4. Cache frozen DINOv3 features on GPU 0

First run a six-page smoke cache (two pages from each split):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python cache_dinov3.py \
  --dataset-root /home/USER/datasets/Manga109s_released_2023_12_07 \
  --data-dir data --cache-dir cache/smoke --max-pages 2
```

Then run one tiny end-to-end training check:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --data-dir data --cache-dir cache/smoke --output-dir runs/smoke \
  --epochs 1 --max-pages 2 --num-workers 0 \
  --hidden-dim 128 --layers 1 --heads 4
```

Only after both commands succeed, build the full cache:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python cache_dinov3.py \
  --dataset-root /home/USER/datasets/Manga109s_released_2023_12_07 \
  --data-dir data \
  --cache-dir cache/dinov3_vitb16_l896
```

Resume is automatic: existing valid page cache files are skipped.

## 5. Train the relation Transformer

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --data-dir data \
  --cache-dir cache/dinov3_vitb16_l896 \
  --output-dir runs/vitb16_relation_lr1e4 \
  --epochs 50 \
  --lr 1e-4 --min-lr 1e-6 \
  --warmup-epochs 3 --cosine-epochs 40 \
  --early-stopping-patience 10 \
  --grad-accum 8 --amp bf16
```

The defaults match the command above. The learning rate warms up linearly for
three epochs, then decays to `1e-6` over 40 epochs. Training stops when
validation Top-1 fails to improve for ten consecutive epochs. `best.pt` always
contains the highest validation Top-1 checkpoint, `last.pt` contains the most
recent completed epoch, and the untouched test split is evaluated from
`best.pt` after either early stopping or the requested maximum epoch.

Use a new output directory for this schedule. Resuming a checkpoint produced by
the old 100-epoch cosine schedule applies the new deterministic learning-rate
curve from the resumed epoch and is not a clean experimental comparison.

The default `--context-grid 0` preserves the complete patch grid (for example
about `56x40=2240` tokens at long side 896). If measured GPU memory or training
speed is unacceptable, retry with `--context-grid 28`, which preserves aspect
ratio and turns a `56x40` grid into about `28x20`. Use `--context-grid 14` only
as the final fallback.

The best validation checkpoint is evaluated on the untouched test books. Main
comparison target: geometry LightGBM Top-1 75.23%, Top-3 95.94%, MRR 85.65%.

## 6. Train the edge-aware bipartite Graph Transformer V2

V2 keeps the same frozen DINOv3 cache, page packs, listwise loss, metrics, and
book-disjoint splits. Each dialogue-character relation remains an independent
edge token. The standardized 45D geometry is used both as an edge embedding and
as a per-head candidate attention bias.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_v2.py \
  --data-dir data \
  --cache-dir cache/dinov3_vitb16_l896 \
  --output-dir runs/vitb16_bipartite_graph_v2 \
  --epochs 50 \
  --hidden-dim 384 --layers 2 --heads 8 \
  --dropout 0.15 --attention-dropout 0.10 \
  --geometry-bias-hidden 128 --geometry-bias-scale-init 0.1 \
  --lr 1e-4 --min-lr 1e-6 \
  --warmup-epochs 3 --cosine-epochs 40 \
  --early-stopping-patience 10 \
  --page-batch-size 4 --grad-accum 2 \
  --num-workers 8 --prefetch-factor 4 \
  --amp bf16
```

These are also the V2 defaults, so only the data, cache, and output paths are
strictly necessary. Do not resume a V1 checkpoint into V2; the training entry
point checks `model_version` and rejects incompatible checkpoints. Keep the V1
and V2 output directories separate for a controlled comparison.

V2 performs a true four-page GPU forward pass. Patch grids, dialogue counts,
and candidate counts are padded independently and protected by attention masks.
The bucketed batch sampler groups pages with similar aspect ratio and `D*C`
workload to reduce padding. `page_batch_size=4` with `grad_accum=2` preserves
the original effective batch of eight pages while exposing substantially more
parallel work to the GPU. The progress bar reports measured `pages_s`.

If 32 GB VRAM is still underused, benchmark `--page-batch-size 8 --grad-accum 1`.
If a rare dense page causes OOM, return to the default batch of four or set
`--eval-page-batch-size 2`. Test `--num-workers 4`, `8`, and `12`; more workers
can be slower when NPZ decompression or storage bandwidth is saturated.

Before uploading, the local CPU structural tests can be run with:

```powershell
python speaker_relation_transformer\tests\test_model_v2.py
```

## 7. Train Geometry + Graph + previous/current/next Text V3

V3 is a controlled text experiment and does not read DINO features. It combines
the standardized 45D D-C geometry edge with frozen Japanese-capable dialogue
embeddings, then uses both candidate-wise and dialogue-wise graph attention.
For dialogue row `i`, the ordered text slots are rows `[i-1, i, i+1]`; page
boundaries use an explicit mask. The order is the existing Manga109Dialog
`speaker_to_text` relation order stored in each page index.

Install the updated requirements and cache frozen text embeddings once:

```bash
cd /home/USER/speaker_relation_transformer
uv pip install --python .venv/bin/python -r requirements.txt
CUDA_VISIBLE_DEVICES=0 .venv/bin/python cache_dialogue_text.py \
  --data-dir data \
  --output-dir cache/text_multilingual_e5_base \
  --model-name intfloat/multilingual-e5-base \
  --batch-size 128 --max-length 128
```

The cache uses attention-mask mean pooling, L2 normalization, and float16
storage. Valid files are skipped on rerun. Its `config.json` records the encoder
and ordering semantics; V3 rejects a mismatched text cache when resuming.

Run a two-page server smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_v3.py \
  --data-dir data \
  --text-cache-dir cache/text_multilingual_e5_base \
  --output-dir runs/geometry_text_graph_v3_smoke \
  --epochs 1 --max-pages 2 --num-workers 0 \
  --hidden-dim 128 --layers 1 --heads 4 --amp bf16
```

Then run the full controlled experiment:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_v3.py \
  --data-dir data \
  --text-cache-dir cache/text_multilingual_e5_base \
  --output-dir runs/geometry_text_graph_v3_prev_current_next \
  --epochs 50 \
  --hidden-dim 384 --layers 2 --heads 8 \
  --dropout 0.15 --attention-dropout 0.10 \
  --geometry-bias-hidden 128 --geometry-bias-scale-init 0.1 \
  --lr 1e-4 --min-lr 1e-6 \
  --warmup-epochs 3 --cosine-epochs 40 \
  --early-stopping-patience 10 \
  --page-batch-size 8 --grad-accum 1 \
  --num-workers 8 --prefetch-factor 4 \
  --amp bf16
```

Local CPU structural tests do not require downloading the text model:

```powershell
python speaker_relation_transformer\tests\test_model_v3.py
```
