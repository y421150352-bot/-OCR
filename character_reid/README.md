# Manga109 supervised character ReID

This project trains a 512-dimensional character embedding from paired Manga109
`face` and `body` boxes. Identity labels are always `book::character_id`.
Training uses ground-truth boxes; deployment consumes the existing
RT-DETRv4-X Manga109-s v2 `detections.json` format.

## 1. Build manifests

Run from this directory on the training server:

```powershell
python build_reid_dataset.py `
  --dataset-root ..\dataset\Manga109s_released_2023_12_07 `
  --output-dir data
```

Linux uses the same arguments, for example:

```bash
python build_reid_dataset.py \
  --dataset-root /data/Manga109s_released_2023_12_07 \
  --output-dir data
```

The deterministic book-disjoint split is 70 train / 8 validation / 9 test.
Each manifest record is one `face+body`, `face-only`, or `body-only` instance.

## 2. Environment and training

The current local DINOv3 checkpoint is reused without downloading:

```powershell
pip install -r requirements.txt
python train.py `
  --backbone ..\speaker_relation_transformer\pretrained\dinov3-vitb16 `
  --p 16 --k 4 --epochs 30
```

On Linux, replace backslashes with slashes. The checkpoint path saved into each
run is resolved to an absolute server path so evaluation and inference do not
depend on the working directory.

The baseline freezes DINOv3 and optimizes Supervised Contrastive Loss. After the
baseline is established, enable the proposed auxiliary losses with
`--arcface-weight 0.5 --triplet-weight 0.2`. Use `--unfreeze-backbone` only with
enough GPU memory and a smaller learning rate.

## 3. Unseen-book evaluation

```powershell
python evaluate.py --checkpoint runs\baseline_supcon\last.pt --gallery-per-id 3
```

This creates gallery/query partitions inside the test books and reports Rank-1,
Rank-5, and mAP. Identities with fewer than two instances are excluded.

## 4. Named character bank and RT-DETR inference

Create an examples JSONL file. Coordinates are original-image `xyxy`; either box
may be omitted:

```json
{"name":"小明","image":"E:/comic/001.jpg","face_box":[10,20,60,80],"body_box":[0,15,110,220]}
```

```powershell
python build_character_bank.py --examples examples.jsonl `
  --checkpoint runs\baseline_supcon\last.pt --output bank.npz

python infer.py `
  --detections ..\rtdetr_manga_test\output\detections.json `
  --image-dir ..\test `
  --checkpoint runs\baseline_supcon\last.pt `
  --bank bank.npz --threshold 0.65
```

Tune the unknown threshold on validation books; `0.65` is only a starting point.
RT-DETR detections do not contain character IDs, so deployment pairing uses
same-panel, face-center containment, and normalized center distance.
