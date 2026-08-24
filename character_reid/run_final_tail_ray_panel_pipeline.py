#!/usr/bin/env python3
"""End-to-end inference for an unlabeled manga.

RT-DETR detections -> face/body instances -> ReID embeddings -> book-level
PCA-HDBSCAN character clusters -> one top-1 speaker instance per text box.

The default speaker ranker is the already trained 45D LightGBM baseline.  A
trained Geometry+Text Graph Transformer V3 checkpoint can be selected with
``--speaker-model v3``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for module_dir in (
    ROOT / "speaker_geometry_baseline",
    ROOT / "speaker_relation_transformer",
):
    if str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))

from build_reid_dataset import Box as ReIDBox, pair_boxes
from evaluate_clustering import (
    merge_seed_clusters,
    normalize,
    pca_features,
    raw_hdbscan_labels,
)
from inference_utils import embed_instance, load_model


@dataclass(frozen=True)
class Box:
    node_id: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    # 计算矩形框中心点x坐标
    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) * 0.5

    # 计算矩形框中心点y坐标,cxcy就是中心点坐标
    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) * 0.5

    # 判断点有没有落在框里
    def contains_point(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


# intersection(a, b) 则是在计算两个 Box 的交集宽度、高度和面积。
def intersection(a: Box, b: Box) -> tuple[float, float, float]:
    width = max(0.0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
    height = max(0.0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
    return width, height, width * height


# 给一个box（比如人物框、对白框），在多个分镜frame里判断它最属于哪一个分镜，并返回这个分镜的下标。
def choose_panel(box: Box, frames: Sequence[Box]) -> int | None:
    containing = [
        i for i, frame in enumerate(frames) if frame.contains_point(box.cx, box.cy)
    ]
    if containing:
        return min(containing, key=lambda i: frames[i].area)
    best_index, best_ratio = None, 0.0
    for index, frame in enumerate(frames):
        _, _, inter = intersection(box, frame)
        ratio = inter / max(box.area, 1.0)
        if ratio > best_ratio:
            best_index, best_ratio = index, ratio
    return best_index if best_ratio > 0 else None


# 把一组数值按照从小到大的排名，转换成 0～1 之间的“归一化排名分数”
def rank_normalized(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    for rank, index in enumerate(order):
        result[index] = rank / denominator
    return result


# 把一个对白框 text 和一个人物框 body 之间的空间关系，转换成固定长度的 45 维数值特征。
def pair_features(
    text: Box,
    body: Box,
    page_width: float,
    page_height: float,
    text_panel_index: int | None,
    body_panel_index: int | None,
    frames: Sequence[Box],
    candidate_count: int,  # 候选人物的数量
    same_panel_count: int,  # 对白和几个人物在一个分镜
    distance_rank: float,  # 距离
    same_panel_rank: float,
    is_nearest: bool,
    is_nearest_same_panel: bool,
) -> list[float]:
    width, height = max(page_width, 1.0), max(page_height, 1.0)
    page_area, page_diagonal = width * height, math.hypot(
        width, height
    )  # 页面对角线长度
    dx, dy = (body.cx - text.cx) / width, (
        body.cy - text.cy
    ) / height  # dx > 0 → 人物在对白右边
    center_distance = (
        math.hypot(body.cx - text.cx, body.cy - text.cy) / page_diagonal
    )  # 两个框中心点的欧氏距离
    gap_x_raw = max(
        0.0, max(text.xmin, body.xmin) - min(text.xmax, body.xmax)
    )  # 两个框在水平方向的边缘间隔
    gap_y_raw = max(
        0.0, max(text.ymin, body.ymin) - min(text.ymax, body.ymax)
    )  # 两个框在竖直方向的边缘间隔
    gap_x, gap_y = gap_x_raw / width, gap_y_raw / height
    edge_distance = (
        math.hypot(gap_x_raw, gap_y_raw) / page_diagonal
    )  ##两个框边缘的欧氏距离
    inter_w, inter_h, inter_area = intersection(text, body)
    union = text.area + body.area - inter_area  # 两个框并集的面积
    same_panel = text_panel_index is not None and text_panel_index == body_panel_index
    panel_dx = panel_dy = panel_distance = panel_area = 0.0
    if same_panel:
        panel = frames[text_panel_index]
        panel_width, panel_height = max(panel.width, 1.0), max(panel.height, 1.0)
        panel_dx = (body.cx - text.cx) / panel_width
        panel_dy = (body.cy - text.cy) / panel_height
        panel_distance = math.hypot(body.cx - text.cx, body.cy - text.cy) / math.hypot(
            panel_width, panel_height
        )
        panel_area = panel.area / page_area
    elif text_panel_index is not None:
        panel_area = frames[text_panel_index].area / page_area
    return [
        text.cx / width,
        text.cy / height,
        text.width / width,
        text.height / height,
        text.area / page_area,
        body.cx / width,
        body.cy / height,
        body.width / width,
        body.height / height,
        body.area / page_area,
        dx,
        dy,
        abs(dx),
        abs(dy),
        center_distance,
        gap_x,
        gap_y,
        edge_distance,
        inter_area / max(union, 1.0),  # IoU
        inter_area / max(text.area, 1.0),  # 对白框有多少比例被 body 覆盖
        inter_area / max(body.area, 1.0),
        # 横向和纵向到底重叠了多少。
        inter_w / max(text.width, 1.0),
        inter_w / max(body.width, 1.0),
        inter_h / max(text.height, 1.0),
        inter_h / max(body.height, 1.0),
        # 人物面积 / 对白面积的对数。
        math.log(max(body.area, 1.0) / max(text.area, 1.0)),
        float(same_panel),
        float(text_panel_index is not None),
        float(body_panel_index is not None),
        panel_dx,
        panel_dy,
        panel_distance,
        panel_area,
        float(candidate_count),  # 当前对白总共有多少个人物候选
        float(same_panel_count),  # 和当前对白处在同 panel 的人物有几个
        distance_rank,  # 当前人物在全部候选人物里的距离排名
        same_panel_rank,
        float(is_nearest),  # 当前人物是不是整页距离对白最近的人
        float(is_nearest_same_panel),
        float(body.cx < text.cx),  # 人物在对白左边
        float(body.cx > text.cx),
        float(body.cy < text.cy),
        float(body.cy > text.cy),
        float(text.contains_point(body.cx, body.cy)),
        float(body.contains_point(text.cx, text.cy)),
    ]


# 参数
DEFAULT_CLUSTER_PARAMETERS = {
    "pca_dimensions": 64,
    "min_cluster_size": 3,
    "min_samples": 2,
    "cluster_selection_method": "eom",
    "merge_threshold": 0.84,
    "assignment_threshold": 0.72,
    "assignment_margin": 0.02,
}
FINAL_SPEAKER_PROTOCOL = "v3_magiv3_single_tail_ray_strict_panel_v1"


# 读取你在命令行里输入的参数，把它们解析成 Python 变量，同时检查参数是否合法。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run character clustering and dialogue-speaker matching"
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--reid-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/new_manga_pipeline")
    )
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="Stop after ReID clustering; do not load a speaker model or analyze dialogues",
    )
    parser.add_argument(
        "--cluster-requires-face",
        action="store_true",
        help="For a cluster-only run, exclude body-only instances from ReID clustering",
    )
    parser.add_argument("--speaker-model", choices=("geometry", "v3"), default="v3")
    parser.add_argument(
        "--geometry-model",
        type=Path,
        default=ROOT / "speaker_geometry_baseline/artifacts/speaker_geometry_lgbm.txt",
    )
    parser.add_argument("--v3-checkpoint", type=Path)
    parser.add_argument(
        "--text-model", type=str, help="Local/Hugging Face path for the V3 text encoder"
    )
    parser.add_argument(
        "--ocr-bundles-dir",
        type=Path,
        help="Optional page_bundles directory containing OCR lines",
    )
    parser.add_argument(
        "--magi-dir",
        type=Path,
        help="Optional MAGI JSON directory providing tail boxes",
    )
    parser.add_argument("--tail-text-max-distance", type=float, default=0.12)
    parser.add_argument(
        "--tail-weight",
        type=float,
        default=6.0,
        help="Tail-ray alignment weight added to V3 logits",
    )
    parser.add_argument(
        "--tail-ray-width",
        type=float,
        default=0.035,
        help="Ray corridor width normalized by page diagonal",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--vlm-top5",
        action="store_true",
        help="Use Gemini to verify the speaker Top-5, then verify the selected speaker's ReID identity Top-5",
    )
    parser.add_argument(
        "--vlm-endpoint",
        default=os.environ.get("MANGA_VLM_ENDPOINT", ""),
        help="Optional Gemini generateContent endpoint override (defaults to Google AI API)",
    )
    parser.add_argument(
        "--vlm-model",
        default=os.environ.get("MANGA_VLM_MODEL", "gemini-3.1-pro-preview"),
        help="Gemini vision model name for --vlm-top5",
    )
    parser.add_argument(
        "--gemini-api-key-env",
        default="GEMINI_API_KEY",
        help="Environment-variable name holding the Gemini API key; the key is never stored in results",
    )
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument("--vlm-retries", type=int, default=3)
    parser.add_argument("--vlm-identity-batch-size", type=int, default=5)
    parser.add_argument(
        "--vlm-panel-batch-size",
        type=int,
        default=1,
        help="Number of target text regions analyzed per pass-1 panel request",
    )
    parser.add_argument("--vlm-confidence-threshold", type=float, default=0.70)
    parser.add_argument(
        "--vlm-first-pass-confidence-threshold",
        type=float,
        default=0.80,
        help=(
            "Minimum confidence for pass-1 text type, character-link, and speaker "
            "decisions"
        ),
    )
    parser.add_argument(
        "--vlm-speaker-top-k",
        type=int,
        default=5,
        help="Number of V3 speaker candidates shown to Gemini (maximum 5)",
    )
    parser.add_argument(
        "--vlm-save-boards",
        action="store_true",
        help="Save anonymous speaker and identity Top-5 boards under OUTPUT_DIR for audit",
    )
    parser.add_argument(
        "--vlm-max-dialogues",
        type=int,
        default=0,
        help="Maximum dialogues reviewed by VLM (0 means all; use a small value for a smoke test)",
    )
    parser.add_argument(
        "--vlm-max-pages",
        type=int,
        default=0,
        help="Maximum pages reviewed by the two-pass VLM pipeline (0 means all)",
    )
    parser.add_argument("--vlm-image-size", type=int, default=224)
    parser.add_argument("--largest-cluster-limit", type=float, default=0.55)
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.90,
        help="Minimum cosine similarity for merging mutual-nearest HDBSCAN seed prototypes",
    )
    parser.add_argument(
        "--assignment-threshold",
        type=float,
        default=0.82,
        help="Minimum cosine similarity for attaching an HDBSCAN noise instance to a prototype",
    )
    parser.add_argument(
        "--assignment-margin",
        type=float,
        default=0.05,
        help="Minimum top-1 versus top-2 similarity margin for attaching a noise instance",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        help="Optional reusable .npy cache; defaults to OUTPUT_DIR/reid_embeddings.npy",
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Ignore a matching ReID cache and extract all embeddings again",
    )
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    for name in ("merge_threshold", "assignment_threshold", "assignment_margin"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.cluster_requires_face and not args.cluster_only:
        parser.error("--cluster-requires-face requires --cluster-only")
    if args.vlm_top5 and args.cluster_only:
        parser.error("--vlm-top5 requires dialogue inference; remove --cluster-only")
    if not 0.0 <= args.vlm_confidence_threshold <= 1.0:
        parser.error("--vlm-confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.vlm_first_pass_confidence_threshold <= 1.0:
        parser.error("--vlm-first-pass-confidence-threshold must be between 0 and 1")
    if (
        args.vlm_timeout < 1
        or args.vlm_max_dialogues < 0
        or args.vlm_max_pages < 0
        or args.vlm_image_size < 64
    ):
        parser.error(
            "VLM timeout/image-size must be positive and VLM limits cannot be negative"
        )
    if not 1 <= args.vlm_speaker_top_k <= 5:
        parser.error("--vlm-speaker-top-k must be between 1 and 5")
    if (
        args.vlm_retries < 1
        or args.vlm_identity_batch_size < 1
        or args.vlm_panel_batch_size < 1
    ):
        parser.error("VLM retries and batch sizes must be positive")
    return args


# 把一个 node_id 和一组坐标 values，转换成一个 Box 对象。as_box("C1", [100, 200, 300, 500])=Box("C1", 100.0, 200.0, 300.0, 500.0)
def as_box(node_id: str, values: list[float]) -> Box:
    return Box(node_id, *(float(value) for value in values))


# 算重叠面积a、b 不是 Box 对象，而是用 list 表示的矩形框。
def intersect_area(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


# 根据当前图片名，去指定的 OCR bundle 目录里找到同名 JSON 文件，然后把里面的 ocr_lines 读出来。找不到目录、找不到文件、或者没有 ocr_lines 时，就返回空列表 []。
def load_ocr_lines(bundle_dir: Path | None, image_name: str) -> list[dict[str, Any]]:
    if bundle_dir is None:
        return []
    path = bundle_dir / f"{Path(image_name).stem}.json"
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8-sig")).get("ocr_lines", []))


# 给一个检测到的文字框 text_box，从 OCR 识别出来的多条文字 lines 里，text_box[xmin, ymin, xmax, ymax]
# 找出属于这个文字框的 OCR 行，再按照日漫常见的“从右到左、从上到下”顺序拼成一个字符串。
def text_for_detection(text_box: list[float], lines: list[dict[str, Any]]) -> str:
    selected: list[tuple[float, float, str]] = []
    for line in lines:
        box = [float(value) for value in line.get("box", [])]
        if len(box) != 4:
            continue
        overlap = intersect_area(text_box, box) / max(
            1.0, (box[2] - box[0]) * (box[3] - box[1])
        )
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = text_box[0] <= cx <= text_box[2] and text_box[1] <= cy <= text_box[3]
        if inside or overlap >= 0.35:
            selected.append((box[0], box[1], str(line.get("text_for_review_only", ""))))
    # Japanese manga is commonly vertical and right-to-left.
    selected.sort(key=lambda row: (-row[0], row[1]))
    return "".join(row[2] for row in selected)


# 根据当前漫画图片名，在 MAGI 结果目录里找到对应的 JSON 文件，然后把其中所有 tail（气泡尾巴）框提取出来，最终返回 [[xmin, ymin, xmax, ymax], ...]。
def load_magi_tails(magi_dir: Path | None, image_name: str) -> list[list[float]]:
    if magi_dir is None:
        return []
    stem = Path(image_name).stem
    candidates = (magi_dir / f"{stem}_magiv3_all.json", magi_dir / f"{stem}.json")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [
        [float(value) for value in row["box"]]
        for row in payload.get("tails", [])
        if isinstance(row.get("box"), list) and len(row["box"]) == 4
    ]


def inject_magi_tails(payload: dict[str, Any], magi_dir: Path | None) -> int:
    """把 MAGI 检测出来的所有气泡尾巴 tail，插入到原本的 RT-DETR 检测结果 payload 里面，而且只是在内存里改这个 Python 字典。"""
    inserted = 0
    for page in payload.get("images", []):
        detections = [
            row
            for row in page.get("detections", [])
            if not (row.get("class_name") == "tail" and row.get("source") == "magiv3")
        ]
        tails = load_magi_tails(magi_dir, str(page["image"]))
        for index, box in enumerate(tails):
            detections.append(
                {
                    "class_id": 4,
                    "class_name": "tail",
                    "score": 1.0,
                    "box": box,
                    "area": round(
                        max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]), 2
                    ),
                    "source": "magiv3",
                    "magi_tail_index": index,
                }
            )
            inserted += 1
        page["detections"] = detections
        page.setdefault("counts", {})["tail"] = len(tails)
    payload.setdefault("total_counts", {})["tail"] = inserted
    return inserted


# 算一个点到一个矩形框的最短距离
def point_to_box_distance(x: float, y: float, box: list[float]) -> float:
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


# 算两个矩形框之间的最短边缘距离
def box_gap_distance(left: list[float], right: list[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def estimated_tail_segment(
    tail_box: list[float], text_box: list[float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """对白框 text_box 和气泡尾巴框 tail_box 的相对位置，估算一条“气泡尾巴方向线”，并求出这条线穿过 tail bbox 时的两个端点"""
    text_cx = (text_box[0] + text_box[2]) * 0.5
    text_cy = (text_box[1] + text_box[3]) * 0.5
    tail_cx = (tail_box[0] + tail_box[2]) * 0.5
    tail_cy = (tail_box[1] + tail_box[3]) * 0.5
    outward_x, outward_y = tail_cx - text_cx, tail_cy - text_cy
    outward_norm = max(math.hypot(outward_x, outward_y), 1e-6)
    outward_x, outward_y = outward_x / outward_norm, outward_y / outward_norm
    half_w = max((tail_box[2] - tail_box[0]) * 0.5, 0.5)
    half_h = max((tail_box[3] - tail_box[1]) * 0.5, 0.5)
    # 从 tail 中心沿着这个方向最多走多远，才会第一次碰到 tail bbox 的边。
    extent = min(
        half_w / abs(outward_x) if abs(outward_x) > 1e-9 else float("inf"),
        half_h / abs(outward_y) if abs(outward_y) > 1e-9 else float("inf"),
    )
    root = (
        tail_cx - outward_x * extent,
        tail_cy - outward_y * extent,
    )  # tail 靠近气泡的一端。
    tip = (
        tail_cx + outward_x * extent,
        tail_cy + outward_y * extent,
    )  # 气泡尾巴指向人物的那一端
    return root, tip


def estimated_tail_tip(
    tail_box: list[float], text_box: list[float]
) -> tuple[float, float]:
    return estimated_tail_segment(tail_box, text_box)[1]


def assign_tails_to_texts(
    text_rows: list[dict[str, Any]],
    tail_rows: list[dict[str, Any]],
    frames: list[Box],
    page_width: float,
    page_height: float,
    text_max_distance: float,
) -> dict[int, int]:
    """把一页里的多个 text 框和多个 tail 框，一一匹配起来。要求它们必须在同一个 panel 里，
    而且距离不能太远；最后采用“距离最小优先”的贪心策略，保证一个 text 最多配一个 tail，一个 tail 也最多配一个 text。"""
    diagonal = max(math.hypot(page_width, page_height), 1.0)
    candidates: list[tuple[float, int, int]] = []
    for text_index, text_row in enumerate(text_rows):
        text_box = text_row["box"]
        text_panel = choose_panel(as_box(f"text_{text_index}", text_box), frames)
        if text_panel is None:
            continue
        for tail_index, tail_row in enumerate(tail_rows):
            tail_box = tail_row["box"]
            tail_panel = choose_panel(as_box(f"tail_{tail_index}", tail_box), frames)
            # Strict rule: both objects must resolve to the same smallest
            # detected panel. Unknown panel membership is not accepted.
            if tail_panel is None or text_panel != tail_panel:
                continue
            distance = box_gap_distance(text_box, tail_box) / diagonal
            if distance <= text_max_distance:
                candidates.append((distance, text_index, tail_index))
    assignments: dict[int, int] = {}
    used_tails: set[int] = set()
    for _, text_index, tail_index in sorted(candidates):
        if text_index not in assignments and tail_index not in used_tails:
            assignments[text_index] = tail_index
            used_tails.add(tail_index)
    return assignments


def ray_box_intersection(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    box: list[float],
) -> float | None:
    """气泡尾巴这条射线沿着 tip 指出去，会不会撞到某个人物框。"""
    t_min, t_max = 0.0, float("inf")
    for origin, direction, lower, upper in (
        (origin_x, direction_x, box[0], box[2]),
        (origin_y, direction_y, box[1], box[3]),
    ):
        if abs(direction) < 1e-9:
            if origin < lower or origin > upper:
                return None
            continue
        first, second = (lower - origin) / direction, (upper - origin) / direction
        if first > second:
            first, second = second, first
        t_min, t_max = max(t_min, first), min(t_max, second)
        if t_min > t_max:
            return None
    return t_min if t_max >= 0 else None


def ray_box_interval(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    box: list[float],
) -> tuple[float, float] | None:
    """“什么时候进入？什么时候出去？”"""
    t_enter, t_exit = 0.0, float("inf")
    for origin, direction, lower, upper in (
        (origin_x, direction_x, box[0], box[2]),
        (origin_y, direction_y, box[1], box[3]),
    ):
        if abs(direction) < 1e-9:
            if origin < lower or origin > upper:
                return None
            continue
        first, second = (lower - origin) / direction, (upper - origin) / direction
        if first > second:
            first, second = second, first
        t_enter, t_exit = max(t_enter, first), min(t_exit, second)
        if t_enter > t_exit:
            return None
    if t_exit < 0:
        return None
    return max(0.0, t_enter), t_exit


def ray_box_exit_parameter(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    box: list[float],
) -> float:
    """如果射线起点已经在 box 里面，那么沿着当前方向继续往前走，需要走多远才会第一次碰到这个矩形框的边界并出去。"""
    candidates: list[float] = []
    if direction_x > 1e-9:
        candidates.append((box[2] - origin_x) / direction_x)
    elif direction_x < -1e-9:
        candidates.append((box[0] - origin_x) / direction_x)
    if direction_y > 1e-9:
        candidates.append((box[3] - origin_y) / direction_y)
    elif direction_y < -1e-9:
        candidates.append((box[1] - origin_y) / direction_y)
    positive = [value for value in candidates if value >= 0]
    return min(positive) if positive else 0.0


def one_tail_ray_prior(
    text_box: list[float],
    tail_row: dict[str, Any],
    root: tuple[float, float],
    tip: tuple[float, float],
    diagonal_alignment: float,
    instances: list[dict[str, Any]],
    frames: list[Box],
    page_width: float,
    page_height: float,
    ray_width: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """给一个对白对应的 tail 射线，对这一页所有人物候选打一个 0～1 的 tail-ray prior 分数：
    射线真正打中人物框就给 1.0；没直接打中但离射线很近，就给一个随距离衰减的分数；不同 panel 的人物直接给 0。"""
    page_diagonal = max(math.hypot(page_width, page_height), 1.0)
    tail_box = tail_row["box"]
    (root_x, root_y), (tip_x, tip_y) = root, tip
    direction_x, direction_y = tip_x - root_x, tip_y - root_y
    direction_norm = max(math.hypot(direction_x, direction_y), 1e-6)
    direction_x, direction_y = (
        direction_x / direction_norm,
        direction_y / direction_norm,
    )
    text_panel = choose_panel(as_box("text", text_box), frames)
    tail_panel = choose_panel(as_box("tail", tail_box), frames)
    if text_panel is None or tail_panel is None or text_panel != tail_panel:
        raise ValueError(
            "Strict tail-ray fusion requires text and tail in the same detected panel"
        )
    panel = frames[text_panel]
    panel_box = [panel.xmin, panel.ymin, panel.xmax, panel.ymax]
    panel_exit = ray_box_exit_parameter(
        tip_x, tip_y, direction_x, direction_y, panel_box
    )
    priors = np.zeros(len(instances), dtype=np.float32)
    perpendicular_distances = np.full(len(instances), 1.0, dtype=np.float32)
    projections = np.full(len(instances), -1.0, dtype=np.float32)
    ray_hits = [False] * len(instances)
    face_hits = [False] * len(instances)
    body_hits = [False] * len(instances)
    hit_parameters: list[float | None] = [None] * len(instances)
    center_scores = np.zeros(len(instances), dtype=np.float32)
    crossing_scores = np.zeros(len(instances), dtype=np.float32)
    hit_qualities = np.zeros(len(instances), dtype=np.float32)
    corridor = max(float(ray_width) * page_diagonal, 1.0)
    for body_index, instance in enumerate(instances):
        body_panel = choose_panel(
            as_box(instance["instance_id"], instance["body_box"]), frames
        )
        # Strict panel boundary: unknown-panel candidates and candidates in a
        # neighboring panel receive no tail-ray score.
        if body_panel != text_panel:
            continue
        target = instance.get("face_box") or instance["body_box"]
        hit_t = ray_box_intersection(tip_x, tip_y, direction_x, direction_y, target)
        center_x = (target[0] + target[2]) * 0.5
        center_y = (target[1] + target[3]) * 0.5
        vector_x, vector_y = center_x - tip_x, center_y - tip_y
        projection = vector_x * direction_x + vector_y * direction_y
        perpendicular = abs(vector_x * direction_y - vector_y * direction_x)
        half_diagonal = 0.5 * math.hypot(target[2] - target[0], target[3] - target[1])
        edge_distance = max(0.0, perpendicular - half_diagonal)
        projections[body_index] = projection / page_diagonal
        perpendicular_distances[body_index] = edge_distance / page_diagonal
        if hit_t is not None and hit_t <= panel_exit:
            priors[body_index] = 1.0
            ray_hits[body_index] = True
            hit_parameters[body_index] = round(float(hit_t / page_diagonal), 6)
        elif 0 < projection <= panel_exit:
            # If the finite detection box narrowly misses the ray, use only
            # perpendicular distance to the ray—not distance from the tip.
            priors[body_index] = float(np.exp(-edge_distance / corridor))
    order = np.argsort(-priors, kind="stable")
    return priors, {
        "tail_index": int(tail_row.get("magi_tail_index", -1)),
        "tail_box": tail_box,
        "estimated_tail_root": [round(root_x, 2), round(root_y, 2)],
        "estimated_tail_tip": [round(tip_x, 2), round(tip_y, 2)],
        "ray_origin": [round(tip_x, 2), round(tip_y, 2)],
        "ray_direction": [round(direction_x, 6), round(direction_y, 6)],
        "panel_index": int(text_panel),
        "panel_box": [round(float(value), 2) for value in panel_box],
        "ray_panel_exit": round(float(panel_exit), 6),
        "best_candidate_index": int(order[0]),
        "ray_hits": ray_hits,
        "ray_hit_parameters": hit_parameters,
        "candidate_ray_projections": [round(float(value), 6) for value in projections],
        "candidate_perpendicular_distances": [
            round(float(value), 6) for value in perpendicular_distances
        ],
        "tail_ray_prior": [round(float(value), 6) for value in priors],
    }


def tail_ray_prior(
    text_box: list[float],
    tail_row: dict[str, Any],
    instances: list[dict[str, Any]],
    frames: list[Box],
    page_width: float,
    page_height: float,
    ray_width: float,
    v3_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the single text-center-through-tail-box ray from the final protocol."""
    root, tip = estimated_tail_segment(tail_row["box"], text_box)
    return one_tail_ray_prior(
        text_box,
        tail_row,
        root,
        tip,
        1.0,
        instances,
        frames,
        page_width,
        page_height,
        ray_width,
    )


def build_instances(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    all_instances: list[dict[str, Any]] = []
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page_index, page in enumerate(payload["images"]):
        detections = page["detections"]
        faces = [
            ReIDBox(f"F{index + 1}", "face", "predicted", tuple(item["box"]))
            for index, item in enumerate(
                d for d in detections if d["class_name"] == "face"
            )
        ]
        bodies = [
            ReIDBox(f"B{index + 1}", "body", "predicted", tuple(item["box"]))
            for index, item in enumerate(
                d for d in detections if d["class_name"] == "body"
            )
        ]
        frames = [
            tuple(item["box"]) for item in detections if item["class_name"] == "frame"
        ]
        pairs, _, body_only = pair_boxes(faces, bodies, frames)
        rows = [(face, body) for face, body in pairs] + [
            (None, body) for body in body_only
        ]
        rows.sort(key=lambda row: (row[1].xyxy[1], row[1].xyxy[0]))
        for local_index, (face, body) in enumerate(rows, 1):
            record = {
                "instance_id": f"P{page_index + 1:03d}_C{local_index:03d}",
                "image": page["image"],
                "page_index": page_index,
                "face_box": list(face.xyxy) if face else None,
                "body_box": list(body.xyxy),
                "input_type": "face+body" if face else "body-only",
            }
            all_instances.append(record)
            by_image[page["image"]].append(record)
    return all_instances, by_image


def extract_embeddings(
    instances: list[dict[str, Any]],
    image_dir: Path,
    checkpoint: Path,
    device: torch.device,
) -> np.ndarray:
    model = load_model(checkpoint, device)
    vectors: list[np.ndarray] = []
    current_name = None
    image = None
    for index, record in enumerate(instances, 1):
        if record["image"] != current_name:
            if image is not None:
                image.close()
            image = ImageOps.exif_transpose(
                Image.open(image_dir / record["image"])
            ).convert("RGB")
            current_name = record["image"]
        vectors.append(
            embed_instance(model, image, record["face_box"], record["body_box"], device)
        )
        if index % 25 == 0 or index == len(instances):
            print(f"ReID embeddings: {index}/{len(instances)}", flush=True)
    if image is not None:
        image.close()
    if not vectors:
        raise SystemExit("RT-DETR did not produce any body instances")
    return normalize(np.asarray(vectors, dtype=np.float32))


def embedding_cache_signature(
    instances: list[dict[str, Any]],
    image_dir: Path,
    checkpoint: Path,
) -> str:
    """Fingerprint every input that can change the cached ReID embeddings."""
    image_stats = []
    for image_name in sorted({str(row["image"]) for row in instances}):
        path = image_dir / image_name
        stat = path.stat()
        image_stats.append([image_name, int(stat.st_size), int(stat.st_mtime_ns)])
    checkpoint_stat = checkpoint.stat()
    payload = {
        "version": 1,
        "image_dir": str(image_dir.resolve()),
        "checkpoint": [
            str(checkpoint.resolve()),
            int(checkpoint_stat.st_size),
            int(checkpoint_stat.st_mtime_ns),
        ],
        "images": image_stats,
        "instances": [
            [row["image"], row.get("face_box"), row.get("body_box")]
            for row in instances
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_or_extract_embeddings(
    instances: list[dict[str, Any]],
    image_dir: Path,
    checkpoint: Path,
    device: torch.device,
    cache_path: Path,
    recompute: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse a validated cache or extract and persist normalized embeddings."""
    cache_path = cache_path.resolve()
    metadata_path = cache_path.with_name(f"{cache_path.name}.meta.json")
    signature = embedding_cache_signature(instances, image_dir, checkpoint)

    if not recompute and cache_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.asarray(
                np.load(cache_path, allow_pickle=False), dtype=np.float32
            )
            expected_shape = metadata.get("shape")
            valid_shape = (
                cached.ndim == 2
                and cached.shape[0] == len(instances)
                and expected_shape == list(cached.shape)
            )
            if metadata.get("signature") == signature and valid_shape:
                print(
                    f"ReID embeddings cache hit: {cache_path} ({cached.shape[0]} instances)",
                    flush=True,
                )
                return normalize(cached), {
                    "path": str(cache_path),
                    "signature": signature,
                    "reused": True,
                    "shape": list(cached.shape),
                }
            print(
                f"ReID embeddings cache mismatch; recomputing: {cache_path}", flush=True
            )
        except Exception as error:
            print(f"ReID embeddings cache unreadable; recomputing: {error}", flush=True)

    features = extract_embeddings(instances, image_dir, checkpoint, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    metadata = {
        "version": 1,
        "path": str(cache_path),
        "signature": signature,
        "shape": list(features.shape),
        "instance_count": len(instances),
        "reid_checkpoint": str(checkpoint.resolve()),
        "image_dir": str(image_dir.resolve()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved ReID embeddings cache: {cache_path}", flush=True)
    return features, {**metadata, "reused": False}


def cluster_once(
    features: np.ndarray, parameters: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    reduced = pca_features(features, int(parameters["pca_dimensions"]))
    raw = raw_hdbscan_labels(
        reduced,
        int(parameters["min_cluster_size"]),
        int(parameters["min_samples"]),
        str(parameters["cluster_selection_method"]),
    )
    labels, diagnostics = merge_seed_clusters(
        features,
        raw,
        float(parameters["merge_threshold"]),
        float(parameters["assignment_threshold"]),
        float(parameters["assignment_margin"]),
    )
    largest = max(np.bincount(labels)) / len(labels)
    return labels, {**diagnostics, "largest_cluster_ratio": float(largest)}


def cluster_characters(
    features: np.ndarray,
    largest_limit: float,
    parameter_overrides: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    default_parameters = {**DEFAULT_CLUSTER_PARAMETERS, **(parameter_overrides or {})}
    fallback_parameters = {
        **default_parameters,
        "min_samples": 3,
        "cluster_selection_method": "leaf",
    }
    attempts = []
    selected = None
    for rank, parameters in enumerate((default_parameters, fallback_parameters), 1):
        labels, diagnostics = cluster_once(features, parameters)
        attempts.append(
            {"rank": rank, "parameters": parameters, "diagnostics": diagnostics}
        )
        if (
            selected is None
            or diagnostics["largest_cluster_ratio"]
            < selected[2]["largest_cluster_ratio"]
        ):
            selected = (labels, parameters, diagnostics, rank)
        if diagnostics["largest_cluster_ratio"] <= largest_limit:
            selected = (labels, parameters, diagnostics, rank)
            break
    assert selected is not None
    raw_labels, parameters, diagnostics, rank = selected
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(raw_labels.tolist()):
        groups[int(label)].append(index)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    stable = np.empty(len(raw_labels), dtype=np.int64)
    for cluster_index, members in enumerate(ordered, 1):
        stable[np.asarray(members)] = cluster_index
    return stable, {
        "selected_rank": rank,
        "selected_parameters": parameters,
        "diagnostics": diagnostics,
        "attempts": attempts,
    }


def make_geometry(
    text: Box, bodies: list[Box], frames: list[Box], width: float, height: float
) -> np.ndarray:
    text_panel = choose_panel(text, frames)
    body_panels = [choose_panel(body, frames) for body in bodies]
    distances = [math.hypot(body.cx - text.cx, body.cy - text.cy) for body in bodies]
    distance_ranks = rank_normalized(distances)
    same_indices = [
        i
        for i, panel in enumerate(body_panels)
        if text_panel is not None and panel == text_panel
    ]
    same_values = (
        rank_normalized([distances[i] for i in same_indices]) if same_indices else []
    )
    same_ranks = {
        index: same_values[position] for position, index in enumerate(same_indices)
    }
    nearest = min(range(len(bodies)), key=lambda i: distances[i])
    nearest_same = (
        min(same_indices, key=lambda i: distances[i]) if same_indices else None
    )
    return np.asarray(
        [
            pair_features(
                text,
                body,
                width,
                height,
                text_panel,
                body_panels[index],
                frames,
                len(bodies),
                len(same_indices),
                distance_ranks[index],
                same_ranks.get(index, 1.0),
                index == nearest,
                index == nearest_same,
            )
            for index, body in enumerate(bodies)
        ],
        dtype=np.float32,
    )


def softmax(values: np.ndarray) -> np.ndarray:
    weights = np.exp(values - float(values.max()))
    return weights / max(float(weights.sum()), 1e-12)


class GeometrySpeakerRanker:
    def __init__(self, path: Path):
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=str(path))
        if self.model.num_feature() != 45:
            raise ValueError(
                f"Expected a 45D geometry model, got {self.model.num_feature()} features"
            )

    def score_page(self, geometry: np.ndarray, texts: list[str]) -> np.ndarray:
        dialogues, candidates = geometry.shape[:2]
        return np.asarray(
            self.model.predict(geometry.reshape(-1, 45)), dtype=np.float32
        ).reshape(dialogues, candidates)


class V3SpeakerRanker:
    def __init__(
        self, checkpoint_path: Path, text_model_path: str | None, device: torch.device
    ):
        from model_v3 import SpeakerGeometryTextGraphTransformer
        from transformers import AutoModel, AutoTokenizer

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = dict(checkpoint.get("config", {}))
        state = checkpoint["model"]
        text_dim = int(state["text_input_norm.weight"].numel())
        self.model = SpeakerGeometryTextGraphTransformer(
            text_dim=text_dim,
            hidden_dim=int(config.get("hidden_dim", 384)),
            layers=int(config.get("layers", 2)),
            heads=int(config.get("heads", 8)),
            dropout=float(config.get("dropout", 0.15)),
            attention_dropout=float(config.get("attention_dropout", 0.1)),
            geometry_bias_hidden=int(config.get("geometry_bias_hidden", 128)),
            geometry_bias_scale_init=float(config.get("geometry_bias_scale_init", 0.1)),
            use_text=str(config.get("v3_text_ablation", "full")) == "full",
            use_dialogue_graph=str(config.get("v3_graph_mode", "two_axis"))
            == "two_axis",
        ).to(device)
        self.model.load_state_dict(state)
        self.model.eval()
        metadata = config.get("text_cache") or {}
        model_name_raw = text_model_path or metadata.get("model_name")
        if not model_name_raw:
            raise ValueError(
                "V3 needs --text-model because its checkpoint does not record text_cache.model_name"
            )
        model_name = self.resolve_local_text_model(str(model_name_raw), checkpoint_path)
        self.prefix = str(metadata.get("prefix") or "passage: ")
        self.max_length = int(metadata.get("max_length") or 128)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_name), local_files_only=True
        )
        self.encoder = (
            AutoModel.from_pretrained(str(model_name), local_files_only=True)
            .to(device)
            .eval()
        )
        self.device = device

    @staticmethod
    def resolve_local_text_model(value: str, checkpoint_path: Path) -> Path:
        requested = Path(value).expanduser()
        candidates = [requested]
        if not requested.is_absolute():
            candidates.extend(
                [
                    Path.cwd() / requested,
                    checkpoint_path.resolve().parents[2] / requested,
                    checkpoint_path.resolve().parents[2]
                    / "pretrained"
                    / requested.name,
                    Path.cwd().parent / "speaker_relation_transformer" / requested,
                    Path.cwd().parent / "speaker_relation_transformer" / "pretrained"
                    / requested.name,
                ]
            )
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "config.json").is_file():
                resolved = candidate.resolve()
                print(f"V3 local text model: {resolved}", flush=True)
                return resolved
        checked = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "V3 text model was not found locally. Network downloads are disabled. "
            f"Checked:\n  - {checked}\nPass its absolute directory with --text-model."
        )

    def encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            [self.prefix + text for text in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = self.encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            return torch.nn.functional.normalize(pooled.float(), dim=1)

    def score_page(self, geometry: np.ndarray, texts: list[str]) -> np.ndarray:
        embeddings = self.encode(texts)
        count = len(texts)
        context = torch.zeros(count, 3, embeddings.shape[1], device=self.device)
        mask = torch.zeros(count, 3, dtype=torch.bool, device=self.device)
        context[:, 1], mask[:, 1] = embeddings, True
        if count > 1:
            context[1:, 0], context[:-1, 2] = embeddings[:-1], embeddings[1:]
            mask[1:, 0], mask[:-1, 2] = True, True
        with torch.inference_mode():
            rows = self.model.forward_page(
                torch.from_numpy(geometry).to(self.device), context, mask
            )
        return torch.stack(rows).float().cpu().numpy()


def crop_instance_for_vlm(
    image_dir: Path, row: dict[str, Any], size: int
) -> Image.Image:
    """Return one letterboxed body crop; never send a whole character name to Gemini."""
    with Image.open(image_dir / str(row["image"])) as raw:
        crop = ImageOps.exif_transpose(raw).convert("RGB").crop(tuple(row["body_box"]))
    crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas


def build_cluster_identity_bank(
    instances: list[dict[str, Any]],
    features: np.ndarray,
) -> tuple[list[str], np.ndarray, dict[str, dict[str, Any]]]:
    """Build normalized cluster prototypes and one medoid reference per cluster."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(instances):
        groups[str(row["character_cluster_id"])].append(index)
    cluster_ids = sorted(groups)
    prototypes: list[np.ndarray] = []
    references: dict[str, dict[str, Any]] = {}
    for cluster_id in cluster_ids:
        member_indexes = np.asarray(groups[cluster_id], dtype=np.int64)
        prototype = np.asarray(features[member_indexes].mean(axis=0), dtype=np.float32)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        prototypes.append(prototype)
        medoid_offset = int(np.argmax(features[member_indexes] @ prototype))
        references[cluster_id] = instances[int(member_indexes[medoid_offset])]
    return cluster_ids, np.asarray(prototypes, dtype=np.float32), references


def make_vlm_identity_top5_board(
    image_dir: Path,
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    size: int,
) -> Image.Image:
    """Create an anonymous query-versus-candidates board for visual identity review."""
    labels = ["A", "B", "C", "D", "E"]
    cell_width, label_height = size + 16, 44
    canvas = Image.new("RGB", (cell_width * 6, size + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(14, size // 14))
    query_crop = crop_instance_for_vlm(image_dir, query, size)
    canvas.paste(query_crop, ((cell_width - size) // 2, 0))
    draw.text((8, size + 5), "QUERY", fill="black", font=font)
    for index, candidate in enumerate(candidates):
        crop = crop_instance_for_vlm(image_dir, candidate["reference"], size)
        x = (index + 1) * cell_width + (cell_width - size) // 2
        canvas.paste(crop, (x, 0))
        draw.text(
            ((index + 1) * cell_width + 8, size + 5),
            f"{labels[index]}  ref",
            fill="black",
            font=font,
        )
    return canvas


def make_vlm_speaker_top5_board(
    image_dir: Path,
    page: dict[str, Any],
    dialogue: dict[str, Any],
    candidates: list[dict[str, Any]],
    size: int,
) -> Image.Image:
    """Create a panel board with its text box and anonymous speaker candidates."""
    labels = "ABCDE"
    colors = [
        (35, 95, 220),
        (220, 75, 45),
        (25, 150, 90),
        (180, 55, 175),
        (215, 145, 20),
    ]
    text_box = as_box("text", dialogue["text_box"])
    frames = [
        as_box(f"frame_{index}", box)
        for index, box in enumerate(page.get("frame_boxes", []))
    ]
    frame_index = choose_panel(text_box, frames)
    with Image.open(image_dir / str(page["image"])) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    if frame_index is None:
        boxes = [text_box] + [as_box("body", row["body_box"]) for row in candidates]
        xmin = max(0, int(min(box.xmin for box in boxes) - 20))
        ymin = max(0, int(min(box.ymin for box in boxes) - 20))
        xmax = min(image.width, int(max(box.xmax for box in boxes) + 20))
        ymax = min(image.height, int(max(box.ymax for box in boxes) + 20))
    else:
        frame = frames[frame_index]
        xmin, ymin = max(0, int(frame.xmin)), max(0, int(frame.ymin))
        xmax, ymax = min(image.width, int(frame.xmax)), min(
            image.height, int(frame.ymax)
        )
    cell_width, label_height = size + 16, 38
    panel_height = size * 2
    panel = image.crop((xmin, ymin, max(xmin + 1, xmax), max(ymin + 1, ymax)))
    font = ImageFont.load_default(size=max(14, size // 14))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rectangle(
        (
            text_box.xmin - xmin,
            text_box.ymin - ymin,
            text_box.xmax - xmin,
            text_box.ymax - ymin,
        ),
        outline=(230, 30, 30),
        width=3,
    )
    panel_draw.text((8, 6), "TEXT", fill=(230, 30, 30), font=font)
    for index, candidate in enumerate(candidates):
        box = as_box("body", candidate["body_box"])
        color = colors[index]
        panel_draw.rectangle(
            (box.xmin - xmin, box.ymin - ymin, box.xmax - xmin, box.ymax - ymin),
            outline=color,
            width=4,
        )
        panel_draw.text(
            (box.xmin - xmin + 3, box.ymin - ymin + 3),
            labels[index],
            fill=color,
            font=font,
        )
    panel.thumbnail(
        (cell_width * len(candidates), panel_height), Image.Resampling.LANCZOS
    )
    canvas = Image.new(
        "RGB",
        (cell_width * len(candidates), panel_height + size + label_height),
        "white",
    )
    panel_x = (canvas.width - panel.width) // 2
    canvas.paste(panel, (panel_x, 0))
    draw = ImageDraw.Draw(canvas)
    for index, candidate in enumerate(candidates):
        crop = crop_instance_for_vlm(image_dir, candidate["instance"], size)
        cell_x = index * cell_width + (cell_width - size) // 2
        canvas.paste(crop, (cell_x, panel_height))
        draw.text(
            (index * cell_width + 8, panel_height + size + 4),
            labels[index],
            fill="black",
            font=font,
        )
    return canvas


def parse_vlm_top5_choice(
    text: str, option_count: int
) -> tuple[int | None, float, str | None]:
    """Accept only an A-E option plus a normalized confidence from Gemini JSON."""
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, 0.0, "Gemini did not return valid JSON"
    raw_choice = str(value.get("choice", "unknown")).strip().upper()
    if raw_choice in {"UNKNOWN", "NONE", "NONE_OF_ABOVE", "UNCERTAIN"}:
        return None, 0.0, None
    labels = "ABCDE"[:option_count]
    if raw_choice not in labels:
        return None, 0.0, f"Gemini returned unsupported choice: {raw_choice!r}"
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    confidence = max(0.0, min(1.0, confidence))
    return labels.index(raw_choice), confidence, None


def call_gemini_top5(
    board: Image.Image,
    args: argparse.Namespace,
    option_count: int,
    prompt: str,
) -> tuple[int | None, float, str | None]:
    """Ask Gemini to choose among anonymous visual candidates."""
    try:
        import requests
    except ImportError:
        return None, 0.0, "requests is required for --vlm-top5"
    api_key = os.environ.get(str(args.gemini_api_key_env), "").strip()
    if not api_key or api_key.startswith("YOUR_"):
        return (
            None,
            0.0,
            f"Set {args.gemini_api_key_env}=YOUR_GEMINI_API_KEY before enabling --vlm-top5",
        )
    buffer = io.BytesIO()
    board.save(buffer, format="JPEG", quality=90)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    endpoint = str(args.vlm_endpoint).strip()
    if not endpoint:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{args.vlm_model}:generateContent"
        )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": encoded_image}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    try:
        response = requests.post(
            endpoint,
            headers={"x-goog-api-key": api_key},
            json=payload,
            timeout=max(1, int(args.vlm_timeout)),
        )
        response.raise_for_status()
        body = response.json()
        parts = body["candidates"][0]["content"]["parts"]
        text = "".join(str(part.get("text", "")) for part in parts)
    except Exception as error:
        return None, 0.0, f"Gemini request failed: {error}"
    return parse_vlm_top5_choice(text, option_count)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from a Gemini response."""
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Gemini response must be a JSON object")
    return value


def parse_vlm_confidence(
    value: Any,
    field_name: str,
) -> tuple[float | None, str | None]:
    """Parse a numeric VLM confidence without silently replacing invalid data."""
    if value is None:
        return None, f"{field_name} is missing"
    if isinstance(value, bool):
        return None, f"{field_name} must be numeric, not boolean"
    try:
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None, f"{field_name} is empty"
            if cleaned.endswith("%"):
                confidence = float(cleaned[:-1].strip()) / 100.0
            else:
                confidence = float(cleaned)
        else:
            confidence = float(value)
    except (TypeError, ValueError):
        return None, f"{field_name} is not numeric: {value!r}"
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None, f"{field_name} is outside [0, 1]: {value!r}"
    return confidence, None


def append_vlm_audit_record(path: Path, record: dict[str, Any]) -> None:
    """Append one prompt/response audit record without storing credentials or images."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_gemini_json(
    images: list[Image.Image],
    prompt: str,
    args: argparse.Namespace,
    *,
    audit_path: Path | None = None,
    audit_context: dict[str, Any] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Send multiple page-level images to Gemini and return one JSON object."""
    try:
        import requests
    except ImportError:
        return None, "requests is required for --vlm-top5"
    api_key = os.environ.get(str(args.gemini_api_key_env), "").strip()
    if not api_key or api_key.startswith("YOUR_"):
        return (
            None,
            f"Set {args.gemini_api_key_env}=YOUR_GEMINI_API_KEY before enabling --vlm-top5",
        )
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for image in images:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                }
            }
        )
    endpoint = str(args.vlm_endpoint).strip()
    if not endpoint:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{args.vlm_model}:generateContent"
        )
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    uses_response_schema = response_schema is not None and not str(
        args.vlm_model
    ).startswith("gemini-3.")
    if uses_response_schema:
        payload["generationConfig"]["responseSchema"] = response_schema
    last_error = "unknown Gemini request error"
    for attempt in range(1, int(args.vlm_retries) + 1):
        try:
            response = requests.post(
                endpoint,
                headers={"x-goog-api-key": api_key},
                json=payload,
                timeout=max(1, int(args.vlm_timeout)),
            )
            if response.status_code >= 400:
                response_body = response.text.strip()
                api_error = RuntimeError(
                    f"Gemini HTTP {response.status_code}: {response_body[:4000]}"
                )
                api_error.response = response
                raise api_error
            response.raise_for_status()
            body = response.json()
            candidates = body.get("candidates", [])
            candidate = candidates[0] if candidates else {}
            content = candidate.get("content")
            if not isinstance(content, dict):
                finish_reason = candidate.get("finishReason", "unknown")
                prompt_feedback = body.get("promptFeedback", {})
                raise RuntimeError(
                    "Gemini returned no content "
                    f"(finishReason={finish_reason}, promptFeedback={prompt_feedback})"
                )
            response_parts = content.get("parts", [])
            response_text = "".join(
                str(part.get("text", "")) for part in response_parts
            )
            parsed_response = parse_json_object(response_text)
            if audit_path is not None:
                append_vlm_audit_record(
                    audit_path,
                    {
                        **(audit_context or {}),
                        "model": str(args.vlm_model),
                        "response_schema_enabled": uses_response_schema,
                        "attempt": attempt,
                        "status": "success",
                        "prompt": prompt,
                        "response_text": response_text,
                        "parsed_response": parsed_response,
                    },
                )
            return parsed_response, None
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if status_code is not None and 400 <= int(status_code) < 500:
                break
            if attempt < int(args.vlm_retries):
                time.sleep(min(8, 2**attempt))
    if audit_path is not None:
        append_vlm_audit_record(
            audit_path,
            {
                **(audit_context or {}),
                "model": str(args.vlm_model),
                "response_schema_enabled": uses_response_schema,
                "attempt": int(args.vlm_retries),
                "status": "error",
                "prompt": prompt,
                "error": last_error,
            },
        )
    return (
        None,
        f"Gemini request failed after {args.vlm_retries} attempts: {last_error}",
    )


def make_page_vlm_images(
    image_dir: Path,
    page: dict[str, Any],
) -> tuple[Image.Image, Image.Image, dict[str, str]]:
    """Return an original page, an annotated copy, and instance-to-label mapping."""
    with Image.open(image_dir / str(page["image"])) as raw:
        original = ImageOps.exif_transpose(raw).convert("RGB")
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(18, min(original.size) // 55))
    instance_ids: list[str] = []
    instance_boxes: dict[str, list[float]] = {}
    for dialogue in page.get("dialogues", []):
        for candidate in dialogue.get("top_candidates", []):
            instance_id = str(candidate["speaker_instance_id"])
            if instance_id not in instance_boxes:
                instance_ids.append(instance_id)
                instance_boxes[instance_id] = list(candidate["body_box"])
    body_labels = {
        instance_id: f"B{index + 1}" for index, instance_id in enumerate(instance_ids)
    }
    for index, dialogue in enumerate(page.get("dialogues", []), 1):
        box = list(dialogue["text_box"])
        label = f"T{index}"
        dialogue["vlm_text_id"] = label
        draw.rectangle(box, outline=(230, 35, 35), width=4)
        draw.text(
            (box[0] + 3, max(0, box[1] - 24)),
            label,
            fill=(230, 35, 35),
            font=font,
            stroke_width=2,
            stroke_fill="white",
        )
    for instance_id, label in body_labels.items():
        box = instance_boxes[instance_id]
        draw.rectangle(box, outline=(30, 100, 230), width=4)
        draw.text(
            (box[0] + 3, box[1] + 3),
            label,
            fill=(30, 100, 230),
            font=font,
            stroke_width=2,
            stroke_fill="white",
        )
    return original, annotated, body_labels


def crop_image_region(
    image: Image.Image,
    box: list[float],
    padding: int = 12,
) -> Image.Image:
    """Crop one region from an image with bounded padding."""
    xmin = max(0, int(box[0]) - padding)
    ymin = max(0, int(box[1]) - padding)
    xmax = min(image.width, int(box[2]) + padding)
    ymax = min(image.height, int(box[3]) + padding)
    return image.crop((xmin, ymin, max(xmin + 1, xmax), max(ymin + 1, ymax)))


def make_text_crop_grid(
    original: Image.Image,
    dialogues: list[dict[str, Any]],
) -> Image.Image:
    """Create large, separately labeled text crops without covering their glyphs."""
    cell_width, cell_height, caption_height = 340, 360, 42
    columns = min(3, max(1, len(dialogues)))
    rows = math.ceil(len(dialogues) / columns)
    grid = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=22)
    for index, dialogue in enumerate(dialogues):
        crop = crop_image_region(original, list(dialogue["text_box"]), padding=16)
        crop.thumbnail(
            (cell_width - 20, cell_height - caption_height - 16),
            Image.Resampling.LANCZOS,
        )
        column = index % columns
        row = index // columns
        x = column * cell_width
        y = row * cell_height
        paste_x = x + (cell_width - crop.width) // 2
        paste_y = y + caption_height + (cell_height - caption_height - crop.height) // 2
        grid.paste(crop, (paste_x, paste_y))
        draw.text(
            (x + 10, y + 8),
            str(dialogue["vlm_text_id"]),
            fill=(210, 25, 25),
            font=font,
        )
    return grid


def panel_dialogue_batches(
    page: dict[str, Any],
) -> list[tuple[int | None, list[dict[str, Any]]]]:
    """Group text regions by panel and order panels as right-to-left manga rows."""
    frames = [
        as_box(f"frame_{index}", box)
        for index, box in enumerate(page.get("frame_boxes", []))
    ]
    grouped: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for dialogue in page.get("dialogues", []):
        panel_index = choose_panel(as_box("text", dialogue["text_box"]), frames)
        dialogue["vlm_panel_index"] = panel_index
        grouped[panel_index].append(dialogue)

    indexed_frames = [
        (index, list(box)) for index, box in enumerate(page.get("frame_boxes", []))
    ]
    rows: list[list[tuple[int, list[float]]]] = []
    for frame in sorted(indexed_frames, key=lambda item: (item[1][1], -item[1][2])):
        _, box = frame
        height = max(1.0, box[3] - box[1])
        best_row = None
        best_overlap = 0.0
        for row_index, row in enumerate(rows):
            row_top = min(item[1][1] for item in row)
            row_bottom = max(item[1][3] for item in row)
            overlap = max(0.0, min(box[3], row_bottom) - max(box[1], row_top))
            overlap_ratio = overlap / min(height, max(1.0, row_bottom - row_top))
            if overlap_ratio >= 0.35 and overlap_ratio > best_overlap:
                best_row = row_index
                best_overlap = overlap_ratio
        if best_row is None:
            rows.append([frame])
        else:
            rows[best_row].append(frame)

    rows.sort(key=lambda row: min(item[1][1] for item in row))
    panel_order = [
        index
        for row in rows
        for index, _ in sorted(
            row,
            key=lambda item: -((item[1][0] + item[1][2]) / 2.0),
        )
        if index in grouped
    ]
    if None in grouped:
        panel_order.append(None)
    return [(panel_index, grouped[panel_index]) for panel_index in panel_order]


def build_page_body_labels(page: dict[str, Any]) -> dict[str, str]:
    """Assign deterministic page-global B labels to every V3 candidate body."""
    instance_boxes: dict[str, list[float]] = {}
    for dialogue in page.get("dialogues", []):
        for candidate in dialogue.get("top_candidates", []):
            instance_boxes.setdefault(
                str(candidate["speaker_instance_id"]),
                list(candidate["body_box"]),
            )
    ordered_ids = sorted(
        instance_boxes,
        key=lambda instance_id: (
            (instance_boxes[instance_id][1] + instance_boxes[instance_id][3]) / 2.0,
            -(instance_boxes[instance_id][0] + instance_boxes[instance_id][2]) / 2.0,
            instance_id,
        ),
    )
    return {
        instance_id: f"B{index + 1}" for index, instance_id in enumerate(ordered_ids)
    }


def make_panel_vlm_images(
    image_dir: Path,
    page: dict[str, Any],
    panel_index: int | None,
    dialogues: list[dict[str, Any]],
    previous_panel_index: int | None,
    next_panel_index: int | None,
    body_labels: dict[str, str],
    instance_role_labels: dict[str, str],
) -> tuple[list[Image.Image], Image.Image, dict[str, str], list[str]]:
    """Build full-page and sliding-panel context for one small Gemini batch."""
    with Image.open(image_dir / str(page["image"])) as raw:
        original = ImageOps.exif_transpose(raw).convert("RGB")
    frames = [list(box) for box in page.get("frame_boxes", [])]
    instance_boxes: dict[str, list[float]] = {}
    for dialogue in dialogues:
        for candidate in dialogue.get("top_candidates", []):
            instance_id = str(candidate["speaker_instance_id"])
            instance_boxes.setdefault(instance_id, list(candidate["body_box"]))

    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(18, min(original.size) // 55))
    for dialogue in dialogues:
        box = list(dialogue["text_box"])
        draw.rectangle(box, outline=(230, 35, 35), width=4)
        draw.text(
            (box[0] + 3, max(0, box[1] - 26)),
            str(dialogue["vlm_text_id"]),
            fill=(230, 35, 35),
            font=font,
            stroke_width=2,
            stroke_fill="white",
        )
    for instance_id, label in body_labels.items():
        if instance_id not in instance_boxes:
            continue
        box = instance_boxes[instance_id]
        role_label = instance_role_labels.get(instance_id, "R?")
        draw.rectangle(box, outline=(30, 100, 230), width=4)
        draw.text(
            (box[0] + 3, max(0, box[1] - 26)),
            f"{label}/{role_label}",
            fill=(30, 100, 230),
            font=font,
            stroke_width=2,
            stroke_fill="white",
        )

    if panel_index is not None and 0 <= panel_index < len(frames):
        focus_box = frames[panel_index]
    else:
        relevant_boxes = [list(row["text_box"]) for row in dialogues] + list(
            instance_boxes.values()
        )
        focus_box = [
            min(box[0] for box in relevant_boxes),
            min(box[1] for box in relevant_boxes),
            max(box[2] for box in relevant_boxes),
            max(box[3] for box in relevant_boxes),
        ]
    images = [
        original,
        annotated,
        crop_image_region(annotated, focus_box, padding=20),
        make_text_crop_grid(original, dialogues),
    ]
    descriptions = [
        "Image 1: complete original page",
        "Image 2: complete page with target T labels and stable B/provisional-R labels",
        "Image 3: enlarged labeled target panel",
        "Image 4: enlarged text crops labeled by T ID",
    ]
    if previous_panel_index is not None and 0 <= previous_panel_index < len(frames):
        images.append(
            crop_image_region(annotated, frames[previous_panel_index], padding=12)
        )
        descriptions.append(
            "Image 5: previous panel with stable B/provisional-R labels"
        )
    if next_panel_index is not None and 0 <= next_panel_index < len(frames):
        images.append(
            crop_image_region(annotated, frames[next_panel_index], padding=12)
        )
        descriptions.append(
            f"Image {len(images)}: next panel with stable B/provisional-R labels"
        )
    return images, annotated, body_labels, descriptions


def make_captioned_identity_board(
    board: Image.Image,
    dialogue_id: str,
) -> Image.Image:
    """Add a stable dialogue identifier above an identity candidate board."""
    caption_height = 42
    canvas = Image.new("RGB", (board.width, board.height + caption_height), "white")
    canvas.paste(board, (0, caption_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    draw.text((10, 8), dialogue_id, fill="black", font=font)
    return canvas


def run_panel_semantic_pass(
    pages: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    instance_index: dict[str, int],
    image_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Transcribe and link character-authored text in small sliding-panel batches."""
    batch_specs: list[
        tuple[
            dict[str, Any],
            int,
            int | None,
            list[dict[str, Any]],
            int | None,
            int | None,
        ]
    ] = []
    cluster_ids = sorted(
        {
            str(instance["character_cluster_id"])
            for instance in instances
            if instance.get("character_cluster_id") is not None
        }
    )
    cluster_role_labels = {
        cluster_id: f"R{index + 1}" for index, cluster_id in enumerate(cluster_ids)
    }
    instance_role_labels = {
        str(instance["instance_id"]): cluster_role_labels.get(
            str(instance.get("character_cluster_id")), "R?"
        )
        for instance in instances
    }
    page_body_labels: dict[str, dict[str, str]] = {}
    for page in pages:
        for index, dialogue in enumerate(page.get("dialogues", []), 1):
            dialogue["vlm_text_id"] = f"T{index}"
        page_key = str(page["image"])
        page_body_labels[page_key] = build_page_body_labels(page)
        panel_groups = panel_dialogue_batches(page)
        request_index = 0
        for panel_position, (panel_index, panel_dialogues) in enumerate(panel_groups):
            previous_panel = (
                panel_groups[panel_position - 1][0] if panel_position > 0 else None
            )
            next_panel = (
                panel_groups[panel_position + 1][0]
                if panel_position + 1 < len(panel_groups)
                else None
            )
            for start in range(0, len(panel_dialogues), args.vlm_panel_batch_size):
                batch_specs.append(
                    (
                        page,
                        request_index,
                        panel_index,
                        panel_dialogues[start : start + args.vlm_panel_batch_size],
                        previous_panel,
                        next_panel,
                    )
                )
                request_index += 1

    reviewed = failures = filtered_non_dialogue = 0
    recent_dialogues: list[dict[str, Any]] = []
    progress = tqdm(
        batch_specs,
        desc="Gemini pass 1/2: panel text + speaker",
        unit="target",
        dynamic_ncols=True,
    )
    semantic_response_schema = {
        "type": "OBJECT",
        "properties": {
            "texts": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "text_id": {"type": "STRING"},
                        "text": {"type": "STRING"},
                        "text_confidence": {"type": "NUMBER"},
                        "type": {
                            "type": "STRING",
                            "enum": [
                                "dialogue",
                                "thought",
                                "narration",
                                "sound_effect",
                                "sign",
                                "unknown",
                            ],
                        },
                        "type_confidence": {"type": "NUMBER"},
                        "requires_character_link": {"type": "BOOLEAN"},
                        "link_confidence": {"type": "NUMBER"},
                        "speaker_choice": {"type": "STRING"},
                        "speaker_confidence": {"type": "NUMBER"},
                        "speaker_evidence": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING",
                                "enum": [
                                    "bubble_tail",
                                    "mouth_pose",
                                    "dialogue_semantics",
                                    "cross_panel_continuity",
                                ],
                            },
                        },
                        "speaker_reason": {"type": "STRING"},
                    },
                    "required": [
                        "text_id",
                        "text",
                        "text_confidence",
                        "type",
                        "type_confidence",
                        "requires_character_link",
                        "link_confidence",
                        "speaker_choice",
                        "speaker_confidence",
                        "speaker_evidence",
                        "speaker_reason",
                    ],
                },
            }
        },
        "required": ["texts"],
    }
    for (
        page,
        batch_index,
        panel_index,
        dialogues,
        previous_panel,
        next_panel,
    ) in progress:
        body_labels = page_body_labels[str(page["image"])]
        images, annotated, body_labels, image_descriptions = make_panel_vlm_images(
            image_dir,
            page,
            panel_index,
            dialogues,
            previous_panel,
            next_panel,
            body_labels,
            instance_role_labels,
        )
        reverse_body_labels = {label: key for key, label in body_labels.items()}
        candidate_sets = []
        allowed_labels_by_text_id: dict[str, set[str]] = {}
        for dialogue in dialogues:
            candidates: list[dict[str, str]] = []
            dialogue["speaker_body_top5"] = []
            for candidate in dialogue.get("top_candidates", []):
                instance_id = str(candidate["speaker_instance_id"])
                label = body_labels.get(instance_id)
                if label is None:
                    continue
                candidate_record = {
                    "label": label,
                    "provisional_role": instance_role_labels.get(instance_id, "R?"),
                    **{
                        key: candidate[key]
                        for key in (
                            "rank",
                            "speaker_instance_id",
                            "v3_score",
                            "fused_score",
                            "softmax_share",
                            "body_box",
                        )
                    },
                }
                dialogue["speaker_body_top5"].append(candidate_record)
                candidates.append(
                    {
                        "body_label": label,
                        "provisional_role": instance_role_labels.get(instance_id, "R?"),
                    }
                )
            text_id = str(dialogue["vlm_text_id"])
            allowed_labels_by_text_id[text_id.upper()] = {
                candidate["body_label"] for candidate in candidates
            }
            candidate_sets.append(
                {
                    "text_id": text_id,
                    "speaker_candidates": candidates,
                }
            )
        prompt = f"""You are analyzing a small target-panel batch from one manga page.
The images are supplied in this order:
{json.dumps(image_descriptions, ensure_ascii=False)}

For every target T region, do all tasks without omission:
1. Transcribe its exact visible text from the enlarged crop. Preserve punctuation,
   repeated symbols, historical character forms, and Japanese kana/kanji exactly.
   Never rewrite, translate, or substitute a more common name spelling. Use "□"
   only for a genuinely unreadable character.
2. Classify it as dialogue, thought, narration, sound_effect, sign, or unknown.
3. Decide whether the text has a specific character source. Spoken dialogue,
   thoughts, laughs, screams, grunts, and character vocalizations require a
   character link. Narration, environmental motion sounds, signs, and scene labels
   do not. Return requires_character_link accordingly.
4. When a character link is required, choose only from that T region's allowed B
   candidates, or choose offscreen/unknown. Candidates may be visible in the
   target, previous, or next panel. Use the complete page and recent dialogue
   context; do not reject a candidate merely for being in an adjacent panel.
   B labels are stable across every request for this page. R labels are provisional
   ReID identities: the same R suggests the same role across panels/pages, but ReID
   clustering can be wrong, so never use R alone as decisive evidence.
5. Rerank only the supplied speaker candidates; never invent a body label. Prefer
   the best supported supplied candidate. Use offscreen only when the dialogue is
   clearly spoken by someone outside the visible candidates, and use unknown only
   when none of the supplied candidates has reliable support. A non-unknown speaker
   choice must be supported by at least one explicit evidence
   type: bubble_tail, mouth_pose, dialogue_semantics, or cross_panel_continuity.
   State the concrete visual/semantic reason in speaker_reason. Generic proximity,
   gender, clothing color, or a candidate rank is not sufficient evidence. When no
   explicit evidence supports one candidate, set speaker_choice to unknown, return
   an empty speaker_evidence list, and explain the ambiguity. An unknown or
   low-confidence answer preserves the existing V3/geometry prediction.
6. Every confidence must be a JSON number from 0.00 to 1.00, not a percentage or a
   word. Estimate it independently. Do not copy an example or placeholder value.

Allowed candidates:
{json.dumps(candidate_sets, ensure_ascii=False)}

Recent linked text context (page, text, stable B on that page, provisional R):
{json.dumps(recent_dialogues[-5:], ensure_ascii=False)}

Return JSON only:
{{
  "texts": [
    {{
      "text_id": "T1",
      "text": "exact transcription",
      "text_confidence": 0.91,
      "type": "dialogue|thought|narration|sound_effect|sign|unknown",
      "type_confidence": 0.88,
      "requires_character_link": true,
      "link_confidence": 0.86,
      "speaker_choice": "B1|offscreen|unknown|null",
      "speaker_confidence": 0.84,
      "speaker_evidence": ["bubble_tail", "dialogue_semantics"],
      "speaker_reason": "The bubble tail points to B1 and the reply fits B1's prior turn."
    }}
  ]
}}
"""
        image_path = Path(str(page["image"]))
        response, error = call_gemini_json(
            images,
            prompt,
            args,
            audit_path=output_dir / "vlm_raw_responses.jsonl",
            audit_context={
                "stage": "panel_speaker",
                "page": str(page["image"]),
                "panel_batch": batch_index + 1,
                "text_ids": [dialogue["vlm_text_id"] for dialogue in dialogues],
            },
            response_schema=semantic_response_schema,
        )
        if args.vlm_save_boards:
            board_path = (
                output_dir
                / "vlm_panel_boards"
                / image_path.parent
                / f"{image_path.stem}_panel_{batch_index + 1:02d}.jpg"
            )
            board_path.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(board_path, quality=92)
            crop_path = board_path.with_name(f"{board_path.stem}_text_crops.jpg")
            images[3].save(crop_path, quality=95)
        if error is not None or response is None:
            failures += 1
            for dialogue in dialogues:
                reviewed += 1
                dialogue["page_vlm"] = {"status": "error", "error": error}
                dialogue["speaker_vlm_top5"] = {
                    "status": "fallback",
                    "error": error,
                    "fallback": "v3_geometry",
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
            progress.set_postfix(
                texts=reviewed, filtered=filtered_non_dialogue, errors=failures
            )
            continue
        rows_by_id = {
            str(row.get("text_id", "")).strip().upper(): row
            for row in response.get("texts", [])
            if isinstance(row, dict)
        }
        for dialogue in dialogues:
            reviewed += 1
            text_id = str(dialogue["vlm_text_id"]).upper()
            row = rows_by_id.get(text_id)
            if row is None:
                failures += 1
                dialogue["page_vlm"] = {
                    "status": "error",
                    "error": "missing text region",
                }
                dialogue["speaker_vlm_top5"] = {
                    "status": "fallback",
                    "error": "missing text region",
                    "fallback": "v3_geometry",
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            recognized_text = str(row.get("text", "unknown")).strip() or "unknown"
            text_type = str(row.get("type", "unknown")).strip().lower()
            requires_character_link = bool(row.get("requires_character_link", False))
            confidence_fields = (
                "text_confidence",
                "type_confidence",
                "link_confidence",
                "speaker_confidence",
            )
            parsed_confidences = {
                field: parse_vlm_confidence(row.get(field), field)
                for field in confidence_fields
            }
            text_confidence = parsed_confidences["text_confidence"][0]
            type_confidence = parsed_confidences["type_confidence"][0]
            link_confidence = parsed_confidences["link_confidence"][0]
            speaker_confidence = parsed_confidences["speaker_confidence"][0]
            confidence_errors = {
                field: parsed[1]
                for field, parsed in parsed_confidences.items()
                if parsed[1] is not None
            }
            speaker_choice = str(row.get("speaker_choice", "unknown")).strip().upper()
            allowed_speaker_labels = allowed_labels_by_text_id.get(text_id, set())
            valid_evidence_types = {
                "bubble_tail",
                "mouth_pose",
                "dialogue_semantics",
                "cross_panel_continuity",
            }
            raw_evidence = row.get("speaker_evidence", [])
            speaker_evidence = (
                [
                    str(item).strip().lower()
                    for item in raw_evidence
                    if str(item).strip().lower() in valid_evidence_types
                ]
                if isinstance(raw_evidence, list)
                else []
            )
            speaker_reason = str(row.get("speaker_reason", "")).strip()
            dialogue["ocr_text_before_vlm"] = dialogue.get("ocr_text", "")
            dialogue["ocr_text"] = recognized_text
            dialogue["recognized_text"] = recognized_text
            dialogue["text_type"] = text_type
            dialogue["requires_character_link"] = requires_character_link
            dialogue["page_vlm"] = {
                "status": "accepted",
                "text_confidence": (
                    round(text_confidence, 4) if text_confidence is not None else None
                ),
                "type": text_type,
                "type_confidence": (
                    round(type_confidence, 4) if type_confidence is not None else None
                ),
                "requires_character_link": requires_character_link,
                "link_confidence": (
                    round(link_confidence, 4) if link_confidence is not None else None
                ),
                "confidence_errors": confidence_errors,
            }
            confirmed_non_character_text = (
                text_type in {"narration", "sound_effect"}
                and not requires_character_link
                and type_confidence is not None
                and type_confidence >= args.vlm_first_pass_confidence_threshold
                and link_confidence is not None
                and link_confidence >= args.vlm_first_pass_confidence_threshold
            )
            if confirmed_non_character_text:
                filtered_non_dialogue += 1
                mark_dialogue_unknown(dialogue, f"no character link: {text_type}")
                dialogue["speaker_vlm_top5"] = {
                    "status": "filtered_non_dialogue",
                    "choice": None,
                    "confidence": (
                        round(speaker_confidence, 4)
                        if speaker_confidence is not None
                        else None
                    ),
                    "evidence": speaker_evidence,
                    "reason": speaker_reason,
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            if (
                not requires_character_link
                or link_confidence is None
                or link_confidence < args.vlm_first_pass_confidence_threshold
                or type_confidence is None
                or type_confidence < args.vlm_first_pass_confidence_threshold
            ):
                dialogue["speaker_vlm_top5"] = {
                    "status": "fallback",
                    "choice": "unknown",
                    "confidence": (
                        round(speaker_confidence, 4)
                        if speaker_confidence is not None
                        else None
                    ),
                    "evidence": speaker_evidence,
                    "reason": speaker_reason,
                    "confidence_errors": confidence_errors,
                    "fallback": "v3_geometry",
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            instance_id = reverse_body_labels.get(speaker_choice)
            has_explicit_evidence = bool(speaker_evidence and speaker_reason)
            confirmed_offscreen = (
                speaker_choice == "OFFSCREEN"
                and speaker_confidence is not None
                and speaker_confidence >= args.vlm_first_pass_confidence_threshold
                and has_explicit_evidence
            )
            if confirmed_offscreen:
                mark_dialogue_unknown(dialogue, "Gemini confirmed an offscreen speaker")
                dialogue["speaker_vlm_top5"] = {
                    "status": "offscreen",
                    "choice": "offscreen",
                    "confidence": round(speaker_confidence, 4),
                    "threshold": args.vlm_first_pass_confidence_threshold,
                    "evidence": speaker_evidence,
                    "reason": speaker_reason,
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            if (
                instance_id is None
                or speaker_choice not in allowed_speaker_labels
                or speaker_confidence is None
                or speaker_confidence < args.vlm_first_pass_confidence_threshold
                or not has_explicit_evidence
                or instance_id not in instance_index
            ):
                reason = (
                    "Gemini gave no explicit speaker evidence"
                    if not has_explicit_evidence
                    else "Gemini speaker choice was invalid or insufficient"
                )
                dialogue["speaker_vlm_top5"] = {
                    "status": "fallback",
                    "choice": speaker_choice.lower(),
                    "confidence": (
                        round(speaker_confidence, 4)
                        if speaker_confidence is not None
                        else None
                    ),
                    "threshold": args.vlm_first_pass_confidence_threshold,
                    "evidence": speaker_evidence,
                    "reason": speaker_reason,
                    "confidence_errors": confidence_errors,
                    "fallback": "v3_geometry",
                    "fallback_reason": reason,
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            selected_body = instances[instance_index[instance_id]]
            dialogue.setdefault(
                "speaker_instance_id_before_vlm", dialogue.get("speaker_instance_id")
            )
            dialogue.setdefault(
                "character_cluster_id_before_vlm", dialogue.get("character_cluster_id")
            )
            dialogue.setdefault(
                "character_name_before_vlm", dialogue.get("character_name")
            )
            dialogue["speaker_instance_id"] = instance_id
            dialogue["speaker_body"] = selected_body["body_box"]
            dialogue["speaker_source"] = "gemini_panel_speaker"
            dialogue["character_cluster_id"] = selected_body["character_cluster_id"]
            dialogue["character_name"] = selected_body["character_name"]
            dialogue["identity_source"] = "gemini_panel_speaker_cluster"
            dialogue.pop("unknown_reason", None)
            dialogue["speaker_vlm_top5"] = {
                "status": "accepted",
                "choice": speaker_choice,
                "confidence": round(speaker_confidence, 4),
                "threshold": args.vlm_first_pass_confidence_threshold,
                "evidence": speaker_evidence,
                "reason": speaker_reason,
                "selected_instance_id": instance_id,
            }
            recent_dialogues.append(
                {
                    "page": str(page["image"]),
                    "text_id": text_id,
                    "text": recognized_text,
                    "speaker_body_label": body_labels.get(instance_id),
                    "provisional_role": instance_role_labels.get(instance_id, "R?"),
                }
            )
        progress.set_postfix(
            texts=reviewed, filtered=filtered_non_dialogue, errors=failures
        )
    return {
        "reviewed": reviewed,
        "failures": failures,
        "filtered_non_dialogue": filtered_non_dialogue,
        "panel_batches": len(batch_specs),
    }


def mark_dialogue_unknown(dialogue: dict[str, Any], reason: str) -> None:
    """Clear speaker identity only for a confirmed non-character/offscreen result."""
    dialogue.setdefault(
        "speaker_instance_id_before_vlm", dialogue.get("speaker_instance_id")
    )
    dialogue.setdefault(
        "character_cluster_id_before_vlm", dialogue.get("character_cluster_id")
    )
    dialogue.setdefault("character_name_before_vlm", dialogue.get("character_name"))
    dialogue["speaker_instance_id"] = "unknown"
    dialogue["speaker_body"] = "unknown"
    dialogue["character_cluster_id"] = "unknown"
    dialogue["character_name"] = "unknown"
    dialogue["identity_source"] = "unknown"
    dialogue["unknown_reason"] = reason


def mark_identity_unknown(dialogue: dict[str, Any], reason: str) -> None:
    """Keep a verified speaker while abstaining from an uncertain identity."""
    dialogue.setdefault(
        "character_cluster_id_before_vlm", dialogue.get("character_cluster_id")
    )
    dialogue.setdefault("character_name_before_vlm", dialogue.get("character_name"))
    dialogue["character_cluster_id"] = "unknown"
    dialogue["character_name"] = "unknown"
    dialogue["identity_source"] = "unknown"
    dialogue["unknown_reason"] = reason


def build_identity_top5(
    query_index: int,
    instances: list[dict[str, Any]],
    features: np.ndarray,
    cluster_ids: list[str],
    prototypes: np.ndarray,
    references: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retrieve five identities without reusing the query crop as its reference."""
    scores = features[query_index] @ prototypes.T
    candidates: list[dict[str, Any]] = []
    query_instance_id = str(instances[query_index]["instance_id"])
    for index in np.argsort(-scores, kind="stable").tolist():
        cluster_id = cluster_ids[int(index)]
        reference = references[cluster_id]
        if str(reference["instance_id"]) == query_instance_id:
            alternate_indexes = [
                member_index
                for member_index, instance in enumerate(instances)
                if member_index != query_index
                and str(instance.get("character_cluster_id")) == cluster_id
            ]
            if not alternate_indexes:
                continue
            reference_index = max(
                alternate_indexes,
                key=lambda member_index: float(
                    features[query_index] @ features[member_index]
                ),
            )
            reference = instances[reference_index]
        candidates.append(
            {
                "cluster_id": cluster_id,
                "character_name": str(reference.get("character_name", cluster_id)),
                "reid_similarity": round(float(scores[int(index)]), 6),
                "reference": reference,
            }
        )
        if len(candidates) == 5:
            break
    return candidates


def verify_dialogues_with_vlm(
    pages: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    features: np.ndarray,
    image_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Verify speaker bodies first, then verify their ReID identity candidates."""
    cluster_ids, prototypes, references = build_cluster_identity_bank(
        instances, features
    )
    instance_index = {
        str(row["instance_id"]): index for index, row in enumerate(instances)
    }
    reviewed = accepted = changed = unknown = failures = unreviewed = 0
    for page in pages:
        for dialogue in page.get("dialogues", []):
            if args.vlm_max_dialogues and reviewed >= args.vlm_max_dialogues:
                mark_dialogue_unknown(
                    dialogue, "not reviewed because --vlm-max-dialogues was reached"
                )
                dialogue["speaker_vlm_top5"] = {"status": "not_reviewed"}
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                unknown += 1
                unreviewed += 1
                continue
            original_speaker_id = str(dialogue.get("speaker_instance_id") or "")
            original_cluster_id = str(dialogue.get("character_cluster_id") or "")
            speaker_candidates = []
            for candidate in dialogue.get("top_candidates", [])[
                : args.vlm_speaker_top_k
            ]:
                candidate_index = instance_index.get(
                    str(candidate["speaker_instance_id"])
                )
                if candidate_index is None:
                    continue
                speaker_candidates.append(
                    {**candidate, "instance": instances[candidate_index]}
                )
            dialogue["speaker_body_top5"] = [
                {
                    key: candidate[key]
                    for key in (
                        "rank",
                        "speaker_instance_id",
                        "v3_score",
                        "fused_score",
                        "softmax_share",
                        "body_box",
                    )
                }
                for candidate in speaker_candidates
            ]
            if not speaker_candidates:
                failures += 1
                unknown += 1
                mark_dialogue_unknown(dialogue, "speaker candidates are unavailable")
                dialogue["speaker_vlm_top5"] = {
                    "status": "error",
                    "error": "speaker candidates are unavailable",
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            speaker_board = make_vlm_speaker_top5_board(
                image_dir, page, dialogue, speaker_candidates, args.vlm_image_size
            )
            speaker_board_path: Path | None = None
            if args.vlm_save_boards:
                speaker_board_path = (
                    output_dir
                    / "vlm_speaker_top5_boards"
                    / f"{dialogue['dialogue_id']}.jpg"
                )
                speaker_board_path.parent.mkdir(parents=True, exist_ok=True)
                speaker_board.save(speaker_board_path, quality=90)
            speaker_prompt = """You are deciding who speaks a manga dialogue.
The upper image is the complete panel. The red rectangle is the dialogue text or
speech bubble. A-E are the only candidate character bodies, with matching colored
rectangles in the panel and crops below. Choose the character most likely to speak
that text using bubble position, tail direction, pose, gaze, and panel composition.
Do not guess from gender or proximity alone. If the speaker cannot be determined
reliably, choose unknown. Return JSON only:
{"choice":"A|B|C|D|E|unknown","confidence":0.0}
"""
            selected_speaker, speaker_confidence, speaker_error = call_gemini_top5(
                speaker_board, args, len(speaker_candidates), speaker_prompt
            )
            reviewed += 1
            speaker_evidence: dict[str, Any] = {
                "model": args.vlm_model,
                "candidate_count": len(speaker_candidates),
                "choice": (
                    "unknown" if selected_speaker is None else "ABCDE"[selected_speaker]
                ),
                "confidence": round(speaker_confidence, 4),
            }
            if speaker_board_path is not None:
                speaker_evidence["board"] = str(
                    speaker_board_path.relative_to(output_dir)
                )
            if speaker_error:
                failures += 1
                unknown += 1
                speaker_evidence.update(status="error", error=speaker_error)
                dialogue["speaker_vlm_top5"] = speaker_evidence
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                mark_dialogue_unknown(dialogue, "Gemini speaker verification failed")
                continue
            if (
                selected_speaker is None
                or speaker_confidence < args.vlm_confidence_threshold
            ):
                unknown += 1
                speaker_evidence.update(
                    status="unknown" if selected_speaker is None else "below_threshold",
                    threshold=args.vlm_confidence_threshold,
                )
                dialogue["speaker_vlm_top5"] = speaker_evidence
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                mark_dialogue_unknown(
                    dialogue, "Gemini speaker confidence is insufficient"
                )
                continue
            selected_body = speaker_candidates[selected_speaker]["instance"]
            query_index = instance_index[str(selected_body["instance_id"])]
            dialogue["speaker_instance_id_before_vlm"] = original_speaker_id
            dialogue["speaker_instance_id"] = selected_body["instance_id"]
            dialogue["speaker_body"] = selected_body["body_box"]
            dialogue["speaker_source"] = "gemini_speaker_top5"
            speaker_evidence.update(
                status="accepted", selected_instance_id=selected_body["instance_id"]
            )
            dialogue["speaker_vlm_top5"] = speaker_evidence
            identity_candidates = build_identity_top5(
                query_index, instances, features, cluster_ids, prototypes, references
            )
            dialogue["identity_top5"] = [
                {
                    "rank": rank,
                    "character_cluster_id": row["cluster_id"],
                    "character_name": row["character_name"],
                    "reid_similarity": row["reid_similarity"],
                    "reference_instance_id": row["reference"]["instance_id"],
                }
                for rank, row in enumerate(identity_candidates, 1)
            ]
            if not identity_candidates:
                failures += 1
                unknown += 1
                dialogue["identity_vlm_top5"] = {
                    "status": "error",
                    "error": "character library is empty",
                }
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                mark_identity_unknown(
                    dialogue, "ReID identity candidates are unavailable"
                )
                continue
            identity_board = make_vlm_identity_top5_board(
                image_dir, selected_body, identity_candidates, args.vlm_image_size
            )
            identity_board_path: Path | None = None
            if args.vlm_save_boards:
                identity_board_path = (
                    output_dir
                    / "vlm_identity_top5_boards"
                    / f"{dialogue['dialogue_id']}.jpg"
                )
                identity_board_path.parent.mkdir(parents=True, exist_ok=True)
                identity_board.save(identity_board_path, quality=90)
            identity_prompt = """You are verifying the identity of a manga character.
The first image cell is QUERY: the speaker selected after dialogue verification.
Cells A-E are anonymous ReID candidate-character references. Choose the same
character only when facial features, hairstyle, accessories, and clothing design
support it. Do not choose based only on gender, pose, or generic art style.
If the query is a back view, too small, occluded, or none of A-E is reliably the
same person, choose unknown. Return JSON only:
{"choice":"A|B|C|D|E|unknown","confidence":0.0}
"""
            selected_identity, identity_confidence, identity_error = call_gemini_top5(
                identity_board, args, len(identity_candidates), identity_prompt
            )
            identity_evidence: dict[str, Any] = {
                "model": args.vlm_model,
                "candidate_count": len(identity_candidates),
                "choice": (
                    "unknown"
                    if selected_identity is None
                    else "ABCDE"[selected_identity]
                ),
                "confidence": round(identity_confidence, 4),
            }
            if identity_board_path is not None:
                identity_evidence["board"] = str(
                    identity_board_path.relative_to(output_dir)
                )
            if identity_error:
                failures += 1
                unknown += 1
                identity_evidence.update(status="error", error=identity_error)
                dialogue["identity_vlm_top5"] = identity_evidence
                dialogue["vlm_top5"] = identity_evidence
                mark_identity_unknown(dialogue, "Gemini identity verification failed")
                continue
            if (
                selected_identity is None
                or identity_confidence < args.vlm_confidence_threshold
            ):
                unknown += 1
                identity_evidence.update(
                    status=(
                        "unknown" if selected_identity is None else "below_threshold"
                    ),
                    threshold=args.vlm_confidence_threshold,
                )
                dialogue["identity_vlm_top5"] = identity_evidence
                dialogue["vlm_top5"] = identity_evidence
                mark_identity_unknown(
                    dialogue, "Gemini identity confidence is insufficient"
                )
                continue
            selected_candidate = identity_candidates[selected_identity]
            dialogue["character_cluster_id_before_vlm"] = original_cluster_id
            dialogue["character_name_before_vlm"] = dialogue.get("character_name")
            dialogue["character_cluster_id"] = selected_candidate["cluster_id"]
            dialogue["character_name"] = selected_candidate["character_name"]
            dialogue["identity_source"] = "gemini_identity_top5"
            identity_evidence.update(
                status="accepted", selected_cluster_id=selected_candidate["cluster_id"]
            )
            dialogue["identity_vlm_top5"] = identity_evidence
            dialogue["vlm_top5"] = identity_evidence
            accepted += 1
            changed += int(
                original_speaker_id != selected_body["instance_id"]
                or original_cluster_id != selected_candidate["cluster_id"]
            )
            if reviewed % 25 == 0:
                print(f"Gemini Top-5 reviewed: {reviewed}", flush=True)
    return {
        "reviewed": reviewed,
        "accepted": accepted,
        "changed": changed,
        "unknown": unknown,
        "failures": failures,
        "unreviewed": unreviewed,
        "limited": int(unreviewed > 0),
    }


def verify_pages_with_vlm(
    pages: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    features: np.ndarray,
    image_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Run one semantic request and one identity request for every selected page."""
    cluster_ids, prototypes, references = build_cluster_identity_bank(
        instances, features
    )
    instance_index = {
        str(row["instance_id"]): index for index, row in enumerate(instances)
    }
    page_limit = args.vlm_max_pages or len(pages)
    selected_pages = pages[: min(page_limit, len(pages))]
    unselected_pages = pages[len(selected_pages) :]
    unreviewed = 0
    for page in unselected_pages:
        for dialogue in page.get("dialogues", []):
            mark_dialogue_unknown(
                dialogue, "not reviewed because --vlm-max-pages was reached"
            )
            dialogue["page_vlm"] = {"status": "not_reviewed"}
            dialogue["speaker_vlm_top5"] = {"status": "not_reviewed"}
            dialogue["identity_vlm_top5"] = {"status": "not_run"}
            dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
            unreviewed += 1

    reviewed = accepted = changed = failures = filtered_non_dialogue = 0
    recent_dialogues: list[dict[str, str]] = []
    pass_one = tqdm(
        selected_pages,
        desc="Gemini pass 1/2: page text + speaker",
        unit="page",
        dynamic_ncols=True,
    )
    for page in pass_one:
        original, annotated, body_labels = make_page_vlm_images(image_dir, page)
        reverse_body_labels = {label: key for key, label in body_labels.items()}
        candidate_sets = []
        for dialogue in page.get("dialogues", []):
            labels = [
                body_labels[str(candidate["speaker_instance_id"])]
                for candidate in dialogue.get("top_candidates", [])
                if str(candidate["speaker_instance_id"]) in body_labels
            ]
            dialogue["speaker_body_top5"] = [
                {
                    "label": body_labels[str(candidate["speaker_instance_id"])],
                    **{
                        key: candidate[key]
                        for key in (
                            "rank",
                            "speaker_instance_id",
                            "v3_score",
                            "fused_score",
                            "softmax_share",
                            "body_box",
                        )
                    },
                }
                for candidate in dialogue.get("top_candidates", [])
                if str(candidate["speaker_instance_id"]) in body_labels
            ]
            candidate_sets.append(
                {
                    "text_id": dialogue["vlm_text_id"],
                    "speaker_candidates": labels,
                }
            )
        prompt = f"""You are analyzing one complete manga page.
Image 1 is the original high-resolution page. Image 2 is the same page with red
text-region labels T1... and blue body labels B1.... Read Image 1 for exact text;
use Image 2 only to locate regions and candidates.

For every T region, do all of the following without omission:
1. Transcribe the visible text exactly. Preserve punctuation and repeated symbols.
   Do not rewrite, translate, normalize, or infer unreadable characters. Use □ for
   an unreadable character.
2. Classify it as dialogue, thought, narration, sound_effect, sign, or unknown.
3. Only for dialogue, choose its speaker from that T region's allowed body labels.
   You may instead choose offscreen or unknown. Never attach narration, sound
   effects, or signs to a character. Do not guess from proximity alone.

Allowed speaker candidates by text region:
{json.dumps(candidate_sets, ensure_ascii=False)}

Recent dialogue context from preceding pages:
{json.dumps(recent_dialogues[-5:], ensure_ascii=False)}

Return JSON only:
{{
  "texts": [
    {{
      "text_id": "T1",
      "text": "exact transcription",
      "text_confidence": 0.0,
      "type": "dialogue|thought|narration|sound_effect|sign|unknown",
      "type_confidence": 0.0,
      "speaker_choice": "B1|offscreen|unknown|null",
      "speaker_confidence": 0.0
    }}
  ]
}}
"""
        response, error = call_gemini_json([original, annotated], prompt, args)
        if args.vlm_save_boards:
            board_path = output_dir / "vlm_page_boards" / str(page["image"])
            board_path.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(board_path, quality=92)
        if error is not None or response is None:
            failures += 1
            for dialogue in page.get("dialogues", []):
                mark_dialogue_unknown(dialogue, "Gemini page analysis failed")
                dialogue["page_vlm"] = {"status": "error", "error": error}
                dialogue["speaker_vlm_top5"] = {"status": "error", "error": error}
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
            pass_one.set_postfix(ok=reviewed, errors=failures)
            continue
        rows = response.get("texts", [])
        rows_by_id = {
            str(row.get("text_id", "")).strip().upper(): row
            for row in rows
            if isinstance(row, dict)
        }
        for dialogue in page.get("dialogues", []):
            reviewed += 1
            text_id = str(dialogue["vlm_text_id"]).upper()
            row = rows_by_id.get(text_id)
            if row is None:
                mark_dialogue_unknown(dialogue, "Gemini omitted this text region")
                dialogue["page_vlm"] = {
                    "status": "error",
                    "error": "missing text region",
                }
                dialogue["speaker_vlm_top5"] = {
                    "status": "error",
                    "error": "missing text region",
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                failures += 1
                continue
            recognized_text = str(row.get("text", "unknown")).strip() or "unknown"
            text_type = str(row.get("type", "unknown")).strip().lower()
            try:
                text_confidence = float(row.get("text_confidence", 0.0))
                type_confidence = float(row.get("type_confidence", 0.0))
                speaker_confidence = float(row.get("speaker_confidence", 0.0))
            except (TypeError, ValueError):
                text_confidence = type_confidence = speaker_confidence = 0.0
            speaker_choice = str(row.get("speaker_choice", "unknown")).strip().upper()
            dialogue["ocr_text_before_vlm"] = dialogue.get("ocr_text", "")
            dialogue["ocr_text"] = recognized_text
            dialogue["recognized_text"] = recognized_text
            dialogue["text_type"] = text_type
            dialogue["page_vlm"] = {
                "status": "accepted",
                "text_confidence": round(max(0.0, min(1.0, text_confidence)), 4),
                "type": text_type,
                "type_confidence": round(max(0.0, min(1.0, type_confidence)), 4),
            }
            if (
                text_type != "dialogue"
                and type_confidence >= args.vlm_confidence_threshold
            ):
                filtered_non_dialogue += 1
                mark_dialogue_unknown(dialogue, f"filtered text type: {text_type}")
                dialogue["speaker_vlm_top5"] = {
                    "status": "filtered_non_dialogue",
                    "choice": None,
                    "confidence": round(speaker_confidence, 4),
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            if (
                text_type != "dialogue"
                or type_confidence < args.vlm_confidence_threshold
            ):
                mark_dialogue_unknown(dialogue, "Gemini text type is uncertain")
                dialogue["speaker_vlm_top5"] = {
                    "status": "unknown",
                    "choice": "unknown",
                    "confidence": round(speaker_confidence, 4),
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            instance_id = reverse_body_labels.get(speaker_choice)
            if (
                instance_id is None
                or speaker_confidence < args.vlm_confidence_threshold
                or instance_id not in instance_index
            ):
                reason = (
                    "Gemini selected an offscreen speaker"
                    if speaker_choice == "OFFSCREEN"
                    else "Gemini speaker confidence is insufficient"
                )
                mark_dialogue_unknown(dialogue, reason)
                dialogue["speaker_vlm_top5"] = {
                    "status": (
                        "offscreen" if speaker_choice == "OFFSCREEN" else "unknown"
                    ),
                    "choice": speaker_choice.lower(),
                    "confidence": round(speaker_confidence, 4),
                }
                dialogue["identity_vlm_top5"] = {"status": "not_run"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            selected_body = instances[instance_index[instance_id]]
            dialogue.setdefault(
                "speaker_instance_id_before_vlm", dialogue.get("speaker_instance_id")
            )
            dialogue.setdefault(
                "character_cluster_id_before_vlm", dialogue.get("character_cluster_id")
            )
            dialogue.setdefault(
                "character_name_before_vlm", dialogue.get("character_name")
            )
            dialogue["speaker_instance_id"] = instance_id
            dialogue["speaker_body"] = selected_body["body_box"]
            dialogue["speaker_source"] = "gemini_page_speaker"
            dialogue["speaker_vlm_top5"] = {
                "status": "accepted",
                "choice": speaker_choice,
                "confidence": round(speaker_confidence, 4),
                "selected_instance_id": instance_id,
            }
            recent_dialogues.append(
                {"text": recognized_text, "speaker_instance_id": instance_id}
            )
        pass_one.set_postfix(
            texts=reviewed, filtered=filtered_non_dialogue, errors=failures
        )

    pass_two = tqdm(
        selected_pages,
        desc="Gemini pass 2/2: ReID identity",
        unit="page",
        dynamic_ncols=True,
    )
    for page in pass_two:
        identity_jobs: list[
            tuple[dict[str, Any], list[dict[str, Any]], Image.Image]
        ] = []
        for dialogue in page.get("dialogues", []):
            if dialogue.get("speaker_vlm_top5", {}).get("status") != "accepted":
                continue
            query_index = instance_index.get(str(dialogue.get("speaker_instance_id")))
            if query_index is None:
                mark_identity_unknown(
                    dialogue, "selected speaker instance is unavailable"
                )
                dialogue["identity_vlm_top5"] = {"status": "error"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                failures += 1
                continue
            identity_candidates = build_identity_top5(
                query_index,
                instances,
                features,
                cluster_ids,
                prototypes,
                references,
            )
            dialogue["identity_top5"] = [
                {
                    "rank": rank,
                    "character_cluster_id": row["cluster_id"],
                    "character_name": row["character_name"],
                    "reid_similarity": row["reid_similarity"],
                    "reference_instance_id": row["reference"]["instance_id"],
                }
                for rank, row in enumerate(identity_candidates, 1)
            ]
            if not identity_candidates:
                mark_identity_unknown(
                    dialogue, "ReID identity candidates are unavailable"
                )
                dialogue["identity_vlm_top5"] = {"status": "error"}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                failures += 1
                continue
            board = make_vlm_identity_top5_board(
                image_dir,
                instances[query_index],
                identity_candidates,
                args.vlm_image_size,
            )
            board = make_captioned_identity_board(board, dialogue["dialogue_id"])
            identity_jobs.append((dialogue, identity_candidates, board))
            if args.vlm_save_boards:
                board_path = (
                    output_dir
                    / "vlm_identity_top5_boards"
                    / f"{dialogue['dialogue_id']}.jpg"
                )
                board_path.parent.mkdir(parents=True, exist_ok=True)
                board.save(board_path, quality=92)
        if not identity_jobs:
            pass_two.set_postfix(accepted=accepted, errors=failures)
            continue
        identity_prompt = f"""You are verifying manga character identities for one page.
Each image after the first page image is labeled with a dialogue ID. Within each
identity board, QUERY is the speaker selected in pass 1 and A-E are anonymous ReID
character references. For every dialogue ID, choose the same character only when
visual identity evidence is reliable. Do not choose by gender, pose, or generic art
style. Choose unknown for back views, tiny or occluded queries, or no reliable match.

Dialogue IDs in image order:
{json.dumps([job[0]['dialogue_id'] for job in identity_jobs], ensure_ascii=False)}

Return JSON only:
{{
  "identities": [
    {{
      "dialogue_id": "P001_T001",
      "choice": "A|B|C|D|E|unknown",
      "confidence": 0.0
    }}
  ]
}}
"""
        with Image.open(image_dir / str(page["image"])) as raw:
            page_image = ImageOps.exif_transpose(raw).convert("RGB")
        response, error = call_gemini_json(
            [page_image] + [job[2] for job in identity_jobs], identity_prompt, args
        )
        if error is not None or response is None:
            failures += 1
            for dialogue, _, _ in identity_jobs:
                mark_identity_unknown(dialogue, "Gemini page identity analysis failed")
                dialogue["identity_vlm_top5"] = {"status": "error", "error": error}
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
            pass_two.set_postfix(accepted=accepted, errors=failures)
            continue
        rows = response.get("identities", [])
        rows_by_id = {
            str(row.get("dialogue_id", "")): row
            for row in rows
            if isinstance(row, dict)
        }
        for dialogue, candidates, _ in identity_jobs:
            row = rows_by_id.get(str(dialogue["dialogue_id"]))
            if row is None:
                mark_identity_unknown(dialogue, "Gemini omitted this identity decision")
                dialogue["identity_vlm_top5"] = {
                    "status": "error",
                    "error": "missing identity decision",
                }
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                failures += 1
                continue
            choice = str(row.get("choice", "unknown")).strip().upper()
            try:
                confidence = float(row.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            selected = "ABCDE"[: len(candidates)].find(choice)
            evidence: dict[str, Any] = {
                "model": args.vlm_model,
                "choice": choice.lower() if choice == "UNKNOWN" else choice,
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "candidate_count": len(candidates),
            }
            if selected < 0 or confidence < args.vlm_confidence_threshold:
                mark_identity_unknown(
                    dialogue, "Gemini identity confidence is insufficient"
                )
                evidence.update(
                    status="unknown" if selected < 0 else "below_threshold",
                    threshold=args.vlm_confidence_threshold,
                )
                dialogue["identity_vlm_top5"] = evidence
                dialogue["vlm_top5"] = evidence
                continue
            selected_candidate = candidates[selected]
            original_speaker = str(dialogue.get("speaker_instance_id_before_vlm"))
            original_cluster = str(dialogue.get("character_cluster_id_before_vlm"))
            dialogue["character_cluster_id"] = selected_candidate["cluster_id"]
            dialogue["character_name"] = selected_candidate["character_name"]
            dialogue["identity_source"] = "gemini_page_identity_top5"
            dialogue.pop("unknown_reason", None)
            evidence.update(
                status="accepted",
                selected_cluster_id=selected_candidate["cluster_id"],
            )
            dialogue["identity_vlm_top5"] = evidence
            dialogue["vlm_top5"] = evidence
            accepted += 1
            changed += int(
                original_speaker != str(dialogue.get("speaker_instance_id"))
                or original_cluster != selected_candidate["cluster_id"]
            )
        pass_two.set_postfix(accepted=accepted, errors=failures)

    unknown = sum(
        dialogue.get("character_name") == "unknown"
        for page in pages
        for dialogue in page.get("dialogues", [])
    )
    return {
        "reviewed": reviewed,
        "reviewed_pages": len(selected_pages),
        "accepted": accepted,
        "changed": changed,
        "unknown": int(unknown),
        "filtered_non_dialogue": filtered_non_dialogue,
        "failures": failures,
        "unreviewed": unreviewed,
        "limited": int(bool(unselected_pages)),
    }


def run_batched_identity_pass(
    pages: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    features: np.ndarray,
    cluster_ids: list[str],
    prototypes: np.ndarray,
    references: dict[str, dict[str, Any]],
    instance_index: dict[str, int],
    image_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Verify ReID candidates in bounded batches instead of one oversized page call."""
    batches: list[
        tuple[
            dict[str, Any],
            list[tuple[dict[str, Any], list[dict[str, Any]], Image.Image]],
        ]
    ] = []
    failures = 0
    for page in pages:
        page_jobs: list[tuple[dict[str, Any], list[dict[str, Any]], Image.Image]] = []
        for dialogue in page.get("dialogues", []):
            if dialogue.get("speaker_vlm_top5", {}).get("status") != "accepted":
                continue
            query_index = instance_index.get(str(dialogue.get("speaker_instance_id")))
            if query_index is None:
                failures += 1
                dialogue["identity_vlm_top5"] = {
                    "status": "fallback",
                    "error": "selected speaker instance is unavailable",
                    "fallback": "reid_v3_geometry",
                }
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            candidates = build_identity_top5(
                query_index,
                instances,
                features,
                cluster_ids,
                prototypes,
                references,
            )
            dialogue["identity_top5"] = [
                {
                    "rank": rank,
                    "character_cluster_id": row["cluster_id"],
                    "character_name": row["character_name"],
                    "reid_similarity": row["reid_similarity"],
                    "reference_instance_id": row["reference"]["instance_id"],
                }
                for rank, row in enumerate(candidates, 1)
            ]
            if not candidates:
                failures += 1
                dialogue["identity_vlm_top5"] = {
                    "status": "fallback",
                    "error": "ReID identity candidates are unavailable",
                    "fallback": "reid_v3_geometry",
                }
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            board = make_vlm_identity_top5_board(
                image_dir,
                instances[query_index],
                candidates,
                args.vlm_image_size,
            )
            board = make_captioned_identity_board(board, dialogue["dialogue_id"])
            page_jobs.append((dialogue, candidates, board))
            if args.vlm_save_boards:
                board_path = (
                    output_dir
                    / "vlm_identity_top5_boards"
                    / f"{dialogue['dialogue_id']}.jpg"
                )
                board_path.parent.mkdir(parents=True, exist_ok=True)
                board.save(board_path, quality=92)
        batch_size = int(args.vlm_identity_batch_size)
        for start in range(0, len(page_jobs), batch_size):
            batches.append((page, page_jobs[start : start + batch_size]))

    accepted = changed = 0
    progress = tqdm(
        batches,
        desc="Gemini pass 2/2: ReID identity",
        unit="batch",
        dynamic_ncols=True,
    )
    identity_response_schema = {
        "type": "OBJECT",
        "properties": {
            "identities": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "dialogue_id": {"type": "STRING"},
                        "choice": {
                            "type": "STRING",
                            "enum": ["A", "B", "C", "D", "E", "unknown"],
                        },
                        "confidence": {"type": "NUMBER"},
                        "reason": {"type": "STRING"},
                    },
                    "required": ["dialogue_id", "choice", "confidence", "reason"],
                },
            }
        },
        "required": ["identities"],
    }
    for page, jobs in progress:
        prompt = f"""You are verifying manga character identities.
Image 1 is the complete page. Every following image is an identity board labeled
with a dialogue ID. QUERY is the pass-1 speaker; A-E are anonymous ReID character
references. For each dialogue ID, choose the same character only with reliable
visual identity evidence. Do not choose by gender, pose, or generic art style.
Choose unknown for back views, tiny or occluded queries, or no reliable match.
Rerank only A-E and never invent a choice. An unknown or low-confidence answer
preserves the existing ReID/V3/geometry identity.
Return confidence as a JSON number from 0.00 to 1.00, not a percentage or word.
Estimate it independently and do not copy an example or placeholder value. Explain
the concrete identity evidence briefly in reason.

Dialogue IDs in image order:
{json.dumps([job[0]['dialogue_id'] for job in jobs], ensure_ascii=False)}

Return JSON only:
{{
  "identities": [
    {{
      "dialogue_id": "P001_T001",
      "choice": "A|B|C|D|E|unknown",
      "confidence": 0.83,
      "reason": "The face shape, hairline, and eye design match reference A."
    }}
  ]
}}
"""
        with Image.open(image_dir / str(page["image"])) as raw:
            page_image = ImageOps.exif_transpose(raw).convert("RGB")
        response, error = call_gemini_json(
            [page_image] + [job[2] for job in jobs],
            prompt,
            args,
            audit_path=output_dir / "vlm_raw_responses.jsonl",
            audit_context={
                "stage": "reid_identity",
                "page": str(page["image"]),
                "dialogue_ids": [job[0]["dialogue_id"] for job in jobs],
            },
            response_schema=identity_response_schema,
        )
        if error is not None or response is None:
            failures += 1
            for dialogue, _, _ in jobs:
                dialogue["identity_vlm_top5"] = {
                    "status": "fallback",
                    "error": error,
                    "fallback": "reid_v3_geometry",
                }
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
            progress.set_postfix(accepted=accepted, errors=failures)
            continue
        rows_by_id = {
            str(row.get("dialogue_id", "")): row
            for row in response.get("identities", [])
            if isinstance(row, dict)
        }
        for dialogue, candidates, _ in jobs:
            row = rows_by_id.get(str(dialogue["dialogue_id"]))
            if row is None:
                failures += 1
                dialogue["identity_vlm_top5"] = {
                    "status": "fallback",
                    "error": "missing identity decision",
                    "fallback": "reid_v3_geometry",
                }
                dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
                continue
            choice = str(row.get("choice", "unknown")).strip().upper()
            confidence, confidence_error = parse_vlm_confidence(
                row.get("confidence"), "confidence"
            )
            identity_reason = str(row.get("reason", "")).strip()
            selected = "ABCDE"[: len(candidates)].find(choice)
            evidence: dict[str, Any] = {
                "model": args.vlm_model,
                "choice": choice.lower() if choice == "UNKNOWN" else choice,
                "confidence": round(confidence, 4) if confidence is not None else None,
                "candidate_count": len(candidates),
                "reason": identity_reason,
            }
            if confidence_error is not None:
                evidence["confidence_error"] = confidence_error
            if (
                selected < 0
                or confidence is None
                or confidence < args.vlm_confidence_threshold
            ):
                evidence.update(
                    status="fallback",
                    threshold=args.vlm_confidence_threshold,
                    fallback="reid_v3_geometry",
                    fallback_reason=(
                        "Gemini returned unknown"
                        if selected < 0
                        else "Gemini identity confidence is insufficient"
                    ),
                )
                dialogue["identity_vlm_top5"] = evidence
                dialogue["vlm_top5"] = evidence
                continue
            selected_candidate = candidates[selected]
            original_speaker = str(dialogue.get("speaker_instance_id_before_vlm"))
            original_cluster = str(dialogue.get("character_cluster_id_before_vlm"))
            dialogue["character_cluster_id"] = selected_candidate["cluster_id"]
            dialogue["character_name"] = selected_candidate["character_name"]
            dialogue["identity_source"] = "gemini_batched_identity_top5"
            dialogue.pop("unknown_reason", None)
            evidence.update(
                status="accepted",
                selected_cluster_id=selected_candidate["cluster_id"],
            )
            dialogue["identity_vlm_top5"] = evidence
            dialogue["vlm_top5"] = evidence
            accepted += 1
            changed += int(
                original_speaker != str(dialogue.get("speaker_instance_id"))
                or original_cluster != selected_candidate["cluster_id"]
            )
        progress.set_postfix(accepted=accepted, errors=failures)
    return {
        "accepted": accepted,
        "changed": changed,
        "failures": failures,
        "identity_batches": len(batches),
    }


def verify_panel_pages_with_vlm(
    pages: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    features: np.ndarray,
    image_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Run sliding-panel semantic batches followed by bounded identity batches."""
    cluster_ids, prototypes, references = build_cluster_identity_bank(
        instances, features
    )
    instance_index = {
        str(row["instance_id"]): index for index, row in enumerate(instances)
    }
    page_limit = args.vlm_max_pages or len(pages)
    selected_pages = pages[: min(page_limit, len(pages))]
    unselected_pages = pages[len(selected_pages) :]
    unreviewed = 0
    for page in unselected_pages:
        for dialogue in page.get("dialogues", []):
            dialogue["page_vlm"] = {"status": "not_reviewed"}
            dialogue["speaker_vlm_top5"] = {
                "status": "fallback",
                "fallback": "v3_geometry",
                "fallback_reason": "--vlm-max-pages was reached",
            }
            dialogue["identity_vlm_top5"] = {"status": "not_run"}
            dialogue["vlm_top5"] = dialogue["identity_vlm_top5"]
            unreviewed += 1
    semantic = run_panel_semantic_pass(
        selected_pages,
        instances,
        instance_index,
        image_dir,
        output_dir,
        args,
    )
    identity = run_batched_identity_pass(
        selected_pages,
        instances,
        features,
        cluster_ids,
        prototypes,
        references,
        instance_index,
        image_dir,
        output_dir,
        args,
    )
    unknown = sum(
        dialogue.get("character_name") == "unknown"
        for page in pages
        for dialogue in page.get("dialogues", [])
    )
    return {
        "reviewed": semantic["reviewed"],
        "reviewed_pages": len(selected_pages),
        "accepted": identity["accepted"],
        "changed": identity["changed"],
        "unknown": int(unknown),
        "filtered_non_dialogue": semantic["filtered_non_dialogue"],
        "failures": semantic["failures"] + identity["failures"],
        "unreviewed": unreviewed,
        "limited": int(bool(unselected_pages)),
        "panel_batches": semantic["panel_batches"],
        "identity_batches": identity["identity_batches"],
    }


def reading_order(
    texts: list[dict[str, Any]], frames: list[Box]
) -> list[dict[str, Any]]:
    """Japanese manga order: panel rows top-down/right-left, text right-left/top-down."""
    if not frames:
        return sorted(
            texts,
            key=lambda row: (
                -as_box("text", row["box"]).cx,
                as_box("text", row["box"]).cy,
            ),
        )

    # Group panels into visually overlapping rows. Irregular panels that span
    # rows keep their own top coordinate, while panels on the same band are
    # ordered from right to left.
    panel_rows: list[dict[str, Any]] = []
    for panel_index in sorted(
        range(len(frames)), key=lambda index: (frames[index].ymin, -frames[index].xmax)
    ):
        panel = frames[panel_index]
        best_row = None
        best_overlap = 0.0
        for row_index, band in enumerate(panel_rows):
            overlap = max(
                0.0, min(panel.ymax, band["ymax"]) - max(panel.ymin, band["ymin"])
            )
            ratio = overlap / max(1.0, min(panel.height, band["ymax"] - band["ymin"]))
            if ratio >= 0.35 and ratio > best_overlap:
                best_row, best_overlap = row_index, ratio
        if best_row is None:
            panel_rows.append(
                {"ymin": panel.ymin, "ymax": panel.ymax, "panels": [panel_index]}
            )
        else:
            band = panel_rows[best_row]
            band["ymin"] = min(band["ymin"], panel.ymin)
            band["ymax"] = max(band["ymax"], panel.ymax)
            band["panels"].append(panel_index)
    panel_rows.sort(key=lambda band: band["ymin"])
    panel_rank: dict[int, int] = {}
    rank = 0
    for band in panel_rows:
        for panel_index in sorted(
            band["panels"], key=lambda index: frames[index].xmax, reverse=True
        ):
            panel_rank[panel_index] = rank
            rank += 1

    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        box = as_box("text", row["box"])
        panel = choose_panel(box, frames)
        if panel is None:
            return (float(len(frames)) + box.cy, -box.cx, box.cy)
        return (float(panel_rank.get(panel, len(frames))), -box.cx, box.cy)

    return sorted(texts, key=key)


def match_dialogues(
    payload: dict[str, Any],
    by_image: dict[str, list[dict[str, Any]]],
    ocr_dir: Path | None,
    ranker: Any,
    top_k: int,
    tail_text_max_distance: float,
    tail_weight: float,
    tail_ray_width: float,
) -> list[dict[str, Any]]:
    pages = []
    for page_index, page in enumerate(payload["images"]):
        instances = by_image.get(page["image"], [])
        bodies = [as_box(row["instance_id"], row["body_box"]) for row in instances]
        frames = [
            as_box(f"frame_{i + 1}", item["box"])
            for i, item in enumerate(page["detections"])
            if item["class_name"] == "frame"
        ]
        text_rows = [
            item for item in page["detections"] if item["class_name"] == "text"
        ]
        text_rows = reading_order(text_rows, frames)
        ocr_lines = load_ocr_lines(ocr_dir, page["image"])
        tail_rows = [
            item for item in page["detections"] if item["class_name"] == "tail"
        ]
        tail_assignments = assign_tails_to_texts(
            text_rows,
            tail_rows,
            frames,
            page["width"],
            page["height"],
            tail_text_max_distance,
        )
        dialogues = []
        if not bodies:
            pages.append(
                {
                    "image": page["image"],
                    "dialogues": [],
                    "warning": "no body candidates",
                    "frame_boxes": [
                        [frame.xmin, frame.ymin, frame.xmax, frame.ymax]
                        for frame in frames
                    ],
                }
            )
            continue
        geometries = (
            np.stack(
                [
                    make_geometry(
                        as_box(f"T{i + 1}", row["box"]),
                        bodies,
                        frames,
                        page["width"],
                        page["height"],
                    )
                    for i, row in enumerate(text_rows)
                ]
            )
            if text_rows
            else np.empty((0, len(bodies), 45), dtype=np.float32)
        )
        texts = [text_for_detection(row["box"], ocr_lines) for row in text_rows]
        scores = (
            ranker.score_page(geometries, texts)
            if len(text_rows)
            else np.empty((0, len(bodies)))
        )
        for text_index, (text_row, text_value, row_scores) in enumerate(
            zip(text_rows, texts, scores)
        ):
            dialogue_index = text_index + 1
            v3_scores = np.asarray(row_scores, dtype=np.float32)
            v3_order = np.argsort(-v3_scores, kind="stable")
            v3_margin = (
                float(row_scores[v3_order[0]] - row_scores[v3_order[1]])
                if len(v3_order) > 1
                else None
            )
            tail_index = tail_assignments.get(text_index)
            if tail_index is not None:
                tail_prior, tail_evidence = tail_ray_prior(
                    text_row["box"],
                    tail_rows[tail_index],
                    instances,
                    frames,
                    page["width"],
                    page["height"],
                    tail_ray_width,
                    v3_scores,
                )
                fused_scores = v3_scores + float(tail_weight) * tail_prior
                order = np.argsort(-fused_scores, kind="stable")
                speaker_source = "v3_tail_fusion"
                tail_evidence["tail_weight"] = float(tail_weight)
                tail_evidence["matched_tail_detection_index"] = int(tail_index)
            else:
                fused_scores = v3_scores
                order = v3_order
                speaker_source = "v3_fallback"
                tail_evidence = None
            shares = softmax(fused_scores)
            order = order[: min(top_k, len(bodies))]
            candidates = []
            for rank, body_index in enumerate(order.tolist(), 1):
                instance = instances[body_index]
                candidates.append(
                    {
                        "rank": rank,
                        "speaker_instance_id": instance["instance_id"],
                        "character_cluster_id": instance["character_cluster_id"],
                        "character_name": instance["character_name"],
                        "v3_score": round(float(v3_scores[body_index]), 6),
                        "fused_score": round(float(fused_scores[body_index]), 6),
                        "softmax_share": round(float(shares[body_index]), 6),
                        "body_box": instance["body_box"],
                    }
                )
            dialogues.append(
                {
                    "dialogue_id": f"P{page_index + 1:03d}_T{dialogue_index:03d}",
                    "display_id": f"D{dialogue_index}",
                    "text_box": text_row["box"],
                    "ocr_text": text_value,
                    "speaker_instance_id": candidates[0]["speaker_instance_id"],
                    "character_cluster_id": candidates[0]["character_cluster_id"],
                    "character_name": candidates[0]["character_name"],
                    "speaker_source": speaker_source,
                    "tail_evidence": tail_evidence,
                    "v3_top1_margin": (
                        round(v3_margin, 6) if v3_margin is not None else None
                    ),
                    "top_candidates": candidates,
                }
            )
        pages.append(
            {
                "image": page["image"],
                "frame_boxes": [
                    [frame.xmin, frame.ymin, frame.xmax, frame.ymax] for frame in frames
                ],
                "magi_tails": tail_rows,
                "matched_tail_indexes": sorted(set(tail_assignments.values())),
                "dialogues": dialogues,
            }
        )
    return pages


def save_cluster_crops(
    instances: list[dict[str, Any]], image_dir: Path, output_dir: Path, crop_size: int
) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        groups[row["character_cluster_id"]].append(row)
    for cluster_id, members in groups.items():
        for index, row in enumerate(members, 1):
            with Image.open(image_dir / row["image"]) as raw:
                crop = (
                    ImageOps.exif_transpose(raw)
                    .convert("RGB")
                    .crop(tuple(row["body_box"]))
                )
            crop.thumbnail((crop_size, crop_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (crop_size, crop_size), "white")
            canvas.paste(
                crop, ((crop_size - crop.width) // 2, (crop_size - crop.height) // 2)
            )
            destination = (
                output_dir
                / "character_clusters"
                / cluster_id
                / f"{index:03d}_{Path(row['image']).stem}.jpg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(destination, quality=92)


def draw_pages(pages: list[dict[str, Any]], image_dir: Path, output_dir: Path) -> None:
    font = ImageFont.load_default(size=16)
    for page in pages:
        with Image.open(image_dir / page["image"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        draw = ImageDraw.Draw(image)
        matched_tail_indexes = set(page.get("matched_tail_indexes", []))
        for index, tail in enumerate(page.get("magi_tails", [])):
            color = (170, 30, 200) if index in matched_tail_indexes else (155, 135, 165)
            draw.rectangle(tail["box"], outline=color, width=3)
            draw.text(
                (tail["box"][0], max(0, tail["box"][1] - 15)),
                f"tail{index}",
                fill=color,
                font=font,
                stroke_width=1,
                stroke_fill="white",
            )
        for dialogue in page.get("dialogues", []):
            text_box = dialogue["text_box"]
            body_box = dialogue.get("speaker_body")
            if body_box is None and dialogue.get("top_candidates"):
                body_box = dialogue["top_candidates"][0]["body_box"]
            draw.rectangle(text_box, outline=(230, 30, 30), width=3)
            if isinstance(body_box, list) and len(body_box) == 4:
                draw.rectangle(body_box, outline=(30, 90, 230), width=3)
                tx, ty = (text_box[0] + text_box[2]) / 2, (
                    text_box[1] + text_box[3]
                ) / 2
                bx, by = (body_box[0] + body_box[2]) / 2, (
                    body_box[1] + body_box[3]
                ) / 2
                draw.line((tx, ty, bx, by), fill=(255, 170, 0), width=3)
            tail = dialogue.get("tail_evidence")
            if tail:
                draw.rectangle(tail["tail_box"], outline=(170, 30, 200), width=3)
                root_x, root_y = tail["estimated_tail_root"]
                tip_x, tip_y = tail["estimated_tail_tip"]
                direction_x, direction_y = tail["ray_direction"]
                ray_length = float(tail["ray_panel_exit"])
                end_x, end_y = (
                    tip_x + direction_x * ray_length,
                    tip_y + direction_y * ray_length,
                )
                draw.line((root_x, root_y, tip_x, tip_y), fill=(230, 30, 40), width=5)
                draw.ellipse(
                    (root_x - 4, root_y - 4, root_x + 4, root_y + 4), fill=(230, 30, 40)
                )
                draw.line((tip_x, tip_y, end_x, end_y), fill=(170, 30, 200), width=4)
                draw.ellipse(
                    (tip_x - 5, tip_y - 5, tip_x + 5, tip_y + 5), fill=(170, 30, 200)
                )
            label = f"{dialogue.get('display_id', dialogue['dialogue_id'])} -> {dialogue['character_name']}"
            draw.text(
                (text_box[0], max(0, text_box[1] - 19)),
                label,
                fill=(180, 0, 0),
                font=font,
                stroke_width=2,
                stroke_fill="white",
            )
        destination = (
            output_dir / "annotated_pages" / f"{Path(page['image']).stem}_linked.jpg"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=94)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.detections.read_text(encoding="utf-8"))
    injected_tail_count = inject_magi_tails(payload, args.magi_dir)
    enriched_detection_path = args.output_dir / "detections_with_magiv3_tails.json"
    enriched_detection_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_instances, by_image = build_instances(payload)
    device = torch.device(args.device)
    embeddings_cache = args.embeddings_cache or (
        args.output_dir / "reid_embeddings.npy"
    )
    all_features, embeddings = load_or_extract_embeddings(
        all_instances,
        args.image_dir,
        args.reid_checkpoint,
        device,
        embeddings_cache,
        args.recompute_embeddings,
    )
    instances = all_instances
    features = all_features
    if args.cluster_requires_face:
        face_indexes = [
            index
            for index, row in enumerate(all_instances)
            if row["face_box"] is not None
        ]
        instances = [all_instances[index] for index in face_indexes]
        if not instances:
            raise SystemExit(
                "No face+body instances available for --cluster-requires-face"
            )
        features = all_features[np.asarray(face_indexes, dtype=np.int64)]
        embeddings = {
            **embeddings,
            "source_instance_count": len(all_instances),
            "selected_instance_count": len(instances),
            "selection": "face+body_only",
        }
        print(
            f"Clustering face+body instances only: {len(instances)}/{len(all_instances)} "
            f"(excluded body-only={len(all_instances) - len(instances)})",
            flush=True,
        )
    labels, clustering = cluster_characters(
        features,
        args.largest_cluster_limit,
        {
            "merge_threshold": args.merge_threshold,
            "assignment_threshold": args.assignment_threshold,
            "assignment_margin": args.assignment_margin,
        },
    )

    names_path = args.output_dir / "character_names.json"
    if names_path.is_file():
        names = json.loads(names_path.read_text(encoding="utf-8"))
    else:
        cluster_ids = [
            f"character_{label:03d}" for label in sorted(set(labels.tolist()))
        ]
        names = {cluster_id: cluster_id for cluster_id in cluster_ids}
        names_path.write_text(
            json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for row, label in zip(instances, labels.tolist()):
        cluster_id = f"character_{label:03d}"
        row["character_cluster_id"] = cluster_id
        row["character_name"] = str(names.get(cluster_id, cluster_id))

    speaker_candidate_count = max(
        args.top_k, args.vlm_speaker_top_k if args.vlm_top5 else args.top_k
    )
    if args.cluster_only:
        pages = [
            {"image": page["image"], "dialogues": []} for page in payload["images"]
        ]
        speaker_model_name = "disabled"
        speaker_protocol = None
    elif args.speaker_model == "geometry":
        ranker = GeometrySpeakerRanker(args.geometry_model)
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
        speaker_model_name = args.speaker_model
        speaker_protocol = FINAL_SPEAKER_PROTOCOL
    else:
        if args.v3_checkpoint is None:
            raise SystemExit("--speaker-model v3 requires --v3-checkpoint")
        ranker = V3SpeakerRanker(args.v3_checkpoint, args.text_model, device)
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
        speaker_model_name = args.speaker_model
        speaker_protocol = FINAL_SPEAKER_PROTOCOL

    vlm_top5 = {
        "enabled": False,
        "reviewed": 0,
        "reviewed_pages": 0,
        "accepted": 0,
        "changed": 0,
        "unknown": 0,
        "failures": 0,
        "unreviewed": 0,
        "filtered_non_dialogue": 0,
        "panel_batches": 0,
        "identity_batches": 0,
        "limited": 0,
    }
    if args.vlm_top5:
        print(
            f"Gemini two-pass page verification: model={args.vlm_model} "
            f"page_limit={args.vlm_max_pages or 'all'} "
            f"first_pass_threshold={args.vlm_first_pass_confidence_threshold:.2f} "
            f"identity_threshold={args.vlm_confidence_threshold:.2f}",
            flush=True,
        )
        vlm_top5 = {
            "enabled": True,
            "model": args.vlm_model,
            "panel_batch_size": args.vlm_panel_batch_size,
            "first_pass_confidence_threshold": (
                args.vlm_first_pass_confidence_threshold
            ),
            "identity_confidence_threshold": args.vlm_confidence_threshold,
            "raw_responses": str(args.output_dir / "vlm_raw_responses.jsonl"),
            **verify_panel_pages_with_vlm(
                pages,
                instances,
                features,
                args.image_dir,
                args.output_dir,
                args,
            ),
        }

    result = {
        "protocol": "unlabeled_new_manga_clustering_then_v3_speaker_top5_gemini_then_reid_top5_gemini",
        "speaker_protocol": speaker_protocol,
        "detections": str(args.detections.resolve()),
        "reid_checkpoint": str(args.reid_checkpoint.resolve()),
        "reid_embeddings": embeddings,
        "speaker_model": speaker_model_name,
        "vlm_top5": vlm_top5,
        "clustering": clustering,
        "summary": {
            "pages": len(payload["images"]),
            "character_instances": len(instances),
            "detected_body_instances": len(all_instances),
            "excluded_body_only_instances": len(all_instances) - len(instances),
            "character_clusters": len(set(labels.tolist())),
            "cluster_only": args.cluster_only,
            "cluster_requires_face": args.cluster_requires_face,
            "dialogues": sum(len(page["dialogues"]) for page in pages),
            "magiv3_tails_injected": injected_tail_count,
            "tail_fused_dialogues": sum(
                row.get("speaker_source") == "v3_tail_fusion"
                for page in pages
                for row in page["dialogues"]
            ),
            "vlm_top5_reviewed": int(vlm_top5["reviewed"]),
            "vlm_pages_reviewed": int(vlm_top5["reviewed_pages"]),
            "vlm_top5_changed": int(vlm_top5["changed"]),
            "vlm_top5_unknown": int(vlm_top5["unknown"]),
            "vlm_filtered_non_dialogue": int(vlm_top5["filtered_non_dialogue"]),
            "vlm_panel_batches": int(vlm_top5["panel_batches"]),
            "vlm_identity_batches": int(vlm_top5["identity_batches"]),
        },
        "character_instances": instances,
        "pages": pages,
    }
    result_path = args.output_dir / "pipeline_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_cluster_crops(instances, args.image_dir, args.output_dir, args.crop_size)
    if not args.cluster_only:
        draw_pages(pages, args.image_dir, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Names: {names_path}")
    print(f"Result: {result_path}")


if __name__ == "__main__":
    main()
