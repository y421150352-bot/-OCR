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
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for module_dir in (ROOT / "speaker_geometry_baseline", ROOT / "speaker_relation_transformer"):
    if str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))

from build_reid_dataset import Box as ReIDBox, pair_boxes
from evaluate_clustering import merge_seed_clusters, normalize, pca_features, raw_hdbscan_labels
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

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) * 0.5

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) * 0.5

    def contains_point(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


def intersection(a: Box, b: Box) -> tuple[float, float, float]:
    width = max(0.0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
    height = max(0.0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
    return width, height, width * height


def choose_panel(box: Box, frames: Sequence[Box]) -> int | None:
    containing = [i for i, frame in enumerate(frames) if frame.contains_point(box.cx, box.cy)]
    if containing:
        return min(containing, key=lambda i: frames[i].area)
    best_index, best_ratio = None, 0.0
    for index, frame in enumerate(frames):
        _, _, inter = intersection(box, frame)
        ratio = inter / max(box.area, 1.0)
        if ratio > best_ratio:
            best_index, best_ratio = index, ratio
    return best_index if best_ratio > 0 else None


def rank_normalized(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    for rank, index in enumerate(order):
        result[index] = rank / denominator
    return result


def pair_features(
    text: Box,
    body: Box,
    page_width: float,
    page_height: float,
    text_panel_index: int | None,
    body_panel_index: int | None,
    frames: Sequence[Box],
    candidate_count: int,
    same_panel_count: int,
    distance_rank: float,
    same_panel_rank: float,
    is_nearest: bool,
    is_nearest_same_panel: bool,
) -> list[float]:
    width, height = max(page_width, 1.0), max(page_height, 1.0)
    page_area, page_diagonal = width * height, math.hypot(width, height)
    dx, dy = (body.cx - text.cx) / width, (body.cy - text.cy) / height
    center_distance = math.hypot(body.cx - text.cx, body.cy - text.cy) / page_diagonal
    gap_x_raw = max(0.0, max(text.xmin, body.xmin) - min(text.xmax, body.xmax))
    gap_y_raw = max(0.0, max(text.ymin, body.ymin) - min(text.ymax, body.ymax))
    gap_x, gap_y = gap_x_raw / width, gap_y_raw / height
    edge_distance = math.hypot(gap_x_raw, gap_y_raw) / page_diagonal
    inter_w, inter_h, inter_area = intersection(text, body)
    union = text.area + body.area - inter_area
    same_panel = text_panel_index is not None and text_panel_index == body_panel_index
    panel_dx = panel_dy = panel_distance = panel_area = 0.0
    if same_panel:
        panel = frames[text_panel_index]
        panel_width, panel_height = max(panel.width, 1.0), max(panel.height, 1.0)
        panel_dx = (body.cx - text.cx) / panel_width
        panel_dy = (body.cy - text.cy) / panel_height
        panel_distance = math.hypot(body.cx - text.cx, body.cy - text.cy) / math.hypot(panel_width, panel_height)
        panel_area = panel.area / page_area
    elif text_panel_index is not None:
        panel_area = frames[text_panel_index].area / page_area
    return [
        text.cx / width, text.cy / height, text.width / width, text.height / height, text.area / page_area,
        body.cx / width, body.cy / height, body.width / width, body.height / height, body.area / page_area,
        dx, dy, abs(dx), abs(dy), center_distance,
        gap_x, gap_y, edge_distance,
        inter_area / max(union, 1.0), inter_area / max(text.area, 1.0), inter_area / max(body.area, 1.0),
        inter_w / max(text.width, 1.0), inter_w / max(body.width, 1.0),
        inter_h / max(text.height, 1.0), inter_h / max(body.height, 1.0),
        math.log(max(body.area, 1.0) / max(text.area, 1.0)),
        float(same_panel), float(text_panel_index is not None), float(body_panel_index is not None),
        panel_dx, panel_dy, panel_distance, panel_area,
        float(candidate_count), float(same_panel_count), distance_rank, same_panel_rank,
        float(is_nearest), float(is_nearest_same_panel),
        float(body.cx < text.cx), float(body.cx > text.cx), float(body.cy < text.cy), float(body.cy > text.cy),
        float(text.contains_point(body.cx, body.cy)), float(body.contains_point(text.cx, text.cy)),
    ]


DEFAULT_CLUSTER_PARAMETERS = {
    "pca_dimensions": 64,
    "min_cluster_size": 3,
    "min_samples": 2,
    "cluster_selection_method": "eom",
    "merge_threshold": 0.84,
    "assignment_threshold": 0.72,
    "assignment_margin": 0.02,
}
FALLBACK_CLUSTER_PARAMETERS = {
    **DEFAULT_CLUSTER_PARAMETERS,
    "min_samples": 3,
    "cluster_selection_method": "leaf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run character clustering and dialogue-speaker matching")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--reid-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/new_manga_pipeline"))
    parser.add_argument("--speaker-model", choices=("geometry", "v3"), default="v3")
    parser.add_argument("--geometry-model", type=Path, default=ROOT / "speaker_geometry_baseline/artifacts/speaker_geometry_lgbm.txt")
    parser.add_argument("--v3-checkpoint", type=Path)
    parser.add_argument("--text-model", type=str, help="Local/Hugging Face path for the V3 text encoder")
    parser.add_argument("--ocr-bundles-dir", type=Path, help="Optional page_bundles directory containing OCR lines")
    parser.add_argument("--magi-dir", type=Path, help="Optional MAGI JSON directory providing tail boxes")
    parser.add_argument("--tail-text-max-distance", type=float, default=0.12)
    parser.add_argument("--tail-weight", type=float, default=6.0, help="Tail-ray alignment weight added to V3 logits")
    parser.add_argument("--tail-ray-width", type=float, default=0.035, help="Ray corridor width normalized by page diagonal")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--largest-cluster-limit", type=float, default=0.55)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def as_box(node_id: str, values: list[float]) -> Box:
    return Box(node_id, *(float(value) for value in values))


def intersect_area(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def load_ocr_lines(bundle_dir: Path | None, image_name: str) -> list[dict[str, Any]]:
    if bundle_dir is None:
        return []
    path = bundle_dir / f"{Path(image_name).stem}.json"
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8-sig")).get("ocr_lines", []))


def text_for_detection(text_box: list[float], lines: list[dict[str, Any]]) -> str:
    selected: list[tuple[float, float, str]] = []
    for line in lines:
        box = [float(value) for value in line.get("box", [])]
        if len(box) != 4:
            continue
        overlap = intersect_area(text_box, box) / max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = text_box[0] <= cx <= text_box[2] and text_box[1] <= cy <= text_box[3]
        if inside or overlap >= 0.35:
            selected.append((box[0], box[1], str(line.get("text_for_review_only", ""))))
    # Japanese manga is commonly vertical and right-to-left.
    selected.sort(key=lambda row: (-row[0], row[1]))
    return "".join(row[2] for row in selected)


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
    """Insert every MAGI tail box into the RT-DETR detection payload in memory."""
    inserted = 0
    for page in payload.get("images", []):
        detections = [
            row for row in page.get("detections", [])
            if not (row.get("class_name") == "tail" and row.get("source") == "magiv3")
        ]
        tails = load_magi_tails(magi_dir, str(page["image"]))
        for index, box in enumerate(tails):
            detections.append({
                "class_id": 4,
                "class_name": "tail",
                "score": 1.0,
                "box": box,
                "area": round(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]), 2),
                "source": "magiv3",
                "magi_tail_index": index,
            })
            inserted += 1
        page["detections"] = detections
        page.setdefault("counts", {})["tail"] = len(tails)
    payload.setdefault("total_counts", {})["tail"] = inserted
    return inserted


def point_to_box_distance(x: float, y: float, box: list[float]) -> float:
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def box_gap_distance(left: list[float], right: list[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def candidate_tail_segments(
    tail_box: list[float], text_box: list[float]
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """Return both box diagonals oriented from the dialogue toward the speaker."""
    text_cx = (text_box[0] + text_box[2]) * 0.5
    text_cy = (text_box[1] + text_box[3]) * 0.5
    tail_cx = (tail_box[0] + tail_box[2]) * 0.5
    tail_cy = (tail_box[1] + tail_box[3]) * 0.5
    outward_x, outward_y = tail_cx - text_cx, tail_cy - text_cy
    outward_norm = max(math.hypot(outward_x, outward_y), 1e-6)
    outward_x, outward_y = outward_x / outward_norm, outward_y / outward_norm
    top_left = (tail_box[0], tail_box[1])
    top_right = (tail_box[2], tail_box[1])
    bottom_left = (tail_box[0], tail_box[3])
    bottom_right = (tail_box[2], tail_box[3])
    diagonals = ((top_left, bottom_right), (top_right, bottom_left))

    candidates: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for first, second in diagonals:
        # Orient each diagonal so root is nearer the text and tip is farther.
        first_distance = math.hypot(first[0] - text_cx, first[1] - text_cy)
        second_distance = math.hypot(second[0] - text_cx, second[1] - text_cy)
        root, tip = (first, second) if first_distance <= second_distance else (second, first)
        diagonal_x, diagonal_y = tip[0] - root[0], tip[1] - root[1]
        diagonal_norm = max(math.hypot(diagonal_x, diagonal_y), 1e-6)
        alignment = (diagonal_x / diagonal_norm) * outward_x + (diagonal_y / diagonal_norm) * outward_y
        candidates.append((root, tip, float(alignment)))
    return candidates


def estimated_tail_segment(
    tail_box: list[float], text_box: list[float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    root, tip, _ = max(candidate_tail_segments(tail_box, text_box), key=lambda row: row[2])
    return root, tip


def estimated_tail_tip(tail_box: list[float], text_box: list[float]) -> tuple[float, float]:
    return estimated_tail_segment(tail_box, text_box)[1]


def assign_tails_to_texts(
    text_rows: list[dict[str, Any]],
    tail_rows: list[dict[str, Any]],
    frames: list[Box],
    page_width: float,
    page_height: float,
    text_max_distance: float,
) -> dict[int, int]:
    """Greedy minimum-distance one-to-one text-tail matching within panels."""
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
    """Return the first non-negative ray parameter entering an axis-aligned box."""
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
    """Return forward ray entry/exit parameters through an axis-aligned box."""
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
    """Distance along a unit ray from an inside point to the box boundary."""
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
    """Score candidates by intersection/alignment with the outward tail ray."""
    page_diagonal = max(math.hypot(page_width, page_height), 1.0)
    tail_box = tail_row["box"]
    (root_x, root_y), (tip_x, tip_y) = root, tip
    direction_x, direction_y = tip_x - root_x, tip_y - root_y
    direction_norm = max(math.hypot(direction_x, direction_y), 1e-6)
    direction_x, direction_y = direction_x / direction_norm, direction_y / direction_norm
    text_panel = choose_panel(as_box("text", text_box), frames)
    tail_panel = choose_panel(as_box("tail", tail_box), frames)
    if text_panel is None or tail_panel is None or text_panel != tail_panel:
        raise ValueError("Strict tail-ray fusion requires text and tail in the same detected panel")
    panel = frames[text_panel]
    panel_box = [panel.xmin, panel.ymin, panel.xmax, panel.ymax]
    panel_exit = ray_box_exit_parameter(tip_x, tip_y, direction_x, direction_y, panel_box)
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
        body_panel = choose_panel(as_box(instance["instance_id"], instance["body_box"]), frames)
        # Strict panel boundary: unknown-panel candidates and candidates in a
        # neighboring panel receive no tail-ray score.
        if body_panel != text_panel:
            continue
        face_box = instance.get("face_box")
        body_box = instance["body_box"]
        # Face is stronger evidence, while body remains available when the
        # face is missing or the ray only enters the body silhouette.
        face_interval = (
            ray_box_interval(tip_x, tip_y, direction_x, direction_y, face_box)
            if face_box is not None else None
        )
        body_interval = ray_box_interval(tip_x, tip_y, direction_x, direction_y, body_box)
        face_valid_hit = face_interval is not None and face_interval[0] <= panel_exit
        body_valid_hit = body_interval is not None and body_interval[0] <= panel_exit
        target = face_box if face_valid_hit else body_box
        interval = face_interval if face_valid_hit else body_interval if body_valid_hit else None
        center_x = (target[0] + target[2]) * 0.5
        center_y = (target[1] + target[3]) * 0.5
        vector_x, vector_y = center_x - tip_x, center_y - tip_y
        projection = vector_x * direction_x + vector_y * direction_y
        perpendicular = abs(vector_x * direction_y - vector_y * direction_x)
        half_diagonal = 0.5 * math.hypot(target[2] - target[0], target[3] - target[1])
        edge_distance = max(0.0, perpendicular - half_diagonal)
        projections[body_index] = projection / page_diagonal
        perpendicular_distances[body_index] = edge_distance / page_diagonal
        target_half_diagonal = max(0.5 * math.hypot(target[2] - target[0], target[3] - target[1]), 1.0)
        center_score = math.exp(-0.5 * (perpendicular / (0.45 * target_half_diagonal)) ** 2)
        center_scores[body_index] = center_score
        if interval is not None:
            hit_enter, hit_exit = interval
            crossing = min(max(0.0, min(hit_exit, panel_exit) - hit_enter) / (2.0 * target_half_diagonal), 1.0)
            crossing_scores[body_index] = crossing
            # Exact face hits dominate body hits. Within the same class, a ray
            # through the center, a longer chord, and the first hit all score
            # higher than a grazing or distant intersection.
            hit_base = 1.25 if face_valid_hit else 0.78
            first_hit_bonus = 0.12 * (1.0 - min(hit_enter / max(panel_exit, 1.0), 1.0))
            quality = hit_base + 0.42 * center_score + 0.16 * crossing + first_hit_bonus
            priors[body_index] = quality
            hit_qualities[body_index] = quality
            ray_hits[body_index] = True
            face_hits[body_index] = bool(face_valid_hit)
            body_hits[body_index] = bool(body_valid_hit)
            hit_parameters[body_index] = round(float(hit_enter / page_diagonal), 6)
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
        "face_hits": face_hits,
        "body_hits": body_hits,
        "ray_hit_parameters": hit_parameters,
        "candidate_center_scores": [round(float(value), 6) for value in center_scores],
        "candidate_crossing_scores": [round(float(value), 6) for value in crossing_scores],
        "candidate_hit_qualities": [round(float(value), 6) for value in hit_qualities],
        "candidate_ray_projections": [round(float(value), 6) for value in projections],
        "candidate_perpendicular_distances": [round(float(value), 6) for value in perpendicular_distances],
        "tail_ray_prior": [round(float(value), 6) for value in priors],
        "diagonal_alignment": round(float(diagonal_alignment), 6),
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
    """Evaluate both box diagonals, then select using ray evidence and V3."""
    candidates = []
    for diagonal_index, (root, tip, alignment) in enumerate(
        candidate_tail_segments(tail_row["box"], text_box)
    ):
        prior, evidence = one_tail_ray_prior(
            text_box, tail_row, root, tip, alignment, instances, frames,
            page_width, page_height, ray_width,
        )
        hits = int(sum(evidence["ray_hits"]))
        face_hits = int(sum(evidence["face_hits"]))
        max_hit_quality = max(evidence["candidate_hit_qualities"], default=0.0)
        positive_hit_parameters = [
            value for value in evidence["ray_hit_parameters"] if value is not None
        ]
        first_hit_parameter = min(positive_hit_parameters, default=1.0)
        if v3_scores is None:
            joint = float(prior.max())
        else:
            normalized_v3 = softmax(np.asarray(v3_scores, dtype=np.float64))
            # Choose the diagonal whose ray supports the most plausible V3
            # candidate. Exact intersection takes precedence over near misses.
            joint = float(np.max(prior * (0.5 + normalized_v3)))
        # Tail geometry decides first. V3 is only a tie-breaker after face/body
        # hit class, center/crossing quality, and first-hit order.
        selection_key = (
            face_hits > 0,
            max_hit_quality,
            hits > 0,
            -first_hit_parameter,
            joint,
            alignment,
        )
        evidence["diagonal_index"] = diagonal_index
        evidence["diagonal_selection_score"] = round(joint, 6)
        candidates.append((selection_key, prior, evidence))
    _, selected_prior, selected_evidence = max(candidates, key=lambda row: row[0])
    selected_evidence["diagonal_candidates"] = [
        {
            "diagonal_index": evidence["diagonal_index"],
            "root": evidence["estimated_tail_root"],
            "tip": evidence["estimated_tail_tip"],
            "ray_direction": evidence["ray_direction"],
            "ray_hits": evidence["ray_hits"],
            "face_hits": evidence["face_hits"],
            "body_hits": evidence["body_hits"],
            "center_scores": evidence["candidate_center_scores"],
            "hit_qualities": evidence["candidate_hit_qualities"],
            "selection_score": evidence["diagonal_selection_score"],
        }
        for _, _, evidence in candidates
    ]
    return selected_prior, selected_evidence


def build_instances(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    all_instances: list[dict[str, Any]] = []
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page_index, page in enumerate(payload["images"]):
        detections = page["detections"]
        faces = [
            ReIDBox(f"F{index + 1}", "face", "predicted", tuple(item["box"]))
            for index, item in enumerate(d for d in detections if d["class_name"] == "face")
        ]
        bodies = [
            ReIDBox(f"B{index + 1}", "body", "predicted", tuple(item["box"]))
            for index, item in enumerate(d for d in detections if d["class_name"] == "body")
        ]
        frames = [tuple(item["box"]) for item in detections if item["class_name"] == "frame"]
        pairs, _, body_only = pair_boxes(faces, bodies, frames)
        rows = [(face, body) for face, body in pairs] + [(None, body) for body in body_only]
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


def extract_embeddings(instances: list[dict[str, Any]], image_dir: Path, checkpoint: Path, device: torch.device) -> np.ndarray:
    model = load_model(checkpoint, device)
    vectors: list[np.ndarray] = []
    current_name = None
    image = None
    for index, record in enumerate(instances, 1):
        if record["image"] != current_name:
            if image is not None:
                image.close()
            image = ImageOps.exif_transpose(Image.open(image_dir / record["image"])).convert("RGB")
            current_name = record["image"]
        vectors.append(embed_instance(model, image, record["face_box"], record["body_box"], device))
        if index % 25 == 0 or index == len(instances):
            print(f"ReID embeddings: {index}/{len(instances)}", flush=True)
    if image is not None:
        image.close()
    if not vectors:
        raise SystemExit("RT-DETR did not produce any body instances")
    return normalize(np.asarray(vectors, dtype=np.float32))


def cluster_once(features: np.ndarray, parameters: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
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


def cluster_characters(features: np.ndarray, largest_limit: float) -> tuple[np.ndarray, dict[str, Any]]:
    attempts = []
    selected = None
    for rank, parameters in enumerate((DEFAULT_CLUSTER_PARAMETERS, FALLBACK_CLUSTER_PARAMETERS), 1):
        labels, diagnostics = cluster_once(features, parameters)
        attempts.append({"rank": rank, "parameters": parameters, "diagnostics": diagnostics})
        if selected is None or diagnostics["largest_cluster_ratio"] < selected[2]["largest_cluster_ratio"]:
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


def make_geometry(text: Box, bodies: list[Box], frames: list[Box], width: float, height: float) -> np.ndarray:
    text_panel = choose_panel(text, frames)
    body_panels = [choose_panel(body, frames) for body in bodies]
    distances = [math.hypot(body.cx - text.cx, body.cy - text.cy) for body in bodies]
    distance_ranks = rank_normalized(distances)
    same_indices = [i for i, panel in enumerate(body_panels) if text_panel is not None and panel == text_panel]
    same_values = rank_normalized([distances[i] for i in same_indices]) if same_indices else []
    same_ranks = {index: same_values[position] for position, index in enumerate(same_indices)}
    nearest = min(range(len(bodies)), key=lambda i: distances[i])
    nearest_same = min(same_indices, key=lambda i: distances[i]) if same_indices else None
    return np.asarray([
        pair_features(
            text, body, width, height, text_panel, body_panels[index], frames,
            len(bodies), len(same_indices), distance_ranks[index], same_ranks.get(index, 1.0),
            index == nearest, index == nearest_same,
        )
        for index, body in enumerate(bodies)
    ], dtype=np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    weights = np.exp(values - float(values.max()))
    return weights / max(float(weights.sum()), 1e-12)


class GeometrySpeakerRanker:
    def __init__(self, path: Path):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=str(path))
        if self.model.num_feature() != 45:
            raise ValueError(f"Expected a 45D geometry model, got {self.model.num_feature()} features")

    def score_page(self, geometry: np.ndarray, texts: list[str]) -> np.ndarray:
        dialogues, candidates = geometry.shape[:2]
        return np.asarray(self.model.predict(geometry.reshape(-1, 45)), dtype=np.float32).reshape(dialogues, candidates)


class V3SpeakerRanker:
    def __init__(self, checkpoint_path: Path, text_model_path: str | None, device: torch.device):
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
            use_dialogue_graph=str(config.get("v3_graph_mode", "two_axis")) == "two_axis",
        ).to(device)
        self.model.load_state_dict(state)
        self.model.eval()
        metadata = config.get("text_cache") or {}
        model_name_raw = text_model_path or metadata.get("model_name")
        if not model_name_raw:
            raise ValueError("V3 needs --text-model because its checkpoint does not record text_cache.model_name")
        model_name = self.resolve_local_text_model(str(model_name_raw), checkpoint_path)
        self.prefix = str(metadata.get("prefix") or "passage: ")
        self.max_length = int(metadata.get("max_length") or 128)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_name), local_files_only=True)
        self.encoder = AutoModel.from_pretrained(str(model_name), local_files_only=True).to(device).eval()
        self.device = device

    @staticmethod
    def resolve_local_text_model(value: str, checkpoint_path: Path) -> Path:
        requested = Path(value).expanduser()
        candidates = [requested]
        if not requested.is_absolute():
            candidates.extend([
                Path.cwd() / requested,
                checkpoint_path.resolve().parents[2] / requested,
                checkpoint_path.resolve().parents[2] / "pretrained" / requested.name,
                Path.cwd().parent / "speaker_relation_transformer" / requested,
                Path.cwd().parent
                / "speaker_relation_transformer"
                / "pretrained"
                / requested.name,
            ])
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
            [self.prefix + text for text in texts], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
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
            rows = self.model.forward_page(torch.from_numpy(geometry).to(self.device), context, mask)
        return torch.stack(rows).float().cpu().numpy()


def reading_order(texts: list[dict[str, Any]], frames: list[Box]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        box = as_box("text", row["box"])
        panel = choose_panel(box, frames)
        panel_y = frames[panel].ymin if panel is not None else box.ymin
        panel_x = frames[panel].xmax if panel is not None else box.xmax
        return (panel_y, -panel_x, -box.cx + box.cy * 1e-3)
    return sorted(texts, key=key)


def match_dialogues(
    payload: dict[str, Any], by_image: dict[str, list[dict[str, Any]]],
    ocr_dir: Path | None, ranker: Any, top_k: int,
    tail_text_max_distance: float, tail_weight: float, tail_ray_width: float,
) -> list[dict[str, Any]]:
    pages = []
    for page_index, page in enumerate(payload["images"]):
        instances = by_image.get(page["image"], [])
        bodies = [as_box(row["instance_id"], row["body_box"]) for row in instances]
        frames = [as_box(f"frame_{i + 1}", item["box"]) for i, item in enumerate(page["detections"]) if item["class_name"] == "frame"]
        text_rows = [item for item in page["detections"] if item["class_name"] == "text"]
        text_rows = reading_order(text_rows, frames)
        ocr_lines = load_ocr_lines(ocr_dir, page["image"])
        tail_rows = [item for item in page["detections"] if item["class_name"] == "tail"]
        tail_assignments = assign_tails_to_texts(
            text_rows, tail_rows, frames, page["width"], page["height"], tail_text_max_distance,
        )
        dialogues = []
        if not bodies:
            pages.append({"image": page["image"], "dialogues": [], "warning": "no body candidates"})
            continue
        geometries = np.stack([
            make_geometry(as_box(f"T{i + 1}", row["box"]), bodies, frames, page["width"], page["height"])
            for i, row in enumerate(text_rows)
        ]) if text_rows else np.empty((0, len(bodies), 45), dtype=np.float32)
        texts = [text_for_detection(row["box"], ocr_lines) for row in text_rows]
        scores = ranker.score_page(geometries, texts) if len(text_rows) else np.empty((0, len(bodies)))
        for text_index, (text_row, text_value, row_scores) in enumerate(zip(text_rows, texts, scores)):
            dialogue_index = text_index + 1
            v3_scores = np.asarray(row_scores, dtype=np.float32)
            v3_order = np.argsort(-v3_scores, kind="stable")
            v3_margin = float(row_scores[v3_order[0]] - row_scores[v3_order[1]]) if len(v3_order) > 1 else None
            tail_index = tail_assignments.get(text_index)
            if tail_index is not None:
                tail_prior, tail_evidence = tail_ray_prior(
                    text_row["box"], tail_rows[tail_index], instances, frames,
                    page["width"], page["height"], tail_ray_width, v3_scores,
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
            order = order[:min(top_k, len(bodies))]
            candidates = []
            for rank, body_index in enumerate(order.tolist(), 1):
                instance = instances[body_index]
                candidates.append({
                    "rank": rank,
                    "speaker_instance_id": instance["instance_id"],
                    "character_cluster_id": instance["character_cluster_id"],
                    "character_name": instance["character_name"],
                    "v3_score": round(float(v3_scores[body_index]), 6),
                    "fused_score": round(float(fused_scores[body_index]), 6),
                    "softmax_share": round(float(shares[body_index]), 6),
                    "body_box": instance["body_box"],
                })
            dialogues.append({
                "dialogue_id": f"P{page_index + 1:03d}_T{dialogue_index:03d}",
                "text_box": text_row["box"],
                "ocr_text": text_value,
                "speaker_instance_id": candidates[0]["speaker_instance_id"],
                "character_cluster_id": candidates[0]["character_cluster_id"],
                "character_name": candidates[0]["character_name"],
                "speaker_source": speaker_source,
                "tail_evidence": tail_evidence,
                "v3_top1_margin": round(v3_margin, 6) if v3_margin is not None else None,
                "top_candidates": candidates,
            })
        pages.append({
            "image": page["image"],
            "magi_tails": tail_rows,
            "matched_tail_indexes": sorted(set(tail_assignments.values())),
            "dialogues": dialogues,
        })
    return pages


def save_cluster_crops(instances: list[dict[str, Any]], image_dir: Path, output_dir: Path, crop_size: int) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        groups[row["character_cluster_id"]].append(row)
    for cluster_id, members in groups.items():
        for index, row in enumerate(members, 1):
            with Image.open(image_dir / row["image"]) as raw:
                crop = ImageOps.exif_transpose(raw).convert("RGB").crop(tuple(row["body_box"]))
            crop.thumbnail((crop_size, crop_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (crop_size, crop_size), "white")
            canvas.paste(crop, ((crop_size - crop.width) // 2, (crop_size - crop.height) // 2))
            destination = output_dir / "character_clusters" / cluster_id / f"{index:03d}_{Path(row['image']).stem}.jpg"
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
                (tail["box"][0], max(0, tail["box"][1] - 15)), f"tail{index}",
                fill=color, font=font, stroke_width=1, stroke_fill="white",
            )
        for dialogue in page.get("dialogues", []):
            text_box = dialogue["text_box"]
            body_box = dialogue["top_candidates"][0]["body_box"]
            draw.rectangle(text_box, outline=(230, 30, 30), width=3)
            draw.rectangle(body_box, outline=(30, 90, 230), width=3)
            tx, ty = (text_box[0] + text_box[2]) / 2, (text_box[1] + text_box[3]) / 2
            bx, by = (body_box[0] + body_box[2]) / 2, (body_box[1] + body_box[3]) / 2
            draw.line((tx, ty, bx, by), fill=(255, 170, 0), width=3)
            tail = dialogue.get("tail_evidence")
            if tail:
                draw.rectangle(tail["tail_box"], outline=(170, 30, 200), width=3)
                root_x, root_y = tail["estimated_tail_root"]
                tip_x, tip_y = tail["estimated_tail_tip"]
                direction_x, direction_y = tail["ray_direction"]
                ray_length = float(tail["ray_panel_exit"])
                end_x, end_y = tip_x + direction_x * ray_length, tip_y + direction_y * ray_length
                draw.line((root_x, root_y, tip_x, tip_y), fill=(230, 30, 40), width=5)
                draw.ellipse((root_x - 4, root_y - 4, root_x + 4, root_y + 4), fill=(230, 30, 40))
                draw.line((tip_x, tip_y, end_x, end_y), fill=(170, 30, 200), width=4)
                draw.ellipse((tip_x - 5, tip_y - 5, tip_x + 5, tip_y + 5), fill=(170, 30, 200))
            label = f"{dialogue['dialogue_id']} -> {dialogue['character_name']}"
            draw.text((text_box[0], max(0, text_box[1] - 19)), label, fill=(180, 0, 0), font=font, stroke_width=2, stroke_fill="white")
        destination = output_dir / "annotated_pages" / f"{Path(page['image']).stem}_linked.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=94)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.detections.read_text(encoding="utf-8"))
    injected_tail_count = inject_magi_tails(payload, args.magi_dir)
    enriched_detection_path = args.output_dir / "detections_with_magiv3_tails.json"
    enriched_detection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    instances, by_image = build_instances(payload)
    device = torch.device(args.device)
    features = extract_embeddings(instances, args.image_dir, args.reid_checkpoint, device)
    labels, clustering = cluster_characters(features, args.largest_cluster_limit)

    names_path = args.output_dir / "character_names.json"
    if names_path.is_file():
        names = json.loads(names_path.read_text(encoding="utf-8"))
    else:
        cluster_ids = [f"character_{label:03d}" for label in sorted(set(labels.tolist()))]
        names = {cluster_id: cluster_id for cluster_id in cluster_ids}
        names_path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    for row, label in zip(instances, labels.tolist()):
        cluster_id = f"character_{label:03d}"
        row["character_cluster_id"] = cluster_id
        row["character_name"] = str(names.get(cluster_id, cluster_id))

    if args.speaker_model == "geometry":
        ranker = GeometrySpeakerRanker(args.geometry_model)
    else:
        if args.v3_checkpoint is None:
            raise SystemExit("--speaker-model v3 requires --v3-checkpoint")
        ranker = V3SpeakerRanker(args.v3_checkpoint, args.text_model, device)
    pages = match_dialogues(
        payload, by_image, args.ocr_bundles_dir, ranker, args.top_k,
        args.tail_text_max_distance, args.tail_weight, args.tail_ray_width,
    )

    result = {
        "protocol": "unlabeled_new_manga_book_level_clustering_then_dialogue_top1_speaker",
        "detections": str(args.detections.resolve()),
        "reid_checkpoint": str(args.reid_checkpoint.resolve()),
        "speaker_model": args.speaker_model,
        "clustering": clustering,
        "summary": {
            "pages": len(payload["images"]),
            "character_instances": len(instances),
            "character_clusters": len(set(labels.tolist())),
            "dialogues": sum(len(page["dialogues"]) for page in pages),
            "magiv3_tails_injected": injected_tail_count,
            "tail_fused_dialogues": sum(
                row.get("speaker_source") == "v3_tail_fusion"
                for page in pages for row in page["dialogues"]
            ),
        },
        "character_instances": instances,
        "pages": pages,
    }
    result_path = args.output_dir / "pipeline_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_cluster_crops(instances, args.image_dir, args.output_dir, args.crop_size)
    draw_pages(pages, args.image_dir, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Names: {names_path}")
    print(f"Result: {result_path}")


if __name__ == "__main__":
    main()
