#!/usr/bin/env python3
"""Build Face+Body ReID manifests with the speaker model's exact book split."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    annotation_id: str
    character_id: str
    xyxy: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def contains(box: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def node_box(node: ET.Element) -> Box:
    return Box(
        annotation_id=str(node.attrib["id"]),
        character_id=str(node.attrib["character"]),
        xyxy=tuple(
            int(float(node.attrib[key]))
            for key in ("xmin", "ymin", "xmax", "ymax")
        ),  # type: ignore[arg-type]
    )


def frame_index(
    box: Box, frames: list[tuple[int, int, int, int]]
) -> int | None:
    matches = []
    for index, frame in enumerate(frames):
        if contains(frame, box.center):
            area = max(1, frame[2] - frame[0]) * max(1, frame[3] - frame[1])
            matches.append((area, index))
    return min(matches)[1] if matches else None


def pair_face_body(
    faces: list[Box],
    bodies: list[Box],
    frames: list[tuple[int, int, int, int]],
    max_panel_distance: float,
) -> tuple[list[tuple[Box, Box]], list[Box], list[Box]]:
    """One-to-one spatial pairing; same character ID alone is not sufficient."""
    candidates: list[tuple[tuple[float, ...], int, int]] = []
    for face_index, face in enumerate(faces):
        face_panel = frame_index(face, frames)
        for body_index, body in enumerate(bodies):
            if face.character_id != body.character_id:
                continue
            body_panel = frame_index(body, frames)
            same_panel = face_panel is not None and face_panel == body_panel
            inside = contains(body.xyxy, face.center)
            fx, fy = face.center
            bx, by = body.center
            bw = max(1, body.xyxy[2] - body.xyxy[0])
            bh = max(1, body.xyxy[3] - body.xyxy[1])
            distance = ((fx - bx) ** 2 + (fy - by) ** 2) ** 0.5 / (
                bw * bw + bh * bh
            ) ** 0.5
            if inside or (same_panel and distance <= max_panel_distance):
                candidates.append(
                    ((0.0 if inside else 1.0, 0.0 if same_panel else 1.0, distance),
                     face_index, body_index)
                )
    used_faces: set[int] = set()
    used_bodies: set[int] = set()
    pairs: list[tuple[Box, Box]] = []
    for _, face_index, body_index in sorted(candidates):
        if face_index in used_faces or body_index in used_bodies:
            continue
        used_faces.add(face_index)
        used_bodies.add(body_index)
        pairs.append((faces[face_index], bodies[body_index]))
    return (
        pairs,
        [face for index, face in enumerate(faces) if index not in used_faces],
        [body for index, body in enumerate(bodies) if index not in used_bodies],
    )


def parse_book(
    xml_path: Path, dataset_root: Path, max_panel_distance: float
) -> list[dict[str, object]]:
    root = ET.parse(xml_path).getroot()
    book = str(root.attrib["title"])
    characters = root.find("characters")
    names = {
        str(node.attrib["id"]): str(node.attrib.get("name", ""))
        for node in (characters or [])
    }
    pages = root.find("pages")
    records: list[dict[str, object]] = []
    for page in pages or []:
        page_index = int(page.attrib["index"])
        relative_image = Path("images") / book / f"{page_index:03d}.jpg"
        if not (dataset_root / relative_image).is_file():
            continue
        frames = [
            tuple(
                int(float(node.attrib[key]))
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
            for node in page.findall("frame")
        ]
        faces = [node_box(node) for node in page.findall("face")]
        bodies = [node_box(node) for node in page.findall("body")]
        pairs, face_only, body_only = pair_face_body(
            faces, bodies, frames, max_panel_distance
        )
        instances: list[tuple[Box | None, Box | None]] = [
            (face, body) for face, body in pairs
        ]
        instances.extend((face, None) for face in face_only)
        instances.extend((None, body) for body in body_only)
        for face, body in instances:
            anchor = body or face
            assert anchor is not None
            character_id = anchor.character_id
            records.append(
                {
                    "key": f"{book}/{page_index:03d}/{anchor.annotation_id}",
                    "book": book,
                    "page": page_index,
                    "character_id": character_id,
                    "character_name": names.get(character_id, ""),
                    "identity": f"{book}::{character_id}",
                    "image": relative_image.as_posix(),
                    "width": int(page.attrib["width"]),
                    "height": int(page.attrib["height"]),
                    "body_box": list(body.xyxy) if body else None,
                    "face_box": list(face.xyxy) if face else None,
                    "body_annotation_id": body.annotation_id if body else None,
                    "face_annotation_id": face.annotation_id if face else None,
                    "input_type": (
                        "face+body" if face and body else "face-only" if face else "body-only"
                    ),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reid"))
    parser.add_argument("--max-panel-distance", type=float, default=0.75)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    split_books = json.loads(args.splits_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for split in ("train", "val", "test"):
        records = [
            record
            for book in split_books[split]
            for record in parse_book(
                dataset_root / "annotations" / f"{book}.xml",
                dataset_root,
                args.max_panel_distance,
            )
        ]
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as writer:
            for record in records:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        aligned_candidates = 0
        page_index_path = args.splits_json.parent / f"{split}_pages.jsonl"
        if page_index_path.is_file():
            available = {
                (
                    str(record["book"]),
                    int(record["page"]),
                    str(record["body_annotation_id"]),
                )
                for record in records
                if record.get("body_annotation_id") is not None
            }
            with page_index_path.open("r", encoding="utf-8") as reader:
                for line in reader:
                    page_record = json.loads(line)
                    for candidate_id in page_record["candidate_ids"]:
                        key = (
                            str(page_record["book"]),
                            int(page_record["page_index"]),
                            str(candidate_id),
                        )
                        if key not in available:
                            raise ValueError(
                                f"Speaker candidate missing from ReID manifest: {key}"
                            )
                        aligned_candidates += 1
        summary[split] = {
            "books": len(split_books[split]),
            "instances": len(records),
            "identities": len({str(record["identity"]) for record in records}),
            "paired": sum(record["input_type"] == "face+body" for record in records),
            "face_only": sum(record["input_type"] == "face-only" for record in records),
            "body_only": sum(record["input_type"] == "body-only" for record in records),
            "speaker_candidates_aligned": aligned_candidates,
        }
    metadata = {
        "dataset_root": str(dataset_root),
        "splits_json": str(args.splits_json.resolve()),
        "max_panel_distance": args.max_panel_distance,
        "summary": summary,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
