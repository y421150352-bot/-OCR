#!/usr/bin/env python3
"""Convert Manga109 XML annotations into paired face/body ReID manifests."""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    annotation_id: str
    kind: str
    character_id: str
    xyxy: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return max(1, x2 - x1) * max(1, y2 - y1)


def contains(box: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def frame_for(box: Box, frames: list[tuple[int, int, int, int]]) -> int | None:
    candidates = [(max(1, f[2] - f[0]) * max(1, f[3] - f[1]), i) for i, f in enumerate(frames) if contains(f, box.center)]
    return min(candidates)[1] if candidates else None


def pair_boxes(
    faces: list[Box],
    bodies: list[Box],
    frames: list[tuple[int, int, int, int]],
    max_panel_distance: float = 0.75,
) -> tuple[list[tuple[Box, Box]], list[Box], list[Box]]:
    """Strict one-to-one matching; uncertain pairs remain single-modality.

    A pair is accepted when the face center is inside the body, or when both
    boxes belong to the same panel and their normalized center distance is no
    greater than ``max_panel_distance``. Same-page/same-ID alone is not enough.
    """
    candidates: list[tuple[tuple[float, ...], int, int]] = []
    for fi, face in enumerate(faces):
        face_frame = frame_for(face, frames)
        for bi, body in enumerate(bodies):
            if face.character_id != body.character_id:
                continue
            body_frame = frame_for(body, frames)
            same_panel = face_frame is not None and face_frame == body_frame
            inside = contains(body.xyxy, face.center)
            fx, fy = face.center
            bx, by = body.center
            bw = max(1, body.xyxy[2] - body.xyxy[0])
            bh = max(1, body.xyxy[3] - body.xyxy[1])
            distance = ((fx - bx) ** 2 + (fy - by) ** 2) ** 0.5 / (bw * bw + bh * bh) ** 0.5
            if not inside and not (same_panel and distance <= max_panel_distance):
                continue
            candidates.append(((0 if inside else 1, 0 if same_panel else 1, distance), fi, bi))
    used_faces: set[int] = set()
    used_bodies: set[int] = set()
    pairs: list[tuple[Box, Box]] = []
    for _, fi, bi in sorted(candidates):
        if fi not in used_faces and bi not in used_bodies:
            used_faces.add(fi)
            used_bodies.add(bi)
            pairs.append((faces[fi], bodies[bi]))
    return pairs, [x for i, x in enumerate(faces) if i not in used_faces], [x for i, x in enumerate(bodies) if i not in used_bodies]


def element_box(node: ET.Element, kind: str) -> Box:
    return Box(
        annotation_id=node.attrib["id"], kind=kind, character_id=node.attrib["character"],
        xyxy=tuple(int(node.attrib[k]) for k in ("xmin", "ymin", "xmax", "ymax")),  # type: ignore[arg-type]
    )


def parse_book(xml_path: Path, dataset_root: Path, max_panel_distance: float) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    book = root.attrib["title"]
    characters_node = root.find("characters")
    pages_node = root.find("pages")
    names = {node.attrib["id"]: node.attrib.get("name", "") for node in characters_node} if characters_node is not None else {}
    records: list[dict] = []
    for page in pages_node if pages_node is not None else []:
        page_index = int(page.attrib["index"])
        image = dataset_root / "images" / book / f"{page_index:03d}.jpg"
        if not image.is_file():
            continue
        frames = [tuple(int(node.attrib[k]) for k in ("xmin", "ymin", "xmax", "ymax")) for node in page.findall("frame")]
        faces = [element_box(node, "face") for node in page.findall("face")]
        bodies = [element_box(node, "body") for node in page.findall("body")]
        pairs, face_only, body_only = pair_boxes(faces, bodies, frames, max_panel_distance)
        instances: list[tuple[Box | None, Box | None]] = [(f, b) for f, b in pairs]
        instances += [(f, None) for f in face_only]
        instances += [(None, b) for b in body_only]
        for face, body in instances:
            character_id = (face or body).character_id  # type: ignore[union-attr]
            records.append({
                "key": f"{book}/{page_index:03d}/{(body or face).annotation_id}",
                "book": book, "page": page_index, "character_id": character_id,
                "character_name": names.get(character_id, ""), "identity": f"{book}::{character_id}",
                "image": image.relative_to(dataset_root).as_posix(),
                "width": int(page.attrib["width"]), "height": int(page.attrib["height"]),
                "body_box": list(body.xyxy) if body else None, "face_box": list(face.xyxy) if face else None,
                "body_annotation_id": body.annotation_id if body else None,
                "face_annotation_id": face.annotation_id if face else None,
                "input_type": "face+body" if face and body else ("face-only" if face else "body-only"),
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset/Manga109s_released_2023_12_07"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-books", type=int, default=70)
    parser.add_argument("--val-books", type=int, default=8)
    parser.add_argument("--max-panel-distance", type=float, default=0.75)
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    xml_dir = args.dataset_root / "annotations"
    books = sorted(path.stem for path in xml_dir.glob("*.xml"))
    random.Random(args.seed).shuffle(books)
    split_books = {"train": books[:args.train_books], "val": books[args.train_books:args.train_books + args.val_books], "test": books[args.train_books + args.val_books:]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for split, selected in split_books.items():
        records = [record for book in selected for record in parse_book(xml_dir / f"{book}.xml", args.dataset_root, args.max_panel_distance)]
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary[split] = {"books": len(selected), "instances": len(records), "identities": len({r["identity"] for r in records})}
    metadata = {
        "dataset_root": str(args.dataset_root), "seed": args.seed,
        "pairing": {"protocol": "inside_or_same_panel_near", "max_panel_distance": args.max_panel_distance},
        "split_books": split_books, "summary": summary,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
