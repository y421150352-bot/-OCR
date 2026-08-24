#!/usr/bin/env python3

import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


WORKSPACE_ROOT = Path(os.environ.get("MANGA_OCR_WORKSPACE", "manga_ppocrv6_dataset"))
DATA = WORKSPACE_ROOT / "component_overlap_analysis.json"

IMAGE_ROOT = Path(os.environ.get("MANGA109_ROOT", "datasets/Manga109s_released_2023_12_07")) / "images"

OUTPUT = WORKSPACE_ROOT / "review"

CATEGORIES = {
    "near_duplicate",
    "near_group_duplicate",
    "partial_overlap",
    "different_overlap",
    "complex_overlap",
    "many_to_many",
}

MARGIN = 80


def clamp(value, low, high):
    return max(low, min(high, value))


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    image_output = OUTPUT / "images"
    image_output.mkdir(parents=True, exist_ok=True)

    rows = []

    components = [
        row
        for row in data["components"]
        if row["category"] in CATEGORIES
    ]

    # 同类别中优先把文本相似度高的放前面。
    components.sort(
        key=lambda row: (
            row["category"],
            -float(row.get("similarity", 0)),
            -float(row.get("max_iom", 0)),
            row["book"],
            row["page"],
        )
    )

    for index, row in enumerate(components, 1):
        book = row["book"]
        page = int(row["page"])

        source = IMAGE_ROOT / book / f"{page:03d}.jpg"

        if not source.is_file():
            print("Missing:", source)
            continue

        with Image.open(source) as raw:
            page_image = ImageOps.exif_transpose(raw).convert("RGB")

        all_boxes = (
            [x["bbox"] for x in row["manga"]]
            + [x["bbox"] for x in row["coo"]]
        )

        xmin = min(b[0] for b in all_boxes)
        ymin = min(b[1] for b in all_boxes)
        xmax = max(b[2] for b in all_boxes)
        ymax = max(b[3] for b in all_boxes)

        crop_left = int(
            clamp(xmin - MARGIN, 0, page_image.width)
        )
        crop_top = int(
            clamp(ymin - MARGIN, 0, page_image.height)
        )
        crop_right = int(
            clamp(xmax + MARGIN, 0, page_image.width)
        )
        crop_bottom = int(
            clamp(ymax + MARGIN, 0, page_image.height)
        )

        crop = page_image.crop(
            (crop_left, crop_top, crop_right, crop_bottom)
        )

        draw = ImageDraw.Draw(crop)

        # Manga109 = red
        for number, item in enumerate(row["manga"], 1):
            x1, y1, x2, y2 = item["bbox"]

            box = [
                x1 - crop_left,
                y1 - crop_top,
                x2 - crop_left,
                y2 - crop_top,
            ]

            draw.rectangle(
                box,
                outline=(230, 30, 30),
                width=4,
            )

            draw.text(
                (box[0] + 3, box[1] + 3),
                f"M{number}",
                fill=(230, 30, 30),
            )

        # COO = blue
        for number, item in enumerate(row["coo"], 1):
            x1, y1, x2, y2 = item["bbox"]

            box = [
                x1 - crop_left,
                y1 - crop_top,
                x2 - crop_left,
                y2 - crop_top,
            ]

            draw.rectangle(
                box,
                outline=(30, 80, 240),
                width=4,
            )

            draw.text(
                (box[0] + 3, box[1] + 18),
                f"C{number}",
                fill=(30, 80, 240),
            )

        filename = (
            f"{index:05d}_"
            f"{row['category']}_"
            f"{book}_{page:03d}.jpg"
        )

        crop.save(
            image_output / filename,
            quality=92,
        )

        manga_text = "<br>".join(
            f"M{i}: <b>{html.escape(str(x['text']))}</b>"
            for i, x in enumerate(row["manga"], 1)
        )

        coo_text = "<br>".join(
            f"C{i}: <b>{html.escape(str(x['text']))}</b>"
            for i, x in enumerate(row["coo"], 1)
        )

        rows.append(f"""
        <tr>
          <td>{index}</td>
          <td>{html.escape(row["category"])}</td>
          <td>{html.escape(book)} p{page}</td>
          <td>{row["max_iom"]}</td>
          <td>{row["similarity"]}</td>
          <td>{manga_text}</td>
          <td>{coo_text}</td>
          <td>
            <img src="images/{filename}" style="max-width:500px">
          </td>
        </tr>
        """)

    page = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Manga109 + COO overlap review</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 20px;
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
        }}
        td, th {{
          border: 1px solid #bbb;
          padding: 8px;
          vertical-align: top;
        }}
        th {{
          position: sticky;
          top: 0;
          background: white;
        }}
        img {{
          display: block;
        }}
        .legend {{
          margin-bottom: 20px;
          font-size: 18px;
        }}
      </style>
    </head>
    <body>

    <h1>Manga109 + COO Overlap Review</h1>

    <div class="legend">
      <span style="color:#e61e1e">红框 = Manga109</span>
      &nbsp;&nbsp;
      <span style="color:#1e50f0">蓝框 = COO</span>
    </div>

    <table>
      <tr>
        <th>#</th>
        <th>Category</th>
        <th>Book/Page</th>
        <th>IoM</th>
        <th>Similarity</th>
        <th>Manga109</th>
        <th>COO</th>
        <th>Image</th>
      </tr>

      {''.join(rows)}

    </table>
    </body>
    </html>
    """

    (OUTPUT / "index.html").write_text(
        page,
        encoding="utf-8",
    )

    print()
    print(f"Review components: {len(rows)}")
    print(f"Saved: {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
