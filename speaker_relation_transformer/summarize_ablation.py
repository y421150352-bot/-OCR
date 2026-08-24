from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE = {"top1": 0.752261, "top3": 0.959397, "mrr": 0.856472}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V2 Graph Transformer ablation table from test_metrics.json files."
    )
    parser.add_argument(
        "--full",
        type=Path,
        default=Path("runs/vitb16_bipartite_graph_v2_b4/test_metrics.json"),
    )
    parser.add_argument(
        "--geometry-only",
        type=Path,
        default=Path(
            "runs/vitb16_bipartite_graph_v2_geometry_only_b4/test_metrics.json"
        ),
    )
    parser.add_argument(
        "--visual-only",
        type=Path,
        default=Path("runs/vitb16_bipartite_graph_v2_visual_only_b4/test_metrics.json"),
    )
    parser.add_argument(
        "--no-geometry-bias",
        type=Path,
        default=Path(
            "runs/vitb16_bipartite_graph_v2_no_geometry_bias_b4/test_metrics.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/v2_ablation_table.md"),
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    test = result.get("test")
    if not isinstance(test, dict) or "top1" not in test:
        raise ValueError(f"Missing test metrics in {path}")
    return result


def percent(value: float | None) -> str:
    return "?" if value is None else f"{value * 100:.2f}%"


def signed_percent(value: float | None) -> str:
    return "?" if value is None else f"{value * 100:+.2f} pp"


def metric(result: dict[str, Any] | None, key: str) -> float | None:
    if result is None:
        return None
    return float(result["test"][key])


def main() -> None:
    args = parse_args()
    experiments = [
        ("Graph Transformer", "geometry only", args.geometry_only),
        ("Graph Transformer", "visual only", args.visual_only),
        ("Graph Transformer", "geometry + visual", args.full),
        (
            "Graph Transformer",
            "geometry + visual, without geometry bias",
            args.no_geometry_bias,
        ),
    ]

    rows = [
        (
            "LightGBM",
            "geometry",
            percent(BASELINE["top1"]),
            percent(BASELINE["top3"]),
            percent(BASELINE["mrr"]),
            "+0.00 pp",
            "-",
        )
    ]
    missing: list[Path] = []
    for model_name, inputs, path in experiments:
        result = load_result(path)
        if result is None:
            missing.append(path)
        top1 = metric(result, "top1")
        rows.append(
            (
                model_name,
                inputs,
                percent(top1),
                percent(metric(result, "top3")),
                percent(metric(result, "mrr")),
                signed_percent(None if top1 is None else top1 - BASELINE["top1"]),
                "?" if result is None else str(result.get("best_epoch", "?")),
            )
        )

    lines = [
        "# V2 Graph Transformer ablation",
        "",
        "| Model | Inputs / setting | Top-1 | Top-3 | MRR | Δ Top-1 vs LightGBM | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    if missing:
        lines.extend(["", "Missing result files:"])
        lines.extend(f"- `{path}`" for path in missing)

    report = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
