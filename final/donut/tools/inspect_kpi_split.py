"""Inspect train/test label placement for one Donut-format KPI CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show label counts for possible train/test splits."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Donut-format KPI CSV with timestamp,value,label columns.",
    )
    parser.add_argument(
        "--portions",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated train portions to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    portions = [float(x.strip()) for x in args.portions.split(",") if x.strip()]

    rows = []
    with args.input.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    int(float(row["timestamp"])),
                    float(row["value"]),
                    int(row["label"]),
                )
            )

    total_labels = sum(label for _, _, label in rows)
    print(f"file: {args.input}")
    print(f"points: {len(rows)}")
    print(f"label_points: {total_labels}")
    print("portion,train_points,train_labels,test_points,test_labels")
    for portion in portions:
        train_n = int(len(rows) * portion)
        train_labels = sum(label for _, _, label in rows[:train_n])
        test_labels = sum(label for _, _, label in rows[train_n:])
        print(
            f"{portion:.3f},{train_n},{train_labels},"
            f"{len(rows) - train_n},{test_labels}"
        )


if __name__ == "__main__":
    main()
