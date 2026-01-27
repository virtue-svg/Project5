from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import pandas as pd
from PIL import Image

def _find_project_root(start: Path) -> Path:
    cur = start
    while True:
        if (cur / 'requirements.txt').exists() or (cur / '.git').exists():
            return cur
        if cur.parent == cur:
            return start
        cur = cur.parent

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import build_records
from src.text_utils import clean_text, read_text


def _first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def _default_data_dir(root: Path) -> Path:
    if (root / "data" / "data").exists():
        return root / "data" / "data"
    if (root / "data" / "project5" / "data").exists():
        return root / "data" / "project5" / "data"
    return root / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset statistics.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing data/ and outputs/.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _image_info(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def _collect(records: list, split: str) -> tuple[pd.DataFrame, dict]:
    label_counter = Counter()
    missing_text = 0
    missing_image = 0
    rows = []

    for rec in records:
        if rec.label:
            label_counter[rec.label] += 1

        if rec.text_path is None or not rec.text_path.exists():
            missing_text += 1
            text_len = None
            clean_len = None
        else:
            raw = read_text(rec.text_path)
            text_len = len(raw)
            clean_len = len(clean_text(raw))

        if rec.image_path is None or not rec.image_path.exists():
            missing_image += 1
            width, height = None, None
        else:
            width, height = _image_info(rec.image_path)

        rows.append(
            {
                "guid": rec.guid,
                "split": split,
                "label": rec.label,
                "text_len": text_len,
                "text_clean_len": clean_len,
                "image_width": width,
                "image_height": height,
            }
        )

    summary = {
        "split": split,
        "count": len(records),
        "labels": dict(label_counter),
        "missing_text": missing_text,
        "missing_image": missing_image,
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    data_dir = args.data_dir or _default_data_dir(root)
    output_dir = args.output_dir or (root / "outputs" / "processed")

    train_file = args.train_file or _first_existing(
        [
            root / "data" / "train.txt",
            root / "data" / "project5" / "train.txt",
            root / "train.txt",
        ]
    )
    test_file = args.test_file or _first_existing(
        [
            root / "data" / "test_without_label.txt",
            root / "data" / "project5" / "test_without_label.txt",
            root / "test_without_label.txt",
        ]
    )

    train_records = build_records(train_file, data_dir, has_label=True)
    test_records = build_records(test_file, data_dir, has_label=False)

    train_df, train_summary = _collect(train_records, "train")
    test_df, test_summary = _collect(test_records, "test")

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "data_stats.csv"
    summary_path = output_dir / "data_stats_summary.json"

    pd.concat([train_df, test_df], ignore_index=True).to_csv(detail_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"train": train_summary, "test": test_summary}, f, ensure_ascii=False, indent=2)

    print(f"Saved detail stats to: {detail_path}")
    print(f"Saved summary stats to: {summary_path}")


if __name__ == "__main__":
    main()
