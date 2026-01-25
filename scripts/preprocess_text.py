from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import build_records
from src.text_utils import clean_text, clean_text_advanced, read_text


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
    parser = argparse.ArgumentParser(description="Clean text and export CSV.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing data/ and outputs/.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with text/image files named by guid.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=None,
        help="Path to train.txt (guid + label).",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Path to test_without_label.txt (guid + null).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Enable light stopword removal in advanced cleaning.",
    )
    return parser.parse_args()


def _process(records: list, split: str, remove_stopwords: bool) -> pd.DataFrame:
    rows = []
    missing = 0
    for rec in records:
        if rec.text_path is None:
            missing += 1
            continue
        raw = read_text(rec.text_path)
        cleaned = clean_text(raw)
        cleaned_adv = clean_text_advanced(
            raw, replace_emoji=True, collapse_repeats=True, remove_stopwords=remove_stopwords
        )
        rows.append(
            {
                "guid": rec.guid,
                "split": split,
                "label": rec.label,
                "text": raw.strip(),
                "text_clean": cleaned,
                "text_clean_adv": cleaned_adv,
                "len_raw": len(raw),
                "len_clean": len(cleaned),
                "len_clean_adv": len(cleaned_adv),
            }
        )
    if missing:
        print(f"[warn] {split}: {missing} missing text files")
    return pd.DataFrame(rows)


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

    train_df = _process(train_records, "train", args.remove_stopwords)
    test_df = _process(test_records, "test", args.remove_stopwords)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "text_cleaned.csv"
    pd.concat([train_df, test_df], ignore_index=True).to_csv(out_path, index=False)
    print(f"Saved cleaned text to: {out_path}")


if __name__ == "__main__":
    main()
