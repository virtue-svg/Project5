# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge bad case augmentations into training CSV.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("outputs/splits/train.csv"),
    )
    parser.add_argument(
        "--badcase-csv",
        type=Path,
        default=Path("outputs/optimize/bad_cases.csv"),
    )
    parser.add_argument(
        "--badcase-image-csv",
        type=Path,
        default=Path("outputs/optimize/badcase_aug/badcase_augmented.csv"),
    )
    parser.add_argument(
        "--badcase-text-csv",
        type=Path,
        default=Path("outputs/optimize/badcase_text_aug.csv"),
    )
    parser.add_argument(
        "--text-output-dir",
        type=Path,
        default=Path("outputs/optimize/badcase_text_aug"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/optimize/train_aug.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    bad_df = pd.read_csv(args.badcase_csv)

    guid_to_text = dict(zip(bad_df["guid"], bad_df["text_path"]))
    guid_to_image = dict(zip(bad_df["guid"], bad_df["image_path"]))
    guid_to_label = dict(zip(bad_df["guid"], bad_df["true_label"]))

    new_rows = []

    if args.badcase_image_csv.exists():
        img_df = pd.read_csv(args.badcase_image_csv)
        for idx, row in img_df.iterrows():
            guid = row.get("guid", "")
            if guid not in guid_to_label:
                continue
            new_rows.append(
                {
                    "guid": f"{guid}_ia{idx}",
                    "text_path": guid_to_text.get(guid, ""),
                    "image_path": row["image_path"],
                    "label": guid_to_label.get(guid, ""),
                }
            )

    if args.badcase_text_csv.exists():
        txt_df = pd.read_csv(args.badcase_text_csv)
        out_dir = args.text_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in txt_df.iterrows():
            guid = row.get("guid", "")
            if guid not in guid_to_label:
                continue
            text_aug = str(row.get("text_aug", "")).strip()
            if not text_aug:
                continue
            out_path = out_dir / f"{guid}_ta{idx}.txt"
            out_path.write_text(text_aug, encoding="utf-8")
            new_rows.append(
                {
                    "guid": f"{guid}_ta{idx}",
                    "text_path": str(out_path),
                    "image_path": guid_to_image.get(guid, ""),
                    "label": guid_to_label.get(guid, ""),
                }
            )

    if new_rows:
        aug_df = pd.DataFrame(new_rows)
        merged = pd.concat([train_df, aug_df], ignore_index=True)
    else:
        merged = train_df.copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Saved merged training CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
