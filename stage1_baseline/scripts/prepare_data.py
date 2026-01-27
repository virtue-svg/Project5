# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 生成训练/验证/测试划分文件。
# 流程: 根据 guid 对齐文本/图像路径并分层划分。
# 输出: outputs/splits/train.csv、val.csv、test.csv。

import argparse
from pathlib import Path
import sys

def _find_project_root(start: Path) -> Path:
    # 从当前脚本向上查找项目根目录
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

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_utils import build_records, ensure_paths


def _first_existing(candidates: list[Path]) -> Path:
    # 在候选路径中找到第一个存在的文件
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def _default_data_dir(root: Path) -> Path:
    # 自动识别 data 目录位置
    if (root / "data" / "data").exists():
        return root / "data" / "data"
    if (root / "data" / "project5" / "data").exists():
        return root / "data" / "project5" / "data"
    return root / "data"


def parse_args() -> argparse.Namespace:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits.")
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
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    # 主流程：读取标签文件并生成划分
    args = parse_args()
    root = args.project_root.resolve()

    data_dir = args.data_dir or _default_data_dir(root)
    output_dir = args.output_dir or (root / "outputs" / "splits")

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

    ensure_paths(train_records)
    ensure_paths(test_records)

    train_ids = [r.guid for r in train_records]
    train_idx, val_idx = train_test_split(
        list(range(len(train_ids))),
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=[r.label for r in train_records],
    )

    def to_frame(records: list) -> pd.DataFrame:
        # 生成标准化的 CSV 结构
        return pd.DataFrame(
            {
                "guid": [r.guid for r in records],
                "text_path": [str(r.text_path) for r in records],
                "image_path": [str(r.image_path) for r in records],
                "label": [r.label for r in records],
            }
        )

    train_df = to_frame([train_records[i] for i in train_idx])
    val_df = to_frame([train_records[i] for i in val_idx])
    test_df = to_frame(test_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Saved splits to: {output_dir}")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")


if __name__ == "__main__":
    main()
