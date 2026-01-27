# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 对 bad cases 的文本进行轻量增强。
# 流程: emoji 标记、重复字符折叠、可选小写/空白规范化。
# 输出: badcase_text_aug.csv。

import argparse
from pathlib import Path
import sys

import pandas as pd

def _find_project_root(start: Path) -> Path:
    cur = start
    while True:
        if (cur / "requirements.txt").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return start
        cur = cur.parent

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment bad case texts.")
    parser.add_argument("--badcase-csv", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/optimize/badcase_text_aug.csv"),
    )
    parser.add_argument("--replace-emoji", action="store_true", default=True)
    parser.add_argument("--collapse-repeats", action="store_true", default=True)
    parser.add_argument("--lowercase", action="store_true", default=True)
    return parser.parse_args()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.badcase_csv)
    rows = []
    for _, row in df.iterrows():
        text_path = Path(row["text_path"])
        raw = _read_text(text_path)
        text = raw
        if args.lowercase:
            text = text.lower()
        # 轻量清洗：用通用规则处理 emoji / 重复字符
        from src.text_utils import clean_text_advanced

        aug = clean_text_advanced(
            text, replace_emoji=args.replace_emoji, collapse_repeats=args.collapse_repeats
        )
        rows.append(
            {
                "guid": row.get("guid", ""),
                "true_label": row.get("true_label", ""),
                "pred_label": row.get("pred_label", ""),
                "text_path": str(text_path),
                "text_raw": raw.strip(),
                "text_aug": aug,
            }
        )

    out_path = args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved text augmented samples: {out_path}")


if __name__ == "__main__":
    main()
