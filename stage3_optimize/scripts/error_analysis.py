# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 按文本长度/emoji/图像尺寸分桶统计错误率并可视化。
# 流程: 读取 val.csv 与 bad_cases.csv，生成分桶统计与图表。
# 输出: outputs/optimize/error_analysis_*.csv 与 .png。

import argparse
import re
from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# emoji 粗略匹配（与文本清洗一致的范围）
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Error analysis by buckets.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--badcase-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _default_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "outputs" / "splits" / "val.csv",
        root / "outputs" / "optimize" / "bad_cases.csv",
        root / "outputs" / "optimize",
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _bucket_text_len(n: int) -> str:
    if n <= 20:
        return "0-20"
    if n <= 50:
        return "21-50"
    if n <= 100:
        return "51-100"
    return "100+"


def _bucket_image_area(area: int, q1: float, q2: float) -> str:
    if area <= q1:
        return "small"
    if area <= q2:
        return "medium"
    return "large"


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(df[x], df[y], color="#4C72B0", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_ylim(0, max(0.05, df[y].max() * 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    default_val, default_bad, default_out = _default_paths(root)
    val_csv = args.val_csv or default_val
    bad_csv = args.badcase_csv or default_bad
    output_dir = args.output_dir or default_out
    output_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(val_csv)
    bad_df = pd.read_csv(bad_csv)
    bad_set = set(bad_df["guid"].astype(str).tolist())

    # 计算文本长度与 emoji 是否存在
    text_lens = []
    emoji_flags = []
    areas = []
    for _, row in val_df.iterrows():
        text = _read_text(Path(row["text_path"]))
        text_lens.append(len(text))
        emoji_flags.append(int(_EMOJI_RE.search(text) is not None))

        try:
            with Image.open(row["image_path"]) as img:
                areas.append(img.width * img.height)
        except Exception:
            areas.append(0)

    val_df = val_df.copy()
    val_df["text_len"] = text_lens
    val_df["has_emoji"] = emoji_flags
    val_df["image_area"] = areas
    val_df["is_error"] = val_df["guid"].astype(str).isin(bad_set).astype(int)

    # 文本长度分桶
    val_df["text_len_bucket"] = val_df["text_len"].apply(_bucket_text_len)
    text_stats = (
        val_df.groupby("text_len_bucket")
        .agg(total=("guid", "count"), errors=("is_error", "sum"))
        .reset_index()
    )
    text_stats["error_rate"] = text_stats["errors"] / text_stats["total"]
    text_stats.to_csv(output_dir / "error_analysis_text_len.csv", index=False)
    _plot_bar(
        text_stats,
        "text_len_bucket",
        "error_rate",
        "Error Rate by Text Length",
        output_dir / "error_analysis_text_len.png",
    )

    # emoji 分桶
    emoji_stats = (
        val_df.groupby("has_emoji")
        .agg(total=("guid", "count"), errors=("is_error", "sum"))
        .reset_index()
    )
    emoji_stats["has_emoji"] = emoji_stats["has_emoji"].map({0: "no_emoji", 1: "has_emoji"})
    emoji_stats["error_rate"] = emoji_stats["errors"] / emoji_stats["total"]
    emoji_stats.to_csv(output_dir / "error_analysis_emoji.csv", index=False)
    _plot_bar(
        emoji_stats,
        "has_emoji",
        "error_rate",
        "Error Rate by Emoji",
        output_dir / "error_analysis_emoji.png",
    )

    # 图像面积分桶（按 1/3、2/3 分位数）
    q1 = val_df["image_area"].quantile(0.33)
    q2 = val_df["image_area"].quantile(0.66)
    val_df["image_bucket"] = val_df["image_area"].apply(lambda x: _bucket_image_area(x, q1, q2))
    img_stats = (
        val_df.groupby("image_bucket")
        .agg(total=("guid", "count"), errors=("is_error", "sum"))
        .reset_index()
    )
    img_stats["error_rate"] = img_stats["errors"] / img_stats["total"]
    img_stats.to_csv(output_dir / "error_analysis_image_size.csv", index=False)
    _plot_bar(
        img_stats,
        "image_bucket",
        "error_rate",
        "Error Rate by Image Size",
        output_dir / "error_analysis_image_size.png",
    )

    print(f"Saved error analysis to: {output_dir}")


if __name__ == "__main__":
    main()
