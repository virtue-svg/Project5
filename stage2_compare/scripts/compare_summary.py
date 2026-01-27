# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 汇总对比实验结果。
# 流程: 读取各 run 的 history.csv，取最佳 F1。
# 输出: outputs/compare/compare_summary.csv。

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Summarize comparison runs.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    # 主流程：汇总并保存
    args = parse_args()
    root = args.project_root.resolve()
    compare_dir = root / "outputs" / "compare"
    output = args.output or (compare_dir / "compare_summary.csv")

    rows = []
    for run_dir in compare_dir.glob("*"):
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "history.csv"
        if not metrics_path.exists() or not history_path.exists():
            continue
        hist = pd.read_csv(history_path)
        best = hist.loc[hist["val_f1_macro"].idxmax()]
        rows.append(
            {
                "run": run_dir.name,
                "val_acc": best["val_acc"],
                "val_f1_macro": best["val_f1_macro"],
                "val_precision_macro": best["val_precision_macro"],
                "val_recall_macro": best["val_recall_macro"],
            }
        )

    if not rows:
        print("No comparison runs found.")
        return

    df = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    compare_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved comparison summary to: {output}")


if __name__ == "__main__":
    main()
