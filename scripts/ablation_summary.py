from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation results.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _best_from_history(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    best = df.loc[df["val_f1_macro"].idxmax()]
    return {
        "val_acc": best["val_acc"],
        "val_f1_macro": best["val_f1_macro"],
        "val_precision_macro": best["val_precision_macro"],
        "val_recall_macro": best["val_recall_macro"],
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output or (root / "outputs" / "ablation_summary.csv")

    rows = []
    mapping = [
        ("text_only", root / "outputs" / "metrics_text_only.csv"),
        ("image_only", root / "outputs" / "metrics_image_only.csv"),
        ("multimodal", root / "outputs" / "metrics_tfidf_resnet.csv"),
    ]
    for name, path in mapping:
        best = _best_from_history(path)
        if not best:
            continue
        best["model"] = name
        rows.append(best)

    if not rows:
        print("No history files found. Run training first.")
        return

    df = pd.DataFrame(rows)[
        ["model", "val_acc", "val_f1_macro", "val_precision_macro", "val_recall_macro"]
    ]
    df.to_csv(output, index=False)
    print(f"Saved ablation summary to: {output}")


if __name__ == "__main__":
    main()
