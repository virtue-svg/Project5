from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare optimized CLIP with stage2 baseline CLIP.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--baseline-run", type=str, default="clip_openai_clip-vit-base-patch32")
    parser.add_argument("--opt-run", type=str, default="clip_openai_clip-vit-base-patch32_opt")
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
    out = args.output or (root / "outputs" / "optimize" / "compare_opt_vs_base.csv")

    base_hist = root / "outputs" / "compare" / args.baseline_run / "history.csv"
    opt_hist = root / "outputs" / "optimize" / args.opt_run / "history.csv"

    rows = []
    base = _best_from_history(base_hist)
    if base:
        base["model"] = "clip_baseline"
        rows.append(base)
    opt = _best_from_history(opt_hist)
    if opt:
        opt["model"] = "clip_optimized"
        rows.append(opt)

    if not rows:
        print("No history files found.")
        return

    df = pd.DataFrame(rows)[
        ["model", "val_acc", "val_f1_macro", "val_precision_macro", "val_recall_macro"]
    ]
    df.to_csv(out, index=False)
    print(f"Saved comparison to: {out}")


if __name__ == "__main__":
    main()
