from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Prefer common Chinese-capable fonts to avoid missing glyph warnings.
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize comparison results.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_runs(compare_dir: Path) -> list[dict]:
    runs = []
    for run_dir in compare_dir.glob("*"):
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "history.csv"
        if not metrics_path.exists() or not history_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = pd.read_csv(history_path)
        runs.append(
            {
                "name": run_dir.name,
                "metrics": metrics,
                "history": history,
            }
        )
    return runs


def _radar_plot(runs: list[dict], out_path: Path) -> None:
    metrics_keys = ["val_acc", "val_f1_macro", "val_precision_macro", "val_recall_macro"]
    labels = ["Acc", "F1", "Prec", "Recall"]
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(111, polar=True)

    max_vals = np.zeros(len(metrics_keys))
    for run in runs:
        hist = run["history"]
        best = hist.loc[hist["val_f1_macro"].idxmax()]
        vals = np.array([best[k] for k in metrics_keys])
        max_vals = np.maximum(max_vals, vals)

    angles = np.linspace(0, 2 * np.pi, len(metrics_keys), endpoint=False).tolist()
    angles += angles[:1]

    for run in runs:
        hist = run["history"]
        best = hist.loc[hist["val_f1_macro"].idxmax()]
        vals = np.array([best[k] for k in metrics_keys])
        norm = vals / np.maximum(max_vals, 1e-6)
        data = norm.tolist()
        data += data[:1]
        ax.plot(angles, data, linewidth=1.5, label=run["name"])
        ax.fill(angles, data, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Normalized Comprehensive Performance", pad=12)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _params_time_plot(runs: list[dict], out_path: Path) -> None:
    names = [r["name"] for r in runs]
    params = [r["metrics"].get("params_million", 0) for r in runs]
    times = [r["metrics"].get("train_time_sec", 0) / 60.0 for r in runs]

    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    bars = ax1.bar(x, params, color="#4C72B0", alpha=0.85, label="Params (M)")
    ax1.set_ylabel("Parameters (M)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, times, color="#DD8452", marker="o", label="Time (min)")
    ax2.set_ylabel("Training Time (min)")

    for rect in bars:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, height + 0.02, f"{height:.2f}",
                 ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _loss_curves(runs: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for run in runs:
        hist = run["history"]
        ax.plot(hist["epoch"], hist["train_loss"], label=f"{run['name']}")
        ax.scatter(
            [hist["epoch"].iloc[-1]],
            [hist["train_loss"].iloc[-1]],
            s=18,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss Convergence")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _table_image(runs: list[dict], out_path: Path) -> None:
    rows = []
    for run in runs:
        hist = run["history"]
        best = hist.loc[hist["val_f1_macro"].idxmax()]
        rows.append(
            [
                run["name"],
                f"{run['metrics'].get('params_million', 0):.2f}",
                f"{best['val_acc']:.4f}",
                f"{best['val_f1_macro']:.4f}",
                f"{best['val_precision_macro']:.4f}",
                f"{best['val_recall_macro']:.4f}",
                f"{run['metrics'].get('train_time_sec', 0)/60.0:.2f}",
                f"{run['metrics'].get('epochs', 0)}",
            ]
        )

    cols = ["模型", "参数量(M)", "Acc", "F1", "Prec", "Recall", "训练时长(min)", "训练轮数"]
    fig, ax = plt.subplots(figsize=(8, 1 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    compare_dir = root / "outputs" / "compare"
    output_dir = args.output_dir or (compare_dir / "visuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(compare_dir)
    if not runs:
        print("No comparison runs found.")
        return

    _radar_plot(runs, output_dir / "radar.png")
    _params_time_plot(runs, output_dir / "params_time.png")
    _loss_curves(runs, output_dir / "loss_curves.png")
    _table_image(runs, output_dir / "result_table.png")
    print(f"Saved comparison visuals to: {output_dir}")


if __name__ == "__main__":
    main()
