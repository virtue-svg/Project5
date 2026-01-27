# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare preprocessing settings while keeping training hyperparams fixed."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--early-stop", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--improved-remove-stopwords", action="store_true")
    return parser.parse_args()


def _read_best_metrics(history_csv: Path) -> dict:
    if not history_csv.exists():
        return {}
    best_row = None
    with history_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if best_row is None or float(row["val_f1_macro"]) > float(best_row["val_f1_macro"]):
                best_row = row
    return best_row or {}


def _run_case(name: str, cmd: list[str], out_dir: Path) -> dict:
    subprocess.run(cmd, check=True)
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.csv"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_row = _read_best_metrics(history_path)
    return {
        "name": name,
        "best_val_f1_macro": float(best_row.get("val_f1_macro", metrics.get("best_val_f1_macro", 0))),
        "best_val_acc": float(best_row.get("val_acc", 0)),
        "best_val_precision_macro": float(best_row.get("val_precision_macro", 0)),
        "best_val_recall_macro": float(best_row.get("val_recall_macro", 0)),
        "train_time_sec": float(metrics.get("train_time_sec", 0)),
        "params_million": float(metrics.get("params_million", 0)),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    python_exe = Path(sys.executable)
    optimize_script = root / "stage3_optimize" / "scripts" / "optimize_clip.py"
    if not optimize_script.exists():
        raise FileNotFoundError(f"Missing optimize_clip.py at {optimize_script}")

    base_out = args.output_dir or (root / "outputs" / "optimize" / "preprocess_compare")
    base_out.mkdir(parents=True, exist_ok=True)

    base_dir = base_out / "baseline"
    improved_dir = base_out / "improved"
    base_dir.mkdir(parents=True, exist_ok=True)
    improved_dir.mkdir(parents=True, exist_ok=True)

    common = [
        str(python_exe),
        str(optimize_script),
        "--project-root",
        str(root),
        "--model-name",
        args.model_name,
        "--epochs",
        str(args.epochs),
        "--freeze-epochs",
        str(args.freeze_epochs),
        "--early-stop",
        str(args.early_stop),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--max-length",
        str(args.max_length),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.pin_memory:
        common.append("--pin-memory")

    base_cmd = (
        common
        + [
            "--output-dir",
            str(base_dir),
            "--no-replace-emoji",
            "--no-collapse-repeats",
            "--no-remove-stopwords",
        ]
    )
    improved_cmd = common + ["--output-dir", str(improved_dir), "--image-aug"]
    if args.improved_remove_stopwords:
        improved_cmd.append("--remove-stopwords")

    results = []
    results.append(_run_case("baseline", base_cmd, base_dir))
    results.append(_run_case("improved", improved_cmd, improved_dir))

    summary_path = base_out / "preprocess_compare.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "best_val_f1_macro",
                "best_val_acc",
                "best_val_precision_macro",
                "best_val_recall_macro",
                "train_time_sec",
                "params_million",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved preprocessing comparison to: {summary_path}")


if __name__ == "__main__":
    main()
