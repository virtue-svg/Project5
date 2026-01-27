# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 最终训练入口（封装 optimize_clip 参数）。
# 流程: 传入最优超参并调用 optimize_clip.py。
# 输出: outputs/optimize/final_clip/。

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Final train with best CLIP hyperparameters.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--early-stop", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "optimize" / "final_clip",
    )
    return parser.parse_args()


def main() -> None:
    # 主流程：拼接命令并执行
    args = parse_args()
    root = args.project_root.resolve()
    script = root / "stage3_optimize" / "scripts" / "optimize_clip.py"
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    cmd = [
        sys.executable,
        str(script),
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
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        str(output_dir),
    ]
    if args.pin_memory:
        cmd.append("--pin-memory")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
