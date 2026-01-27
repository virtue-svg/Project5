from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for CLIP optimization.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--early-stop", type=int, default=3)
    parser.add_argument("--batch-sizes", type=str, default="8,16")
    parser.add_argument("--lrs", type=str, default="1e-5,2e-5")
    parser.add_argument("--weight-decays", type=str, default="1e-4,5e-4")
    parser.add_argument("--dropouts", type=str, default="0.1,0.2")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    return parser.parse_args()


def _parse_list(s: str, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    bs_list = _parse_list(args.batch_sizes, int)
    lr_list = _parse_list(args.lrs, float)
    wd_list = _parse_list(args.weight_decays, float)
    dp_list = _parse_list(args.dropouts, float)

    results = []
    for bs, lr, wd, dp in product(bs_list, lr_list, wd_list, dp_list):
        run_name = f"bs{bs}_lr{lr}_wd{wd}_dp{dp}"
        output_dir = root / "outputs" / "optimize" / run_name
        cmd = [
            str(root / ".venv" / "Scripts" / "python.exe"),
            str(root / "stage3_optimize" / "scripts" / "optimize_clip.py"),
            "--project-root",
            str(root),
            "--output-dir",
            str(output_dir),
            "--model-name",
            args.model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(bs),
            "--lr",
            str(lr),
            "--weight-decay",
            str(wd),
            "--dropout",
            str(dp),
            "--freeze-epochs",
            str(args.freeze_epochs),
            "--early-stop",
            str(args.early_stop),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.pin_memory:
            cmd.append("--pin-memory")
        print(f"Running: {run_name}")
        try:
            subprocess.run(cmd, check=True)
            results.append({"run": run_name, "batch_size": bs, "lr": lr, "weight_decay": wd, "dropout": dp, "status": "ok"})
        except subprocess.CalledProcessError as exc:
            results.append({"run": run_name, "batch_size": bs, "lr": lr, "weight_decay": wd, "dropout": dp, "status": f"fail:{exc.returncode}"})
            if not args.continue_on_fail:
                raise

    (root / "outputs" / "optimize" / "grid_search.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
