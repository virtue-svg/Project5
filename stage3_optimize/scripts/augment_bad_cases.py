# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 对 bad case 图像进行轻量增强。
# 流程: 随机裁剪/翻转/颜色抖动并保存增强样本。
# 输出: outputs/optimize/badcase_aug 与 badcase_augmented.csv。

import argparse
from pathlib import Path
import random

import pandas as pd
from PIL import Image
import torchvision.transforms as T


def parse_args() -> argparse.Namespace:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Augment images for bad cases.")
    parser.add_argument("--badcase-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/optimize/badcase_aug"),
    )
    parser.add_argument("--num-aug", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    # 主流程：读取 bad cases 并生成增强样本
    args = parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.badcase_csv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    augment = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]
    )

    rows = []
    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                for i in range(args.num_aug):
                    aug_img = augment(img)
                    aug_name = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                    aug_path = out_dir / aug_name
                    aug_img.save(aug_path)
                    rows.append(
                        {
                            "guid": row.get("guid", ""),
                            "true_label": row.get("true_label", ""),
                            "pred_label": row.get("pred_label", ""),
                            "image_path": str(aug_path),
                            "source_image": str(img_path),
                        }
                    )
        except Exception:
            continue

    out_csv = out_dir / "badcase_augmented.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved augmented samples: {out_csv}")


if __name__ == "__main__":
    main()
