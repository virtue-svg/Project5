from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image
import random
from PIL import ImageEnhance, ImageOps

def _find_project_root(start: Path) -> Path:
    cur = start
    while True:
        if (cur / 'requirements.txt').exists() or (cur / '.git').exists():
            return cur
        if cur.parent == cur:
            return start
        cur = cur.parent

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import build_records


def _first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def _default_data_dir(root: Path) -> Path:
    if (root / "data" / "data").exists():
        return root / "data" / "data"
    if (root / "data" / "project5" / "data").exists():
        return root / "data" / "project5" / "data"
    return root / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview conservative image augmentations.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing data/ and outputs/.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    data_dir = args.data_dir or _default_data_dir(root)
    output_dir = args.output_dir or (root / "outputs" / "processed" / "aug_samples")

    train_file = args.train_file or _first_existing(
        [
            root / "data" / "train.txt",
            root / "data" / "project5" / "train.txt",
            root / "train.txt",
        ]
    )

    records = build_records(train_file, data_dir, has_label=True)
    records = [r for r in records if r.image_path is not None and r.image_path.exists()]

    random.seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    def resize_center_crop(img: Image.Image, size: int = 224) -> Image.Image:
        img = img.resize((256, 256))
        left = (img.width - size) // 2
        top = (img.height - size) // 2
        return img.crop((left, top, left + size, top + size))

    def random_resized_crop(img: Image.Image, size: int = 224) -> Image.Image:
        img = img.resize((256, 256))
        scale = random.uniform(0.8, 1.0)
        crop_size = int(256 * scale)
        max_offset = 256 - crop_size
        left = random.randint(0, max_offset) if max_offset > 0 else 0
        top = random.randint(0, max_offset) if max_offset > 0 else 0
        img = img.crop((left, top, left + crop_size, top + crop_size))
        return img.resize((size, size))

    def color_jitter(img: Image.Image) -> Image.Image:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
        return img

    def augment(img: Image.Image) -> Image.Image:
        img = random_resized_crop(img)
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        img = color_jitter(img)
        return img

    for idx, rec in enumerate(records[: args.num_samples], start=1):
        with Image.open(rec.image_path) as img:
            img = img.convert("RGB")
            original = resize_center_crop(img)
            aug1 = augment(img)
            aug2 = augment(img)
            aug3 = augment(img)

        grid = Image.new("RGB", (224 * 2, 224 * 2))
        grid.paste(original, (0, 0))
        grid.paste(aug1, (224, 0))
        grid.paste(aug2, (0, 224))
        grid.paste(aug3, (224, 224))
        grid.save(output_dir / f"{rec.guid}_aug.png")

    print(f"Saved {min(args.num_samples, len(records))} augmentation previews to: {output_dir}")


if __name__ == "__main__":
    main()
