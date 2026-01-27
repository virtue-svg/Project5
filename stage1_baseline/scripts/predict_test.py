from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

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

from src.dataset import LABELS, load_split_csv, MultimodalDataset
from src.image_utils import get_eval_transforms
from src.model_tfidf_resnet import MultimodalClassifier
from src.text_utils import clean_text_advanced, read_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict test labels with best multimodal model.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best_tfidf_resnet.pt",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Original test_without_label.txt for output formatting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path for submission.",
    )
    return parser.parse_args()


def _default_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "outputs" / "splits" / "test.csv",
        root / "outputs" / "models" / "best_tfidf_resnet.pt",
        root / "data" / "project5" / "test_without_label.txt",
    )


def load_texts(samples) -> list[str]:
    texts = []
    for s in samples:
        raw = read_text(s.text_path)
        texts.append(clean_text_advanced(raw, replace_emoji=True, collapse_repeats=True))
    return texts


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    default_test_csv, default_ckpt, default_test_file = _default_paths(root)
    test_csv = Path(args.test_csv) if args.test_csv else default_test_csv
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_ckpt
    test_file = Path(args.test_file) if args.test_file else default_test_file
    output = args.output or (root / "outputs" / "test_with_label.txt")

    samples = load_split_csv(test_csv)
    texts = load_texts(samples)

    ckpt = torch.load(checkpoint, map_location="cpu")
    vectorizer = ckpt["vectorizer"]
    text_feats = vectorizer.transform(texts).toarray()

    dataset = MultimodalDataset(samples, text_feats, get_eval_transforms(args.image_size))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalClassifier(text_dim=text_feats.shape[1], num_classes=len(LABELS))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    preds = []
    with torch.no_grad():
        for images, text_features, _ in loader:
            images = images.to(device)
            text_features = text_features.to(device)
            logits = model(images, text_features)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(pred)

    labels = [LABELS[p] for p in preds]
    guid_to_label = {s.guid: label for s, label in zip(samples, labels)}

    lines = []
    with test_file.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "\t" in line:
                delim = "\t"
                parts = line.split("\t")
            elif "," in line:
                delim = ","
                parts = line.split(",")
            else:
                delim = "\t"
                parts = line.split()
            guid = parts[0].strip()
            if guid.lower() == "guid":
                lines.append(line)
                continue
            label = guid_to_label.get(guid, "neutral")
            lines.append(f"{guid}{delim}{label}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved submission file to: {output}")


if __name__ == "__main__":
    main()
