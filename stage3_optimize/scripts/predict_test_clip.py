# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor


def _find_project_root(start: Path) -> Path:
    cur = start
    while True:
        if (cur / "requirements.txt").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return start
        cur = cur.parent


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compare_clip_blip import ClipFusionClassifier
from src.compare_dataset import LABELS, load_split_csv, RawTextImageDataset
from src.text_utils import clean_text_advanced, read_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict test labels with CLIP multimodal model.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _default_paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "outputs" / "splits" / "test.csv",
        root / "data" / "project5" / "test_without_label.txt",
    )


def _load_texts(samples) -> list[str]:
    texts = []
    for s in samples:
        raw = read_text(s.text_path)
        texts.append(clean_text_advanced(raw, replace_emoji=True, collapse_repeats=True))
    return texts


def collate_clip_batch(processor, batch, max_length: int):
    images = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    text_inputs = processor.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    return (
        text_inputs["input_ids"],
        text_inputs["attention_mask"],
        image_inputs["pixel_values"],
    )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    default_test_csv, default_test_file = _default_paths(root)
    test_csv = args.test_csv or default_test_csv
    test_file = args.test_file or default_test_file
    output = args.output or (root / "outputs" / "test_with_label_clip.txt")

    weights = args.weights if args.weights.is_absolute() else (root / args.weights)
    state = torch.load(weights, map_location="cpu")
    state_args = state.get("args", {}) if isinstance(state, dict) else {}
    dropout = float(state_args.get("dropout", 0.2))
    head_variant = state_args.get("head_variant", "base")
    max_length = int(state_args.get("max_length", args.max_length))

    samples = load_split_csv(Path(test_csv))
    texts = _load_texts(samples)
    ds = RawTextImageDataset(samples, texts)

    processor = CLIPProcessor.from_pretrained(args.model_name)
    clip = CLIPModel.from_pretrained(args.model_name)
    model = ClipFusionClassifier(clip, dropout=dropout, head_variant=head_variant)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=partial(collate_clip_batch, processor, max_length=max_length),
    )

    preds = []
    with torch.no_grad():
        for input_ids, attention_mask, pixel_values in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values = pixel_values.to(device)
            logits = model(input_ids, attention_mask, pixel_values)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(pred)

    labels = [LABELS[p] for p in preds]
    guid_to_label = {s.guid: label for s, label in zip(samples, labels)}

    lines = []
    with Path(test_file).open("r", encoding="utf-8", errors="ignore") as f:
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
