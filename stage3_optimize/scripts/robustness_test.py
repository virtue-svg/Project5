# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader
from functools import partial
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
    parser = argparse.ArgumentParser(description="Robustness test with noise/blur on validation images.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--weights", type=Path, default=Path("outputs/optimize/final_clip/best.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=5.0)
    parser.add_argument("--blur-radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optimize/robustness"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_splits(root: Path) -> tuple[Path, Path]:
    return (
        root / "outputs" / "splits" / "train.csv",
        root / "outputs" / "splits" / "val.csv",
    )


def _load_texts(samples) -> list[str]:
    texts = []
    for s in samples:
        raw = read_text(s.text_path)
        texts.append(clean_text_advanced(raw, replace_emoji=True, collapse_repeats=True))
    return texts


def _apply_perturb(img: Image.Image, mode: str, noise_std: float, blur_radius: float) -> Image.Image:
    if mode == "noise":
        arr = np.array(img).astype(np.float32)
        arr += np.random.normal(0, noise_std, size=arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if mode == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return img


def collate_clip_batch(processor, batch, max_length: int, mode: str, noise_std: float, blur_radius: float):
    images = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    if mode != "clean":
        images = [_apply_perturb(img, mode, noise_std, blur_radius) for img in images]
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
        labels,
    )


def evaluate(model, loader, device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for input_ids, attention_mask, pixel_values, y in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values = pixel_values.to(device)
            logits = model(input_ids, attention_mask, pixel_values)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            labels.extend(y.numpy().tolist())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    return {"acc": acc, "f1_macro": f1, "precision_macro": precision, "recall_macro": recall}


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    set_seed(args.seed)

    _, val_csv = _default_splits(root)
    val_samples = load_split_csv(Path(val_csv))
    val_texts = _load_texts(val_samples)
    val_ds = RawTextImageDataset(val_samples, val_texts)

    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")

    state = torch.load(args.weights, map_location="cpu")
    state_args = state.get("args", {}) if isinstance(state, dict) else {}
    dropout = float(state_args.get("dropout", 0.2))
    head_variant = state_args.get("head_variant", "base")

    processor = CLIPProcessor.from_pretrained(args.model_name)
    clip = CLIPModel.from_pretrained(args.model_name)
    model = ClipFusionClassifier(clip, dropout=dropout, head_variant=head_variant).eval()
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    results = {}
    for mode in ["clean", "noise", "blur"]:
        collate_fn = partial(
            collate_clip_batch,
            processor,
            max_length=args.max_length,
            mode=mode,
            noise_std=args.noise_std,
            blur_radius=args.blur_radius,
        )
        loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and device.type == "cuda",
            shuffle=False,
            collate_fn=collate_fn,
        )
        results[mode] = evaluate(model, loader, device)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "robustness_metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved robustness metrics to: {out_path}")


if __name__ == "__main__":
    main()
