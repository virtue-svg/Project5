# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 挖掘验证集 bad cases（错误/低置信度样本）。
# 流程: 用最终模型推理并筛选样本。
# 输出: outputs/optimize/bad_cases.csv。

import argparse
import json
from pathlib import Path
import sys
from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor


def _find_project_root(start: Path) -> Path:
    # 向上查找项目根目录
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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Mine bad cases from validation set.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/optimize/final_clip/best.pt"),
    )
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/optimize/bad_cases.csv"),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Also include samples with confidence below this threshold.",
    )
    return parser.parse_args()


def _default_val_split(root: Path) -> Path:
    # 默认验证集路径
    return root / "outputs" / "splits" / "val.csv"


def _load_texts(samples) -> list[str]:
    # 读取并清洗文本
    texts = []
    for s in samples:
        raw = read_text(s.text_path)
        texts.append(clean_text_advanced(raw, replace_emoji=True, collapse_repeats=True))
    return texts


def collate_clip_batch(processor, batch, max_length: int):
    # 将原始文本/图像打包为 CLIP 输入
    images = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
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


def main() -> None:
    # 主流程：推理并输出 bad cases
    args = parse_args()
    root = args.project_root.resolve()
    val_csv = args.val_csv or _default_val_split(root)
    weights = args.weights if args.weights.is_absolute() else (root / args.weights)
    output_csv = args.output_csv if args.output_csv.is_absolute() else (root / args.output_csv)

    samples = load_split_csv(Path(val_csv))
    texts = _load_texts(samples)
    ds = RawTextImageDataset(samples, texts)

    processor = CLIPProcessor.from_pretrained(args.model_name)
    clip = CLIPModel.from_pretrained(args.model_name)
    model = ClipFusionClassifier(clip).eval()

    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=partial(collate_clip_batch, processor, max_length=args.max_length),
    )

    rows = []
    idx = 0
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for input_ids, attention_mask, pixel_values, y in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values = pixel_values.to(device)
            logits = model(input_ids, attention_mask, pixel_values)
            probs = softmax(logits).cpu()
            preds = torch.argmax(probs, dim=1).numpy().tolist()
            confs = probs.max(dim=1).values.numpy().tolist()
            y_list = y.numpy().tolist()

            for j, (pred, conf, true) in enumerate(zip(preds, confs, y_list)):
                sample = samples[idx + j]
                is_wrong = pred != true
                is_low_conf = conf < args.confidence_threshold
                if is_wrong or is_low_conf:
                    rows.append(
                        {
                            "guid": sample.guid,
                            "true_label": LABELS[true],
                            "pred_label": LABELS[pred],
                            "confidence": round(conf, 6),
                            "text_path": str(sample.text_path),
                            "image_path": str(sample.image_path),
                            "reason": "wrong" if is_wrong else "low_conf",
                        }
                    )
            idx += len(y_list)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved bad cases to: {output_csv}")


if __name__ == "__main__":
    main()
