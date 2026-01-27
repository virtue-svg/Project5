from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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

from src.compare_dataset import LABELS, load_split_csv, MultimodalBertDataset
from src.compare_models import FusionClassifier
from src.image_utils import get_eval_transforms, get_train_transforms
from src.text_utils import clean_text_advanced, read_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal comparison models.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="bert-base-chinese")
    parser.add_argument("--image-backbone", type=str, choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--fusion", type=str, choices=["concat", "gated", "late"], default="concat")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-train-aug", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
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


def _load_texts(samples, remove_stopwords: bool = False) -> list[str]:
    texts = []
    for s in samples:
        raw = read_text(s.text_path)
        texts.append(
            clean_text_advanced(
                raw, replace_emoji=True, collapse_repeats=True, remove_stopwords=remove_stopwords
            )
        )
    return texts


def evaluate(model, loader, device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for images, input_ids, attention_mask, y in loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(images, input_ids, attention_mask)
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

    train_csv, val_csv = _default_splits(root)
    train_csv = args.train_csv or train_csv
    val_csv = args.val_csv or val_csv

    train_samples = load_split_csv(Path(train_csv))
    val_samples = load_split_csv(Path(val_csv))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_texts = _load_texts(train_samples)
    val_texts = _load_texts(val_samples)
    train_enc = tokenizer(train_texts, max_length=args.max_length, padding=True, truncation=True)
    val_enc = tokenizer(val_texts, max_length=args.max_length, padding=True, truncation=True)

    train_transform = (
        get_eval_transforms(args.image_size)
        if args.no_train_aug
        else get_train_transforms(args.image_size)
    )
    train_ds = MultimodalBertDataset(train_samples, train_enc, train_transform)
    val_ds = MultimodalBertDataset(val_samples, val_enc, get_eval_transforms(args.image_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = FusionClassifier(
        text_model=args.model_name,
        image_backbone=args.image_backbone,
        fusion=args.fusion,
        pretrained=not args.no_pretrained,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label_counts = np.bincount([s.label for s in train_samples], minlength=len(LABELS))
    class_weights = torch.tensor(
        (label_counts.sum() / np.maximum(label_counts, 1)), dtype=torch.float32
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    history = []
    run_name = f"{args.model_name}_{args.image_backbone}_{args.fusion}"
    output_dir = args.output_dir or (root / "outputs" / "compare" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, input_ids, attention_mask, y in train_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f} "
            f"val_acc {metrics['acc']:.4f} val_f1 {metrics['f1_macro']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_acc": metrics["acc"],
                "val_f1_macro": metrics["f1_macro"],
                "val_precision_macro": metrics["precision_macro"],
                "val_recall_macro": metrics["recall_macro"],
            }
        )
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )

    (output_dir / "history.csv").write_text(
        "\n".join(
            [
                "epoch,train_loss,val_acc,val_f1_macro,val_precision_macro,val_recall_macro"
            ]
            + [
                f"{h['epoch']},{h['train_loss']},{h['val_acc']},{h['val_f1_macro']},"
                f"{h['val_precision_macro']},{h['val_recall_macro']}"
                for h in history
            ]
        ),
        encoding="utf-8",
    )
    elapsed = time.time() - start_time
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_f1_macro": best_f1,
                "epochs": args.epochs,
                "train_time_sec": elapsed,
                "params_million": round(total_params / 1e6, 3),
                "trainable_params_million": round(trainable_params / 1e6, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved best model to: {best_path}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
