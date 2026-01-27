from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

from src.dataset import LABELS, LABEL_TO_ID, load_split_csv, ImageOnlyDataset
from src.image_utils import get_eval_transforms, get_train_transforms
from src.model_ablation import ImageOnlyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image-only ResNet18 model.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-train-aug", action="store_true")
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


def evaluate(model, loader, device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            logits = model(images)
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

    train_transform = (
        get_eval_transforms(args.image_size)
        if args.no_train_aug
        else get_train_transforms(args.image_size)
    )
    train_ds = ImageOnlyDataset(train_samples, train_transform)
    val_ds = ImageOnlyDataset(val_samples, get_eval_transforms(args.image_size))

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

    model = ImageOnlyClassifier(
        num_classes=len(LABELS), pretrained=not args.no_pretrained
    ).to(device)
    label_counts = np.bincount([s.label for s in train_samples], minlength=len(LABELS))
    class_weights = torch.tensor(
        (label_counts.sum() / np.maximum(label_counts, 1)), dtype=torch.float32
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    history = []
    outputs_dir = root / "outputs" / "models"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    best_path = outputs_dir / "best_image_only.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, y in train_loader:
            images = images.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(images)
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
                    "label_map": LABEL_TO_ID,
                    "args": vars(args),
                },
                best_path,
            )

    metrics_path = root / "outputs" / "metrics_image_only.json"
    history_path = root / "outputs" / "metrics_image_only.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(json_dumps({"best_val_f1_macro": best_f1}))
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved best model to: {best_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved history to: {history_path}")


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
