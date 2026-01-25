from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import LABELS, LABEL_TO_ID, load_split_csv, TextOnlyDataset
from src.model_ablation import TextOnlyClassifier
from src.text_utils import clean_text_advanced, read_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text-only TF-IDF model.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-max-df", type=float, default=0.95)
    parser.add_argument("--remove-stopwords", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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


def _load_texts(samples, remove_stopwords: bool) -> list[str]:
    texts = []
    for s in samples:
        raw = read_text(s.text_path)
        texts.append(
            clean_text_advanced(
                raw, replace_emoji=True, collapse_repeats=True, remove_stopwords=remove_stopwords
            )
        )
    return texts


def build_tfidf(train_samples, val_samples, max_features, min_df, max_df, remove_stopwords):
    train_texts = _load_texts(train_samples, remove_stopwords)
    val_texts = _load_texts(val_samples, remove_stopwords)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        stop_words="english" if remove_stopwords else None,
    )
    train_feats = vectorizer.fit_transform(train_texts).toarray()
    val_feats = vectorizer.transform(val_texts).toarray()
    return train_feats, val_feats, vectorizer


def evaluate(model, loader, device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for text_features, y in loader:
            text_features = text_features.to(device)
            logits = model(text_features)
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

    train_feats, val_feats, vectorizer = build_tfidf(
        train_samples,
        val_samples,
        args.tfidf_max_features,
        args.tfidf_min_df,
        args.tfidf_max_df,
        args.remove_stopwords,
    )

    train_ds = TextOnlyDataset(train_samples, train_feats)
    val_ds = TextOnlyDataset(val_samples, val_feats)

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

    model = TextOnlyClassifier(text_dim=train_feats.shape[1], num_classes=len(LABELS)).to(device)
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
    best_path = outputs_dir / "best_text_only.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for text_features, y in train_loader:
            text_features = text_features.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(text_features)
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
                    "vectorizer": vectorizer,
                    "label_map": LABEL_TO_ID,
                    "args": vars(args),
                },
                best_path,
            )

    metrics_path = root / "outputs" / "metrics_text_only.json"
    history_path = root / "outputs" / "metrics_text_only.csv"
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
