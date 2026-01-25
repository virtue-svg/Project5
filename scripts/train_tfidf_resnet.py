from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import LABELS, LABEL_TO_ID, load_split_csv, MultimodalDataset
from src.image_utils import get_eval_transforms, get_train_transforms
from src.model_tfidf_resnet import MultimodalClassifier
from src.text_utils import clean_text_advanced, read_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + ResNet18 multimodal model.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-max-df", type=float, default=0.95)
    parser.add_argument("--remove-stopwords", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _default_splits(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "outputs" / "splits" / "train.csv",
        root / "outputs" / "splits" / "val.csv",
        root / "outputs" / "splits" / "test.csv",
    )


def load_texts(samples, remove_stopwords: bool) -> list[str]:
    texts = []
    for s in samples:
        raw = read_text(s.text_path)
        texts.append(
            clean_text_advanced(
                raw,
                replace_emoji=True,
                collapse_repeats=True,
                remove_stopwords=remove_stopwords,
            )
        )
    return texts


def build_tfidf(
    train_samples, val_samples, test_samples, max_features, min_df, max_df, remove_stopwords
):
    train_texts = load_texts(train_samples, remove_stopwords)
    val_texts = load_texts(val_samples, remove_stopwords)
    test_texts = load_texts(test_samples, remove_stopwords)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        stop_words="english" if remove_stopwords else None,
    )
    train_feats = vectorizer.fit_transform(train_texts).toarray()
    val_feats = vectorizer.transform(val_texts).toarray()
    test_feats = vectorizer.transform(test_texts).toarray()
    return train_feats, val_feats, test_feats, vectorizer


def evaluate(model, loader, device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for images, text_features, y in loader:
            images = images.to(device)
            text_features = text_features.to(device)
            logits = model(images, text_features)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            labels.extend(y.numpy().tolist())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"acc": acc, "f1_macro": f1}


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    set_seed(args.seed)

    train_csv, val_csv, test_csv = _default_splits(root)
    train_csv = args.train_csv or train_csv
    val_csv = args.val_csv or val_csv
    test_csv = args.test_csv or test_csv

    train_samples = load_split_csv(Path(train_csv))
    val_samples = load_split_csv(Path(val_csv))
    test_samples = load_split_csv(Path(test_csv))

    train_feats, val_feats, test_feats, vectorizer = build_tfidf(
        train_samples,
        val_samples,
        test_samples,
        args.tfidf_max_features,
        args.tfidf_min_df,
        args.tfidf_max_df,
        args.remove_stopwords,
    )

    train_ds = MultimodalDataset(
        train_samples, train_feats, get_train_transforms(args.image_size)
    )
    val_ds = MultimodalDataset(
        val_samples, val_feats, get_eval_transforms(args.image_size)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalClassifier(
        text_dim=train_feats.shape[1],
        num_classes=len(LABELS),
        pretrained=not args.no_pretrained,
    ).to(device)

    label_counts = np.bincount([s.label for s in train_samples], minlength=len(LABELS))
    class_weights = torch.tensor(
        (label_counts.sum() / np.maximum(label_counts, 1)), dtype=torch.float32
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    outputs_dir = root / "outputs" / "models"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    best_path = outputs_dir / "best_tfidf_resnet.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, text_features, y in train_loader:
            images = images.to(device)
            text_features = text_features.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(images, text_features)
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

    metrics_path = root / "outputs" / "metrics_tfidf_resnet.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"best_val_f1_macro": best_f1}, f, ensure_ascii=False, indent=2)
    print(f"Saved best model to: {best_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
