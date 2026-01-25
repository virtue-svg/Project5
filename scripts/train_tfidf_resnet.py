from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-max-df", type=float, default=0.95)
    parser.add_argument("--remove-stopwords", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train-aug", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    probs = []
    with torch.no_grad():
        for images, text_features, y in loader:
            images = images.to(device)
            text_features = text_features.to(device)
            logits = model(images, text_features)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.extend(prob.tolist())
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            labels.extend(y.numpy().tolist())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    return {
        "acc": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "labels": labels,
        "preds": preds,
        "probs": probs,
    }


def _plot_curves(history: list[dict], out_path: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_loss, color="tab:red", label="train_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, color="tab:blue", label="val_acc")
    ax2.set_ylabel("accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion(labels: list[int], preds: list[int], out_path: Path) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_roc(labels: list[int], probs: list[list[float]], out_path: Path) -> None:
    y_true = np.array(labels)
    y_score = np.array(probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    for class_id, class_name in enumerate(LABELS):
        y_bin = (y_true == class_id).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_score[:, class_id])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (OvR)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_correlation(samples, preds: list[int], labels: list[int], out_path: Path) -> None:
    if len(samples) != len(preds):
        return
    rows = []
    for sample, pred, label in zip(samples, preds, labels):
        if not sample.text_path.exists() or not sample.image_path.exists():
            continue
        text_len = len(read_text(sample.text_path))
        with Image.open(sample.image_path) as img:
            width, height = img.size
        rows.append(
            {
                "text_len": text_len,
                "image_area": width * height,
                "correct": int(pred == label),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].scatter(df["text_len"], df["correct"], s=6, alpha=0.5)
    axes[0].set_xlabel("text_len")
    axes[0].set_ylabel("correct")
    axes[0].set_title("Text Length vs Correct")

    axes[1].scatter(df["image_area"], df["correct"], s=6, alpha=0.5)
    axes[1].set_xlabel("image_area")
    axes[1].set_ylabel("correct")
    axes[1].set_title("Image Area vs Correct")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    train_transform = (
        get_eval_transforms(args.image_size)
        if args.no_train_aug
        else get_train_transforms(args.image_size)
    )
    train_ds = MultimodalDataset(train_samples, train_feats, train_transform)
    val_ds = MultimodalDataset(
        val_samples, val_feats, get_eval_transforms(args.image_size)
    )

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
    history = []
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
            f"val_acc {metrics['acc']:.4f} val_f1 {metrics['f1_macro']:.4f} "
            f"val_prec {metrics['precision_macro']:.4f} val_rec {metrics['recall_macro']:.4f}"
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

    metrics_path = root / "outputs" / "metrics_tfidf_resnet.json"
    history_path = root / "outputs" / "metrics_tfidf_resnet.csv"
    visuals_dir = root / "outputs" / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_f1_macro": best_f1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    pd.DataFrame(history).to_csv(history_path, index=False)
    plot_path = root / "outputs" / "metrics_tfidf_resnet.png"
    _plot_curves(history, plot_path)
    _plot_confusion(metrics["labels"], metrics["preds"], visuals_dir / "confusion.png")
    _plot_roc(metrics["labels"], metrics["probs"], visuals_dir / "roc.png")
    _plot_correlation(val_samples, metrics["preds"], metrics["labels"], visuals_dir / "correlation.png")
    print(f"Saved best model to: {best_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved history to: {history_path}")
    print(f"Saved curves to: {plot_path}")
    print(f"Saved visuals to: {visuals_dir}")


if __name__ == "__main__":
    main()
