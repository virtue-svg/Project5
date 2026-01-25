from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


@dataclass
class Sample:
    guid: str
    text_path: Path
    image_path: Path
    label: Optional[int]


def load_split_csv(path: Path) -> List[Sample]:
    df = pd.read_csv(path)
    samples: List[Sample] = []
    for _, row in df.iterrows():
        label = row.get("label")
        label_id = None
        if isinstance(label, str) and label in LABEL_TO_ID:
            label_id = LABEL_TO_ID[label]
        samples.append(
            Sample(
                guid=str(row["guid"]),
                text_path=Path(row["text_path"]),
                image_path=Path(row["image_path"]),
                label=label_id,
            )
        )
    return samples


class MultimodalDataset(Dataset):
    def __init__(self, samples: List[Sample], text_features, image_transform=None):
        self.samples = samples
        self.text_features = text_features
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            if self.image_transform:
                img = self.image_transform(img)
        text_feat = torch.tensor(self.text_features[idx], dtype=torch.float32)
        label = sample.label if sample.label is not None else -1
        return img, text_feat, label
