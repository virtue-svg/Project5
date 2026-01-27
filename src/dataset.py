# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 阶段一数据集封装（多模态/仅文本/仅图像）。
# 流程: 从划分 CSV 读取样本并返回模型所需张量。
# 输出: 图像张量 + 文本特征 + 标签。

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
    # 读取划分 CSV，构造 Sample 列表
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
        # 样本数量
        return len(self.samples)

    def __getitem__(self, idx: int):
        # 返回图像张量 + 文本特征 + 标签
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            if self.image_transform:
                img = self.image_transform(img)
        text_feat = torch.tensor(self.text_features[idx], dtype=torch.float32)
        label = sample.label if sample.label is not None else -1
        return img, text_feat, label


class TextOnlyDataset(Dataset):
    def __init__(self, samples: List[Sample], text_features):
        self.samples = samples
        self.text_features = text_features

    def __len__(self) -> int:
        # 样本数量
        return len(self.samples)

    def __getitem__(self, idx: int):
        # 返回文本特征 + 标签
        sample = self.samples[idx]
        text_feat = torch.tensor(self.text_features[idx], dtype=torch.float32)
        label = sample.label if sample.label is not None else -1
        return text_feat, label


class ImageOnlyDataset(Dataset):
    def __init__(self, samples: List[Sample], image_transform=None):
        self.samples = samples
        self.image_transform = image_transform

    def __len__(self) -> int:
        # 样本数量
        return len(self.samples)

    def __getitem__(self, idx: int):
        # 返回图像张量 + 标签
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            if self.image_transform:
                img = self.image_transform(img)
        label = sample.label if sample.label is not None else -1
        return img, label
