# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 消融模型（仅文本 / 仅图像）。
# 流程: 文本用 MLP，图像用 ResNet 特征后接分类头。
# 输出: 分类 logits。

import torch
import torch.nn as nn
from torchvision import models


class TextOnlyClassifier(nn.Module):
    def __init__(self, text_dim: int, num_classes: int = 3, hidden_dim: int = 256, dropout: float = 0.2):
        # 文本特征 -> MLP 分类
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        # 前向：文本特征分类
        return self.net(text_features)


class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.2, pretrained: bool = True):
        # 图像分支：ResNet18 去掉分类头
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 前向：图像特征分类
        feat = self.features(images).flatten(1)
        return self.classifier(feat)
