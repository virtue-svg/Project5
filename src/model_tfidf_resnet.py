from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class TextMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int,
        num_classes: int = 3,
        text_hidden: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(weights=None)
        self.image_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.image_dim = 512
        self.text_mlp = TextMLP(text_dim, hidden_dim=text_hidden, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.image_dim + text_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_backbone(images).flatten(1)
        text_feat = self.text_mlp(text_features)
        fused = torch.cat([img_feat, text_feat], dim=1)
        return self.classifier(fused)
