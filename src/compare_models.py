# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 定义 BERT+ResNet 的多种融合结构（concat / gated / late）。
# 流程: 文本与图像分别编码后融合，再送入分类头。
# 输出: 分类 logits（用于计算损失与评估）。

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        # 文本编码器：加载预训练 BERT/等价模型
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 若有 pooler_output 则优先使用，否则做 masked mean pooling
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        # 图像编码器：ResNet18/50 去掉分类头
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            self.output_dim = 2048
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.output_dim = 512
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 输出 [B, C] 的全局特征向量
        return self.features(images).flatten(1)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        text_model: str,
        image_backbone: str = "resnet18",
        num_classes: int = 3,
        fusion: str = "concat",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        # fusion: concat/gated/late 控制融合方式
        super().__init__()
        self.text_encoder = TextEncoder(text_model)
        self.image_encoder = ImageEncoder(image_backbone, pretrained=pretrained)
        self.fusion = fusion

        self.text_proj = nn.Linear(self.text_encoder.model.config.hidden_size, hidden_dim)
        self.image_proj = nn.Linear(self.image_encoder.output_dim, hidden_dim)

        if fusion == "late":
            self.text_head = nn.Linear(hidden_dim, num_classes)
            self.image_head = nn.Linear(hidden_dim, num_classes)
        else:
            if fusion == "gated":
                self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, images, input_ids, attention_mask):
        # 分别提取文本/图像特征并投影到同一维度
        text_feat = self.text_encoder(input_ids, attention_mask)
        image_feat = self.image_encoder(images)
        text_feat = self.dropout(torch.relu(self.text_proj(text_feat)))
        image_feat = self.dropout(torch.relu(self.image_proj(image_feat)))

        if self.fusion == "late":
            # late fusion：两路各自分类后平均
            text_logits = self.text_head(text_feat)
            image_logits = self.image_head(image_feat)
            return (text_logits + image_logits) / 2.0

        if self.fusion == "gated":
            # gated fusion：学习门控权重决定模态贡献
            gate = torch.sigmoid(self.gate(torch.cat([text_feat, image_feat], dim=1)))
            fused = torch.cat([gate * image_feat, (1 - gate) * text_feat], dim=1)
        else:
            # concat fusion：直接拼接
            fused = torch.cat([image_feat, text_feat], dim=1)
        return self.classifier(fused)
