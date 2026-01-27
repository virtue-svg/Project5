# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 定义 CLIP/BLIP 融合分类器，供对比训练调用。
# 流程: 文本与图像分别编码后做投影，再拼接进入分类头。
# 输出: 分类 logits（未做 softmax）。

import torch
import torch.nn as nn


class ClipFusionClassifier(nn.Module):
    def __init__(
        self,
        clip_model,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.2,
        head_variant: str = "base",
    ):
        # 初始化 CLIP 文本/图像投影层与分类头
        super().__init__()
        self.clip = clip_model
        self.head_variant = head_variant
        # CLIP 默认投影维度（不同模型可能不同）
        embed_dim = getattr(clip_model.config, "projection_dim", 512)
        self.text_proj = nn.Linear(embed_dim, hidden_dim)
        self.image_proj = nn.Linear(embed_dim, hidden_dim)
        # 可选：轻量门控与 LayerNorm
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim) if head_variant == "gated" else None
        self.layernorm = nn.LayerNorm(hidden_dim * 2) if head_variant == "ln" else None
        if head_variant == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, pixel_values):
        # 前向：提取图文特征并拼接分类
        # 文本/图像各自编码，再拼接分类
        outputs = self.clip(
            input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values
        )
        text_feat = outputs.text_embeds
        image_feat = outputs.image_embeds
        text_feat = self.dropout(torch.relu(self.text_proj(text_feat)))
        image_feat = self.dropout(torch.relu(self.image_proj(image_feat)))
        if self.head_variant == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([text_feat, image_feat], dim=1)))
            fused = torch.cat([gate * image_feat, (1 - gate) * text_feat], dim=1)
        else:
            fused = torch.cat([text_feat, image_feat], dim=1)
        if self.layernorm is not None:
            fused = self.layernorm(fused)
        return self.classifier(fused)


class BlipFusionClassifier(nn.Module):
    def __init__(self, blip_model, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.2):
        # 初始化 BLIP 文本/图像投影层与分类头
        super().__init__()
        self.blip = blip_model
        # 使用 LazyLinear 适配不同 BLIP 变体的隐藏维度
        self.text_proj = nn.LazyLinear(hidden_dim)
        self.image_proj = nn.LazyLinear(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, pixel_values):
        # 前向：兼容不同 BLIP 输出字段，融合后分类
        outputs = self.blip(
            input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values
        )
        text_feat = getattr(outputs, "text_embeds", None)
        image_feat = getattr(outputs, "image_embeds", None)
        if text_feat is None:
            text_feat = outputs.text_model_output.last_hidden_state[:, 0]
        if image_feat is None:
            vision_out = outputs.vision_model_output
            if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
                image_feat = vision_out.pooler_output
            else:
                image_feat = vision_out.last_hidden_state.mean(dim=1)

        text_feat = self.dropout(torch.relu(self.text_proj(text_feat)))
        image_feat = self.dropout(torch.relu(self.image_proj(image_feat)))
        fused = torch.cat([text_feat, image_feat], dim=1)
        return self.classifier(fused)
