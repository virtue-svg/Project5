from __future__ import annotations

import torch
import torch.nn as nn


class ClipFusionClassifier(nn.Module):
    def __init__(self, clip_model, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.clip = clip_model
        embed_dim = getattr(clip_model.config, "projection_dim", 512)
        self.text_proj = nn.Linear(embed_dim, hidden_dim)
        self.image_proj = nn.Linear(embed_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values
        )
        text_feat = outputs.text_embeds
        image_feat = outputs.image_embeds
        text_feat = self.dropout(torch.relu(self.text_proj(text_feat)))
        image_feat = self.dropout(torch.relu(self.image_proj(image_feat)))
        fused = torch.cat([text_feat, image_feat], dim=1)
        return self.classifier(fused)


class BlipFusionClassifier(nn.Module):
    def __init__(self, blip_model, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.blip = blip_model
        # Use lazy projections to adapt to BLIP variants with different hidden sizes.
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
