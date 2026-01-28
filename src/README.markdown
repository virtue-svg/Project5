# src
本目录包含实验项目的核心代码与公共模块。

## 文件说明
- `compare_clip_blip.py`：CLIP/BLIP 融合分类器与轻量结构变体。
- `compare_dataset.py`：多模态数据集封装与统一标签映射。
- `compare_models.py`：BERT+ResNet18 融合模型实现（concat/gated/late）。
- `data_utils.py`：数据读取与分层划分工具。
- `dataset.py`：基线模型数据集与 DataLoader 支持。
- `image_utils.py`：图像预处理与增强配置。
- `model_ablation.py`：仅文本/仅图像等消融模型。
- `model_tfidf_resnet.py`：基线多模态模型（TF‑IDF + ResNet18）。
- `text_utils.py`：文本清洗与读取工具。
