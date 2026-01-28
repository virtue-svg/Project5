# 阶段二 多模态模型对比

## 目标
在统一数据划分与训练设置下，对比不同融合策略，选出本次实验最佳模型，再进行单独优化。

## 对比模型（基于已有研究与常见结构）
1) Concat Fusion（Early Fusion）
   - 模型: BERT + ResNet18 + 拼接 + MLP
   - 理由: 早期融合能直接建模跨模态关联，是常见基线扩展。
2) Gated Fusion（门控融合）
   - 模型: BERT + ResNet18 + 门控融合
   - 理由: 通过门控学习模态权重，缓解单模态噪声影响。
3) Late Fusion（后融合）
   - 模型: BERT 分支 + ResNet18 分支，各自分类后取平均
   - 理由: 结构简单、可解释，作为后融合代表。
4) CLIP: 使用 `openai/clip-vit-base-patch32` 获取图文对齐特征，再做分类头。
5) BLIP: 使用 `Salesforce/blip-image-captioning-base` 获取图文特征，再做分类头。

## 公平对比策略
- 固定训练/验证划分: 使用 `outputs/splits/train.csv` 与 `outputs/splits/val.csv`
- 固定随机种子: 42
- 固定超参: epoch、batch size、优化器、学习率
- 固定输入: 文本最大长度、图像分辨率
- 仅改变模型结构与融合策略

## 评估指标
- Acc
- Macro-F1（主指标）
- Precision / Recall（macro）

## 结果整理
运行结束后汇总到:
`outputs/compare/compare_summary.csv`
