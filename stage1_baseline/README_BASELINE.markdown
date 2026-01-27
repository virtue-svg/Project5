﻿## 阶段一：基线模型（运行说明）

本阶段包含数据准备、基线模型训练、消融实验与测试集预测，按照顺序即可复现基线模型。
基线模型的目的：
1. 提供**可量化的参考点**，供多模态对比时进行模型的选择。
2. 作为**流程正确性验证**（数据、训练、评估全链路能跑通）

## 环境准备
```powershell
.\.venv\Scripts\Activate.ps1
```

## 安装依赖：
```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 脚本说明与运行方式

### 1) 数据划分：`prepare_data.py`
- 作用：读取 `train.txt` / `test_without_label.txt`，生成 train/val/test 划分 CSV。
- 运行：
```powershell
.\.venv\Scripts\python stage1_baseline/scripts/prepare_data.py --project-root .
```
- 输出：
  - `outputs/splits/train.csv`：训练集索引与路径。
  - `outputs/splits/val.csv`：验证集索引与路径。
  - `outputs/splits/test.csv`：测试集索引与路径。

### 2) 文本预处理：`preprocess_text.py`
- 作用：文本清洗（基础 + 进阶），输出清洗文本与长度统计。
- 运行：
```powershell
.\.venv\Scripts\python stage1_baseline/scripts/preprocess_text.py --project-root .
```
- 输出：
  - `outputs/processed/text_cleaned.csv`：原文、清洗文本、长度统计。

### 3) 数据统计：`data_stats.py`
- 作用：统计标签分布、文本长度、图像尺寸等。
- 运行：
```powershell
.\.venv\Scripts\python stage1_baseline/scripts/data_stats.py --project-root .
```
- 输出：
  - `outputs/processed/data_stats.csv`：样本级统计明细。
  - `outputs/processed/data_stats_summary.json`：总体统计摘要。

### 4) 图像增强可视化：`preview_augmentations.py`
- 作用：生成“原图 + 增强图”样例，用于报告展示。
- 运行：
```powershell
.\.venv\Scripts\python stage1_baseline/scripts/preview_augmentations.py --project-root . --num-samples 20
```
- 输出：
  - `outputs/processed/aug_samples/*.png`：增强样例拼图。

### 5) 基线模型训练：`train_tfidf_resnet.py`
- 作用：训练基线模型（TF-IDF + ResNet18 + concat 融合）。
- 运行：
```powershell
.\.venv\Scripts\python stage1_baseline/scripts/train_tfidf_resnet.py --project-root . --batch-size 64 --num-workers 4 --pin-memory
```
- 输出：
  - `outputs/models/best_tfidf_resnet.pt`：最佳权重。
  - `outputs/metrics_tfidf_resnet.json`：最佳 Macro-F1 摘要。
  - `outputs/metrics_tfidf_resnet.csv`：每轮指标与损失。
  - `outputs/metrics_tfidf_resnet.png`：loss/acc 曲线图。
  - `outputs/visuals/confusion.png`：混淆矩阵。
  - `outputs/visuals/roc.png`：ROC 曲线。
  - `outputs/visuals/correlation.png`：长度/尺寸相关性图。

### 6) 消融：仅文本 `train_text_only.py`
- 作用：仅文本分支（TF-IDF + MLP）。
- 运行：
```powershell
.\.venv\Scripts\python stage1_baseline/scripts/train_text_only.py --project-root . --batch-size 64 --num-workers 2 --pin-memory
```
- 输出：
  - `outputs/models/best_text_only.pt`：最佳权重。
  - `outputs/metrics_text_only.json`：最佳 Macro-F1 摘要。
  - `outputs/metrics_text_only.csv`：每轮指标与损失。

### 7) 消融：仅图像 `train_image_only.py`
- 作用：仅图像分支（ResNet18）。
- 运行：
```powershell
.\.venv\Scripts\python stage1_baseline/scripts/train_image_only.py --project-root . --batch-size 32 --num-workers 4 --pin-memory
```
- 输出：
  - `outputs/models/best_image_only.pt`：最佳权重。
  - `outputs/metrics_image_only.json`：最佳 Macro-F1 摘要。
  - `outputs/metrics_image_only.csv`：每轮指标与损失。

### 8) 消融对比汇总：`ablation_summary.py`
- 作用：汇总三种模型最佳指标，用于报告表格。
- 运行：
```powershell
.\.venv\Scripts\python stage1_baseline/scripts/ablation_summary.py --project-root .
```
- 输出：
  - `outputs/ablation_summary.csv`：三模型对比表。

### 9) 测试集预测：`predict_test.py`
- 作用：生成提交文件（替换 `null` 标签）。
- 运行：
```powershell
.\.venv\Scripts\python stage1_baseline/scripts/predict_test.py --project-root .
```
- 输出：
  - `outputs/test_with_label.txt`：最终提交文件。

## 常见问题（阶段一）
- 预测结果全为 `neutral`：多由 `test_without_label.txt` 使用逗号分隔导致解析失败，需使用修复后的脚本版本。
- 相关性图报错 `array length 400 does not match index length 4000`：需确保相关性图仅使用验证集样本生成。
- PyTorch 导入 `WinError 1114`：固定 `torch==2.1.2+cu121` 与 `numpy==1.26.4`。
