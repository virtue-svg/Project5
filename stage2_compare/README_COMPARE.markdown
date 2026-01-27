﻿## 阶段二 多模态模型对比

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

## 训练对比模型
### 1) Concat Fusion（早期融合）
```powershell
.\.venv\Scripts\python stage2_compare/scripts/compare_train.py --project-root . --model-name bert-base-chinese --image-backbone resnet18 --fusion concat --epochs 5 --batch-size 16 --num-workers 2 --pin-memory
```
- 作用：BERT（文本）+ ResNet18（图像）+ concat 融合训练。
- 输出：`outputs/compare/bert-base-chinese_resnet18_concat/`（含 `best.pt`, `metrics.json`, `history.csv`）

### 2) Gated Fusion（门控融合）
```powershell
.\.venv\Scripts\python stage2_compare/scripts/compare_train.py --project-root . --model-name bert-base-chinese --image-backbone resnet18 --fusion gated --epochs 5 --batch-size 16 --num-workers 2 --pin-memory
```
- 作用：BERT + ResNet18 + 门控融合训练。
- 输出：`outputs/compare/bert-base-chinese_resnet18_gated/`（含 `best.pt`, `metrics.json`, `history.csv`）

### 3) Late Fusion（后融合）
```powershell
.\.venv\Scripts\python stage2_compare/scripts/compare_train.py --project-root . --model-name bert-base-chinese --image-backbone resnet18 --fusion late --epochs 5 --batch-size 16 --num-workers 2 --pin-memory
```
- 作用：BERT 与 ResNet18 分别分类，后融合取平均。
- 输出：`outputs/compare/bert-base-chinese_resnet18_late/`（含 `best.pt`, `metrics.json`, `history.csv`）

### 4) CLIP 对比
```powershell
.\.venv\Scripts\python stage2_compare/scripts/compare_clip.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 5 --batch-size 16 --num-workers 2 --pin-memory
```
- 作用：CLIP 图文编码 + 分类头训练。
- 输出：`outputs/compare/clip_openai_clip-vit-base-patch32/`（含 `best.pt`, `metrics.json`, `history.csv`）

### 5) BLIP 对比
```powershell
.\.venv\Scripts\python stage2_compare/scripts/compare_blip.py --project-root . --model-name Salesforce/blip-image-captioning-base --epochs 5 --batch-size 8 --num-workers 2 --pin-memory
```
- 作用：BLIP 图文编码 + 分类头训练。
- 输出：`outputs/compare/blip_Salesforce_blip-image-captioning-base/`（含 `best.pt`, `metrics.json`, `history.csv`）

## 汇总对比结果
```powershell
.\.venv\Scripts\python stage2_compare/scripts/compare_summary.py --project-root .
```

输出: `outputs/compare/compare_summary.csv`
作用：汇总所有对比模型的最佳指标（Acc/F1/Prec/Recall）。

## 对比可视化
```powershell
.\.venv\Scripts\python stage2_compare/scripts/compare_visualize.py --project-root .
```

输出目录: `outputs/compare/visuals/`
作用：生成雷达图、参数量-时间图、loss 曲线与结果表格图。

## 常见问题（阶段二）
- `AutoModel requires the PyTorch library`：transformers 5.x 需要 torch>=2.2，建议固定 `transformers==4.39.3`。
- DataLoader 报 `Can't pickle local object`：Windows 多进程 + lambda collate，需改为 `functools.partial` 或 `--num-workers 0`。
- CLIP/BLIP 训练中 CUDA 报错（CUBLAS/illegal instruction）：降 batch，设置 `--num-workers 0`，必要时加 `CUDA_LAUNCH_BLOCKING=1`。
- BLIP 权重部分未初始化提示：属于正常提示，需微调以适配下游任务。
- 重构后脚本路径变更：请使用 `stage2_compare/scripts/` 下的新路径。
