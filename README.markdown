# 实验五：多模态情感分类（Project5）

本仓库按“三阶段”组织：基线 → 多模态对比 → 最优模型优化，并给出完整可复现的训练与评估流程。

## 目录
- 项目概览
- 文件夹说明
- 代码结构
- 完整执行流程
- 参考与致谢

## 项目概览
本项目完成多模态情感分类任务，涵盖：
- 基线模型训练与消融实验
- 多模态模型结构对比（BERT+ResNet18 / CLIP / BLIP）
- 最优模型优化、稳定性/泛化测试、bad case 定向增强与再训练
- 测试集预测文件生成

## 文件夹说明
- `src/`：核心模型与通用工具（数据处理、模型定义、清洗与增强）。
- `stage1_baseline/`：基线流程脚本与说明（数据划分、训练、消融、预测）。
- `stage2_compare/`：多模态对比流程脚本与说明（融合对比、可视化汇总）。
- `stage3_optimize/`：最优模型优化与扩展流程脚本与说明（网格搜索、结构改动、稳定性/泛化、bad case）。
- `data/`：数据放置目录（仅保留说明文件）。
- `outputs/`：训练输出目录（仅保留说明文件）。
- `report/`：实验报告与阶段计划、工作日志。

## 代码结构
```
.
├── src/
├── stage1_baseline/
│   ├── README_BASELINE.markdown
│   └── scripts/
├── stage2_compare/
│   ├── README_COMPARE.markdown
│   └── scripts/
├── stage3_optimize/
│   ├── README_OPTIMIZE.markdown
│   └── scripts/
├── data/README.markdown
├── outputs/README.markdown
├── report/
│   ├── REPORT.markdown
│   ├── WORK_LOG.markdown
│   ├── STAGE1_PLAN.markdown
│   ├── COMPARE_PLAN.markdown
│   └── STAGE3_PLAN.markdown
├── requirements.txt
└── README.markdown
```

## 完整执行流程
**建议按阶段 README 依次执行，流程如下：**

### 0) 环境准备
```powershell
.
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 1) 阶段一：基线
详见 `stage1_baseline/README_BASELINE.markdown`，包含：
- 数据划分、文本清洗、数据统计、增强可视化
- 基线模型训练（TF‑IDF + ResNet18 concat）
- 消融实验（仅文本 / 仅图像）
- 基线测试集预测

### 2) 阶段二：多模态对比
详见 `stage2_compare/README_COMPARE.markdown`，包含：
- BERT+ResNet18（concat/gated/late）
- CLIP / BLIP 对比训练
- 对比汇总与可视化图表

### 3) 阶段三：最优模型优化
详见 `stage3_optimize/README_OPTIMIZE.markdown`，包含：
- 网格搜索 / 结构小改动 / 预处理对比
- 稳定性与泛化测试
- bad case 挖掘与定向增强
- 合并增强样本再训练
- 测试集预测文件生成

## 参考与致谢
**模型与论文参考：**
- **ResNet18** — He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition*. CVPR 2016, pp. 770-778. DOI: 10.1109/CVPR.2016.90。
  - 说明：图像分支使用 ResNet18 作为视觉特征提取 backbone。  
- **BERT** — Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv: 1810.04805。
  - 说明：文本分支采用 BERT 编码（`bert-base-chinese`）。  
- **CLIP** — Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. arXiv: 2103.00020。
  - 说明：使用 `openai/clip-vit-base-patch32` 进行对比与最终优化。  
- **BLIP** — Li, J. et al. *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. arXiv: 2201.12086。
  - 说明：使用 `Salesforce/blip-image-captioning-base` 参与对比。  

**代码与工具参考：**
- PyTorch：模型训练、优化器与张量运算。  
- HuggingFace Transformers：加载 BERT/CLIP/BLIP 的预训练模型与 tokenizer/processor。  
- scikit‑learn：TF‑IDF 向量化与部分基础统计处理。  
- torchvision：图像增强与预处理变换（Resize、Crop、ColorJitter 等）。  
- pandas / matplotlib：数据统计、结果表格与可视化输出。  

如需复现实验细节、运行参数或结果解读，请直接查看各阶段 README 与 `report/REPORT.markdown`。
