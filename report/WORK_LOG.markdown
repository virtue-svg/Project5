# 工作日志（实验五）

更新时间：2026-01-28

## 一、已完成工作
### 阶段一：基线
- 完成数据解压与组织：以 `train.txt` / `test_without_label.txt` 的 guid 为准，忽略 data 内冗余文件。
- 完成文本清洗与统计：输出 `outputs/processed/text_cleaned.csv`、`data_stats.csv` 等。
- 完成保守图像增强可视化：生成 `outputs/processed/aug_samples/*.png`。
- 基线模型训练：TF‑IDF + ResNet18 concat；生成混淆矩阵、ROC、相关性图。
- 消融实验：仅文本 / 仅图像 / 多模态基线，并生成对比表。
- 生成测试集预测文件（基线版本）。

### 阶段二：多模态对比
- 完成 BERT+ResNet18（concat/gated/late）对比训练。
- 完成 CLIP、BLIP 对比训练与汇总。
- 生成对比可视化：雷达图、参数‑时间图、loss 曲线、结果表。
- 选出 CLIP 作为后续优化对象。

### 阶段三：优化与扩展
- 超参网格搜索与微调：确定 CLIP 最优配置。
- 结构小改动对比：base / ln / mlp / gated，ln 最优。
- 稳定性测试：三种 seed 复现并统计均值与方差。
- 泛化测试：噪声/模糊扰动评估并记录指标。
- Bad case 挖掘、错误类型分析可视化（文本长度 / emoji / 图像尺寸）。
- 定向增强：图像增强 + 文本增强；合并训练集后再训练。
- 增强后再训练：Macro‑F1 显著提升（记录于报告）。
- 生成 CLIP 最终测试集预测文件。

### 文档与结构
- 三阶段 README 已补全，阶段三新增流程与常见问题。
- `data/`、`outputs/`、`report/` 目录新增 README 说明文件。
- 报告 `REPORT.markdown` 已完善到摘要与总结，并同步最新实验结果。

## 二、关键输出
- 训练划分：`outputs/splits/train.csv`、`val.csv`、`test.csv`
- 基线指标与可视化：`outputs/metrics_tfidf_resnet.*`、`outputs/visuals/*`
- 对比汇总：`outputs/compare/compare_summary.csv` 与 `outputs/compare/visuals/*`
- 稳定性与泛化：`outputs/optimize/stability/*`、`outputs/optimize/robustness_ln/*`
- Bad case：`outputs/optimize/bad_cases.csv`、错误分析图
- 定向增强：`outputs/optimize/badcase_aug/*`、`badcase_text_aug.csv`
- 增强后再训练：`outputs/optimize/final_clip_aug/*`
- 最终测试集预测：`outputs/test_with_label_clip_aug.txt`

## 三、问题与解决
- CUDA 报错（CUBLAS/illegal instruction）：降低 batch，`num-workers=0`，必要时开启 `CUDA_LAUNCH_BLOCKING=1`。
- BLIP 权重/维度问题：使用 LazyLinear 并在前向后统计参数。
- Windows 多进程 DataLoader 报错：避免 lambda，改用 `functools.partial`。
- bad case 挖掘权重加载冲突：从权重中读取 `head_variant`，并使用 `strict=False`。
- HuggingFace 下载超时：依赖自动重试或使用缓存继续。


