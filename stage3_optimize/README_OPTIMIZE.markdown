﻿## 阶段三：最优模型优化（CLIP）

本阶段基于阶段二的最优模型（CLIP），进行进一步优化与微调，并完成稳定性/泛化测试、bad case 定向增强与最终预测。

## 训练脚本：`optimize_clip.py`
- 作用：在固定划分下，对 CLIP 进行优化训练（可冻结编码器、早停、结构小改动与消融）。
- 常用参数：`--head-variant ln`、`--loss ce|focal|cb|combo`、`--use-cosine --warmup-ratio`
- 运行示例（最终模型配置）：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/optimize_clip.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 12 --freeze-epochs 1 --early-stop 3 --batch-size 8 --lr 8e-6 --weight-decay 1e-4 --dropout 0.15 --num-workers 0 --head-variant ln
```

## 超参搜索：`clip_grid_search.py`
- 作用：对学习率、batch size、权重衰减、dropout 做网格搜索。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/clip_grid_search.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 8 --freeze-epochs 1 --early-stop 3 --batch-sizes 4,8 --lrs 1e-5,2e-5 --weight-decays 1e-4,5e-4 --dropouts 0.1,0.2 --num-workers 0
```

## 结构小改动对比：`head_variant_compare.py`
- 作用：对比 base / ln / mlp / gated 头部结构。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/head_variant_compare.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 12 --freeze-epochs 1 --early-stop 3 --batch-size 8 --lr 8e-6 --weight-decay 1e-4 --dropout 0.15 --num-workers 0 --variants base,ln,mlp,gated
```

## 预处理改进对比：`preprocess_compare.py`
- 作用：仅调整清洗与增强规则，其余训练设置不变，对比 Macro‑F1。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/preprocess_compare.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 12 --freeze-epochs 1 --early-stop 3 --batch-size 8 --lr 8e-6 --weight-decay 1e-4 --dropout 0.15 --num-workers 0
```

## 稳定性/泛化测试：`robustness_test.py`
- 作用：在验证集加入噪声/模糊评估鲁棒性。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/robustness_test.py --project-root . --weights outputs/optimize/head_variant_compare/head_ln/best.pt --batch-size 16 --num-workers 0 --output-dir outputs/optimize/robustness_ln
```

## Bad case 挖掘：`mine_bad_cases.py`
- 作用：在验证集上找错误/低置信度样本，生成 `bad_cases.csv`（guid/真实/预测/置信度/路径）。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/mine_bad_cases.py --project-root . --model-name openai/clip-vit-base-patch32 --weights outputs/optimize/head_variant_compare/head_ln/best.pt --batch-size 16 --num-workers 0 --output-csv outputs/optimize/bad_cases.csv
```

## Bad case 图像增强：`augment_bad_cases.py`
- 作用：对 bad cases 的图像做轻量增强，输出增强样本与索引表。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/augment_bad_cases.py --badcase-csv outputs/optimize/bad_cases.csv --output-dir outputs/optimize/badcase_aug --num-aug 1
```

## Bad case 文本增强：`augment_bad_cases_text.py`
- 作用：对 bad cases 的文本做轻量增强（emoji 标记、重复字符折叠等），输出增强文本表。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/augment_bad_cases_text.py --badcase-csv outputs/optimize/bad_cases.csv --output-csv outputs/optimize/badcase_text_aug.csv
```

## 增强样本合并：`merge_badcase_aug.py`
- 作用：合并 bad case 增强样本与原训练集，生成 `train_aug.csv`。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/merge_badcase_aug.py --train-csv outputs/splits/train.csv --badcase-csv outputs/optimize/bad_cases.csv --badcase-image-csv outputs/optimize/badcase_aug/badcase_augmented.csv --badcase-text-csv outputs/optimize/badcase_text_aug.csv --output-csv outputs/optimize/train_aug.csv
```

## 再训练：`optimize_clip.py`
- 作用：在增强后的训练集上再训练。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/optimize_clip.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 12 --freeze-epochs 1 --early-stop 3 --batch-size 4 --lr 8e-6 --weight-decay 1e-4 --dropout 0.15 --num-workers 0 --head-variant ln --train-csv outputs/optimize/train_aug.csv --output-dir outputs/optimize/final_clip_aug
```

## 测试集预测：`predict_test_clip.py`
- 作用：使用 CLIP 模型生成最终测试集提交文件。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/predict_test_clip.py --project-root . --model-name openai/clip-vit-base-patch32 --weights outputs/optimize/final_clip_aug/best.pt --output outputs/test_with_label_clip_aug.txt --batch-size 32 --num-workers 0
```

## 错误类型分析（可视化）：`error_analysis.py`
- 作用：按文本长度、emoji、图像尺寸分桶统计错误率，并生成图表。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/error_analysis.py --project-root . --badcase-csv outputs/optimize/bad_cases.csv
```

## 结果对比：`compare_with_baseline.py`
- 作用：与阶段二 CLIP 基线做指标对比（Acc/F1/Prec/Recall）。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/compare_with_baseline.py --project-root .
```

## 主要输出
- 最优权重: `outputs/optimize/head_variant_compare/head_ln/best.pt`
- 结构对比: `outputs/optimize/head_variant_compare/`
- 预处理对比: `outputs/optimize/preprocess_compare/`
- 稳定性/泛化: `outputs/optimize/robustness_ln/`
- Bad case 清单: `outputs/optimize/bad_cases.csv`
- Bad case 增强: `outputs/optimize/badcase_aug/`
- Bad case 文本增强: `outputs/optimize/badcase_text_aug.csv`
- 增强后训练: `outputs/optimize/final_clip_aug/`
- 测试集预测: `outputs/test_with_label_clip_aug.txt`
- 错误分析图表: `outputs/optimize/error_analysis_*.png`

## 常见问题（阶段三）
- **CUDA 报错（CUBLAS / illegal instruction）**：优先将 `--num-workers 0`，并适当降低 batch（如 8→4）。
- **bad case 挖掘加载权重报错（layernorm keys）**：确保使用 ln 头的权重，或脚本已启用 `strict=False` 兼容加载。
- **HuggingFace 下载超时**：等待自动重试，必要时检查网络或使用缓存后继续运行。
