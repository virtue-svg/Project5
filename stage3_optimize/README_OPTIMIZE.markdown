﻿## 阶段三：最优模型优化（CLIP）

本阶段基于阶段二的最优模型（CLIP），进行进一步优化与微调。

## 训练脚本：`optimize_clip.py`
- 作用：在固定划分下，对 CLIP 进行优化训练（可冻结编码器、早停等）。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/optimize_clip.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 8 --batch-size 16 --num-workers 2 --pin-memory --freeze-epochs 1 --early-stop 3 --dropout 0.2
```

## 最终训练：`final_train_clip.py`
- 作用：用“最优超参”进行最终训练，并统一输出到 `outputs/optimize/final_clip/`。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/final_train_clip.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 8 --freeze-epochs 1 --early-stop 3 --batch-size 8 --lr 1e-5 --weight-decay 1e-4 --dropout 0.2 --num-workers 0 --output-dir outputs/optimize/final_clip
```

## 超参搜索：`clip_grid_search.py`
- 作用：对学习率、batch size、权重衰减、dropout 做网格搜索。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/clip_grid_search.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 8 --freeze-epochs 1 --early-stop 3 --batch-sizes 8,16 --lrs 1e-5,2e-5 --weight-decays 1e-4,5e-4 --dropouts 0.1,0.2 --num-workers 2 --pin-memory
```

## Bad case 挖掘：`mine_bad_cases.py`
- 作用：在验证集上找错误/低置信度样本，生成 `bad_cases.csv`（guid/真实/预测/置信度/路径）。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/mine_bad_cases.py --project-root . --model-name openai/clip-vit-base-patch32 --weights outputs/optimize/final_clip/best.pt --batch-size 16 --num-workers 0 --output-csv outputs/optimize/bad_cases.csv
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
- 模型权重: `outputs/optimize/clip_openai_clip-vit-base-patch32_opt/best.pt`
- 指标: `outputs/optimize/clip_openai_clip-vit-base-patch32_opt/metrics.json`
- 训练曲线: `outputs/optimize/clip_openai_clip-vit-base-patch32_opt/history.csv`
- 最终训练输出: `outputs/optimize/final_clip/`
- Bad case 清单: `outputs/optimize/bad_cases.csv`
- Bad case 增强: `outputs/optimize/badcase_aug/`
- Bad case 文本增强: `outputs/optimize/badcase_text_aug.csv`
- 错误分析图表: `outputs/optimize/error_analysis_*.png`
