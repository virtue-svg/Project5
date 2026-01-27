# 阶段三：最优模型优化（CLIP）

本阶段基于阶段二的最优模型（CLIP），进行进一步优化与微调。

## 环境准备
```powershell
.\.venv\Scripts\Activate.ps1
```

## 训练脚本：`optimize_clip.py`
- 作用：在固定划分下，对 CLIP 进行优化训练（可冻结编码器、早停等）。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/optimize_clip.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 8 --batch-size 16 --num-workers 2 --pin-memory --freeze-epochs 1 --early-stop 3 --dropout 0.2
```

## 超参搜索：`clip_grid_search.py`
- 作用：对学习率、batch size、权重衰减、dropout 做网格搜索。
- 运行示例：
```powershell
.\.venv\Scripts\python stage3_optimize/scripts/clip_grid_search.py --project-root . --model-name openai/clip-vit-base-patch32 --epochs 8 --freeze-epochs 1 --early-stop 3 --batch-sizes 8,16 --lrs 1e-5,2e-5 --weight-decays 1e-4,5e-4 --dropouts 0.1,0.2 --num-workers 2 --pin-memory
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
