# 阶段三：最优模型优化计划（CLIP）

## 目标
在阶段二选出的最佳模型基础上提升性能，验证稳定性与泛化能力，并通过数据驱动策略进一步提升指标，最终生成测试集预测结果。

## 核心策略
1) 超参搜索与微调：确定最优超参组合  
2) 结构小改动：比较 base / ln / mlp / gated 头部  
3) 稳定性测试：多随机种子复现  
4) 泛化测试：噪声/模糊扰动评估  
5) Bad case 挖掘与定向增强  
6) 增强后再训练与测试集预测

## 执行步骤
1) 网格搜索与微调，确定最优超参  
2) 结构消融：比较不同头部结构  
3) 选择最终模型（CLIP + ln）  
4) 稳定性测试（seed=42/123/2024）  
5) 泛化测试（noise/blur）  
6) 挖掘 bad cases 并生成错误分析可视化  
7) 定向增强（图像+文本），合并训练集再训练  
8) 使用增强后权重生成测试集提交文件

## 公平性控制
- 固定 train/val 划分与随机种子  
- 统一训练轮数与评估指标  
- 结构对比仅改变头部，其他超参保持一致  
- 定向增强放在对比实验之后，避免污染公平性

## 预期输出
- 最优权重与指标：`outputs/optimize/head_variant_compare/`  
- 稳定性/泛化：`outputs/optimize/stability/`、`outputs/optimize/robustness_ln/`  
- Bad case：`outputs/optimize/bad_cases.csv` 与错误分析图  
- 增强后训练：`outputs/optimize/final_clip_aug/`  
- 测试集预测：`outputs/test_with_label_clip_aug.txt`
