# 阶段一：基线模型计划

## 目标
构建可复现的多模态基线流程，验证数据读取、训练、评估链路完整可跑，并产出可量化的基准指标，为后续对比与优化提供参照。

## 模型与方法
- 文本：TF‑IDF 表示  
- 图像：ResNet18 预训练特征  
- 融合：特征拼接（concat）+ MLP 分类头  
- 损失：带类别权重的交叉熵  
- 评估：Acc / Macro‑F1 / Precision / Recall

## 执行步骤
1) 数据划分：生成 train/val/test 的 CSV 索引  
2) 文本清洗：输出清洗文本与长度统计  
3) 数据统计：标签分布、文本长度、图像尺寸  
4) 图像增强可视化：生成增强样例  
5) 训练基线模型：保存权重与指标  
6) 消融实验：仅文本 / 仅图像  
7) 测试集预测：生成提交文件

## 公平性控制
- 固定划分（train/val），固定随机种子  
- 统一训练轮数与评估指标  
- 仅作为基准，不做结构或超参搜索

## 预期输出
- `outputs/models/best_tfidf_resnet.pt`  
- `outputs/metrics_tfidf_resnet.*`  
- `outputs/visuals/*`  
- `outputs/ablation_summary.csv`  
- `outputs/test_with_label.txt`
