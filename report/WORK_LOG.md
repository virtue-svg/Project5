# 实验五进度记录

更新日期: 2026-01-26

## 已完成工作
- 初始化项目结构: 创建 `src/`、`scripts/`、`outputs/`、`report/`，配置 `.gitignore`。
- 安装基础依赖并冻结到 `requirements.txt`。
- 数据解压与组织: 将 `project5.zip` 解压到 `data/project5/`。
- 数据读取与划分:
  - 以 `train.txt`/`test_without_label.txt` 的 guid 为准，忽略 data 内冗余文件。
  - 分层划分训练/验证集，输出 `outputs/splits/train.csv`、`val.csv`、`test.csv`。
- 文本预处理:
  - 基础清洗: 去 URL、@提及、# 处理、空白规范化、转小写。
  - 进阶清洗: emoji 标记、重复字符折叠、可选停用词过滤。
  - 输出 `outputs/processed/text_cleaned.csv`。
- 数据统计:
  - 统计标签分布、文本长度、图像尺寸、缺失情况。
  - 输出 `outputs/processed/data_stats.csv` 与 `data_stats_summary.json`。
- 图像增强（仅训练时使用，不修改原图）:
  - 定义保守增强策略: 随机裁剪/缩放、水平翻转、轻微颜色抖动。
  - 生成可视化样例到 `outputs/processed/aug_samples/`。
- 修复 PyTorch 导入问题并稳定环境:
  - 由 `torch==2.10.0` + `numpy==2.x` 组合导致 DLL 初始化失败，改为 `torch==2.1.2+cpu` 与 `numpy==1.26.4`。
  - 更新 `requirements.txt`。

## 已生成关键文件
- `outputs/splits/train.csv`
- `outputs/splits/val.csv`
- `outputs/splits/test.csv`
- `outputs/processed/text_cleaned.csv`
- `outputs/processed/data_stats.csv`
- `outputs/processed/data_stats_summary.json`
- `outputs/processed/aug_samples/*.png`

## 过程中遇到的问题与解决方案
- 问题: `Expand-Archive` 解压数据集超时。
  - 解决: 使用 Python `zipfile` 直接解压，成功完成。
- 问题: `train.txt` 含表头 `guid` 导致缺失样本报错。
  - 解决: 读取时跳过 `guid` 表头行。
- 问题: `torch` 导入报错 `WinError 1114` (c10.dll 依赖失败)。
  - 解决: 改用 `torch==2.1.2+cpu` 并降级 `numpy==1.26.4`，导入正常。
- 问题: `torch 2.1.2` 与 `numpy 2.x` 不兼容。
  - 解决: 锁定 `numpy==1.26.4`，并同步 `pandas==2.2.3`。

## 训练集概要
- 样本数: 4000
- 标签分布: positive 2388 / negative 1193 / neutral 419
- 文本与图片缺失: 0

## 下一阶段计划
- 建立多模态模型与训练流程脚手架（dataset / dataloader / train / eval）。
