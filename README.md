# 精神疾病辅助诊断结果预测 - Baseline

这是一个精神疾病辅助诊断结果预测的baseline项目，实现了多种文本分类方法来预测患者的精神疾病诊断类别。

## 项目概述

### 任务描述
基于医患对话数据，预测患者的精神疾病诊断类别。这是一个4分类问题：
- 0: 抑郁相关 (重性抑郁发作等)
- 1: 双相情感障碍 (双相I型、双相II型等)
- 2: 焦虑相关 (广泛性焦虑障碍等)
- 3: 多动障碍 (ADHD等)

### 数据集
- 数据位于 `./DiagnosisData/` 目录
- 包含1500个JSON格式的医患对话文件
- 按patient_id进行80%-20%分割，确保同一患者的数据不会同时出现在训练集和测试集中
- 处理类别不平衡问题

## 实现的Baseline方法

### 1. TF-IDF + 传统机器学习
- **逻辑回归 (Logistic Regression)**
- **随机森林 (Random Forest)**
- **支持向量机 (SVM)**

特点：
- 使用jieba进行中文分词
- TF-IDF特征提取
- 网格搜索超参数优化
- 处理类别不平衡

### 2. ChineseBERT Fine-tuning
- 基于 `hfl/chinese-bert-wwm-ext` 预训练模型
- 使用transformers库进行fine-tuning
- 支持GPU加速训练
- 早停机制防止过拟合

### 3. Qwen3 Zero-shot
- 通过OpenAI API格式调用本地部署的Qwen3-8B模型
- 无需训练，直接推理
- 专门设计的提示工程
- 支持批量预测和进度保存

## 项目结构

```
challenge_baseline/
├── DataSyn/                     # 原始数据目录
├── processed_data/              # 预处理后的数据
├── models/                      # 模型实现
│   ├── __init__.py
│   ├── tfidf_baseline.py       # TF-IDF基线模型
│   ├── chinesebert_baseline.py # ChineseBERT基线模型
│   └── qwen3_zeroshot.py       # Qwen3零样本模型
├── models/saved/                # 保存的模型
├── results/                     # 结果输出
├── create_4class_labels.py     # 标签映射工具
├── data_preprocessing.py       # 数据预处理
├── run_all_baselines.py       # 主运行脚本
├── requirements.txt            # 依赖包
└── README.md                   # 项目说明
```

## 安装与使用

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt

# 如果使用GPU，请安装对应版本的PyTorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 运行方式

#### 运行所有baseline
```bash
python run_all_baselines.py
```

#### 运行特定模型
```bash
# 只运行TF-IDF模型
python run_all_baselines.py --only-tfidf

# 只运行ChineseBERT模型
python run_all_baselines.py --only-bert

# 只运行Qwen3模型
python run_all_baselines.py --only-qwen3

# 跳过数据预处理
python run_all_baselines.py --skip-preprocessing

# 只比较已有结果
python run_all_baselines.py --compare-only
```

#### 单独运行各模块
```bash
# 数据预处理
python data_preprocessing.py

# TF-IDF模型
python models/tfidf_baseline.py

# ChineseBERT模型
python models/chinesebert_baseline.py

# Qwen3模型（需要先启动Qwen3服务）
python models/qwen3_zeroshot.py
```

### 3. Qwen3服务配置

使用Qwen3 zero-shot需要先启动本地服务：

```bash
# 设置环境变量
export OPENAI_API_BASE="xxx"
export MODEL_NAME="xxx"

# 确保Qwen3服务在指定端口运行
# 服务应该兼容OpenAI API格式
```

## 数据预处理详情

### 患者ID提取
从文件名提取患者ID：
- 文件格式：`dialogue_1.json`, `dialogue_2.json` 等
- 患者ID：从 `DiagnosisData` 文件夹下的文件名中提取
- 提取规则：文件名中的数字部分作为患者标识

### 数据分割策略
1. 按患者ID分组
2. 计算每个患者的主要诊断标签
3. 分层采样确保训练集和测试集的标签分布相似
4. 80%-20%分割比例

### 标签映射
将复杂的诊断结果映射为4个主要类别：
- 优先级：双相 > 抑郁 > 焦虑 > 多动
- 支持多种诊断表述的识别

## 结果输出

### 评估指标
- **准确率 (Accuracy)**
- **F1 Score (weighted)** - 考虑类别不平衡
- **F1 Score (macro)** - 各类别平均表现
- **分类报告** - 每个类别的详细指标
- **混淆矩阵** - 可视化分类结果
