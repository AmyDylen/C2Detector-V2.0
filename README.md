# C2Detector-V2.0
## 项目简介

这是一个基于深度学习的C2（Command and Control）流量检测系统，结合了多种先进技术和架构设计，包括元属性交叉注意力机制、Transformer架构、卷积神经网络以及数据增强技术，并支持GPU加速训练。

## 主要特性

- **元属性交叉注意力机制**
: 对不同特征组（方向、大小、时间、标志、比率、差值）使用交叉注意力，提高特征交互效果
- **多层次架构**
: 结合Transformer、卷积层和注意力机制的混合模型
- **动态区间桶化**
: 自动对数值特征进行动态区间分桶，适应不同量级的数据
- **GPU加速**
: 支持CUDA和混合精度训练，显著提升训练速度
- **数据增强**
: 在数据加载阶段支持样本限制和数据平衡
- **全面评估**
: 包含准确率、精确率、召回率、F1分数、误报率等多维度评估指标
- **错误分析**
: 提供详细的错误分类分析和混淆矩阵

## 技术架构

### 模型结构
- **特征分组处理**
: 将输入特征分为三组（方向/大小/时间、方向/大小/时间、组合属性）
- **组内注意力**
: 前两组使用多头注意力机制，第三组直接投影
- **跨组注意力**
: 在前两组之间应用跨组注意力
- **卷积块**
: 使用1D卷积进行特征提取和降维
- **全局Transformer**
: 对卷积输出应用全局Transformer编码
- **分类器**
: 最终的全连接层分类器

### 特征处理
- **方向特征**
: 使用嵌入层处理方向信息
- **大小特征**
: 动态区间桶化处理大小特征
- **时间特征**
: 多阈值时间区间桶化
- **组合属性**
: 包括标志、比率、差值等组合特征

## 依赖库

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```
CSV文件应包含以下列：
- `Session ID`: 会话ID
- `direction1`, `direction2`: 方向特征
- `size1`, `size2`: 大小特征
- `time1`, `time2`: 时间特征
- `flag`: 标志特征
- `ratio`: 比率特征
- `diff`: 差值特征

## 使用方法

### 1. 基本训练

```python
from 元属性交叉注意力+组合属性+transfomer+卷积+数据增强-GPU加速 import train_model

# 设置数据路径
DATA_PATH = "path/to/your/data"

# 启动训练
train_model(DATA_PATH)
```

### 2. 自定义配置

```python
# 自定义配置参数
CUSTOM_CONFIG = {
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 100,
    'max_seq_len': 50,
    'd_model': 128,
    'min_samples': 10,
    'patience': 5,
    'train_ratio': 0.9,
    'val_ratio': 0.1,
    'max_samples_per_class': 10000,
    'use_amp': True,
    'num_workers': 4
}

train_model(DATA_PATH, split_params=CUSTOM_CONFIG)
```

### 3. 配置参数说明

- `batch_size`: 批次大小，默认32
- `lr`: 学习率，默认1e-4
- `epochs`: 训练轮数，默认100
- `max_seq_len`: 最大序列长度，默认50
- `d_model`: 模型维度，默认128
- `min_samples`: 每类最小样本数，默认10
- `patience`: 早停耐心值，默认5
- `train_ratio`: 训练集比例，默认0.9
- `val_ratio`: 验证集比例，默认0.1
- `max_samples_per_class`: 每类最大样本数，默认10000
- `use_amp`: 是否使用混合精度训练，默认True
- `num_workers`: DataLoader工作进程数

## 输出文件

- **训练日志**: `training_and_Test_log_[timestamp].txt`
- **最佳模型**: `best_model_epoch_[epoch]_val_acc_[accuracy]_[timestamp].pth`
- **最终模型**: `final_model_test_acc_[accuracy]_[timestamp].pth`
- **详细测试结果**: `detailed_test_results_[timestamp].json`

## 评估指标

- **准确率 (Accuracy)**
- **精确率 (Precision)**
- **召回率 (Recall)**
- **F1分数 (F1 Score)**
- **误报率 (False Positive Rate)**
- **混淆矩阵**

## GPU加速

该模型支持GPU加速，包括：
- CUDA设备检测和使用
- 混合精度训练 (AMP)
- 张量内存优化
- 并行数据加载

## 模型保存与加载

训练过程中会自动保存：
- 每轮最佳验证精度的模型
- 最终测试精度的模型
- 包含训练配置和状态的完整模型文件

## 错误分析

系统提供详细的错误分类分析，包括：
- 错误分类模式统计
- 各类别错误率分析
- 具体错误样本详情
- 预测置信度分析

## 注意事项

1. 确保数据路径正确且数据格式符合要求
2. GPU内存充足时启用混合精度训练以提高效率
3. 根据数据集大小调整`max_samples_per_class`参数
4. 根据系统资源调整`num_workers`参数
5. 监控训练过程中的早停机制
