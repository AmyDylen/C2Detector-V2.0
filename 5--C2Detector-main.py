import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from functools import partial
import datetime
import sys
from torch.cuda.amp import autocast, GradScaler

# 创建日志记录器类
class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

# 设备配置函数
def get_device():
    """检测并返回可用的计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("⚠️  未检测到GPU，使用CPU")
    return device
        
# 数据加载模块
class CustomDataset(Dataset):
    def __init__(self, data_root, max_seq_len=50, 
                min_samples=10, max_samples_per_class=None):
        """
        参数说明：
        min_samples: int - 每个类别最小样本数 (默认10)
        max_samples_per_class: int - 每个类别最大样本数，用于快速训练 (None表示不限制)
        """
        self.sequences = []
        self.labels = []
        self.sample_info = []  # 新增：存储样本来源信息
        self.max_seq_len = max_seq_len
        self.max_samples_per_class = max_samples_per_class
        
        # 按类别收集数据
        class_data = {}  # {class_idx: [(seq_data, label, info), ...]}
        sample_info_data = {}  # {class_idx: [sample_info, ...]}

        # 遍历文件夹结构
        for class_idx, class_name in enumerate(sorted(os.listdir(data_root))):
            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_data[class_idx] = []
            sample_info_data[class_idx] = []
                
            # 处理每个类别的CSV文件
            for csv_file in os.listdir(class_dir):
                if not csv_file.endswith('.csv') or 'UDP' in csv_file:
                    continue
                    
                file_path = os.path.join(class_dir, csv_file)
                try:
                    # 读取并预处理数据
                    df = pd.read_csv(file_path)
                    
                    # 新增列名校验
                    required_columns = {'Session ID', 'direction1', 'size1', 'time1',
                                      'direction2', 'size2', 'time2', 'flag', 'ratio', 'diff'}
                    if not required_columns.issubset(df.columns):
                        print(f"文件 {file_path} 缺少必要列，已跳过")
                        continue
                        
                    # 按Session ID分组处理
                    for session_id, group in df.groupby('Session ID'):
                        # 提取特征数据
                        seq_data = group[['direction1', 'size1', 'time1',
                                       'direction2', 'size2', 'time2',
                                       'flag', 'ratio', 'diff']].values.astype(np.float32)
                        
                        # 修改后的序列分割逻辑
                        sequence_length = len(seq_data)
                        
                        # 原始长度有效性检查
                        if sequence_length < 1:
                            continue
                            
                        # 存储样本信息和来源
                        sample_info = {
                            'file_path': file_path,
                            'csv_file': csv_file,
                            'class_name': class_name
                        }
                        
                        class_data[class_idx].append((seq_data, class_idx))
                        sample_info_data[class_idx].append(sample_info)
                        
                        # 检查是否达到该类别的最大样本数限制
                        if self.max_samples_per_class and len(class_data[class_idx]) >= self.max_samples_per_class:
                            break
                        
                except Exception as e:
                    print(f"加载文件 {file_path} 时发生错误: {str(e)}")
                    continue
            
            # 如果该类别设置了最大样本数限制，进行随机采样
            if self.max_samples_per_class and len(class_data[class_idx]) > self.max_samples_per_class:
                print(f"类别 {class_idx} 原始样本数: {len(class_data[class_idx])}, 采样到: {self.max_samples_per_class}")
                # 随机采样
                sampled_indices = np.random.choice(len(class_data[class_idx]), 
                                                 self.max_samples_per_class, replace=False)
                class_data[class_idx] = [class_data[class_idx][i] for i in sampled_indices]
                sample_info_data[class_idx] = [sample_info_data[class_idx][i] for i in sampled_indices]
        
        # 将所有类别的数据合并
        for class_idx, data_list in class_data.items():
            self.sequences.extend([item[0] for item in data_list])
            self.labels.extend([item[1] for item in data_list])
            self.sample_info.extend(sample_info_data[class_idx])
                    
        # 检查每个类别的样本数，如果小于min_samples则报错
        self._check_min_samples(min_samples)
        
        # 最终数据检查
        print(f"\n数据加载完成，总样本数：{len(self.sequences)}")
        if len(self.sequences) == 0:
            raise RuntimeError("未加载到任何有效数据，请检查数据路径和文件格式")
            
    def _check_min_samples(self, min_samples):
        """检查每个类别的样本数是否满足最小要求"""
        # 按类别收集样本索引
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        print(f"\n检查每个类别的样本数，最小样本数要求: {min_samples}")
        print("类别分布：")
        for cls, indices in class_indices.items():
            count = len(indices)
            print(f"类别 {cls}: {count} 样本")
            
            # 如果样本数小于最小要求，直接报错
            if count < min_samples:
                raise ValueError(f"类别 {cls} 样本数不足：{count} < {min_samples}，无法进行训练")
        
        print(f"✓ 所有类别样本数均满足最小要求 (≥ {min_samples})")
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            'features': self.sequences[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'sample_info': self.sample_info[idx]  # 新增：返回样本信息
        }

# 修改后的模型定义
class HierarchicalAttentionClassifier(nn.Module):
    def __init__(self, d_model=96, n_classes=5):
        super().__init__()
        self.d_model = d_model
        
        self.time_thresholds =    [1000,10000, 100000, 1000000, 5000000,  10000000, 30000000, 60000000]
        self.time_bucket_widths = [10,  100,   100,    1000,    2000,    100000,   1000000,   2000000]
        time_total_buckets = self._calculate_total_buckets(self.time_thresholds, self.time_bucket_widths)
        
        self.size_thresholds = [10000, 100000, 1000000]
        self.size_bucket_widths = [1,  100,  10000]
        size_total_buckets = self._calculate_total_buckets(self.size_thresholds, self.size_bucket_widths)
        
        self.ratio_thresholds =    [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000]
        self.ratio_bucket_widths = [100,  100,   100,    1000,    10000,    100000,   1000000,    100000000,   10000000000,  100000000000]
        ratio_total_buckets = self._calculate_total_buckets(self.ratio_thresholds, self.ratio_bucket_widths)
        
        self.diff_thresholds =    [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 6000000000000]
        self.diff_bucket_widths = [100,   100,  100,    1000,    10000,    100000,    1000000,    100000000,   10000000000,  100000000000]
        diff_total_buckets = self._calculate_total_buckets(self.diff_thresholds, self.diff_bucket_widths)
        
        # 初始化嵌入层
        self.dir_emb = nn.Embedding(2, d_model)
        self.size_emb = nn.Embedding(size_total_buckets, d_model)
        self.time_emb = nn.Embedding(time_total_buckets, d_model)
        self.flag_emb = nn.Embedding(2, d_model)
        # 对数分箱
        self.ratio_emb = nn.Embedding(ratio_total_buckets, d_model) 
        self.diff_emb = nn.Embedding(diff_total_buckets, d_model)
        
        # 特征处理器（前两组带注意力，第三组直接投影）
        self.group_processors = nn.ModuleList([
            # 前两组处理器
            nn.Sequential(
                nn.MultiheadAttention(d_model, 4, batch_first=True),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model*2),
                nn.GELU(),
                nn.Linear(d_model*2, d_model*2)
            ) for _ in range(2)
        ] + [
            # 第三组处理器
            nn.Sequential(
                nn.Linear(d_model, d_model*2),
                nn.GELU(),
                nn.LayerNorm(d_model*2)
            )
        ])
        
        # 跨组注意力层
        self.cross_attn = nn.MultiheadAttention(d_model*2, 4, batch_first=True)
        
        # 修正卷积层输入维度
        self.conv_block = nn.Sequential(
            nn.Conv1d(2*(d_model*2) + (d_model*2), 512, 5, padding=2),  # 实际输入维度
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.AdaptiveMaxPool1d(100),
            nn.Dropout(0.2)
        )
        
        # 全局Transformer
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x, mask=None):
        # 特征分组处理
        B = x.size(0)  # batch size
        
        # 第一组（方向/尺寸/时间）
        group1 = torch.stack([
            self.dir_emb(self._safe_index(x[:,:,0])),
            self.size_emb(self._log_bucketize(x[:,:,1])),
            self.time_emb(self._time_bucketize(x[:,:,2]))
        ], dim=2)
        
        # 第二组（方向/尺寸/时间）
        group2 = torch.stack([
            self.dir_emb(self._safe_index(x[:,:,3])),
            self.size_emb(self._log_bucketize(x[:,:,4])),
            self.time_emb(self._time_bucketize(x[:,:,5]))
        ], dim=2)
        
        # 第三组（组合属性）
        group3 = torch.stack([
            self.flag_emb(self._flag_bucketize(x[:,:,6])),
            self.ratio_emb(self._ratio_bucketize(x[:,:,7])),
            self.diff_emb(self._diff_bucketize(x[:,:,8]))
        ], dim=2)
        
        # 组内处理
        g1 = self._process_group(group1, 0)  # [B, S, d_model*2]
        g2 = self._process_group(group2, 1)  # [B, S, d_model*2]
        g3 = self._process_group(group3, 2)  # [B, S, d_model*2]
        
        # 跨组注意力（仅前两组）
        attn_out, _ = self.cross_attn(g1, g2, g2, key_padding_mask=mask)
        
        # 正确的特征拼接方式
        final_combined = torch.cat([g1, g2, g3], dim=-1)  # [B, S, 3*d_model*2]
        
        # 维度调整
        conv_input = final_combined.transpose(1, 2)  # [B, 3*d_model*2, S]
        conv_feat = self.conv_block(conv_input)      # [B, 512, 100]
        
        # 全局Transformer处理
        trans_input = conv_feat.transpose(1, 2)  # [B, 100, 512]
        trans_out = self.global_transformer(trans_input)  # [B, 100, 512]
        
        # 分类决策
        aggregated, _ = torch.max(trans_out, dim=1)  # [B, 512]
        return self.classifier(aggregated)  # [B, n_classes]
        
    def _process_group(self, group, processor_id):
        """修改后的组处理逻辑"""
        B, S = group.shape[0], group.shape[1]
        reshaped = group.view(B*S, 3, self.d_model)  # [B*S, 3, d_model]
        
        if processor_id < 2:  # 前两组使用注意力
            # 注意力处理
            attn_out, _ = self.group_processors[processor_id][0](reshaped, reshaped, reshaped)
            norm_out = self.group_processors[processor_id][1](attn_out)
            
            # 特征聚合
            aggregated = norm_out.mean(dim=1)  # [B*S, d_model]
            
            # 维度扩展
            proj_out = self.group_processors[processor_id][2](aggregated)  # [B*S, d_model*2]
        else:  # 第三组直接处理
            # 平均聚合
            aggregated = reshaped.mean(dim=1)  # [B*S, d_model]
            
            # 直接投影
            proj_out = self.group_processors[processor_id][0](aggregated)  # [B*S, d_model*2]
            proj_out = self.group_processors[processor_id][2](proj_out)  # LayerNorm
        
        return proj_out.view(B, S, -1)  # [B, S, d_model*2]
    
    # 添加计算总桶数量的方法
    def _calculate_total_buckets(self, thresholds, bucket_widths, min_value = 1):
        start_buckets = [0]
        # 计算每个区间的起始桶编号
        for i in range(1, len(thresholds)):
            if i == 1:
                range_start = min_value
                range_end = thresholds[i-1]
            else:
                range_start = thresholds[i-2] + 1
                range_end = thresholds[i-1]
            
            bucket_count = ((range_end - range_start) // bucket_widths[i-1]) + 1
            bucket_count = max(1, bucket_count)
            next_start = start_buckets[-1] + bucket_count
            start_buckets.append(next_start)
        
        # 计算最后一个区间的桶数量
        last_range_start = thresholds[-2] + 1 if len(thresholds) > 1 else min_value
        last_range_end = thresholds[-1]
        last_bucket_count = ((last_range_end - last_range_start) // bucket_widths[-1]) + 1
        
        return start_buckets[-1] + max(1, last_bucket_count)
        
    def _auto_dynamic_range_bucketize(self, values, thresholds, bucket_widths, min_value=1):
        """自动动态区间编码，无需手动指定起始桶编号
        
        参数:
            values: 输入的时间值张量
            thresholds: 区间边界列表，升序排列
            bucket_widths: 每个区间的桶宽度列表
            min_value: 最小值，默认为1
            
        返回:
            bucketed: 编码后的桶索引张量
        """
        # 确保thresholds和bucket_widths长度一致
        assert len(thresholds) == len(bucket_widths), "阈值和桶宽度数量必须一致"
        
        # 计算每个区间的起始桶编号
        start_buckets = [0]  # 第一个区间的起始桶编号为0
        
        # 计算每个区间的桶数量并累加得到起始桶编号
        for i in range(1, len(thresholds)):
            # 计算当前区间的数值范围
            if i == 1:
                # 第一个区间：从最小值到第一个阈值
                range_start = min_value
                range_end = thresholds[i-1]
            else:
                # 中间区间：从上一个阈值到当前阈值
                range_start = thresholds[i-2] + 1  # +1避免重叠
                range_end = thresholds[i-1]
            
            # 计算该区间的桶数量
            bucket_count = ((range_end - range_start) // bucket_widths[i-1]) + 1
            # 确保至少有一个桶
            bucket_count = max(1, bucket_count)
            # 计算下一个区间的起始桶编号
            next_start = start_buckets[-1] + bucket_count
            start_buckets.append(next_start)
        
        # 初始化结果张量
        bucketed = torch.zeros_like(values, dtype=torch.long)
        
        # 对每个区间应用不同的桶化策略
        for i in range(len(thresholds)):
            if i == 0:
                # 第一个区间：小于等于第一个阈值
                mask = values <= thresholds[i]
            elif i == len(thresholds) - 1:
                # 最后一个区间：大于上一个阈值
                mask = values > thresholds[i-1]
            else:
                # 中间区间：大于上一个阈值且小于等于当前阈值
                mask = (values > thresholds[i-1]) & (values <= thresholds[i])
            
            if mask.any():
                if i == 0:
                    # 第一个区间：从最小值开始计算
                    bucket_values = (values[mask] - min_value) // bucket_widths[i]
                else:
                    # 其他区间：从上一个阈值+1开始计算
                    bucket_values = (values[mask] - (thresholds[i-1] + 1)) // bucket_widths[i]
                
                # 确保桶索引非负
                bucket_values = torch.clamp(bucket_values, 0)
                # 计算最终桶索引并确保类型匹配
                bucketed[mask] = (start_buckets[i] + bucket_values).long()
        
        # 确保最终索引不超过Embedding层大小
        return torch.clamp(bucketed, 0, self.time_emb.num_embeddings - 1)
    
    # 辅助方法保持不变
    def _safe_index(self, tensor):
        return torch.clamp(((tensor + 1) // 2).long(), 0, 1)
    
    def _log_bucketize(self, values):
        valid_values = torch.clamp(values, 1, 999999)
        bucketed = self._auto_dynamic_range_bucketize(valid_values, self.size_thresholds, self.size_bucket_widths, min_value=1)
        return bucketed
    
    def _time_bucketize(self, values):
    # 首先确保输入是有效的,然后应用桶化策略
        valid_values = torch.clamp(values, 1, 59999999)
        bucketed = self._auto_dynamic_range_bucketize(valid_values, self.time_thresholds, self.time_bucket_widths, min_value=1)
        return bucketed
    
    def _flag_bucketize(self, values):
        return torch.clamp(values.long(), 0, 1)
    
    def _ratio_bucketize(self, values):
        valid_values = torch.clamp(values*1000000, 1, 59999999)
        bucketed = self._auto_dynamic_range_bucketize(valid_values, self.ratio_thresholds, self.ratio_bucket_widths, min_value=1)
        return bucketed
    
    def _diff_bucketize(self, values):
        valid_values = torch.clamp(values*100000, 1, 59999999)
        bucketed = self._auto_dynamic_range_bucketize(valid_values, self.diff_thresholds, self.diff_bucket_widths, min_value=1)
        return bucketed  # 
    
# 数据预处理和训练模块（修复collate_fn函数）
def collate_fn(batch, max_seq_len=50):
    features = [item['features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # 新增：提取sample_info字段
    sample_infos = []
    for item in batch:
        if 'sample_info' in item:
            sample_infos.append(item['sample_info'])
        else:
            sample_infos.append(None)
    
    processed_features = []
    masks = []
    for f in features:
        seq_len = f.shape[0]
        if seq_len > max_seq_len:
            truncated = f[:max_seq_len]
            mask = torch.zeros(max_seq_len).bool()
        else:
            truncated = np.zeros((max_seq_len, 9), dtype=np.float32)
            truncated[:seq_len] = f
            mask = torch.cat([torch.zeros(seq_len), 
                            torch.ones(max_seq_len - seq_len)]).bool()
        processed_features.append(torch.FloatTensor(truncated))
        masks.append(mask)
    
    return {
        'features': torch.stack(processed_features),
        'label': labels,
        'mask': torch.stack(masks),
        'sample_info': sample_infos  # 新增：传递样本信息
    }
    
    
def train_model(data_root, split_params=None):
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"training_and_Test_log_{timestamp}.txt"
    
    # 设置日志记录器
    logger = Logger(log_file_path)
    sys.stdout = logger
    
    try:
        print(f"🚀 开始训练 - {timestamp}")
        print(f"📁 日志文件: {log_file_path}")
        print("=" * 80)
        
        # 获取设备（GPU或CPU）
        device = get_device()
                 
        config = {
            'batch_size': 32,
            'lr': 1e-4,
            'epochs': 100,
            'max_seq_len': 50,
            'd_model': 128,
            # 默认参数
            'min_samples': 10,
            'patience': 5,
            'save_best_model': True,
            # 训练集划分比例（从train文件夹中划分出验证集）
            'train_ratio': 0.9,    # 90%训练集（从train文件夹中）
            'val_ratio': 0.1,      # 10%验证集（从train文件夹中）
            'max_samples_per_class': 10000,  # 新增：每个类别最大样本数
            'use_amp': True,  # 新增：是否使用混合精度训练
            'num_workers': 4 if torch.cuda.is_available() else 0,  # DataLoader工作进程数
        }
        
        # 合并用户自定义参数
        if split_params:
            config.update(split_params)
        
        # 构建训练集和测试集路径
        train_dir = os.path.join(data_root, 'train')
        test_dir = os.path.join(data_root, 'test')
        
        # 检查文件夹是否存在
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"训练集文件夹不存在: {train_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"测试集文件夹不存在: {test_dir}")
        
        print(f"数据路径:")
        print(f"  训练集: {train_dir}")
        print(f"  测试集: {test_dir}")
        
        # 加载训练集
        try:
            print("\n正在加载训练集...")
            train_dataset = CustomDataset(
                train_dir,
                max_seq_len=config['max_seq_len'],
                min_samples=config['min_samples'],
                max_samples_per_class=config['max_samples_per_class']
            )
            print(f"训练集加载完成，共 {len(train_dataset)} 样本")
        except Exception as e:
            print(f"训练集加载失败: {str(e)}")
            return
        
        # 加载测试集
        try:
            print("\n正在加载测试集...")
            test_dataset = CustomDataset(
                test_dir,
                max_seq_len=config['max_seq_len'],
                min_samples=config['min_samples']
            )
            print(f"测试集加载完成，共 {len(test_dataset)} 样本")
        except Exception as e:
            print(f"测试集加载失败: {str(e)}")
            return
        
        # 验证数据集
        n_classes = len(np.unique(train_dataset.labels))
        print(f"\n检测到 {n_classes} 个有效类别")
        
        if n_classes < 2:
            raise ValueError(f"至少需要2个类别，当前检测到{n_classes}个")
        
        # 检查训练集样本总数
        total_train_samples = len(train_dataset)
        if total_train_samples < 5:
            raise ValueError(
                f"训练集需要至少5个样本进行训练，当前只有{total_train_samples}个样本。"
            )
        
        # 检查测试集样本总数
        total_test_samples = len(test_dataset)
        if total_test_samples < 1:
            raise ValueError(
                f"测试集需要至少1个样本进行评估，当前只有{total_test_samples}个样本。"
            )
        
        # 打印数据集信息
        print(f"\n数据集信息:")
        print(f"  训练集总样本数: {total_train_samples}")
        print(f"  测试集总样本数: {total_test_samples}")
        
        # 检查训练集类别分布
        unique_train_labels, train_counts = np.unique(train_dataset.labels, return_counts=True)
        print("\n训练集类别分布：")
        for label, count in zip(unique_train_labels, train_counts):
            print(f"  类别 {label}: {count} 样本")
        
        # 检查测试集类别分布
        unique_test_labels, test_counts = np.unique(test_dataset.labels, return_counts=True)
        print("\n测试集类别分布：")
        for label, count in zip(unique_test_labels, test_counts):
            print(f"  类别 {label}: {count} 样本")
        
        # 从训练集中划分出验证集
        try:
            train_idx, val_idx = train_test_split(
                np.arange(total_train_samples),
                test_size=config['val_ratio'],
                stratify=train_dataset.labels if total_train_samples > 10 else None,
                random_state=42
            )
        except ValueError as e:
            # 分层抽样失败时改用简单抽样
            print("分层抽样失败，使用简单随机抽样")
            train_idx, val_idx = train_test_split(
                np.arange(total_train_samples),
                test_size=config['val_ratio'],
                random_state=42
            )
        
        print(f"\n训练集划分:")
        print(f"  训练集: {len(train_idx)} 样本 ({len(train_idx)/total_train_samples:.1%})")
        print(f"  验证集: {len(val_idx)} 样本 ({len(val_idx)/total_train_samples:.1%})")
        print(f"  测试集: {total_test_samples} 样本 (独立测试集)")
        
        # 创建数据子集
        train_set = torch.utils.data.Subset(train_dataset, train_idx)
        val_set = torch.utils.data.Subset(train_dataset, val_idx)
        # 测试集直接使用完整的test_dataset
        
        # 创建DataLoader（添加GPU加速相关参数）
        collate = partial(collate_fn, max_seq_len=config['max_seq_len'])
        train_loader = DataLoader(
            train_set, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            collate_fn=collate,
            num_workers=config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=config['batch_size'],
            collate_fn=collate,
            num_workers=config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'],
            collate_fn=collate,
            num_workers=config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        # 初始化模型并移动到设备
        model = HierarchicalAttentionClassifier(config['d_model'], n_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        # 初始化混合精度训练
        scaler = GradScaler() if config['use_amp'] and torch.cuda.is_available() else None
        if scaler:
            print(f"✓ 启用混合精度训练 (AMP)")
        
        # 早停机制相关变量
        best_val_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        best_epoch = 0
        
        print(f"\n开始训练，共 {config['epochs']} 轮，早停耐心值: {config['patience']}")
        print("=" * 80)
        
        # 训练循环
        for epoch in range(config['epochs']):
            model.train()
            total_loss = 0
            batch_count = 0
            
            # 训练阶段
            for batch in train_loader:
                # 将数据移动到设备
                features = batch['features'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                mask = batch['mask'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # 使用混合精度训练
                if scaler:
                    with autocast():
                        outputs = model(features, mask)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(features, mask)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            
            # 在验证集上评估（不是测试集！）
            val_metrics = evaluate(model, val_loader, device)
            current_val_accuracy = val_metrics['accuracy']
            
            # 早停逻辑：检查验证精度是否提升
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0  # 重置计数器
                
                # 保存最佳模型
                if config['save_best_model']:
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_save_path = f"best_model_epoch_{epoch+1}_val_acc_{best_val_accuracy:.4f}_{current_time}.pth"
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': best_val_accuracy,
                        'config': config,
                        'n_classes': n_classes,
                        'train_idx': train_idx,
                        'val_idx': val_idx,
                        'test_dataset_size': len(test_dataset)
                    }, model_save_path)
                    print(f"  ✓ 保存最佳模型: {model_save_path}")
            else:
                patience_counter += 1
            
            # 打印训练进度
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val P_mac: {val_metrics['precision_macro']:.4f} | "
                f"Val R_mac: {val_metrics['recall_macro']:.4f} | "
                f"Val F1_mac: {val_metrics['f1_macro']:.4f} | "
                f"Val FPR_mac: {val_metrics['fpr_macro']:.4f} | "
                f"Best Val Acc: {best_val_accuracy:.4f} (Epoch {best_epoch}) | "
                f"Patience: {patience_counter}/{config['patience']}")
            
            # 早停检查
            if patience_counter >= config['patience']:
                print(f"\n⚡ 早停触发！连续 {config['patience']} 轮验证精度无提升")
                print(f"最佳验证精度: {best_val_accuracy:.4f} (第 {best_epoch} 轮)")
                break
        
        # 训练结束，恢复最佳模型并在测试集上最终评估
        final_test_metrics = None
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n🎯 训练完成！已恢复最佳模型 (第 {best_epoch} 轮, 验证精度: {best_val_accuracy:.4f})")
            
            # 在测试集上进行最终评估（从未使用过的数据）
            print("\n🧪 在独立测试集上进行最终评估...")
            final_test_metrics = evaluate(model, test_loader, device, return_details=True)
            
            # 打印混淆矩阵
            print_confusion_matrix(final_test_metrics['confusion_matrix'])
            
            # 调用错误分类分析函数，传递每个类别的总样本数
            if 'misclassified' in final_test_metrics:
                # 从混淆矩阵计算每个类别的总样本数
                cm = final_test_metrics['confusion_matrix']
                total_samples_per_class = [cm[i, :].sum() for i in range(cm.shape[0])]
                
                analyze_misclassifications(final_test_metrics['misclassified'], 
                                        total_samples_per_class=total_samples_per_class)
            else:
                print("⚠️ 没有错误分类数据可用")
            
            print("📊 最终测试结果:")
            print(f"  准确率: {final_test_metrics['accuracy']:.4f}")
            print(f"  精确率(宏): {final_test_metrics['precision_macro']:.4f}")
            print(f"  召回率(宏): {final_test_metrics['recall_macro']:.4f}")
            print(f"  F1分数(宏): {final_test_metrics['f1_macro']:.4f}")
            print(f"  误报率(宏): {final_test_metrics['fpr_macro']:.4f}")
            print(f"  精确率(微): {final_test_metrics['precision_micro']:.4f}")
            print(f"  召回率(微): {final_test_metrics['recall_micro']:.4f}")
            print(f"  F1分数(微): {final_test_metrics['f1_micro']:.4f}")
            print(f"  误报率(微): {final_test_metrics['fpr_micro']:.4f}")
            
            # 保存最终模型
            # 获取当前时间的时间戳
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_model_path = f"final_model_test_acc_{final_test_metrics['accuracy']:.4f}_{current_time}.pth"
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_accuracy,
                'test_accuracy': final_test_metrics['accuracy'],
                'config': config,
                'n_classes': n_classes,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_dataset_size': len(test_dataset)
            }, final_model_path)
            print(f"💾 最终模型已保存: {final_model_path}")
        
        print("=" * 80)
        print("🎉 训练流程总结:")
        print(f"  • 训练集: {len(train_idx)} 样本")
        print(f"  • 验证集: {len(val_idx)} 样本")
        print(f"  • 测试集: {len(test_dataset)} 样本")
        print(f"  • 最佳验证精度: {best_val_accuracy:.4f} (第 {best_epoch} 轮)")
        if final_test_metrics:
            print(f"  • 最终测试精度: {final_test_metrics['accuracy']:.4f}")
            print(f"  • 泛化差距: {(best_val_accuracy - final_test_metrics['accuracy']):.4f}")
        
        return model, best_val_accuracy, final_test_metrics
    
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None
    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout = logger.terminal
        logger.close()    
        
def evaluate(model, loader, device, return_details=False):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    sample_indices = []
    sample_infos = []  # 新增：存储样本信息
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # 将数据移动到设备
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            
            outputs = model(features, mask)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # 移回CPU进行后续处理
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 记录批次索引和样本在批次中的位置
            for i in range(len(labels)):
                sample_indices.append((batch_idx, i))
                # 存储样本信息
                if 'sample_info' in batch:
                    sample_infos.append(batch['sample_info'][i])
                else:
                    sample_infos.append(None)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    n_classes = cm.shape[0]
    
    # 计算误报率 (False Positive Rate)
    fpr_per_class = []
    for i in range(n_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_per_class.append(fpr)
    
    fpr_macro = np.mean(fpr_per_class)
    
    total_fp = 0
    total_tn = 0
    for i in range(n_classes):
        total_fp += cm[:, i].sum() - cm[i, i]
        total_tn += cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
    
    fpr_micro = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
    
    result = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_micro': precision_score(all_labels, all_preds, average='micro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_micro': recall_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'fpr_macro': fpr_macro,
        'fpr_micro': fpr_micro,
        'confusion_matrix': cm,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs,
        'sample_indices': sample_indices,
        'sample_infos': sample_infos  # 新增：样本信息
    }
    
    if return_details:
        # 分析错误分类
        misclassified = []
        for i, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
            if true_label != pred_label:
                misclassified.append({
                    'sample_index': i,
                    'batch_info': sample_indices[i],
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': all_probs[i][pred_label],
                    'true_class_confidence': all_probs[i][true_label],
                    'sample_info': sample_infos[i]  # 新增：样本来源信息
                })
        
        result['misclassified'] = misclassified
        result['misclassification_rate'] = len(misclassified) / len(all_labels)
    
    return result
  
def print_confusion_matrix(cm, class_names=None):
    """打印格式化的混淆矩阵"""
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    print("\n📊 混淆矩阵:")
    print(" " * 8, end="")
    for name in class_names:
        print(f"{name:>8}", end="")
    print("  (预测)")
    
    for i in range(n_classes):
        print(f"{class_names[i]:>8}", end="")
        for j in range(n_classes):
            print(f"{cm[i, j]:>8}", end="")
        print()
    
    # 打印每行的百分比
    print("\n📈 混淆矩阵 (行百分比):")
    print(" " * 8, end="")
    for name in class_names:
        print(f"{name:>8}", end="")
    print("  (预测)")
    
    for i in range(n_classes):
        row_sum = cm[i, :].sum()
        print(f"{class_names[i]:>8}", end="")
        for j in range(n_classes):
            percentage = (cm[i, j] / row_sum * 100) if row_sum > 0 else 0
            print(f"{percentage:>7.1f}%", end="")
        print()
        
def analyze_misclassifications(misclassified, class_names=None, total_samples_per_class=None):
    """分析错误分类模式
    
    Args:
        misclassified: 错误分类的样本列表
        class_names: 类别名称列表
        total_samples_per_class: 每个类别的总样本数（可选）
    """
    if not misclassified:
        print("🎉 没有错误分类的样本！")
        return
    
    # 获取类别数量
    n_classes = max(max(mc['true_label'] for mc in misclassified), 
                   max(mc['predicted_label'] for mc in misclassified)) + 1
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # 统计错误分类模式
    error_patterns = {}
    for mc in misclassified:
        pattern = (mc['true_label'], mc['predicted_label'])
        error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
    
    print(f"\n🔍 错误分类分析 (共 {len(misclassified)} 个错误样本):")
    print("=" * 60)
    
    # 按频率排序
    sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
    
    for (true_label, pred_label), count in sorted_patterns:
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        print(f"  {true_name} → {pred_name}: {count} 次")
    
    # 分析每个类别的错误情况
    print(f"\n📋 各类别错误统计:")
    for i in range(n_classes):
        # 统计真实标签为i的错误样本
        class_errors = [mc for mc in misclassified if mc['true_label'] == i]
        
        if total_samples_per_class is not None and i < len(total_samples_per_class):
            # 如果提供了每个类别的总样本数，使用准确的计算
            total_class_samples = total_samples_per_class[i]
            error_rate = len(class_errors) / total_class_samples if total_class_samples > 0 else 0
            print(f"  {class_names[i]}: {len(class_errors)} 个错误, 错误率: {error_rate:.2%}")
        else:
            # 如果没有提供总样本数，只显示错误数量
            print(f"  {class_names[i]}: {len(class_errors)} 个错误")
        
        # 这个类别最常被误分为哪些类别
        if class_errors:
            pred_counts = {}
            for mc in class_errors:
                pred_counts[mc['predicted_label']] = pred_counts.get(mc['predicted_label'], 0) + 1
            
            if pred_counts:
                most_common = max(pred_counts.items(), key=lambda x: x[1])
                print(f"    最常误分为: {class_names[most_common[0]]} ({most_common[1]} 次)")
    
    # 显示具体的错误分类样本详细信息（包含文件名）
    print(f"\n🔎 具体错误分类样本信息:")
    print("=" * 60)
    for i, mc in enumerate(misclassified): 
        true_name = class_names[mc['true_label']]
        pred_name = class_names[mc['predicted_label']]
        print(f"样本 #{i+1}:")
        print(f"  • 样本索引: {mc['sample_index']}")
        print(f"  • 批次信息: {mc['batch_info']}")
        print(f"  • 真实类别: {true_name} ")
        print(f"  • 预测类别: {pred_name} ")
        print(f"  • 预测置信度: {mc['confidence']:.4f}")
        print(f"  • 真实类别置信度: {mc['true_class_confidence']:.4f}")
        print(f"  • 置信度差异: {mc['confidence'] - mc['true_class_confidence']:.4f}")
        
        # 显示样本来源信息
        if mc['sample_info']:
            info = mc['sample_info']
            print(f"  • 来源文件: {info['csv_file']}")
            print(f"  • 类别文件夹: {info['class_name']}")
        print()
    
         
if __name__ == "__main__":
    DATA_PATH = r"F:\序列\datacon2020_augmented"  # 修改为实际路径
    
    # 完整配置示例
    CUSTOM_CONFIG = {
        'train_ratio': 0.9,  # 从train文件夹中划分90%作为训练集
        'val_ratio': 0.1,     # 从train文件夹中划分10%作为验证集
        'min_samples': 10,   # 可自定义最小样本数
        'patience': 5,  # 设置5轮耐心值
        'save_best_model': True,
        'use_amp': True,  # 启用混合精度训练（GPU加速）
        'num_workers': 4  # DataLoader工作进程数
    }
    
    # 使用自定义配置启动训练和评估
    train_model(DATA_PATH, split_params=CUSTOM_CONFIG)
