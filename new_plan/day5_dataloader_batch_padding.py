import torch
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data import WeightedRandomSampler


# 1. 定义简单的数据集类（Dataset是PyTorch的基础类，必须实现__len__和__getitem__）
class TextDataset(Dataset):
    def __init__(self, texts, vocab, tokenize_fn, max_len=10):
        self.texts = texts  # 原始文本列表
        self.vocab = vocab  # 之前构建的词表
        self.tokenize_fn = tokenize_fn  # 分词函数
        self.max_len = max_len  # 最大序列长度

    # 核心1：返回数据集长度
    def __len__(self):
        return len(self.texts)

    # 核心2：返回单个样本（文本转索引+padding/截断）
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 1. 分词+转索引（承接之前的text2idx）
        tokens = self.tokenize_fn(text)
        idx_list = []
        unk_idx = self.vocab.token2idx[self.vocab.UNK]
        for token in tokens[:self.max_len]:  # 截断超长文本
            idx_list.append(self.vocab.token2idx.get(token, unk_idx))
        
        pad_idx = 0  # 约定PAD的索引为0
        '''
        # 2. padding补长（核心！）
        # 【低效写法】循环append补零
        while len(idx_list) < self.max_len:
            idx_list.append(pad_idx)

        # 转成tensor（模型需要tensor格式）
        return torch.tensor(idx_list, dtype=torch.long)
        '''

        # 【高效写法】直接创建固定长度张量（推荐）
        # 1. 创建全为pad_idx的固定长度张量（核心优化）
        padded_tensor = torch.full((self.max_len,), pad_idx, dtype=torch.long)
        valid_len = len(idx_list)
        padded_tensor[:valid_len] = torch.tensor(idx_list, dtype=torch.long)
        return padded_tensor

# 2. 准备测试数据（模拟不同长度的文本）
texts = [
    "我爱吃螺蛳粉",  # 长度5
    "今天学习DataLoader",  # 长度7
    "Python真好用",  # 长度4
    "batch和padding是啥",  # 长度8
    "采样能让训练更稳"  # 长度7
]

# 3. 复用之前的词表（简化版，已包含UNK/PAD）
class SimpleVocab:
    def __init__(self):
        self.token2idx = {"<UNK>":1, "<PAD>":0, "我":2, "爱":3, "吃":4, "螺":5, "蛳":6, "粉":7, "今":8, "天":9, "学":10, "习":11}
        self.UNK = "<UNK>"

        
vocab = SimpleVocab()

# 4. 分词函数（中文按字分）
def tokenize_chinese(text):
    return list(text)

# 如果你的样本包含 “文本 + 标签”，可自定义collate_fn统一处理 batch 的 padding：
def collate_fn(batch):
    # batch是[(文本索引, 标签), (文本索引, 标签), ...]
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # 统一padding（这里可写更灵活的逻辑）
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, torch.tensor(labels)

# 传入DataLoader
#dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# 1. 创建数据集实例
dataset = TextDataset(texts, vocab, tokenize_chinese, max_len=8)

# 2. 定义采样器（可选，这里用随机采样）
# 简单随机采样：直接在DataLoader里设shuffle=True即可
# 如果要处理数据不平衡，可自定义Sampler，这里先讲基础

# 3. 创建DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# 4. 测试加载效果
print("=== 每个batch的输出（batch_size=2，seq_len=8）===")
for batch_idx, batch_data in enumerate(dataloader):
    print(f"第{batch_idx+1}个batch: ")
    print(f"shape: {batch_data.shape}")
    print(f"data: \n{batch_data}\n")


# 采样的扩展（数据不平衡时用）
# 比如你的数据里 “正面评论 90 条，负面评论 10 条”，可自定义采样器
# 假设labels是标签列表（0=负面，1=正面）
labels = [0]*10 + [1]*90
# 计算权重：让负面样本被采样的概率更高
weights = [10 if label==0 else 1 for label in labels]
sampler = WeightedRandomSampler(weights, num_samples=100, replacement=True)
# 传入DataLoader
#dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)