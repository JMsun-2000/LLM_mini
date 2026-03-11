import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 第一步：判断设备（CPU/GPU），周1基础复用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义字符级文本Dataset
class TextDataset(Dataset):
    def __init__(self, text):
        # 1. 初始化：构建字符词汇表，做字符→数字编码
        self.chars = sorted(list(set(text)))  # 去重并排序，得到所有唯一字符
        self.vocab_size = len(self.chars)     # 词汇表大小（大模型的vocab_size参数）
        # 字符→数字 / 数字→字符 映射（必备，用于编码/解码）
        self.char2idx = {c:i for i,c in enumerate(self.chars)}
        self.idx2char = {i:c for i,c in enumerate(self.chars)}
        # 2. 文本转数字序列（核心：把原始文本变成张量可处理的数值）
        self.data = [self.char2idx[c] for c in text]

    def __len__(self):
        # 返回数据集总长度：这里按「序列长度-1」定，为后续自回归预测做准备（Day4铺垫）
        # 自回归：用前n个字符预测第n+1个，所以数据长度要减1
        return len(self.data) - 1

    def __getitem__(self, idx):
        # 按索引取单条数据：输入x（前n个字符），标签y（第n+1个字符）
        # 大模型训练的核心是「预测下一个token」，这是基础逻辑
        x = self.data[idx]
        y = self.data[idx + 1]
        # 转成张量并迁移设备，返回单条(x,y)
        return torch.tensor(x, dtype=torch.long).to(device), torch.tensor(y, dtype=torch.long).to(device)

# 测试：用简单文本构建数据集（可替换成任意文本，如小说/诗歌）
if __name__ == "__main__":
    text = "Hello GPT! This is my first PyTorch data loader demo."
    dataset = TextDataset(text)
    # 测试核心方法
    print(f"词汇表大小：{dataset.vocab_size}")
    print(f"数据集总长度：{len(dataset)}")
    print(f"索引0的单条数据：{dataset[0]}")  # (x=字符H的编码, y=字符e的编码)
    print(f"字符→数字映射：{dataset.char2idx}")

    # 构建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,  # 批次大小4，每次返回4条数据的张量
        shuffle=True,  # 打乱数据
        num_workers=0, # Windows设0，Linux/Mac可设4/8
        drop_last=True # 丢弃不完整批次
    )
    # 迭代取批次数据（大模型训练的循环方式）
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        print(f"第{batch_idx+1}个批次")
        print(f"批次输入x（形状：{batch_x.shape}）：{batch_x}")
        print(f"批次标签y（形状：{batch_y.shape}）：{batch_y}")
        print("-"*50)