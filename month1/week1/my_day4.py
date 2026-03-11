import torch
from torch.utils.data import Dataset, DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WordDataset(Dataset):
    def __init__(self, text, seq_len):
        self.words = sorted(list(set(text.split())))
        self.vocab_size = len(self.words)
        self.word2idx = {word:i for i, word in enumerate(self.words)}
        self.idx2word = {i:word for i, word in enumerate(self.words)}
        self.data = [self.word2idx[word] for word in text.split()]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.long).to(device), torch.tensor(y, dtype=torch.long).to(device)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "my_day4_words.txt")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            my_file = f.read()
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        # 提供测试文本，避免文件不存在时程序崩溃
        my_file = "hello world hello pytorch hello python hello data"
    dataset = WordDataset(my_file, seq_len=5)

    print(f"词汇表大小：{dataset.vocab_size}")
    print(f"数据集总长度：{len(dataset)}")
    print(f"索引0的单条数据：{dataset[0]}")
    print(f"字符→数字映射：{dataset.word2idx}")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=True)
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        print(f"第{batch_idx+1}个批次")
        print(f"批次输入x（形状：{batch_x.shape}）：{batch_x}")
        print(f"批次标签y（形状：{batch_y.shape}）：{batch_y}")
        print("-"*50)
        # 只打印前3个批次，避免输出过长
        if batch_idx >= 2:
            break