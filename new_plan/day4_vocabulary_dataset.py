from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import jieba

class BasicVocab:
    def __init__(self):
        # 初始化特殊标记：PAD(填充)、UNK(未知词)
        self.PAD = "<PAD>"  # 索引0
        self.UNK = "<UNK>"  # 索引1
        self.token2idx = {self.PAD: 0, self.UNK: 1}  # 词→索引
        self.idx2token = {0: self.PAD, 1: self.UNK}  # 索引→词
        self.token_count = defaultdict(int)  # 统计词频（可选，用于过滤低频词

    # 统计文本中的所有词/字
    def add_text(self, text, tokenize_fn):
        """
        text: 单条文本（如"我爱自然语言处理"）
        tokenize_fn: 分词函数（中文按字分/按词分，英文按空格分）
        """
        tokens = tokenize_fn(text)
        for token in tokens:
            self.token_count[token] += 1    

    # 构建最终词表（可选过滤低频词）
    def build_vocab(self, min_freq=1):
        # 按词频过滤，只保留出现次数≥min_freq的词
        filtered_tokens = [token for token, count in self.token_count.items() if count >= min_freq]
        # 给每个词分配索引
        for token in filtered_tokens:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    # 把单条文本转成数字序列
    def text2idx(self, text, tokenize_fn, max_len=10):
        tokens = tokenize_fn(text)
        # 1. 转索引（未知词用UNK的索引）
        idx_list = [self.token2idx.get(token, self.token2idx[self.UNK]) for token in tokens]
        # 2. 截断/填充到固定长度（保证张量维度一致）
        if len(idx_list) > max_len:
            idx_list = idx_list[:max_len]  # 截断
        else:
            idx_list += [self.token2idx[self.PAD]] * (max_len - len(idx_list))  # 填充
        return idx_list

# --------------------------
# 测试词表功能（中文按字分词）
# --------------------------
# 定义分词函数（中文简单按字分，后续可替换成jieba分词）
def tokenize_chinese(text):
    return tokenize_chinese_jieba(text)
    #return list(text)  # 按字拆分，如"我爱NLP"→["我","爱","N","L","P"]

def tokenize_chinese_jieba(text):
    return jieba.lcut(text)  # 按词拆分，如"自然语言处理"→["自然语言","处理"]

# 构建词表
vocab = BasicVocab()
# 语料库（模拟训练数据）
corpus = [
    "自然语言处理很有趣",
    "PyTorch是好用的框架",
    "文本转张量是NLP的基础",
    "我爱学习深度学习"
]

# 1. 统计所有词
for text in corpus:
    vocab.add_text(text, tokenize_chinese)

# 2. 构建词表（过滤出现次数≥1的词）
vocab.build_vocab(min_freq=1)

# 测试文本转索引
text = "自然语言处理入门"
idx_list = vocab.text2idx(text, tokenize_chinese, max_len=8)
print(f"原始文本：{text}")
print(f"分词结果：{tokenize_chinese(text)}")
print(f"转索引序列：{idx_list}")
print(f"索引转回文本：{[vocab.idx2token[idx] for idx in idx_list]}")

class TextDataset(Dataset):
    def __init__(self, texts, vocab, tokenize_fn, max_len=10):
        self.texts = texts  # 文本列表
        self.vocab = vocab  # 词表
        self.tokenize_fn = tokenize_fn  # 分词函数
        self.max_len = max_len  # 序列最大长度
    
    # 必须实现：返回第idx条数据
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 文本转索引序列
        idx_list = self.vocab.text2idx(text, self.tokenize_fn, self.max_len)
        # 转成PyTorch张量（模型能处理的格式）
        tensor = torch.tensor(idx_list, dtype=torch.long)  # 索引用long类型
        return tensor
    
    # 必须实现：返回数据集总长度
    def __len__(self):
        return len(self.texts)


# --------------------------
# 测试Dataset和DataLoader
# --------------------------
# 初始化数据集
dataset = TextDataset(
    texts=corpus,
    vocab=vocab,
    tokenize_fn=tokenize_chinese,
    max_len=8
)

# 用DataLoader批量加载（核心：把零散数据拼成批次）
dataloader = DataLoader(
    dataset,
    batch_size=2,  # 每次返回2条数据
    shuffle=True,  # 打乱数据（训练时用）
    num_workers=0  # 新手先设0，避免多进程报错（MPS/Windows下num_workers>0容易出问题）
)

# 遍历数据加载器
print("\n批量加载数据：")
for batch_idx, batch_tensor in enumerate(dataloader):
    print(f"批次{batch_idx+1}，张量形状：{batch_tensor.shape}")
    print(f"张量内容：\n{batch_tensor}")
    # 把张量转回文本（验证正确性）
    batch_text = []
    for idx_list in batch_tensor.numpy():
        text = "".join([vocab.idx2token[idx] for idx in idx_list if idx != vocab.token2idx[vocab.PAD]])
        batch_text.append(text)
    print(f"转回文本：{batch_text}\n")