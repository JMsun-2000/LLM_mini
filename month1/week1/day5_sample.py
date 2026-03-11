import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# ====================== 1. 设备配置 & 数据集类（复用Day4修正版） ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WordDataset(Dataset):
    def __init__(self, text, seq_len):
        self.words = sorted(list(set(text.split())))
        self.vocab_size = len(self.words)
        self.word2idx = {word:i for i, word in enumerate(self.words)}
        self.idx2word = {i:word for i, word in enumerate(self.words)}
        self.data = [self.word2idx[word] for word in text.split()]  # 核心修正点
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.long).to(device), torch.tensor(y, dtype=torch.long).to(device)

# ====================== 2. 定义RNN语言模型 ======================
class SimpleRNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_layers=1):
        super().__init__()
        # 1. 词嵌入层：将单词索引转换为稠密向量（解决one-hot稀疏问题）
        '''
        Input (Indices)        Embedding Matrix (Weight)          Output (Vectors)
        [batch_size, seq_len]  [vocab_size, embed_dim]           [batch_size, seq_len, embed_dim]

        +-----+              +-----------------------+         +----------------+

        |  2  | -----------> | Row 0: [0.1, 0.5, ...] | ------> | [0.1, 0.5, ...] | (for index 2)
        +-----+              +-----------------------+         +----------------+

        |  5  | -----------> | Row 1: [0.9, 0.2, ...] | ------> | [0.9, 0.2, ...] | (for index 5)
        +-----+              +-----------------------+         +----------------+

        |  0  | -----------> | Row 2: [0.3, 0.7, ...] | ------> | [0.3, 0.7, ...] | (for index 0)
        +-----+              +-----------------------+         +----------------+

                            | ...                   |
                            +-----------------------+

                            | Row N: [0.4, 0.8, ...] |
                            +-----------------------+

        '''
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 2. RNN层：处理序列数据，捕捉上下文信息
        '''
        
        Input (Vectors)                    Hidden State (H)                     Output (Vectors)
        [batch_size, seq_len, embed_dim]  [num_layers, batch_size, hidden_dim]  [batch_size, seq_len, hidden_dim]

        +-----+              +-----------------------+         +----------------+

        | [0.1, 0.5, ...] | -----------> | Hidden 0: [0.3, 0.7, ...] | ------> | [0.3, 0.7, ...] | (for time step 0)
        +-----+              +-----------------------+         +----------------+

        | [0.9, 0.2, ...] | -----------> | Hidden 1: [0.4, 0.8, ...] | ------> | [0.4, 0.8, ...] | (for time step 1)
        +-----+              +-----------------------+         +----------------+

        | ...              | -----------> | ...                   |
        +-----+              +-----------------------+         +----------------+

        | [0.4, 0.8, ...] | -----------> | Hidden N: [0.5, 0.9, ...] | ------> | [0.5, 0.9, ...] | (for time step N)
        +-----+              +-----------------------+         +----------------+

        '''
        self.rnn = nn.RNN(
            input_size=embed_dim,  # 输入维度=词嵌入维度
            hidden_size=hidden_dim, # 隐藏层维度
            num_layers=num_layers,  # RNN层数
            batch_first=True        # 输入形状：(batch_size, seq_len, embed_dim)
        )
        # 3. 全连接层：将RNN输出映射到词汇表大小（预测每个单词的概率）
        '''
        Input (Vectors)                     Hidden State (H)                    Output (Vectors)
        [batch_size, seq_len, hidden_dim]  [num_layers, batch_size, hidden_dim]  [batch_size, seq_len, hidden_dim]

        +-----+              +-----------------------+         +----------------+

        | [0.3, 0.7, ...] | -----------> | Hidden 0: [0.5, 0.9, ...] | ------> | [0.5, 0.9, ...] | (for time step 0)
        +-----+              +-----------------------+         +----------------+
        '''
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x形状：(batch_size, seq_len)
        embed = self.embedding(x)  # 嵌入后：(batch_size, seq_len, embed_dim)
        # RNN前向传播：output是所有时间步的输出，hidden是最后一个时间步的隐藏状态
        output, hidden = self.rnn(embed)
        # 取最后一个时间步的输出（因为要预测下一个单词）
        last_output = output[:, -1, :]  # 形状：(batch_size, hidden_dim)
        logits = self.fc(last_output)   # 形状：(batch_size, vocab_size)
        return logits

# ====================== 2.2 定义RNN语言模型 ======================
class ManualRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 手动定义RNN的权重
        self.W_ih = nn.Linear(embed_dim, hidden_dim)  # 输入到隐藏层
        self.W_hh = nn.Linear(hidden_dim, hidden_dim) # 隐藏层到隐藏层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)  # (batch, seq_len, embed_dim)
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, hidden_dim).to(x.device)  # 初始化隐藏状态
        # 手动循环每个时间步
        for t in range(embed.shape[1]):
            x_t = embed[:, t, :]  # 取第t个时间步的输入
            h = torch.tanh(self.W_ih(x_t) + self.W_hh(h))  # 核心循环公式
        logits = self.fc(h)
        return logits
        
# ====================== 3. 训练函数 ======================
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()  # 模型设为训练模式
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(batch_x)
            # 计算损失（交叉熵损失：适合分类任务，这里是预测下一个单词）
            loss = criterion(outputs, batch_y)
            # 反向传播 + 优化
            loss.backward()
            optimizer.step()
            # 累计损失
            total_loss += loss.item()
        
        # 打印每个epoch的平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], 平均损失: {avg_loss:.4f}")

# ====================== 4. 预测函数 ======================
def predict_next_word(model, dataset, input_words):
    """
    输入单词序列，预测下一个单词
    :param model: 训练好的模型
    :param dataset: 数据集（提供word2idx/idx2word映射）
    :param input_words: 输入单词列表（长度=seq_len）
    :return: 预测的单词
    """
    model.eval()  # 模型设为评估模式（禁用dropout等）
    with torch.no_grad():  # 禁用梯度计算（提升速度，节省显存）
        # 将输入单词转换为索引
        input_idx = [dataset.word2idx[word] for word in input_words]
        # 转换为tensor并添加batch维度
        input_tensor = torch.tensor([input_idx], dtype=torch.long).to(device)
        # 模型预测
        logits = model(input_tensor)
        # 取概率最大的索引
        pred_idx = torch.argmax(logits, dim=1).item()
        # 索引转单词
        pred_word = dataset.idx2word[pred_idx]
        return pred_word

# ====================== 5. 主程序 ======================
if __name__ == "__main__":
    # 步骤1：准备数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "_my_day4_words.txt")
    # 读取文本（无文件则用测试文本）
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"未找到文件，使用测试文本")
        text = "hello world hello pytorch hello python hello data hello machine hello learning"
    
    seq_len = 5  # 序列长度（和Day4一致）
    dataset = WordDataset(text, seq_len)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

    # 步骤2：初始化模型、损失函数、优化器
    vocab_size = dataset.vocab_size
    model = SimpleRNNLanguageModel(vocab_size, embed_dim=32, hidden_dim=64).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失（分类任务）
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    # 步骤3：训练模型
    print("开始训练模型...")
    train_model(model, dataloader, criterion, optimizer, epochs=20)

    # 步骤4：测试预测
    print("\n开始预测...")
    # 输入一个长度为seq_len的单词序列（需在词汇表中）
    test_input = ["hello", "world", "hello", "pytorch", "hello"]
    pred_word = predict_next_word(model, dataset, test_input)
    print(f"输入序列: {test_input}")
    print(f"预测的下一个单词: {pred_word}")




