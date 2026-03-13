# 极简 Tokenizer（自定义实现，替代第三方库）
class SimpleTokenizer:
    def __init__(self):
        # 基础字典：字符→ID（自定义GPT的vocab）
        self.char2id = {"<PAD>": 0, "<UNK>": 1}  # PAD=0, UNK=1
        self.id2char = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2  # 初始大小

    # 构建字典（用训练文本）
    def fit(self, text_list):
        for text in text_list:
            for char in text:
                if char not in self.char2id:
                    self.char2id[char] = self.vocab_size
                    self.id2char[self.vocab_size] = char
                    self.vocab_size += 1

    # 文本→Token ID序列（tokenizer）
    def encode(self, text, max_len=None):
        token_ids = [self.char2id[char] for char in text]
        if max_len:
            token_ids = token_ids[:max_len]
        return token_ids

    def decode(self, token_ids):
        # join method: the string in front of .join is the separator that gets inserted between the joined items.
        # "".join — no separator, direct concatenation
        # "".join(["a", "b", "c"])   # → "abc"
        return "".join([self.id2char[token_id] for token_id in token_ids])

#步骤 2：核心函数：构建 Input+Target 序列（纯手动）
def build_gpt_sequences(token_ids, max_seq_len, pad_id=0):
    """
    纯手动构建GPT的input和target序列
    :param token_ids: 单条文本的token id序列
    :param max_seq_len: 序列最大长度（自定义GPT的输入长度）
    :param pad_id: padding的token id
    :return: input_ids, target_ids（长度均为max_seq_len）
    """
    # 1. 截断：超过max_seq_len的部分砍掉（避免超长）
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

    # 2. 核心：构建input和target（target右移1位）
    # input：去掉最后一个token（用前n-1个预测第n个）
    input_ids = token_ids[:-1] if len(token_ids) > 1 else []
    # target：去掉第一个token（对应input的下一个token）
    target_ids = token_ids[1:] if len(token_ids) > 1 else []

    # 3. 手动padding：补到max_seq_len-1（因为input/target比原序列短1）
    input_ids += [pad_id] * (max_seq_len - len(input_ids))
    target_ids += [-100] * (max_seq_len - len(target_ids))

    return input_ids, target_ids

# 步骤 3：数据集构建（纯手动，无 DataLoader 依赖）
class GPTDataSet():
    def __init__(self, text_list, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = [] # 存储(input_ids, target_ids)对
        
        # 预处理所有文本
        for text in text_list:
            token_ids = self.tokenizer.encode(text)
            input_ids, target_ids = build_gpt_sequences(token_ids, self.max_seq_len)
            self.data.append((input_ids, target_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
    # 手动生成批次（替代DataLoader）
    def get_batch(self, batch_size):
        batches = []
        for i in range(0, len(self), batch_size):
            batch_data = self.data[i:i+batch_size]
            # 拆分成input批次和target批次
            batch_input = [item[0] for item in batch_data]
            batch_target = [item[1] for item in batch_data]
            batches.append((batch_input, batch_target))
        return batches

# 步骤 4：完整测试（跑通整个流程）
# ========== 测试：纯手写构建GPT训练数据 ==========
if __name__ == "__main__":
    # 1. 准备测试文本（模拟训练数据）
    train_texts = [
        "我喜欢吃苹果。",
        "今天天气很好。",
        "GPT是自回归模型。"
    ]
    
    # 2. 初始化并训练Tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit(train_texts)
    print("词汇表：", tokenizer.char2id)
    
    # 3. 初始化数据集（max_seq_len=10）
    max_seq_len = 10
    dataset = GPTDataSet(train_texts, tokenizer, max_seq_len)
    
    # 4. 查看单条数据的input+target
    print("\n=== 单条数据示例 ===")
    input_ids, target_ids = dataset[0]
    print("原文本：", train_texts[0])
    print("Input序列：", input_ids)
    print("Target序列：", target_ids)
    print("Input解码：", tokenizer.decode(input_ids))
    print("Target解码（忽略-100）：", tokenizer.decode([t for t in target_ids if t != -100]))
    
    # 5. 生成批次数据（适配自定义训练循环）
    print("\n=== 批次数据示例 ===")
    batches = dataset.get_batch(batch_size=2)
    for i, (batch_input, batch_target) in enumerate(batches):
        print(f"批次{i+1} Input：", batch_input)
        print(f"批次{i+1} Target：", batch_target)