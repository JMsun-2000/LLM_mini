"""
把零散的代码整合成端到端流水线，最终输出 GPT 训练能用的「批次数据」，流程如下：
原始文本 → 构建词表 → 文本转Token ID → 生成GPT input/target → Padding/Mask → 批量加载
"""

# ==============================
# 第一步：基础工具（Day1~Day2）
# ==============================
class TextProcessor:
    def __init__(self, corpus):
        # 1. 构建词表（字符级，适配小样本）
        self.vocab = sorted(list(set(corpus)))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_id = 0  # 固定用0作为padding id

    # 文本 → Token ID 序列
    def text2ids(self, text):
        return [self.word2idx[w] for w in text if w in self.word2idx]

    # Token ID 序列 → 文本
    def ids2text(self, ids):
        return ''.join([self.idx2word[i] for i in ids if i in self.idx2word])

# ==============================
# 第二步：GPT数据格式构建（Day6）
# ==============================
def build_gpt_sequences(token_ids, max_len, pad_id=0):
    """
    输入：原始token_ids、最大序列长度
    输出：input_ids, target_ids, pad_mask
    """
    # 1. 截断超长序列
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    
    # 2. 核心：构建GPT的input和target（右移一位）
    input_ids = token_ids[:-1] if len(token_ids) > 1 else []
    target_ids = token_ids[1:] if len(token_ids) > 1 else []
    
    # 3. Padding到max_len-1（因为input/target比原序列短1）
    pad_num = (max_len - 1) - len(input_ids)
    if pad_num > 0:
        input_ids += [pad_id] * pad_num
        target_ids += [-100] * pad_num  # target的padding用-100（后续loss忽略）
    
    # 4. 构建Padding Mask（True=有效，False=padding）
    pad_mask = [1 if tok != pad_id else 0 for tok in input_ids]  # 用1/0替代True/False，更易计算
    
    return input_ids, target_ids, pad_mask

# ==============================
# 第三步：手写DataLoader（Day4+Day7核心）
# ==============================
class SimpleDataLoader:
    def __init__(self, texts, processor, max_len, batch_size=2):
        self.processor = processor
        self.max_len = max_len
        self.batch_size = batch_size
        self.data = []
        
        # 预处理所有文本 → 转换成GPT格式
        for text in texts:
            token_ids = processor.text2ids(text)
            input_ids, target_ids, pad_mask = build_gpt_sequences(
                token_ids, max_len, processor.pad_id
            )
            self.data.append((input_ids, target_ids, pad_mask))
    
    # 生成批次数据（模拟DataLoader的__iter__）
    def get_batches(self):
        batches = []
        # 按batch_size切分数据
        for i in range(0, len(self.data), self.batch_size):
            batch_data = self.data[i:i+self.batch_size]
            # 拆分成input/target/mask的批次
            batch_input = [item[0] for item in batch_data]
            batch_target = [item[1] for item in batch_data]
            batch_mask = [item[2] for item in batch_data]
            batches.append((batch_input, batch_target, batch_mask))
        return batches

# ==============================
# 第四步：周测主流程（端到端运行）
# ==============================
if __name__ == "__main__":
    # 1. 准备测试语料（模拟训练数据）
    train_texts = [
        "我爱吃苹果",
        "我爱吃香蕉",
        "今天天气好",
        "GPT是自回归模型",
        "文本转张量很简单"
    ]
    # 合并语料构建词表
    corpus = ''.join(train_texts)
    
    # 2. 初始化工具类
    max_len = 10  # 序列最大长度
    batch_size = 2
    processor = TextProcessor(corpus)
    
    # 3. 初始化DataLoader
    dataloader = SimpleDataLoader(train_texts, processor, max_len, batch_size)
    
    # 4. 打印关键信息（验证流程）
    print("===== 词表信息 =====")
    print("词表：", processor.vocab)
    print("词表大小：", processor.vocab_size)
    print("\n===== 单条数据示例 =====")
    # 打印第一条文本的处理结果
    first_text = train_texts[0]
    first_ids = processor.text2ids(first_text)
    first_input, first_target, first_mask = build_gpt_sequences(first_ids, max_len, processor.pad_id)
    print("原始文本：", first_text)
    print("Token ID：", first_ids)
    print("GPT Input：", first_input)
    print("GPT Target：", first_target)
    print("Padding Mask：", first_mask)
    
    print("\n===== 批次数据示例（周测核心） =====")
    # 遍历所有批次
    for batch_idx, (batch_input, batch_target, batch_mask) in enumerate(dataloader.get_batches()):
        print(f"\n批次 {batch_idx+1}：")
        print("Input批次：", batch_input)
        print("Target批次：", batch_target)
        print("Mask批次：", batch_mask)