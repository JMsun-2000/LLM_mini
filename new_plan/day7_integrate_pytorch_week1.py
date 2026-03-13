import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

def get_universal_device():
    """
    全平台适配的设备选择函数（字节端侧LLM代码标准写法）
    优先级：CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
# ==============================
# 配置项（统一管理）
# ==============================
DEVICE = get_universal_device()
MAX_SEQ_LEN = 10  # 序列最大长度（Day5）
BATCH_SIZE = 2    # 批次大小（Day5）
PAD_ID = 0        # padding ID（Day6）

# ==============================
# Step1：PyTorch版词表工具（Day4+Day6）
# ==============================
class Vocab:
    def __init__(self, corpus: str):
        # 构建字符级词表（适配小样本）
        self.chars = sorted(list(set(corpus)))
        self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}  # 留0给PAD
        self.char2idx["<PAD>"] = PAD_ID
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
    
    def encode(self, text: str) -> List[int]:
        """文本→Token ID列表（PyTorch输入前置）"""
        return [self.char2idx.get(c, PAD_ID) for c in text]
    
    def decode(self, ids: torch.Tensor) -> str:
        """PyTorch张量→文本（生成结果解码）"""
        # 处理张量（支持CUDA/CPU，先转CPU再转列表）
        if ids.is_cuda or getattr(ids, "is_mps", False):
            ids = ids.cpu()
        ids = ids.tolist()
        return "".join([self.idx2char.get(i, "<PAD>") for i in ids if i != PAD_ID])

# ==============================
# Step2：PyTorch Dataset（Day4+Day5+Day6）
# 核心：输出GPT格式的PyTorch张量（input_ids/target_ids）
# ==============================
class GPTDataSet(Dataset):
    def __init__(self, texts: List[str], vocab: Vocab, max_seq_len: int):
        self.texts = texts
        self.vocab = vocab
        self.max_seq_len = max_seq_len
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：(input_ids, target_ids) → GPT专用格式（Day6）
        均为PyTorch张量，shape=(max_seq_len-1,)
        """
        text = self.texts[idx]
        # 1. 文本→Token ID
        token_ids = self.vocab.encode(text)
        
        # 2. 截断（Day5）
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        
        # 3. 构建GPT input/target（核心：右移一位）
        input_ids = token_ids[:-1] if len(token_ids) > 1 else []
        target_ids = token_ids[1:] if len(token_ids) > 1 else []
        
        # 4. Padding（Day5+Day6）
        pad_len = (self.max_seq_len - 1) - len(input_ids)
        input_ids += [PAD_ID] * pad_len
        target_ids += [-100] * pad_len  # -100：CrossEntropy忽略padding
        
        # 5. 转PyTorch张量（Day1+Day2），并指定设备（Day3）
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(DEVICE)
        target_ids = torch.tensor(target_ids, dtype=torch.long).to(DEVICE)
        
        return input_ids, target_ids

# ==============================
# Step3：数据加载与周测主流程（Day7核心）
# ==============================
def main():
    # 1. 准备测试语料
    train_texts = [
        "我爱吃苹果",
        "我爱吃香蕉",
        "今天天气好",
        "GPT是自回归模型",
        "用PyTorch手写GPT"
    ]
    corpus = "".join(train_texts)
    
    # 2. 初始化词表（Day4）
    vocab = Vocab(corpus)
    print(f"===== 词表信息（PyTorch版） =====")
    print(f"词表大小：{vocab.vocab_size}")
    print(f"字符→ID映射：{vocab.char2idx}")
    
    # 3. 初始化Dataset（Day4+Day5）
    dataset = GPTDataSet(train_texts, vocab, MAX_SEQ_LEN)
    
    # 4. 初始化DataLoader（Day5）
    # collate_fn：自定义批次拼接（这里用默认即可，因为__getitem__已统一长度）
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 打乱数据（训练必备）
        drop_last=False  # 保留最后一个不足批次
    )
    
    # 5. 遍历DataLoader，验证全流程（周测核心）
    print(f"\n===== 批次数据验证（PyTorch张量版） =====")
    for batch_idx, (batch_input, batch_target) in enumerate(dataloader):
        print(f"\n批次 {batch_idx+1}：")
        # 打印张量基本信息（Day1+Day2）
        print(f"Input张量 shape: {batch_input.shape}, device: {batch_input.device}")
        print(f"Target张量 shape: {batch_target.shape}, device: {batch_target.device}")
        
        # 打印具体值（解码验证）
        print(f"Input张量值：\n{batch_input}")
        print(f"Target张量值：\n{batch_target}")
        
        # 解码成文本（验证正确性）
        print(f"Input解码示例：{vocab.decode(batch_input[0])}")
        # 过滤-100后解码target
        target_0 = batch_target[0]
        target_0_valid = target_0[target_0 != -100]
        print(f"Target解码示例：{vocab.decode(target_0_valid)}")
        
        # 张量运算示例（Day2）：统计有效token数
        valid_mask = (batch_input != PAD_ID)
        valid_token_num = valid_mask.sum().item()
        print(f"本批次有效token数：{valid_token_num}")

if __name__ == "__main__":
    # 验证CUDA（Day3）
    print(f"PyTorch版本：{torch.__version__}")
    print(f"CUDA可用：{torch.cuda.is_available()}")
    print(f"当前设备：{DEVICE}")
    
    # 混合精度示例（Day3）：启用AMP
    torch.set_float32_matmul_precision('medium')
    
    # 执行主流程
    main()