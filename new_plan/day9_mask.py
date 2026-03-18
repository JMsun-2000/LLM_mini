import torch
import torch.nn as nn
import math

def create_causal_mask(seq_len, device=None):
    """
    创建因果掩码
    Args:
        seq_len: 序列长度
        device: 设备
    Returns:
        因果掩码，形状为[1, seq_len, seq_len]，布尔类型，True表示需要屏蔽的位置
    """
    # 创建上三角矩阵，diagonal=1表示主对角线以上的元素为True（未来位置）
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    # 扩展batch维度，方便广播
    return mask.unsqueeze(0)


def create_attn_padding_mask(seq, pad_idx):
    """
    创建适用于注意力分数矩阵的填充掩码
    Args:
        seq: 输入序列，形状[batch_size, seq_len]
        pad_idx: PAD token的索引
    Returns:
        填充掩码，形状[batch_size, seq_len, seq_len]，布尔类型，True表示需要屏蔽的位置
    """
    batch_size, seq_len = seq.shape
    # 生成基础掩码 [batch_size, 1, seq_len]
    mask = (seq == pad_idx).unsqueeze(1)
    # 扩展为 [batch_size, seq_len, seq_len]
    return mask.expand(batch_size, seq_len, seq_len)

def combine_masks(causal_mask, padding_mask):
    """
    结合因果掩码和填充掩码
    Args:
        causal_mask: 因果掩码，形状[1, seq_len, seq_len]
        padding_mask: 填充掩码，形状[batch_size, seq_len, seq_len]
    Returns:
        组合后的掩码，形状[batch_size, seq_len, seq_len]
    """
    # 两个掩码取或，只要有一个为True就需要屏蔽
    return torch.logical_or(causal_mask, padding_mask)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

# 测试因果掩码
if __name__ == "__main__":
    seq_len = 5
    causal_mask = create_causal_mask(seq_len)
    print("因果掩码形状:", causal_mask.shape)
    print("因果掩码内容:")
    print(causal_mask)

    padding_mask = create_attn_padding_mask(torch.tensor([[1, 2, 3, 4, 0]]), 0)
    print("填充掩码形状:", padding_mask.shape)
    print("填充掩码内容:")
    print(padding_mask)

    print("================================================================")

    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    pad_idx = 0
    seq_len = seq.shape[1]
    
    causal_mask = create_causal_mask(seq_len)
    padding_mask = create_attn_padding_mask(seq, pad_idx)
    combined_mask = combine_masks(causal_mask, padding_mask)
    
    print("组合掩码形状:", combined_mask.shape)
    print("组合掩码内容:")
    print(combined_mask)

    print("================================================================")

    batch_size = 2
    seq_len = 5
    d_model = 64
    pad_idx = 0
    
    # 模拟输入
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    
    # 创建掩码
    causal_mask = create_causal_mask(seq_len)
    padding_mask = create_attn_padding_mask(seq, pad_idx)
    combined_mask = combine_masks(causal_mask, padding_mask)
    
    # 初始化注意力模块
    attn = ScaledDotProductAttention(dropout=0.1)
    
    # 前向传播
    output, attn_weights = attn(q, k, v, mask=combined_mask)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print("注意力权重（第一样本）:")
    print(attn_weights[0])
    # 验证掩码效果：PAD位置和未来位置的权重应该接近0
    print("注意力权重求和（第一样本，每行和为1）:")
    print(attn_weights[0].sum(dim=-1))