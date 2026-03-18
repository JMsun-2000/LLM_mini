import torch
import torch.nn as nn
import math

# 手动实现版本
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        # 获取键的维度
        d_k = q.size(-1)
        
        # 1. 计算注意力分数：QK^T / √d_k
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. 应用掩码（如果提供）
        if mask is not None:
            # 将掩码位置的分数设置为负无穷，softmax后这些位置权重为0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. 应用softmax和dropout
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        
        # 4. 加权求和得到输出
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights


#使用 PyTorch 内置函数版本
import torch.nn.functional as F

def scaled_dot_product_attention_builtin(q, k, v, mask=None, dropout=0.0):
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=dropout,
        is_causal=False  # 如果是因果掩码，设置为True
    )
    return output


# 测试手动实现的Scaled Dot-Product Attention
if __name__ == "__main__":
    # 模拟输入：batch_size=2, seq_len=5, d_k=d_v=64
    batch_size = 2
    seq_len = 5
    d_model = 64
    
    # 随机生成Q、K、V
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # 初始化注意力模块
    attn = ScaledDotProductAttention(dropout=0.1)
    
    # 前向传播
    output, attn_weights = attn(q, k, v)
    
    print(f"输入Q形状: {q.shape}")
    print(f"输入K形状: {k.shape}")
    print(f"输入V形状: {v.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"注意力权重每行和为1: {torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)))}")
    
    # 测试内置函数
    output_builtin = scaled_dot_product_attention_builtin(q, k, v, dropout=0.1)
    print(f"内置函数输出形状: {output_builtin.shape}")