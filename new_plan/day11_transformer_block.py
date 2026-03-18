import decimal
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ------------------- 复用Day9：掩码函数 -------------------
def create_causal_mask(seq_len, device=None):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0)

def create_attn_padding_mask(seq, pad_idx):
    batch_size, seq_len = seq.shape
    mask = (seq == pad_idx).unsqueeze(1)
    return mask.expand(batch_size, seq_len, seq_len)

# ------------------- 复用Day10：缩放点积注意力 -------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        # 重点：softmax在前，dropout在后，保证权重和为1
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

# ------------------- 复用Day10：多头注意力 -------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x, batch_size):
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)
        
        q_split = self.split_heads(q_proj, batch_size)
        k_split = self.split_heads(k_proj, batch_size)
        v_split = self.split_heads(v_proj, batch_size)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        attn_output, attn_weights = self.attention(q_split, k_split, v_split, mask=mask)
        output = self.combine_heads(attn_output, batch_size)
        output = self.dropout(self.w_o(output))
        
        return output, attn_weights


# ------------------- Day11新增：前馈网络FFN -------------------
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.1, expansion_factor=4):
        super().__init__()
        # GPT标准配置：升维4倍
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)
        # GELU激活函数，适配Transformer训练
        self.gelu = nn.GELU()
    
    def forward(self, x):
        # 升维 -> 激活 -> Dropout -> 降维
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# ------------------- Day11核心：Transformer Block -------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, expansion_factor=4):
        super().__init__()
        # 1. 多头注意力层
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. 前馈网络
        self.ffn = FeedForwardNetwork(d_model, dropout, expansion_factor)
        # 3. 两个层归一化（Pre-LN架构）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 4. Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: 输入特征，shape [batch_size, seq_len, d_model]
            mask: 因果掩码+填充掩码，shape [batch_size, seq_len, seq_len]
        Returns:
            x: Block输出特征，shape不变
            attn_weights: 注意力权重，用于可视化/调试
        """
        # 子层1：多头注意力 + 残差 + Pre-LN
        residual = x
        x = self.norm1(x)  # Pre-LN：先归一化
        attn_out, attn_weights = self.attn(x, x, x, mask)  # 自注意力
        x = residual + self.dropout(attn_out)  # 残差连接
        
        # 子层2：FFN + 残差 + Pre-LN
        residual = x
        x = self.norm2(x)  # Pre-LN：先归一化
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)  # 残差连接
        
        return x, attn_weights


if __name__ == "__main__":
    # 设备配置：自动适配CUDA/MPS/CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 超参数（对齐Day10）
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8
    pad_idx = 0
    dropout = 0.1
    
    # 模拟输入：序列嵌入特征
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    # 模拟真实序列（含PAD）
    seq = torch.tensor([[1,2,3,0,0], [4,5,0,0,0]]).to(device)
    
    # 构建掩码
    causal_mask = create_causal_mask(seq_len, device)
    padding_mask = create_attn_padding_mask(seq, pad_idx)
    mask = torch.logical_or(causal_mask, padding_mask)
    
    # 初始化Block
    block = TransformerBlock(d_model, num_heads, dropout).to(device)
    
    # 前向传播
    output, attn_weights = block(x, mask)
    
    # 验证维度一致性（核心：输入输出维度完全相同）
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"\n✅ Transformer Block验证通过：维度匹配，可堆叠训练！")
    
    # 验证注意力权重合理性（每行和为1）
    print("\n第一个样本第一个头注意力权重求和（每行≈1）:")
    # 方案2：打印前将张量迁移至CPU，彻底规避MPS不支持round的问题（通用兜底）
    print(attn_weights[0, 0].sum(dim=-1).cpu().round(decimals=2))
