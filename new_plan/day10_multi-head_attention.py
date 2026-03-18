import torch
import torch.nn as nn
import math

# 复用Day8：Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（因果掩码/填充掩码）
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)  # 用-1e9避免数值问题
        
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

# 复用Day9：因果掩码和填充掩码创建函数
def create_causal_mask(seq_len, device=None):
    # 因果掩码：屏蔽未来位置，形状[1, seq_len, seq_len]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0)

def create_attn_padding_mask(seq, pad_idx):
    # 适用于注意力分数矩阵的填充掩码，形状[batch_size, seq_len, seq_len]
    batch_size, seq_len = seq.shape
    mask = (seq == pad_idx).unsqueeze(1)
    return mask.expand(batch_size, seq_len, seq_len)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        初始化多头注意力层
        Args:
            d_model: 输入/输出特征维度（必须能被num_heads整除）
            num_heads: 注意力头的数量
            dropout: Dropout概率，用于防止过拟合
        """
        super().__init__()
        # 校验d_model是否能被num_heads整除，确保每个头维度一致
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model          # 总特征维度
        self.num_heads = num_heads      # 注意力头数
        self.d_k = d_model // num_heads # 每个头的特征维度
        
        # 定义线性层：Q、K、V投影层 + 输出融合层
        self.w_q = nn.Linear(d_model, d_model)  # Q投影
        self.w_k = nn.Linear(d_model, d_model)  # K投影
        self.w_v = nn.Linear(d_model, d_model)  # V投影
        self.w_o = nn.Linear(d_model, d_model)  # 输出融合投影（核心：融合所有头的特征，对应理论中“拼接后的线性投影”）
        
        # 注意力计算依赖（复用ScaledDotProductAttention）
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)      # 输出Dropout
    
    def split_heads(self, x, batch_size):
        """
        将输入拆分为多个头，维度转换：[batch_size, seq_len, d_model] → [batch_size, num_heads, seq_len, d_k]
        Args:
            x: 输入张量，形状[batch_size, seq_len, d_model]
            batch_size: 批次大小
        Returns:
            拆分后的多头张量
        """
        # 先拆分d_model为num_heads × d_k，再转置调整维度顺序（便于多头并行计算）
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # 转置后：[batch_size, num_heads, seq_len, d_k]
    
    def combine_heads(self, x, batch_size):
        """
        将多个头的输出拼接融合，维度转换：[batch_size, num_heads, seq_len, d_k] → [batch_size, seq_len, d_model]
        是split_heads的逆操作
        Args:
            x: 多头注意力输出，形状[batch_size, num_heads, seq_len, d_k]
            batch_size: 批次大小
        Returns:
            拼接融合后的张量
        """
        # 先转置还原维度顺序，再拼接为d_model维度
        x = x.transpose(1, 2)  # 转置后：[batch_size, seq_len, num_heads, d_k]
        # contiguous()确保内存连续，避免view报错
        return x.contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, q, k, v, mask=None):
        """
        前向传播（自注意力模式：Q=K=V）
        Args:
            q: 查询张量，形状[batch_size, seq_len_q, d_model]
            k: 键张量，形状[batch_size, seq_len_k, d_model]
            v: 值张量，形状[batch_size, seq_len_v, d_model]（seq_len_k = seq_len_v）
            mask: 掩码张量，形状[batch_size, seq_len_q, seq_len_k]，可选（因果/填充掩码）
        Returns:
            output: 多头注意力最终输出，形状[batch_size, seq_len_q, d_model]
            attn_weights: 所有头的注意力权重，形状[batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)
        
        # 步骤1：Q、K、V线性投影
        q_proj = self.w_q(q)  # [batch_size, seq_len_q, d_model]
        k_proj = self.w_k(k)  # [batch_size, seq_len_k, d_model]
        v_proj = self.w_v(v)  # [batch_size, seq_len_v, d_model]
        
        # 步骤2：拆分多头
        q_split = self.split_heads(q_proj, batch_size)  # [batch_size, num_heads, seq_len_q, d_k]
        k_split = self.split_heads(k_proj, batch_size)  # [batch_size, num_heads, seq_len_k, d_k]
        v_split = self.split_heads(v_proj, batch_size)  # [batch_size, num_heads, seq_len_v, d_k]
        
        # 步骤3：适配掩码（若有），确保掩码能广播到所有头
        if mask is not None:
            # 掩码形状调整为[batch_size, 1, seq_len_q, seq_len_k]，广播到num_heads个头上
            mask = mask.unsqueeze(1)
        
        # 步骤4：每个头独立计算Scaled Dot-Product Attention
        attn_output, attn_weights = self.attention(q_split, k_split, v_split, mask=mask)
        
        # 步骤5：拼接所有头的输出
        output = self.combine_heads(attn_output, batch_size)
        
        # 步骤6：输出线性投影 + Dropout
        output = self.dropout(self.w_o(output))
        
        return output, attn_weights


if __name__ == "__main__":
    # 设备配置：自动选择GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 超参数设置（GPT基础配置）
    batch_size = 2    # 批次大小
    seq_len = 5       # 序列长度
    d_model = 512     # 总特征维度
    num_heads = 8     # 注意力头数（512 ÷ 8 = 64，每个头64维）
    pad_idx = 0       # PAD token索引
    
    # 模拟输入：文本序列嵌入后的张量（实际由Embedding层生成）
    q = torch.randn(batch_size, seq_len, d_model).to(device)
    k = torch.randn(batch_size, seq_len, d_model).to(device)
    v = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 模拟变长序列（含PAD token），用于生成填充掩码
    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]).to(device)
    
    # 步骤1：创建掩码（结合因果掩码和填充掩码，适配GPT自注意力）
    causal_mask = create_causal_mask(seq_len, device)  # [1, 5, 5]
    padding_mask = create_attn_padding_mask(seq, pad_idx)  # [2, 5, 5]
    combined_mask = torch.logical_or(causal_mask, padding_mask)  # 组合掩码
    
    # 步骤2：初始化多头注意力层
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)
    
    # 步骤3：前向传播（自注意力模式，Q=K=V）
    output, attn_weights = mha(q, k, v, mask=combined_mask)
    
    # 步骤4：验证输出形状和掩码效果
    print(f"\n输入形状: {q.shape}")                  # [2, 5, 512]
    print(f"输出形状: {output.shape}")              # [2, 5, 512]（维度不变，可串联）
    print(f"注意力权重形状: {attn_weights.shape}")  # [2, 8, 5, 5]（batch, heads, seq_q, seq_k）
    
    # 验证掩码效果：PAD位置和未来位置的注意力权重接近0
    print("\n第一个样本、第一个头的注意力权重（保留2位小数）:")
    print(attn_weights[0, 0].round(decimals=2))
    print("\n第一个样本注意力权重每行求和（应为1，验证softmax有效性）:")
    print(attn_weights[0].sum(dim=-1).round(decimals=2))