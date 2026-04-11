import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ===================== 复用Day14、Day15基础组件（无修改）=====================
# 掩码工具、注意力、FFN、TransformerBlock、GPTStack 直接沿用
# 此处省略重复代码，直接调用前期已实现模块
# ------------------- 缩放点积注意力（标准无修改）-------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == True, -1e9)
        attn = F.softmax(scores, dim=-1)
        # 仅对注意力权重做Dropout，非输出特征
        attn = self.dropout(attn)
        return torch.matmul(attn, v), attn

# ------------------- 多头注意力（标准版：移除末尾Dropout）-------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_proj = self.split_heads(self.w_q(q), batch_size)
        k_proj = self.split_heads(self.w_k(k), batch_size)
        v_proj = self.split_heads(self.w_v(v), batch_size)
        attn_out, _ = self.attn(q_proj, k_proj, v_proj, mask)
        out = self.combine_heads(attn_out, batch_size)
        # ✅ 标准GPT：此处**不做Dropout**，移除重复Dropout
        return self.w_o(out)

# ------------------- FFN前馈网络（标准版：Dropout放中间，官方设计）-------------------
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.1, expansion=4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model*expansion)
        self.linear2 = nn.Linear(d_model*expansion, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # 标准GPT FFN结构：升维→激活→Dropout→降维，仅这一处Dropout
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

# ------------------- Transformer Block（标准Pre-LN，残差前单次Dropout）-------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 标准GPT Pre-LN结构：先Norm，再注意力，Dropout后加残差
        residual = x
        x = self.norm1(x)
        # ✅ 仅此处1次Dropout：注意力输出→Dropout→残差相加，无重复
        x = residual + self.dropout(self.attn(x, x, x, mask))
        
        # FFN子层：同标准逻辑
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x

# ------------------- Block堆叠模块（标准无修改）-------------------
class GPTStack(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return x

# ------------------- 掩码工具（标准无修改）-------------------
def create_padding_mask(seq, pad_idx=0):
    mask = (seq == pad_idx)
    return mask.unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = ~mask
    return mask.unsqueeze(0).unsqueeze(0)

# ===================== Day16 新增：词嵌入+位置编码 =====================
'''
那为什么要搞 d_model 维，而不是 1 维？
因为：
1维位置信息表达能力太弱，注意力学不到复杂关系。
比如只用一个数字表示位置：pos = 0,1,2,3,4...
那 Q・K^T 里，位置信息只有一个维度参与相似度计算，太容易被淹没。
而 Transformer 需要：位置信息要渗透到每一个特征维度这样注意力才能在每一路特征都学到：
谁在前
谁在后
隔了多远
是什么结构（句子开头 / 中间 / 结尾）
'''
# ------------------- 固定位置编码（sin/cos）-------------------
'''
不是神来之笔，是一步步推导出来的：
要注入位置
必须是 d_model 维向量
不能爆炸、不能训练、能外推
只有周期函数满足
只有 sin/cos 能表示相对位置
奇偶交替刚好构成一组线性基
多频率覆盖多尺度位置
→ 最终自然走到 sin/cos 奇偶编码
'''
class SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
         # 生成位置矩阵 [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        # 计算分母项
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 初始化位置编码 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 偶数位sin，奇数位cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为缓冲区，不参与训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Args:
            x: 词嵌入输出 [batch, seq_len, d_model]
        Returns:
            带位置信息的特征 [batch, seq_len, d_model]
        '''
        # 截取对应序列长度的位置编码 + 嵌入结果
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

# ------------------- 可学习位置编码（GPT首选）-------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 随机初始化位置向量，随训练优化
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
    	# 广播适配batch，与词嵌入相加
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ------------------- GPT嵌入层（标准无修改）-------------------
class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float = 0.1, use_learnable_pe: bool = True):
        super().__init__()
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码（二选一）
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len, dropout) if use_learnable_pe else SinCosPositionalEncoding(d_model, max_len, dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        '''
        Args:
            x: 文本token序列 [batch, seq_len]
        Returns:
            最终输入特征 [batch, seq_len, d_model]
        '''
        # 词嵌入 + 缩放
        token_emb = self.token_embedding(x) * self.scale
        # 加位置编码
        emb_out = self.pos_encoding(token_emb)
        return emb_out

# ===================== 标准架构测试 =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 超参数配置
    batch_size = 2
    seq_len = 6
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 3
    max_len = 5000
    dropout = 0.1

    # 1. 构造模拟文本token（真实场景为文本转token索引）
    token_seq = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    # 初始化模块
    embedding_layer = GPTEmbedding(vocab_size, d_model, max_len, dropout).to(device)
    gpt_stack = GPTStack(num_layers, d_model, num_heads, dropout).to(device)
    # 生成掩码
    pad_mask = create_padding_mask(token_seq).to(device)
    causal_mask = create_causal_mask(seq_len).to(device)
    mask = pad_mask | causal_mask

    # 前向推理
    with torch.no_grad():
        emb_output = embedding_layer(token_seq)
        final_output = gpt_stack(emb_output, mask)

    # 维度验证
    print(f"输入token shape: {token_seq.shape}")
    print(f"嵌入+位置编码 shape: {emb_output.shape}")
    print(f"最终输出 shape: {final_output.shape}")
    print("✅ 标准GPT架构测试通过" if emb_output.shape == final_output.shape else "❌ 维度出错")