import torch
import torch.nn as nn
import os
import torch.nn.functional as F

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def create_padding_mask(seq, pad_idx=0):
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = ~mask
    return mask.unsqueeze(0).unsqueeze(0)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == True, -1e9)

        atten = F.softmax(scores, dim=-1)
        atten = self.dropout(atten)
        return atten@v , atten

# 3. 多头注意力
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
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # self.w_q(q) equals to  q @ W^T + b
        q_proj = self.split_heads(self.w_q(q), batch_size)
        k_proj = self.split_heads(self.w_k(k), batch_size)
        v_proj = self.split_heads(self.w_v(v), batch_size)
        attn_out, _ = self.attn(q_proj, k_proj, v_proj, mask)
        out = self.combine_heads(attn_out, batch_size)
        return self.dropout(self.w_o(out))

# 4. FFN前馈网络
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.1, expansion=4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model*expansion)
        self.linear2 = nn.Linear(d_model*expansion, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

# 5. 单Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 注意力子层
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.attn(x, x, x, mask))
        # FFN子层
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x

# ===================== Day15 新增：多层Block堆叠（GPT解码器骨架）=====================
class GPTStack(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        # 🔥 核心：堆叠多个Transformer Block，用ModuleList封装（支持参数注册）
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # 最终层归一化（GPT架构惯例，堆叠后加一层LN）
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        前向传播：逐层传递特征，掩码全程复用
        Args:
            x: 输入特征 [batch, seq_len, d_model]
            mask: 组合掩码 [batch, 1, seq_len, seq_len]
        Returns:
            堆叠后的输出 [batch, seq_len, d_model]
        """
        # 逐层过Block，特征接力传递
        for block in self.blocks:
            x = block(x, mask)
        # 最终归一化，稳定输出分布
        x = self.final_norm(x)
        return x

# ===================== 测试脚本：验证堆叠效果 =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 超参数配置（小模型，快速测试）
    batch_size = 2
    seq_len = 6
    d_model = 512
    num_heads = 8
    num_layers = 3  # 堆叠3层Block，可自由调整

    '''
    一天必跑完的模型参数
    n_layer = 4~6
    n_head = 4
    d_model = 128~256
    vocab_size = 50257（GPT2 同款）
    参数量 ≈ 10M～30M
    '''
    
    # 构造输入
    x = torch.randint(1, 100, (batch_size, seq_len)).to(device)
    feat = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 生成掩码
    pad_mask = create_padding_mask(x).to(device)
    causal_mask = create_causal_mask(seq_len).to(device)
    mask = pad_mask | causal_mask
    
    # 初始化堆叠模型
    gpt_stack = GPTStack(num_layers, d_model, num_heads).to(device)
    
    # 前向推理
    with torch.no_grad():
        out = gpt_stack(feat, mask)
    
    # 验证维度
    print(f"模型层数: {num_layers}")
    print(f"输入shape: {feat.shape}")
    print(f"输出shape: {out.shape}")
    print("✅ 堆叠测试通过" if feat.shape == out.shape else "❌ 堆叠维度出错")
    # 打印模型结构，查看堆叠的Block
    print("\n模型结构预览:")
    print(gpt_stack)
