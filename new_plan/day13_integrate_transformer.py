import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Mac MPS兼容配置
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# -------------------------- 复用Day9：Mask掩码实现 --------------------------
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """填充掩码：屏蔽pad占位符，shape [batch, 1, 1, seq_len]"""
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """因果掩码（解码器专用）：屏蔽未来信息，下三角矩阵，shape [1,1,seq_len,seq_len]"""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)

# -------------------------- 复用Day8：缩放点积注意力 --------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        d_k = q.size(-1)
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # 掩码屏蔽（置为极小值）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == False, -1e9)
        # 注意力权重+dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 加权求和
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# -------------------------- 复用Day10：多头注意力 --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # QKV投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        # 缩放点积注意力
        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor, batch_size: int):
        """拆分多头：[batch, seq, d_model] → [batch, heads, seq, d_k]"""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor, batch_size: int):
        """拼接多头：[batch, heads, seq, d_k] → [batch, seq, d_model]"""
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        batch_size = q.size(0)
        # QKV线性投影
        q_proj = self.split_heads(self.w_q(q), batch_size)
        k_proj = self.split_heads(self.w_k(k), batch_size)
        v_proj = self.split_heads(self.w_v(v), batch_size)
        # 注意力计算
        attn_out, attn_weights = self.attn(q_proj, k_proj, v_proj, mask)
        # 拼接+输出投影
        out = self.combine_heads(attn_out, batch_size)
        out = self.dropout(self.w_o(out))
        return out, attn_weights

# -------------------------- 复用Day12：FFN前馈网络 --------------------------
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, expansion_factor: int = 4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Xavier初始化（稳定训练）
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # 1. 多头注意力（Day8+9+10）
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. FFN（Day12）
        self.ffn = FeedForwardNetwork(d_model, dropout)
        # 3. LayerNorm（Day11）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 4. Dropout+残差（Day11）
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播（Pre-LN架构）
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
            mask: 组合掩码 [batch_size, 1, seq_len, seq_len]
        Returns:
            模块输出 [batch_size, seq_len, d_model]
        """
        # -------------------------- 注意力子层+残差 --------------------------
        residual = x
        # 先归一化
        x = self.norm1(x)
        # 多头注意力计算
        attn_out, _ = self.attn(x, x, x, mask)
        # Dropout+残差相加
        # ❌ 偏方慎入：千万别给residual加Dropout！纯属极端调奇招，手写GPT必翻车
        x = residual + self.dropout(attn_out)

        # -------------------------- FFN子层+残差 --------------------------
        residual = x
        # 先归一化
        x = self.norm2(x)
        # FFN特征加工
        ffn_out = self.ffn(x)
        # Dropout+残差相加
        x = residual + self.dropout(ffn_out)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # 堆叠多个Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 逐层传播
        for layer in self.layers:
            x = layer(x, mask)
        # 最终归一化
        return self.norm(x)

if __name__ == "__main__":
    # 设备适配
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"测试设备：{device}")

    # 统一超参数
    batch_size = 2
    seq_len = 6
    d_model = 512
    num_heads = 8
    pad_idx = 0

    # 1. 构造模拟输入序列（含pad占位符）
    x = torch.randint(low=1, high=100, size=(batch_size, seq_len), device=device)
    print(f"输入序列: {x}")
    x[0, 4:] = pad_idx  # 手动添加pad
    print(f"输入序列_pad: {x}")
    print(f"输入序列shape：{x.shape}")

    # 2. 生成掩码（padding+因果）
    padding_mask = create_padding_mask(x, pad_idx).to(device)
    causal_mask = create_causal_mask(seq_len).to(device)
    # 组合掩码
    mask = padding_mask & causal_mask
    print(f"padding_mask: {padding_mask}")
    print(f"causal_mask: {causal_mask}")
    print(f"mask: {mask}")
    print(f"组合掩码shape：{mask.shape}")

    # 3. 初始化完整模块
    transformer_block = TransformerBlock(d_model, num_heads).to(device)
    # 构造特征输入（模拟Embedding输出）
    feat = torch.randn(batch_size, seq_len, d_model, device=device)

    # 4. 前向传播
    with torch.no_grad():
        output = transformer_block(feat, mask)

    # 5. 核心验证：维度守恒
    print(f"模块输入特征shape：{feat.shape}")
    print(f"模块输出特征shape：{output.shape}")
    assert feat.shape == output.shape, "❌ 串联失败：维度不匹配"
    print("✅ Transformer全组件串联测试通过！")