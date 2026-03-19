import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

# Mac MPS设备兼容配置
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, expansion_factor: int = 4):
        """
        前馈网络初始化（GPT标准结构）
        Args:
            d_model: 模型特征维度（必须与注意力层、Embedding层一致）
            dropout: Dropout失活概率，防止过拟合
            expansion_factor: 升维扩展因子，GPT默认4
        """
        super().__init__()
        # 升维全连接层：d_model → 4×d_model
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        # 降维全连接层：4×d_model → d_model
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        # GELU激活函数（平滑非线性，适配深层网络）
        self.gelu = nn.GELU()
        # Dropout正则层
        self.dropout = nn.Dropout(dropout)

        # 初始化技巧：Xavier均匀初始化，稳定训练初期梯度
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征，shape [batch_size, seq_len, d_model]
        Returns:
            加工后特征，shape与输入完全一致
        """
        # 第一步：升维变换
        x = self.linear1(x)
        # 第二步：非线性激活
        x = self.gelu(x)
        # 第三步：Dropout正则
        x = self.dropout(x)
        # 第四步：降维还原维度
        x = self.linear2(x)
        return x

    
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

# 复用Day11的Transformer Block（替换FFN为手写版）
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        # 复用Day10多头注意力
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 接入Day12手写FFN
        self.ffn = FeedForwardNetwork(d_model, dropout)
        # Pre-LN层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 注意力子层+残差
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, mask)
        x = residual + self.dropout(attn_out)

        # FFN子层+残差
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        return x

# 联调测试（省略掩码、注意力代码，复用Day9-10即可）
if __name__ == "__main__":
    # 设备适配
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"测试设备：{device}")

    # 超参数（对齐前文）
    batch_size = 2
    seq_len = 5
    d_model = 512

    # 模拟输入：随机特征张量
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    print(f"FFN输入形状：{x.shape}")

    # 初始化FFN
    ffn = FeedForwardNetwork(d_model=d_model).to(device)
    # 前向运算
    output = ffn(x)

    # 核心验证：输入输出维度必须相等
    print(f"FFN输出形状：{output.shape}")
    assert x.shape == output.shape, "❌ FFN维度不匹配，检查升维降维层！"
    print("✅ FFN独立测试通过：维度守恒")


    # 超参数
    d_model = 512
    num_heads = 8
    x = torch.randn(2, 5, d_model).to(device)

    block = TransformerBlock(d_model, num_heads).to(device)
    output = block(x)
    print(f"Block输入：{x.shape}")
    print(f"Block输出：{output.shape}")
    print("✅ FFN+Block联调通过，可进入组件串联！")