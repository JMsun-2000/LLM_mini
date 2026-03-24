import torch
import torch.nn as nn
import os
import torch.nn.functional as F

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def create_padding_mask(seq, pad_idx=0):
	mask = (seq == pad_idx)
	return mask.unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
	mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
	mask = ~mask
	return mask.unsqueeze(0).unsqueeze(0)

class ScaleDotProductAttention(nn.Module):
	def __init__(self, d_model, dropout=0.1):
		super().__init__()
		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v, mask):
		d_k = q.size(-1)
		scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

		if mask is not None:
			scores = scores.masked_fill(mask == True, -1e9)

		atten = F.softmax(scores, dim=-1)

		return self.dropout(atten)@v , atten


class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, num_heads, dropout=0.1):
		super().__init__()
		assert d_model % num_heads == 0
		self.d_model = d_model
		self.dropout = nn.Dropout(dropout)
		self.num_heads = num_heads
		self.d_k = d_model // num_heads

		self.q_w = nn.Linear(d_model, d_model)
		self.k_w = nn.Linear(d_model, d_model)
		self.v_w = nn.Linear(d_model, d_model)
		self.o_w = nn.Linear(d_model, d_model)
		self.atten = ScaleDotProductAttention(dropout)

	def split_heads(self, x, batch_size):
		return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

	def combine_heads(self, x, batch_size):
		return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

	def forward(self, q, k, v, mask=None):
		batch_size = q.size(0)
		q_proj = self.split_heads(self.q_w(q), batch_size)
		k_proj = self.split_heads(self.k_w(k), batch_size)
		v_proj = self.split_heads(self.v_w(v), batch_size)

		atten_out, _ = self.atten(q_proj, k_proj, v_proj, mask)
		out = self.combine_heads(atten_out, batch_size)
		atten_proj = self.dropout(self.o_w(out))
		return atten_proj

class FeedForwardNetwork(nn.Module):
	def __init__(self, d_model, dropout=0.1, expansion=4):
		super().__init__()
		self.d_model = d_model
		self.linear1 = nn.Linear(d_model, d_model * expansion)
		self.linear2 = nn.Linear(d_model * expansion, d_model)
		self.dropout = nn.Dropout(dropout)
		self.gelu = nn.GELU()

	def forward(self, x):
		linear1 = self.linear1(x)
		gelu = self.gelu(linear1)
		return self.linear2(self.dropout(gelu))

class TransformerBlock(nn.Module):
	def __init__(self, d_model, num_heads, dropout=0.1):
		super().__init__()
		self.ffn = FeedForwardNetwork(d_model, dropout)
		self.multiheadatten = MultiHeadAttention(d_model, num_heads, dropout)
		self.dropout = nn.Dropout(dropout)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)

	def forward(self, x, mask=None):
		residual = x
		x = self.norm1(x)
		x = residual + self.dropout(self.multiheadatten(x, x, x, mask))

		residual = x
		x = self.norm2(x)
		x = residual + self.dropout(self.ffn(x))

		return x


# 6. 周测测试脚本
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # 超参数
    batch_size, seq_len, d_model, num_heads = 2, 6, 512, 8
    
    # -------------------------- 变量x作用说明 --------------------------
    # 作用：专门用来**生成Padding掩码**，模拟真实场景中的文本序列（含pad填充符）
    # 1. 真实训练时，x是token化后的文本序列，用于定位pad位置
    # 2. 测试阶段仅用于造掩码、验证掩码逻辑，不参与特征前向传播
    # 3. 属于“造数据的辅助变量”，不是模型输入特征，所以后续没传入block
    # ------------------------------------------------------------------
    x = torch.randint(1, 100, (batch_size, seq_len)).to(device)
    # 构造模型输入特征（模拟Embedding层输出的特征向量）
    feat = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 生成掩码
    pad_mask = create_padding_mask(x).to(device)
    causal_mask = create_causal_mask(seq_len).to(device)
    # 合并掩码：任一位置需要屏蔽，最终就屏蔽
    mask = pad_mask | causal_mask
    # 模型测试
    block = TransformerBlock(d_model, num_heads).to(device)
    with torch.no_grad():
        out = block(feat, mask)
    # 验证结果
    print(f"输入shape: {feat.shape}")
    print(f"输出shape: {out.shape}")
    print("✅ 周测通过" if feat.shape == out.shape else "❌ 代码存在错误")






























