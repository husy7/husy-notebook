# -*- coding: utf-8 -*-
"""
Transformer 结构拆解 —— 典型代码演示
====================================
覆盖知识点：
  1. 缩放点积注意力（Scaled Dot-Product Attention）手写
  2. 多头注意力（Multi-Head Attention）完整实现
  3. 一个最小 Encoder Block（注意力 + FFN + 残差/LayerNorm）
  4. 位置编码（Positional Encoding）与因果掩码
  5. 用 PyTorch 自带 MultiheadAttention 对照

依赖：pip install torch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# =====================================================================
# 一、缩放点积注意力（最核心的算子）
# =====================================================================
def scaled_dot_product_attention(q, k, v, mask=None):
    """缩放点积注意力：
       scores = Q·K^T / sqrt(d_k) → softmax → 加权 V
    Args:
        q,k,v: (batch, seq, d_k)
        mask: 若给掩码，则掩码位置填 -inf（softmax 后为 0）
    """
    d_k = q.shape[-1]
    # Q·K^T 得到 (batch, seq_q, seq_k) 点积相似度
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)   # 除以 sqrt 防梯度消失
    # 位置掩码 → 对未来 token 填 -inf，禁止"偷看"
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)    # 行归一化为注意力权重
    return weights @ v, weights            # 加权求和，返回权重便于观察


# =====================================================================
# 二、多头注意力模块
# =====================================================================
class MultiHeadAttention(nn.Module):
    """多头注意力：把单头注意力复制到 h 个子空间并行计算再拼接。"""
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k = d_model // num_heads   # 每个头的维度

        # Q/K/V 与输出的投影矩阵
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """把 (batch, seq, d_model) 拆成 (batch, heads, seq, d_k)。"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)          # → (batch, heads, seq, d_k)

    def forward(self, x, mask=None):
        batch, seq, _ = x.shape
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)     # 线性投影
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        attn_out, _ = scaled_dot_product_attention(q, k, v, mask)
        # 合并多头：(batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.dropout(self.Wo(attn_out))           # 输出投影


# =====================================================================
# 三、前馈网络 FFN（两个线性层 + ReLU/GELU）并配残差 + LayerNorm
# =====================================================================
def FeedForward(d_model, d_ff):
    """FFN: 升维(d_model→d_ff) → 激活 → 降维(d_ff→d_model)。"""
    return nn.Sequential(
        nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
    )


class EncoderBlock(nn.Module):
    """一个完整的 Encoder Block：
       ① 多头注意力 + 残差 + LayerNorm
       ② 前馈网络 + 残差 + LayerNorm
    """
    def __init__(self, d_model=64, num_heads=8, d_ff=256):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ffn   = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-LN 顺序：先 norm 再 attn，再加残差（比 Post-LN 更稳定，LLM 主流）
        x = x + self.attn(self.norm1(x), mask)      # 残差 1
        x = x + self.ffn(self.norm2(x))             # 残差 2
        return x


# =====================================================================
# 四、位置编码（Sinusoidal 固定编码）
# =====================================================================
def positional_encoding(seq_len, d_model):
    """正弦/余弦位置编码：偶数维用 sin，奇数维用 cos。"""
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1).float()        # (seq,1)
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model))       # 频率分母
    pe[:, 0::2] = torch.sin(pos * div)   # 偶数列
    pe[:, 1::2] = torch.cos(pos * div)   # 奇数列
    return pe.unsqueeze(0)               # (1, seq, d_model)


# =====================================================================
# 五、组装一个"自包含 Encoder"并跑前向
# =====================================================================
d_model, num_heads, seq_len = 64, 8, 12
encoder_block = EncoderBlock(d_model, num_heads, d_ff=256)

# 造一批随机 token 向量
x_input = torch.randn(2, seq_len, d_model)    # (batch=2, seq=12, dim=64)
x_emb = x_input + positional_encoding(seq_len, d_model)   # 加位置编码

out = encoder_block(x_emb)                    # 经过一个 Encoder Block
print("\n[EncoderBlock] 输入 shape", tuple(x_emb.shape),
      "→ 输出 shape", tuple(out.shape))

# 观察输出与位置编码无关（shape 一致、值被注意力混合）
pe = positional_encoding(seq_len, d_model)
print("[位置编码] 前三行第 0/1 维:\n", pe[0, :3, :2])

# =====================================================================
# 六、因果掩码：让第 i 个位置只能看到 ≤i 的 token
# =====================================================================
seq = 6
causal_mask = torch.tril(torch.ones(seq, seq)).unsqueeze(0).unsqueeze(0)
print("\n[因果掩码] 下三角（1=可见, 0=禁用）:\n", causal_mask[0, 0].int())

# 用单头注意力演示掩码效果：输出第 i 行的 softmax 权重只在 [0..i] 有值
bench_logits = torch.randn(1, 1, seq, seq)
masked_scores = bench_logits.masked_fill(causal_mask == 0, float("-inf"))
probs = F.softmax(masked_scores, dim=-1)
print("[掩码后] 第 3 行（位置3）注意力权重（应只在 0~3 非零）:\n",
      torch.round(probs[0, 0, 3], decimals=3))

# =====================================================================
# 七、与 PyTorch 内置 MultiheadAttention 对照
# =====================================================================
builtin = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
got, _ = builtin(x_emb, x_emb, x_emb)        # 自注意力：Q=K=V=x_emb
print("\n[内置 MHA] 输出 shape:", tuple(got.shape), "(与手写一致，均为 2×12×64)")

# =====================================================================
# 小结
# =====================================================================
# Self-Attention 让每个位置看到序列其他地方 → 并行 + 长程依赖；
# 多头 = 多关系视角；位置编码补顺序信息；因果掩码保证自回归不泄漏未来。
