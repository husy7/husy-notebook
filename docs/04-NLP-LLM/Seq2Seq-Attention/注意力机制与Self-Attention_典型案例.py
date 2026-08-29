# -*- coding: utf-8 -*-
"""
注意力机制：从 Bahdanau 到 Self-Attention —— 典型代码演示
==========================================================
覆盖知识点：
  1. 缩放点积注意力（Scaled Dot-Product Attention）手写
  2. Bahdanau 加性注意力的对齐分数计算
  3. Self-Attention：序列自交互，得到上下文向量
  4. 多头注意力的直观对比
  5. 因果掩码（自回归）演示

依赖：pip install torch numpy
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# =====================================================================
# 一、缩放点积注意力（现代主流，Transformer 核心）
# =====================================================================
def scaled_dot_product_attention(q, k, v, mask=None):
    """scores = Q·K^T / sqrt(d_k)，softmax 后加权 V。
    Args:
        q,k,v: (batch, heads_or_1, seq, d_k)
    Returns:
        context, weights
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)   # 缩放防止点积过大
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)                 # 行归一化 → 注意力权重
    context = weights @ v
    return context, weights


# =====================================================================
# 二、Bahdanau 加性注意力（早期 Seq2Seq 对齐方式）
# =====================================================================
def bahdanau_score(decoder_state, encoder_states, W1, W2, v):
    """Bahdanau 对齐分数：
       score(s_{t-1}, h_i) = v^T tanh(W1 h_i + W2 s_{t-1})
    Args:
        decoder_state: (d_dec,)
        encoder_states: (seq_len, d_enc)
    Returns: (seq_len,) 每个源位置的分数
    """
    s = decoder_state                            # (d_dec,)
    e = v.unsqueeze(0) @ F.tanh(W1(encoder_states) + W2(s).unsqueeze(0).expand_as(W1(encoder_states)))
    return e.squeeze(0).squeeze(-1)              # (seq_len,)


# 造参数跑一下 Bahdanau 流程
d_enc, d_dec, seq = 8, 8, 6
W1 = nn.Linear(d_enc, 8); W2 = nn.Linear(d_dec, 8); vv = nn.Parameter(torch.randn(8, 1))
dec_st = torch.randn(d_dec)                      # 解码器当前隐态
enc_st = torch.randn(seq, d_enc)                 # 编码器全部隐态
scores = bahdanau_score(dec_st, enc_st, W1, W2, vv)      # 对齐分数
alphas = F.softmax(scores, dim=-1)                       # 归一化为权重
print("[Bahdanau] 对齐分数:\n", torch.round(scores, decimals=3))
print("[Bahdanau] softmax 后权重(和为1):", round(alphas.sum().item(), 4))

# =====================================================================
# 三、Self-Attention：序列内自交互
# =====================================================================
class SelfAttention(nn.Module):
    """单头自注意力：Q/K/V 都来自同一个输入序列。"""
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        q = self.Wq(x); k = self.Wk(x); v = self.Wv(x)
        context, weights = scaled_dot_product_attention(q, k, v, mask)
        return context, weights

# 演示：一句话"the cat sat on mat"，我们看每个位置如何混合其它位置
d_model, seq = 12, 5
sa = SelfAttention(d_model)
x = torch.randn(1, seq, d_model)                 # 5 个 token
context, weights = sa(x)
print("\n[Self-Attention] 输入 (1, seq=5, dim=%d)" % d_model,
      "→ 输出相同 shape:", tuple(context.shape))
print("[Self-Attention] 注意力权重矩阵 (5×5, 每行和为1):")
print(torch.round(weights[0], decimals=2))

# =====================================================================
# 四、单头 vs 多头注意力的直观对比
# =====================================================================
class MultiHead(nn.Module):
    """极简多头注意力：把 d 维切成 h 个子空间各算一次再拼接。"""
    def __init__(self, d_model, h=4):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        b, s, d = x.shape
        q = self.Wq(x).view(b, s, self.h, self.d_k).transpose(1, 2)  # (b,h,s,d_k)
        k = self.Wk(x).view(b, s, self.h, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(b, s, self.h, self.d_k).transpose(1, 2)
        ctx, _ = scaled_dot_product_attention(q, k, v, mask)
        ctx = ctx.transpose(1, 2).contiguous().view(b, s, d)
        return self.Wo(ctx)

mha = MultiHead(d_model, h=4)
out_mha = mha(x)
print("\n[Multi-Head] 输出 shape:", tuple(out_mha.shape))
print("[Multi-Head] 4 个头并行各关注不同关系（语法/指代/距离），再拼接")


# =====================================================================
# 五、因果掩码：自回归生成时禁止"偷看未来"
# =====================================================================
seq = 5
# 下三角掩码：第 i 行只能看前 i 列（即 ≤i 的 token）
causal = torch.tril(torch.ones(seq, seq)).unsqueeze(0).unsqueeze(0)
print("\n[因果掩码] 下三角矩阵:\n", causal[0, 0].int())

# 演示：未掩码 vs 掩码后第 3 个位置的注意力分布
rand_logits = torch.randn(1, 1, seq, seq)
probs_all = F.softmax(rand_logits, dim=-1)          # 未掩码：能看到整个序列
masked = rand_logits.masked_fill(causal == 0, float("-inf"))
probs_causal = F.softmax(masked, dim=-1)            # 掩码后：只看到前 3 个
print("\n[未掩码] 第3位置权重(有未来值):",
      torch.round(probs_all[0, 0, 3], decimals=2))
print("[掩码后] 第3位置权重(只保留前3个):",
      torch.round(probs_causal[0, 0, 3], decimals=2))

# =====================================================================
# 小结
# =====================================================================
# 注意力 = 按相关性动态加权取用其它位置信息；
# Bahdanau(加性)是早期对齐，Scaled Dot-Product 是现代主流；
# Self-Attention 让序列内充分交互 → Transformer 并行 + 长程依赖的基础。
