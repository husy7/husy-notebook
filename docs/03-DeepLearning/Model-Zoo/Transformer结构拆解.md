---
title: "Transformer 结构拆解"
tags: [深度学习, Transformer, Self-Attention, 架构]
date: 2026-08-29
---

# Transformer 结构拆解

## 一、核心思想

Transformer（Vaswani et al., 2017）完全用**注意力机制（Attention）**取代了 RNN/LSTM 的循环结构来处理序列，实现两点关键突破：

1. **并行化**：不再逐步递归，整个序列可一次同时计算（利于 GPU 加速）。
2. **长程依赖**：任意位置直接能"看到"序列中任意其他位置（不像 RNN 靠逐步传导）。

它是当前所有 LLM、ViT（视觉 Transformer）、多模态模型的基础引擎。

## 二、整体架构

```
        输入序列 x1..xn
              │
         (Token Embedding + 位置编码)
              │
   ┌──────────┴──────────┐
   │   多头自注意力(MHA)   │
   │     残差 + LayerNorm  │
   │ ─────────────────── │
   │  前馈网络(FFN: MLP)  │
   │     残差 + LayerNorm  │
   └──────────────────────┘
              │  （堆叠 N 个 Encoder Block）
              │
        输出表示 → 分类头/解码器
```

每个 block 由 **多头自注意力 + 前馈网络 + 残差/LayerNorm** 构成。

## 三、核心组件

### 3.1 Self-Attention（自注意力）

对每个 token，用三个矩阵把输入映射为 **Q（Query）/ K（Key）/ V（Value）**，然后：

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $QK^\top/sqrt(d_k)$ → 计算每个位置相对**所有其他位置**的相似度（点积）并归一化。
- 除以 $\sqrt{d_k}$ 防止点积过大导致 softmax 饱和、梯度消失。
- 用相似度加权求和 V → 每个 token 的新的、上下文感知的表示。

### 3.2 多头注意力（Multi-Head）

把 Q/K/V 切成 $h$ 个子空间，各自算 attention 再拼接：

```text
多头 = Concat(Head_1, ..., Head_h) W_O
```

不同头可关注不同关系（语法、指代、词间距等），提升表达能力。

### 3.3 位置编码（Positional Encoding）

注意力本身**不感知顺序**（对 token 位置无偏）。需显式注入位置信息，常用：
- 正弦/余弦固定编码。
- 可学习的位置嵌入（如 BERT）——更常用。

## 四、PyTorch 最小实现（单头自注意力）

```python
import torch, torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.d_w = d_model ** 0.5

    def forward(self, x):            # x: (batch, seq, d_model)
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = q @ k.transpose(-2, -1) / self.d_w   # (batch, seq, seq)
        attn = F.softmax(scores, dim=-1)
        return attn @ v

x = torch.randn(2, 5, 16)            # batch=2, 序列长=5, 维=16
print(SelfAttention(16)(x).shape)    # shape(2, 5, 16)，上下文聚合后的表示
```

> ⚠️ 训练须用因果掩码（mask）使第 i 位只看到 ≤i 的 token（自回归），解码端亦然。

## 五、关键超参与选型

| 超参 | 含义 |
|------|------|
| `d_model` | token 表示维度（如 512/768/4096） |
| `num_heads` | 多头数（如 8/12/32），`d_model` 需可被整除 |
| `num_layers` | Encoder/Decoder block 层数 |
| `d_ff` | FFN 中间层维度（常为 4×`d_model`） |

Transformer 有两大流派：
- **Encoder-only**（BERT）→ 理解/表征。
- **Decoder-only**（GPT/LLaMA）→ 自回归生成（当前 LLM 主流）。

## 六、边界与坑

- ❌ 忘记**因果掩码**就训练生成模型 → 提前"偷看"未来 token，指标虚高、生成错乱。✅ decoder 用后续掩码。
- ❌ **Q/K 维度过低**时除以 $\sqrt{d}$ 仍可能软掩码过载 → 数值不稳。✅ 关注缩放后的方差。
- ❌ 长序列自注意力 O(n²) **显存/算力爆炸**。✅ 用窗口/稀疏注意力、FlashAttention（KV 缓存见 04 板块）。
- ❌ Positional encoding 缺失 → 模型无法区分 token 顺序。✅ 必须注入位置信息。
- 边界：Transformer 每 token 计算 O(n²) 注意力，超长序列（>100k）需要专门的长上下文优化。

## 七、关联

- 前置知识：注意力机制、残差连接、LayerNorm。
- 同板块：[ResNet残差网络拆解](..\Model-Zoo\ResNet残差网络拆解.md)（残差思想复用）。
- 跨界：NLP/LLM 的正式引擎，详见 [注意力机制与 Self-Attention](../../04-NLP-LLM/Seq2Seq-Attention/注意力机制与Self-Attention.md)。
- 进阶：ViT（把图片拆 patch 当 token）——在 05-CV 板块展开。

## 八、参考

- Attention Is All You Need — https://arxiv.org/abs/1706.03762
- HuggingFace Transformer 官方文档 — https://huggingface.co/docs/transformers/
