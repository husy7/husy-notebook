---
title: "Transformer 结构拆解"
tags: [深度学习, Transformer, Self-Attention, 架构]
date: 2026-08-29
---

# Transformer 结构拆解

## 定义

Transformer 是 Vaswani 等人在 2017 年论文《Attention Is All You Need》中提出的序列建模架构。它完全用**注意力机制（Attention）**取代 RNN/LSTM 的循环结构来处理序列：输入输出仍是 token 序列，但内部不再沿时间步递归传导隐状态，而是让每个位置直接对序列中所有其他位置计算相似度并加权聚合。

它解决了两大关键问题：

1. **并行化**：整个序列可以一次同时计算，不再逐步递归展开，天然利于 GPU 加速（对比 RNN 必须按时间步串行）。
2. **长程依赖**：任意两个位置之间的信息路径长度为 1，可以直接相互"看见"，不像 RNN 需要沿 n 个时间步逐步传导（远距离信息易衰减、梯度易消失）。

核心特征：Encoder 由 N 个相同的 block 堆叠，每个 block = **多头自注意力（MHA）+ 残差/LayerNorm + 前馈网络（FFN）+ 残差/LayerNorm**；注意力本身不感知位置，需要显式注入位置编码。

适用范畴极广：它是当前所有 LLM（GPT/LLaMA 等）、ViT 视觉 Transformer、多模态模型（CLIP、多模态 LLM）的基础引擎，并衍生出 Encoder-only（BERT，理解/表征）与 Decoder-only（GPT/LLaMA，自回归生成）两大流派。

## 原理

**为什么用注意力替代循环？** RNN 的串行递归带来两个瓶颈：① 无法并行，训练速度受限于时间步展开；② 长依赖信息路径为 O(n)，远端信息沿时间步逐步传导而衰减、梯度消失。注意力把任意两位置间的信息路径压到 O(1)，代价是每层计算/显存为 O(n²)（n 为序列长度），后续靠窗口/稀疏注意力与 FlashAttention 缓解。

**整体架构与数据流**（以 Encoder 为例，堆叠 N 个 block）：

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

**Self-Attention（自注意力）机制**：输入 x 经三个可学习投影矩阵映射为 Q（Query）/ K（Key）/ V（Value），然后计算：

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $QK^\top/\sqrt{d_k}$：对每个位置计算相对**所有其他位置**的点积相似度，再按行做 softmax 归一化得到注意力权重。
- 除以 $\sqrt{d_k}$（d_k 为 K 的维度）：缩放点积，防止维度增大时点积过大、进入 softmax 饱和区导致梯度消失。
- 用注意力权重对 V 加权求和 → 每个 token 得到融合全序列上下文的**上下文感知表示**。

**多头注意力（Multi-Head）**：把 Q/K/V 切成 h 个子空间（每头维度 = d_model/h），各自独立计算 attention 再拼接投影：

```text
多头 = Concat(Head_1, ..., Head_h) W_O
```

不同头可以分别关注不同关系（语法结构、指代消解、词间距等），提升表达能力；注意 d_model 必须能被 h 整除。

**残差 + LayerNorm**：每个子层输出做 `LayerNorm(x + Sublayer(x))`——残差让深层梯度畅通，LayerNorm 稳定激活分布，是它能够堆叠几十上百层的训练保障（残差思想与 ResNet 同源）。

**位置编码（Positional Encoding）**：注意力对 token 集合是置换等变的、本身**不感知顺序**，必须显式注入位置信息，常用：
- 正弦/余弦固定编码（原始 Transformer）。
- 可学习的位置嵌入（如 BERT）——更常用；长上下文场景另有 RoPE 等相对位置编码。

**前馈网络（FFN）**：每个位置独立过两层 MLP（线性 → ReLU/GELU → 线性），中间维度 d_ff 常取 4×d_model，为注意力输出提供非线性变换与逐位置建模能力。

Decoder 与 Encoder 的差别：Decoder 的自注意力带**因果掩码（causal mask）**（第 i 位只能看到 ≤i 的 token，保证自回归），并额外插入一层对 Encoder 输出的交叉注意力（cross-attention）。

## 应用

**典型使用场景**：NLP 几乎所有任务（分类/抽取/生成/翻译）、大规模语言模型预训练与推理、视觉（ViT 把图片拆 patch 当 token）、多模态对齐、语音与代码模型等。

**快速上手步骤**：
1. 分词（tokenize）：把文本切成 token 序列并映射到词表 id。
2. 输入表示：Token Embedding（+ 位置编码）得到 (batch, seq, d_model)。
3. 堆叠 N 个 Encoder/Decoder block（多头注意力 + 残差/LayerNorm + FFN）。
4. 接任务头：分类头/语言模型头；自回归生成时使用因果掩码。
5. 训练/推理：交叉熵等损失反向传播；推理阶段 Decoder-only 逐 token 生成。

**流派选型**：
- **Encoder-only**（BERT）→ 理解/表征（分类、NER、检索 embedding）。
- **Decoder-only**（GPT/LLaMA）→ 自回归生成，是当前 LLM 的主流形态。

**关键超参**：

| 超参 | 含义 |
|------|------|
| `d_model` | token 表示维度（如 512/768/4096） |
| `num_heads` | 多头数（如 8/12/32），`d_model` 需可被整除 |
| `num_layers` | Encoder/Decoder block 层数 |
| `d_ff` | FFN 中间层维度（常为 4×`d_model`） |

**注意事项 / 常见坑**：
- ❌ 忘记**因果掩码**就训练生成模型 → 提前"偷看"未来 token，指标虚高、生成错乱。✅ 训练时给 decoder 加后续掩码，使第 i 位只看到 ≤i 的 token（自回归）。
- ❌ Q/K 维度过低、除以 $\sqrt{d_k}$ 后方差仍失控 → softmax 数值不稳。✅ 关注缩放后的方差。
- ❌ 长序列自注意力 O(n²) **显存/算力爆炸**。✅ 用窗口/稀疏注意力、FlashAttention；超长序列（>100k）需专门的长上下文优化（KV 缓存见 04 板块）。
- ❌ 位置编码缺失 → 模型无法区分 token 顺序。✅ 必须注入位置信息（固定/可学习/相对）。

```python
# 代码示例：PyTorch 最小实现（单头自注意力）+ 案例详解
import torch, torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 三个可学习投影矩阵：把输入分别映射到 Q / K / V 空间
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.d_w = d_model ** 0.5   # 缩放因子 sqrt(d_k)：防止点积过大使 softmax 饱和

    def forward(self, x):            # x: (batch, seq, d_model)
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        # 计算每对位置的点积相似度 (batch, seq, seq)，并除以 sqrt(d_k) 缩放
        scores = q @ k.transpose(-2, -1) / self.d_w   # (batch, seq, seq)
        attn = F.softmax(scores, dim=-1)  # 对每个 query 行归一化 → 注意力权重
        return attn @ v                   # 用权重加权求和 V → 上下文感知表示

# 案例详解：随机造一批数据，batch=2、序列长=5、特征维=16
x = torch.randn(2, 5, 16)            # batch=2, 序列长=5, 维=16
print(SelfAttention(16)(x).shape)    # shape(2, 5, 16)：每个 token 聚合全序列后的表示

# ⚠️ 训练生成模型（自回归）必须加因果掩码：第 i 位只能看到 ≤i 的 token，decoder 端同理
#    实现要点：把 scores 中 (i, j>i) 的位置填 -inf（或 softmax 前用布尔 mask 屏蔽）
```

---
## 关联
- 前置：[[注意力机制与Self-Attention]]（Q/K/V 与点积相似度的基础，见 04-NLP-LLM/Seq2Seq-Attention）；另需残差连接、LayerNorm 前置概念
- 类似：[[ResNet残差网络拆解]]（区别是：ResNet 用残差连接解决深层 CNN 的梯度退化，本文用"残差 + LayerNorm"保障深层注意力训练；主体架构不同——卷积堆叠 vs 注意力 + FFN）
- 进阶：[[ViT 视觉 Transformer]]（把图片拆 patch 当 token，复用 Transformer 编码器做视觉，在 05-CV 板块展开）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（Transformer） | 多头自注意力一次看全序列：QKV 点积相似度 + 加权聚合，信息路径 O(1)、全序列可并行 | 大规模预训练、长程依赖任务、可 GPU 并行的 LLM/多模态骨干 |
| 替代方案（RNN/LSTM） | 沿时间步递归更新隐状态 h_t = f(h_{t-1}, x_t)，顺序逐步传导信息 | 短序列/低资源传统任务、严格流式逐 token 在线处理 |

注：本文方案内部另有 Encoder-only（BERT）与 Decoder-only（GPT/LLaMA）两大流派，分别面向理解表征与自回归生成，选型要点见上文"应用"。

---
## 参考
- [Attention Is All You Need（Transformer 原始论文）](https://arxiv.org/abs/1706.03762)
- [Hugging Face Transformers 官方文档](https://huggingface.co/docs/transformers/)

---
## 具体案例
- [[Transformer 结构拆解 实战示例]](Transformer结构拆解_sample.py)
