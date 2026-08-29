---
title: "注意力机制：从 Bahdanau 到 Self-Attention"
tags: [NLP, 注意力机制, Bahdanau, Self-Attention, RNN]
date: 2026-08-29
---

# 注意力机制：从 Bahdanau 到 Self-Attention

## 一、核心思想

**问题**：经典 RNN/LSTM 把整句压进一个固定大小的"上下文向量"再解码。句子一长，早期信息在逐步传递中丢失 / 信息压缩过载（长程依赖与瓶颈问题）。

**解决**：**注意力机制（Attention）**让解码器在每个生成步**动态选择一个与当前最相关的输入位置**加权取用，而非只依赖一个压扁的向量。

## 二、Seq2Seq 与 RNN 的局限

Seq2Seq 通常用 Encoder-Decoder（RNN/LSTM）：

```
Encoder: 读取输入 → 输出 h1,h2,...,hn（各步隐状态）
Decoder: 逐步生成输出 y1,y2,...，每步参考 Encoder 的"最后一步向量 c"
```

**痛点**：机器翻译里长句容易"头重脚轻"，首批已翻译的内容在 decoder 阶段被冲刷遗忘。

## 三、Bahdanau 注意力（加性注意力）

目标：Decoder 在生成第 t 步时，**给 Encoder 每个输入步算一个权重**，加权求和得到"上下文向量"。

步骤：
1. 算对齐分数：$e_{ti} = \text{score}(s_{t-1}, h_i)$，常用 $e_{ti} = v^\top \tanh(W_1 h_i + W_2 s_{t-1})$。
2. 归一化为权重：$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_j \exp(e_{tj})}$（softmax）。
3. 加权上下文：$c_t = \sum_i \alpha_{ti} h_i$。

```text
Decoder一步:
   s_{t-1} ──┐
  h_1..h_n ──┼─> 算权重 α ──> c_t = Σ α_i·h_i ──> 生成 y_t
```

这样模型**每一步都知道该"看"输入的哪几个词**，长句质量大幅提升。

## 四、从加性注意力到点积注意力

- **Bahdanau**：加性（`tanh` 融合），早期常用，参数稍多。
- **Luong / 点积注意力**：分数直接用 $s_{t-1}^\top W h_i$，计算更简单高效。
- **缩放点积注意力**（Transformer）：$\text{softmax}(\frac{QK^\top}{\sqrt{d_k}})V$，是现在的主流。

## 五、Self-Attention（自注意力）

**Self-Attention = 输入序列自己对自己做注意力**：每个 token 通过 Q/K/V 与序列内**其他所有 token** 交互，获得"充分上下文"的新表示。

- 打破 RNN 的顺序依赖 → 可并行。
- 每个位置直接关联任意距离 → 长程依赖更直接。
- 这正是 Transformer 的核心（详见 03 板块 [Transformer结构拆解](../../03-DeepLearning/Model-Zoo/Transformer结构拆解.md)）。

```python
import torch, torch.nn.functional as F

def scaled_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    return F.softmax(scores, dim=-1) @ v

x = torch.randn(1, 8, 32)     # batch=1, 序列长=8, 维=32
print(scaled_dot_product_attention(x, x, x).shape)  # (1,8,32)
```

## 六、注意力家族总结

| 类型 | 特性 |
|------|------|
| **加性（Bahdanau）** | 早期，Encoder-Decoder 对齐 |
| **点积/缩放点积（Luong/Transformer）** | 高效、主流 |
| **Self-Attention** | 序列内自交互 → Transformer 基础 |
| **多头注意力** | 多子空间，捕捉多种关系 |
| **交叉注意力** | Encoder 与 Decoder 之间（Encoder 的 K/V ≈ 记忆） |

## 七、边界与坑

- ❌ 点积太大 → softmax 过饱和，梯度消失。✅ 除以 $\sqrt{d_k}$（缩放）。
- ❌ Self-Attention **不感知顺序**+未加位置编码 → token 顺序信息丢失。✅ 加位置编码；自回归还要因果掩码。
- ❌ 长序列 O(n²) 注意力算力/显存爆炸。✅ FlashAttention、窗口/稀疏注意力、KV-Cache（推理）。
- ❌ 把当前 token 也加入 Self-Attention 计算时在**解码生成**漏掉因果掩码 → 泄漏未来。✅ 严格上三角 masking。
- 边界：注意力是"软对齐"，没有显式对齐监督，可解释性要靠额外分析。

## 八、关联

- 前置知识：RNN/LSTM、softmax、点积。
- 同板块：[文本预处理与 Tokenizer](../Text-Preprocessing/文本预处理与Tokenizer.md)。
- 跨界：LLM 的注意力 + KV-Cache 详见 [LLM 推理：KV-Cache 与采样策略](../LLM/LLM推理与KV-Cache.md)。

## 九、参考

- Bahdanau et al., Neural Machine Translation by Jointly Learning to Align — https://arxiv.org/abs/1409.0473
- Attention Is All You Need — https://arxiv.org/abs/1706.03762
- Stanford CS224N 注意力讲义
