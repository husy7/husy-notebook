---
title: "注意力机制：从 Bahdanau 到 Self-Attention"
tags: [NLP, 注意力机制, Bahdanau, Self-Attention, RNN]
date: 2026-08-29
---

# 注意力机制：从 Bahdanau 到 Self-Attention

## 定义

注意力机制（Attention Mechanism）是一类让序列模型**在每个输出步动态地为输入各位置计算相关度（对齐分数），并按权重加权聚合输入信息**，从而得到"针对性上下文"的机制族。它解决的核心问题是：经典 RNN/LSTM 的 Seq2Seq 模型被迫把整句压进一个固定大小的"上下文向量 c"再解码，句子一长，早期信息在逐步传递中被冲刷遗忘，或信息压缩过载，造成长程依赖与瓶颈问题。

注意力机制的核心特征有三点：

- **动态选择**：Decoder 生成第 t 步时，对 Encoder 的每个输入步算一个权重，权重随输出步变化，而非始终依赖同一个压扁向量。
- **软对齐（soft alignment）**：权重经 softmax 归一化为概率分布（和为 1），可微、可端到端训练，无需显式对齐标注；代价是可解释性要靠额外分析。
- **全局交互**：在 Self-Attention 形式下，序列内任意两个位置可直接建立联系，打破了 RNN 只能沿时间步逐步传递的顺序约束。

适用范畴：神经机器翻译、文本摘要等 Encoder-Decoder 任务（交叉注意力），以及 Transformer 系列架构（Self-Attention、多头注意力、因果注意力）——它是当代 LLM（GPT、Llama 等）的底层核心运算，也从 NLP 扩展到视觉（ViT）、多模态等领域。

## 原理

**为什么这样设计**：把"整句压缩成单一向量再解码"的信息瓶颈，替换为"每个解码步都保留对所有输入隐状态的访问权，并学会挑选当前最相关的部分"。模型由此获得一条直达任意输入位置的"快捷通路"，长程依赖不再依赖隐状态链式传递。

**Bahdanau 加性注意力（Decoder 第 t 步的完整流程）**：

1. Encoder 读取输入，输出各步隐状态 $h_1, h_2, \dots, h_n$；记 Decoder 上一步状态为 $s_{t-1}$。
2. 逐位置计算对齐分数：$e_{ti} = \text{score}(s_{t-1}, h_i) = v^\top \tanh(W_1 h_i + W_2 s_{t-1})$（加性：用 tanh 把两侧隐状态融合后打分，参数稍多）。
3. softmax 归一化为权重分布：$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_j \exp(e_{tj})}$——表示"这一步该看输入的哪几个词、各看多少"。
4. 加权求和得上下文向量：$c_t = \sum_i \alpha_{ti} h_i$，与 $s_{t-1}$ 一起用于生成 $y_t$。

```text
Decoder一步:
   s_{t-1} ──┐
  h_1..h_n ──┼─> 算权重 α ──> c_t = Σ α_i·h_i ──> 生成 y_t
```

**从加性到点积再到缩放点积**：

- **加性（Bahdanau）**：$\text{score}=v^\top \tanh(W_1 h_i + W_2 s_{t-1})$，早期常用、可表达更复杂的对齐，但参数多。
- **点积（Luong）**：$\text{score} = s_{t-1}^\top W h_i$，直接用点积打分，更简单高效。
- **缩放点积（Transformer，当前主流）**：$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$。除以 $\sqrt{d_k}$ 是为了在维度 $d_k$ 较大时抑制点积数值过大——否则 softmax 进入饱和区、梯度趋近于 0（过饱和导致梯度消失）。

**Self-Attention（自注意力）**：令 $Q=K=V=$ 输入序列自身（或其线性投影），每个 token 通过 Q/K/V 与序列内**其他所有 token** 交互，一次前向即可让每个位置拿到"全序列充分上下文"的新表示。它带来三点质变：① 打破 RNN 的顺序依赖 → 可并行计算；② 每个位置直接关联任意距离 → 长程依赖更直接；③ 成为 Transformer 的核心算子。工程上再叠加**多头注意力**（多子空间捕捉多种关系）与**交叉注意力**（Decoder 的 Q 去查询 Encoder 的 K/V，K/V 相当于外部"记忆"）即构成完整 Transformer。

**注意力家族速览**（源笔记表格保留）：

| 类型 | 特性 |
|------|------|
| **加性（Bahdanau）** | 早期，Encoder-Decoder 对齐 |
| **点积/缩放点积（Luong/Transformer）** | 高效、主流 |
| **Self-Attention** | 序列内自交互 → Transformer 基础 |
| **多头注意力** | 多子空间，捕捉多种关系 |
| **交叉注意力** | Encoder 与 Decoder 之间（Encoder 的 K/V ≈ 记忆） |

## 应用

**典型使用场景**：神经机器翻译 / 文本摘要 / 问答（Encoder-Decoder 交叉注意力）；Transformer 预训练语言模型与 LLM（输入侧堆叠 Self-Attention，位置编码注入顺序信息，解码自回归）；长文本与长程依赖建模；推理阶段的 KV-Cache 加速。

**快速上手步骤**：

1. 先对输入做 token 化与嵌入（参见 [[文本预处理与Tokenizer]]、[[词嵌入与Word2Vec]]），得到序列表示。
2. 线性投影（或直接复用输入）得到 Q、K、V；做 Self-Attention 时 $Q=K=V$。
3. 计算分数 $\text{scores}=QK^\top/\sqrt{d_k}$（**务必除以 $\sqrt{d_k}$ 缩放**）。
4. 沿最后一维做 $\text{softmax}$ 得到权重分布（解码自回归时先对严格上三角施加因果掩码）。
5. 与 V 加权求和得到输出，形状与输入一致、但每个 token 已携带全序列上下文。

**常见坑与边界**（❌ 错误 → ✅ 正确做法）：

- ❌ 点积数值太大 → softmax 过饱和、梯度消失。✅ 除以 $\sqrt{d_k}$（缩放）。
- ❌ Self-Attention **不感知顺序**且未加位置编码 → token 顺序信息丢失（打乱词序结果不变）。✅ 加位置编码；自回归场景还要因果掩码。
- ❌ 长序列 $O(n^2)$ 注意力 → 算力/显存爆炸。✅ FlashAttention、窗口/稀疏注意力、推理期用 KV-Cache。
- ❌ 解码生成时把当前及未来 token 一并放进 Self-Attention → 泄漏未来信息。✅ 严格上三角 masking（未来位置置 $-\infty$，softmax 后权重为 0）。
- 边界提醒：注意力是"软对齐"，无显式对齐监督，"注意力图 = 可解释性"的结论需谨慎，要靠额外归因分析。

```python
import torch, torch.nn.functional as F

def scaled_dot_product_attention(q, k, v):
    # 缩放点积注意力：attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)  # 除以 sqrt(d_k) 防止 softmax 过饱和/梯度消失
    return F.softmax(scores, dim=-1) @ v                     # 按权重分布加权聚合 V

# —— 演示 1：Self-Attention（Q=K=V=输入自身，序列内两两交互）——
x = torch.randn(1, 8, 32)     # batch=1, 序列长=8, 特征维=32
out = scaled_dot_product_attention(x, x, x)
print(out.shape)              # (1,8,32)：形状不变，每个 token 拿到全序列上下文

# —— 演示 2：解码自回归必须加因果掩码，防止"偷看未来"——
def causal_scaled_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    seq_len = q.shape[-2]
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)  # 严格上三角=未来位置
    scores = scores.masked_fill(mask, float("-inf"))  # 未来位置权重置 -inf → softmax 后恰为 0
    return F.softmax(scores, dim=-1) @ v

# 案例详解：以机器翻译 "I love NLP" → "我爱自然语言处理" 为例，
# 每一步 Decoder 并非只看 Encoder 的末态向量，而是对 Encoder 全部隐状态
# 计算权重：译到"爱(love)"时，α 在 love 对应位置取到峰值，再按 α 加权求和
# 得到上下文 c_t 指导当前步生成——这就是"每一步动态选择该看哪个输入词"。

---
## 关联
- 前置：[[RNN与LSTM-note]]（Encoder-Decoder 骨架、隐状态 h_i 与压缩瓶颈的来源）、[[词嵌入与Word2Vec]]（token 向量表示的基础）
- 类似：[[Bahdanau注意力-note]]（区别是____该笔记单点深挖加性注意力及其训练细节，本文横向覆盖从 Bahdanau、Luong 到 Self-Attention 的完整谱系与选型）
- 进阶：[[Transformer结构拆解]]（Self-Attention/多头/交叉注意力如何组装成完整架构）、[[LLM推理与KV-Cache]]（注意力在 LLM 推理中的工程化：KV-Cache 与采样策略）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：Self-Attention（缩放点积） | Q=K=V 取自输入自身，softmax(QK^T/√d_k)V 一次前向建模序列内全部两两关系，可并行 | Transformer 编码/预训练、长程依赖建模、LLM 底座 |
| 替代方案：加性注意力（Bahdanau） | tanh 融合 decoder 状态与 encoder 隐状态后打分，逐步软对齐 | 早期 Encoder-Decoder 机器翻译等 RNN 模型，需显式逐词对齐 |
| 替代方案：点积注意力（Luong） | e = s^T W h，用点积直接打分，省去加性融合参数 | 需要比加性更简单高效的 RNN 解码对齐场景 |
| 替代方案：RNN/LSTM 无注意力基线 | 隐状态逐步压缩全部输入至末态向量 c 再解码 | 短句、低算力基线；无长程/对齐需求时不引入额外参数 |

**选型速查**：要并行 + 长程建模选本文（Self-Attention）；要在 RNN 解码器里做逐步对齐选 Bahdanau/Luong；仅做短序列基线可不加注意力。

---
## 参考
- [Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate (arXiv 1409.0473)](https://arxiv.org/abs/1409.0473)
- [Vaswani et al., Attention Is All You Need (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762)
- [Stanford CS224N 注意力讲义（课程主页）](https://web.stanford.edu/class/cs224n/)

---
## 具体案例
- [[注意力机制与Self-Attention 实战示例]](注意力机制与Self-Attention_sample.py)
