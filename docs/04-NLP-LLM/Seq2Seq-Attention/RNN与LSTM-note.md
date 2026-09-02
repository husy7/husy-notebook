---
title: "RNN 与 LSTM"
tags: [深度学习, 序列建模, NLP]
date: 2026-08-30
---

# RNN 与 LSTM

## 定义

RNN（Recurrent Neural Network，循环神经网络）是一类用**同一套共享参数**按时间步递归处理序列的网络：`h_t = f(W·[x_t; h_{t-1}])`，把"变长历史"压缩进固定维度的隐状态 `h`，因此天然支持任意长度的序列输入。它解决的核心问题是——普通前馈网络无法泛化到变长序列（若给每个位置各配一套参数，训练时见过的长度之外就无法处理），CNN 只能看到局部窗口，而序列数据（文本、语音、时序）需要一种结构上能"记住过去"的机制。

核心特征有四：① **权重共享**，不管序列多长都用同一组 `W_ih, W_hh`；② 结构上是**有向图时间线**，`h_t` 蕴含 `x_1..x_t` 的全部信息；③ 展开（unrolled）后等价于深度 = 序列长度的前馈网，训练采用 BPTT（Backprop Through Time）；④ 时间维天然**串行**，无法像 CNN/Transformer 那样并行。朴素 RNN 的隐状态传递是**矩阵连乘**，梯度沿时间反向传播按谱半径指数放大/衰减，实际上只能记住约 5~10 步。

LSTM（Long Short-Term Memory，长短期记忆网络，Hochreiter & Schmidhuber 1997）属于 RNN 家族，通过**门控 + 线性单元状态**给误差反传开了一条"高速公路"，把可记忆长度推到数百步，代价是参数与计算量上升。适用范畴：语言模型（预测下一个 token）、Seq2Seq 编码器、时间序列预测、语音/流式/生成类需要自回归逐 token 消费并维护状态的场合。

## 原理

**① 递归结构与 BPTT。** RNN 每步执行 `h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b)`，同一组权重在所有时间步复用。训练时先正向展开并缓存每步中间量，再沿时间反向传播（BPTT），所以展开深度 = 序列长度，等价于训练一个非常深的共享权重网络。

**② 梯度灾难的根因。** 误差 `L` 对第 `k` 步输入的梯度，反传路径上每过一步要乘一次 `∂h_t/∂h_{t-1} = diag(f')·W_hh`（约等于循环矩阵）。经过 `T-k` 步连乘后：

- 若 `W_hh` 的主特征值 < 1：梯度随距离**指数衰减**（vanishing）→ 早期输入拿不到梯度，学不到长程依赖；
- 若 > 1：梯度**指数爆炸**（exploding）→ 训练震荡/NaN，需要梯度裁剪（clip）；
- 即便换成 ReLU 去掉 `tanh'` 的饱和区，循环权重的**连乘特性依旧**，问题只是被推迟。

> **坑**：别以为"换激活函数"或"加深"能根治消失问题——根因是**时间维度的连乘**，这正是门控结构（LSTM/GRU）要解决的。

**③ LSTM 的门控机制。** 每个时刻维护两个状态：**细胞状态 `c`（长程记忆载体）**与**隐状态 `h`（输出）**。四组候选更新全部由 `[x_t; h_{t-1}]` 经带偏置的线性层 + 激活生成：

```
i_t = σ(W_i·[x_t; h_{t-1}] + b_i)   输入门：新信息写多少
f_t = σ(W_f·[x_t; h_{t-1}] + b_f)   遗忘门：旧记忆留多少
g_t = tanh(W_g·[x_t; h_{t-1}] + b_g) 候选记忆
o_t = σ(W_o·[x_t; h_{t-1}] + b_o)   输出门：放出多少

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t    # 细胞状态更新：门控的"线性加性"组合
h_t = o_t ⊙ tanh(c_t)
```

**④ 关键洞察（为什么 LSTM 能记住几百步）。** `c_t` 的更新是**加性 + 门控乘法**而非非线性复合：误差经 `c` 通道反传时主要乘 `f_t`（可学习、初始化常设 ≈1），近似"误差原样流过"，远距离梯度不再指数消失。门都是 sigmoid（输出 0~1，可微的"开关"），候选 `g` 用 tanh 保持值域；若 `f=1, i=0`，LSTM 退化为"一直记住"，而何时写入/丢弃/放出完全由输入经门学出来——模型自己决定记忆策略。**GRU**（Cho 2014）把 LSTM 精简为两个门（更新门 `z` ≈ 遗忘+输入合并、重置门 `r`），少一组参数，效果常与 LSTM 相当。

## 应用

**典型场景**：下一 token 预测的语言模型、Seq2Seq 编码器、时间序列/语音建模；凡是需要自回归逐 token 生成并维护隐状态的场合（语音、流式、生成）RNN/LSTM 仍有价值。编码器侧标配**双向**（BiRNN）：正向+反向两个隐状态拼接，每个位置能看到"左右全文"。

**快速上手步骤**：① Tokenizer/词嵌入把输入变成向量序列；② 选 RNN/LSTM/GRU，把 `h0`（LSTM 还有 `c0`）置零并确保每批次重置；③ 变长序列先打包（见下）再正向展开；④ 输出接任务头，BPTT 反传前先梯度裁剪；⑤ 需要全局上下文就换双向结构；⑥ 解码/生成阶段按自回归逐 token 串行调用。

**注意事项 / 常见坑**：

- ❌ 忘设遗忘门偏置 → 默认 `b_f=0` → 初始遗忘门 ≈0.5，训练前期就在"忘"。✅ 初始化 `b_f≈1`（常用 +1~+3），让模型**先记住再学取舍**。
- ❌ 变长 batch 直接喂全 0 padding → pad 位置也参与循环与损失。✅ 用 `pack_padded_sequence` 打包 + `pad_packed_sequence` 解包，只对真实长度计算。
- ❌ 梯度爆炸时不断调学习率 → 先 `clip_grad_norm_(params, max_norm)`（RNN 训练标配）。
- ❌ 隐状态初始化全 0 起步却忘记在批次间清零 `h0` → 跨样本泄漏。
- ✅ 循环权重用正交初始化；输入/输出层用常规初始化。
- 计算图深度 = 序列长度 → 训练无法并行（时间维串行），这是 RNN 在长序列离线任务中被 Transformer 取代的主因；但自回归逐 token 的场合 LSTM 仍具优势。

```python
import torch
import torch.nn as nn

# 1) 朴素 RNN：同一组共享权重沿时间步递归，h_t 压缩 x_1..x_t 的信息
class NaiveRNN(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim + hid_dim, hid_dim)   # 等价 W·[x_t; h_{t-1}]
    def forward(self, xs):                               # xs: (seq_len, batch, in_dim)
        h = xs.new_zeros(xs.size(1), self.fc.out_features)  # 每段序列重置 h0，防跨样本泄漏
        for x in xs:                                     # 时间维只能串行：展开深度=序列长
            h = torch.tanh(self.fc(torch.cat([x, h], dim=-1)))
        return h

# 2) LSTM 单元：三条门(sigmoid) + 候选(tanh) + 线性细胞状态更新
class LSTMCell(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim + hid_dim, hid_dim * 4)   # 一次算 i/f/g/o 的预激活
        # ★ 坑1：遗忘门偏置默认 0 → 初始遗忘门≈0.5 训练前期就在"忘"；
        #   初始化 b_f≈1（常取 +1~+3），让模型先记住再学取舍
        nn.init.constant_(self.fc.bias[hid_dim: hid_dim * 2], 1.0)
    def forward(self, x, h, c):
        i, f, g, o = self.fc(torch.cat([x, h], dim=-1)).chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)                                  # 候选记忆
        c = f * c + i * g        # ★ 加性+门控乘法：误差沿 c 反传主要乘 f_t(≈1) → 不指数消失
        h = o * torch.tanh(c)                              # 输出门决定放出多少
        return h, c

# 3) 工程标配：变长打包 + 梯度裁剪（防梯度爆炸）
# emb = ...                                        # 先做词嵌入/Tokenizer
# packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
# out, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)  # 只算真实长度
# loss.backward()
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)     # ★ 坑2

# 案例详解：手写 RNN/LSTM 前向与 PyTorch 对齐，并用"梯度范数 vs 时间步距离"
# 实验对比朴素 RNN（指数衰减）与 LSTM（近似平坦），完整可运行代码见 RNN与LSTM_sample.py
```

---
## 关联
- 前置：[[词嵌入与Tokenizer]]（把 token 变成输入向量序列）；反向传播与梯度消失是深度学习的共性问题
- 类似：[[注意力机制与Self-Attention]]（区别是注意力任意位置直达、训练可并行，而 RNN 靠隐状态连乘传递历史、时间维串行，长程依赖需逐 token 传递梯度）
- 进阶：[[Bahdanau注意力-note]]（Seq2Seq 中注意力"按需取用"编码器隐状态序列，不再依赖最后一步向量压缩全句）；[[LLM推理与KV-Cache]]（Transformer 解码同样是逐 token 自回归串行，KV-Cache 加速）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| RNN/LSTM（本文） | 共享参数按时间步递归，隐状态连乘传递历史；LSTM 用门控 + 线性细胞状态缓解梯度消失 | 自回归/流式逐 token 场合（语音、流式、生成），需隐状态向量、在线推理 |
| GRU（替代·精简版） | 更新门 z 合并遗忘+输入、重置门 r 的两门结构，少一组参数 | 效果常与 LSTM 相当但更省更快，是更经济的默认选择 |
| Transformer/自注意力（替代） | 位置编码 + 注意力任意距离直达，训练可并行 | 大规模离线长序列语料预训练（GPT/BERT 类），解码仍逐 token 串行 |

---
## 参考
- [Long Short-Term Memory（Hochreiter & Schmidhuber, 1997 原论文）](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Understanding LSTM Networks（Colah's Blog，经典图解）](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch 官方文档：torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

---
## 具体案例
- [[手写 RNN/LSTM 前向与长程梯度实验]](RNN与LSTM_sample.py)（手写 RNN/LSTM 前向与 PyTorch 对齐 + 长程梯度随距离衰减的对照实验）
