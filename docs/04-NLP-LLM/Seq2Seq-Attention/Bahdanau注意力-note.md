---
title: "Bahdanau 注意力：Seq2Seq 的加性对齐机制"
tags: [注意力机制, Seq2Seq, NLP]
date: 2026-08-30
---

# Bahdanau 注意力：Seq2Seq 的加性对齐机制

## 定义

Bahdanau 注意力（Bahdanau Attention，又叫**加性注意力 / MLP 注意力**）由 Bahdanau、Cho 与 Bengio 于 2015 年提出（论文 *Neural Machine Translation by Jointly Learning to Align and Translate*），是第一个被引入 Seq2Seq 模型的注意力机制。

它解决什么问题：经典 Seq2Seq（无注意力）的 Encoder 把整句读成隐状态序列 `h_1..h_n`，Decoder 只能拿到**最后一个**隐状态 `c = h_n` 作为全句摘要开始解码，存在两个致命缺点——① **信息瓶颈**：无论句子多长都压进一个定长向量，长句开头信息在解码中段已被"冲淡"；② 解码器无法"回看"源句的任意位置，长句翻译质量骤降。

核心特征：Bahdanau 注意力让 Decoder 在**每一步**都对 Encoder 的全部隐状态算一个加权平均，作为"当前要看的内容"（上下文向量 `c_t`），等价于每步动态做一次**软对齐（soft alignment）**。其打分器本身是一个可学习的单隐层小网络（因此叫 MLP 注意力），表达能力比"点积"强——点积只是它的一种特例形式。

适用范畴：一切需要 Decoder"逐位置软检索源序列"的 Encoder-Decoder 任务，典型是机器翻译。它与同目录的注意力家族总览笔记互补：总览篇覆盖注意力家族与 Self-Attention 的演进，本篇聚焦 Bahdanau 在 Seq2Seq 里的**完整工作机理**——它为什么出现、每个量是什么形状、为何用 `s_{t-1}` 而非 `s_t`、以及与 Transformer 交叉注意力的关系。

## 原理

设 Encoder 输出隐状态序列 `h_1..h_n`（双向时拼接正反向，维度记为 `enc`），Decoder 第 `t` 步的完整流程分四步：

1. **对齐分数（score/energy）**：用**上一步**的 decoder 隐状态 `s_{t-1}` 去和每个源隐状态 `h_i` 打分：
   `e_{t,i} = vᵀ · tanh(W₁ h_i + W₂ s_{t-1} + b)`
   这是**加性（additive）注意力**：两个不同空间的向量先各自线性映射到公共空间，再相加、过 tanh，最后用向量 `v` 投影成标量——打分器即一个可学习的单隐层小网络（MLP 注意力），比点积打分的表达能力更强。

2. **归一化成权重**：`α_{t,i} = softmax(e_{t,i}) = exp(e_{t,i}) / Σ_j exp(e_{t,j})`。`α_t` 是一个分布在 n 个源位置上的概率向量，即"这一步应该看源句哪里"。

3. **上下文向量**：`c_t = Σ_i α_{t,i} · h_i` —— 源隐状态的加权和（软检索结果）。

4. 把 `c_t` 拼进 Decoder 输入参与生成第 t 个词：`s_t = f([y_{t-1}; c_t], s_{t-1})`，输出层也常再结合 `[s_t; c_t]` 预测 `y_t`。

**训练细节**：训练时 Decoder 用**上一步真值 `y_{t-1}`**（teacher forcing）；损失 = 每步预测对真值下一词的交叉熵之和。

**为什么对齐分数用 `s_{t-1}` 而不是 `s_t`？** 计算 `c_t` 是为了生成 `s_t`/`y_t`，而 `s_t` 本身又依赖 `c_t`——形成鸡生蛋循环。Bahdanau 选择用"上一步状态"打分来打破循环（**先对齐、再更新**）；Luong 注意力则改用当前步 `s_t`（需先算一次不带上下文的过渡状态，或用 input feeding 修正）。实现上很多教程简化为"用 s_t"，也能训练，但要清楚原版语义是 `s_{t-1}`。

**对齐矩阵的可解释性**：把所有步的 `α` 拼成矩阵（行 = 目标位置 t，列 = 源位置 i），机器翻译中它近似**单调的对角线**——模型自己学会了"源词顺序 ↔ 目标词顺序"的对应关系，无需任何对齐标注。这是注意力机制最早被当作"可解释性窗口"的原因；但它只是训练副产品，不是受监督的对齐证据。

**与 Transformer 交叉注意力的关系**：交叉注意力（cross-attention）与 Bahdanau 同构——Q 来自 Decoder 当前层，K/V 来自 Encoder 输出；区别只是加性打分被缩放点积 `QKᵀ/√d` 取代、`h_i` 换成多头 K/V，而"decoder 每步对 encoder 序列做软检索"的骨架完全一致。因此理解了 Bahdanau，交叉注意力只剩投影与缩放是新的。

## 应用

**典型使用场景**：机器翻译（论文原任务是英法/英德 NMT）；任何需要 Decoder 逐步"回看"源序列任意位置的 Seq2Seq 任务；以及把对齐权重 `α` 画成热力图，观察模型自己学到的"源词 ↔ 目标词"词序对应关系。

**快速上手步骤**：
1. Encoder 用双向 RNN/LSTM，把正反向隐状态拼接成 `h_1..h_n`（注意拼接后维度翻倍）。
2. 实现打分器：`W₁`、`W₂` 分别把 `h_i` 与 `s_{t-1}` 映射到公共空间，相加后过 `tanh`，再用向量 `v` 投影成标量 `e_t`。
3. 对 `e_t` 做 softmax 得到 `α_t`，再对源隐状态加权求和得上下文向量 `c_t`。
4. 把 `c_t` 与上一步词嵌入拼接后喂给 Decoder（训练用 teacher forcing；输出层结合 `[s_t; c_t]` 预测词）。
5. 训练完成后把所有步的 `α` 拼成对齐矩阵并可视化，检查是否近似单调对角线。

**注意事项 / 常见坑**：
- ❌ 计算代价：每步对**全部** n 个源位置打分 → 时间 O(T·n)，比纯 RNN 贵；源句超长时可加"只看一个窗口"或截断。
- ❌ softmax 覆盖全序列：很长的源句权重被摊薄 → 长句仍会劣化（现代模型用分块/滑动注意力缓解，是另一个话题）。
- ❌ exposure bias：训练用 teacher forcing、推理用自己的预测 → 训练/推理分布不一致；缓解：scheduled sampling，或直接用整体优化目标。
- ❌ 双向 Encoder 的 `h_i` 是正反向拼接（维度翻倍），打分矩阵尺寸别写错。
- ❌ 别把 `α` 当受监督的对齐证据使用——它只是隐式学出的软对应，可随随机种子漂移。

代码示例（PyTorch 手写 Bahdanau 加性注意力的核心四步；完整的可训练 Seq2Seq 与对齐矩阵可视化见文末 sample.py）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Bahdanau（加性）注意力：Decoder 每步对 Encoder 全部隐状态做软对齐。

    对应公式（原版含偏置 b，已并入各 Linear 内部）：
      e_{t,i} = vᵀ · tanh(W1 h_i + W2 s_{t-1} + b)   # 1) 对齐分数
      α_{t,i} = softmax(e_{t,i})                     # 2) 归一化成权重
      c_t     = Σ_i α_{t,i} · h_i                    # 3) 上下文向量
    """

    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        # 把源隐状态 h_i（双向拼接后 enc_dim 可能已翻倍）映射到公共空间
        self.W1 = nn.Linear(enc_dim, dec_dim)
        # 把"上一步"decoder 状态 s_{t-1} 也映射到同一个公共空间
        self.W2 = nn.Linear(dec_dim, dec_dim)
        # 把 tanh 输出投影成标量分数（vᵀ 的角色）
        self.v = nn.Linear(dec_dim, 1)

    def forward(self, s_prev, h_src):
        # s_prev: (B, dec_dim)             上一步 decoder 隐状态 s_{t-1}
        # h_src:  (B, T_src, enc_dim)      Encoder 全部隐状态 h_1..h_n
        # 1) 打分：W2(s_prev) 扩一维后广播到每个源位置再相加
        score = self.v(torch.tanh(self.W1(h_src) + self.W2(s_prev).unsqueeze(1)))
        #    score: (B, T_src, 1)
        # 2) 归一化成权重 α_t：在 n 个源位置上的概率分布
        alpha = F.softmax(score.squeeze(-1), dim=-1)   # (B, T_src)
        # 3) 上下文向量 c_t = Σ_i α_{t,i} h_i（软检索"这一步要看源句哪里"）
        c_t = torch.bmm(alpha.unsqueeze(1), h_src).squeeze(1)  # (B, enc_dim)
        # 4) 外部再把 c_t 拼进 decoder 输入：s_t = f([y_{t-1}; c_t], s_{t-1})
        return c_t, alpha
```

> 案例详解：上述模块即"打分 → softmax → 加权求和"三步，`W1/W2` 的加法广播是实现加性对齐的关键。完整案例 `Bahdanau注意力_sample.py` 用该模块搭一个带注意力的 Seq2Seq，在"反转数字序列"任务上训练（损失为逐位置交叉熵，teacher forcing 喂真值），并打印行 = 目标位置、列 = 源位置的对齐矩阵，观察其近似单调对角线的学习效果。

---
## 关联
- 前置：[[RNN与LSTM-note]]（Seq2Seq 的底层结构：RNN/LSTM 与梯度问题，`h_i`/`s_t` 的来源）
- 类似：[[注意力机制与Self-Attention]]（区别是它是注意力家族总览，含加性注意力三步公式与 Self-Attention 的演进；本篇只深挖 Bahdanau 在 Seq2Seq 里的完整工作机理）
- 进阶：[[Transformer结构拆解]]（交叉注意力 = Bahdanau 的现代形态：加性打分换成缩放点积与多头，软检索骨架不变）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：Bahdanau（加性/MLP 注意力） | `e = vᵀ tanh(W₁h + W₂s + b)`：不同空间向量各自线性映射到公共空间后相加过 tanh，再投影成标量；打分器是单隐层小网络，表达力最强，位于 Enc-Dec 之间 | 需要最强打分表达力、可接受 O(T·n) 开销的 Enc-Dec 任务（如早期 NMT 与对齐可视化） |
| 替代方案：Luong（乘性/点积注意力） | `sᵀW h` 或 `sᵀh` 打分：计算更快，位于 Enc-Dec 之间，常配合 input feeding 修正 | 对计算速度敏感、可引入 input feeding 的 Enc-Dec 场景 |
| 替代方案：缩放点积注意力（Transformer 交叉注意力） | `QKᵀ/√d`：Decoder 层做 Q、Encoder 输出做 K/V，可多头并行；主流实现，见注意力总览笔记 | 大规模并行训练、长序列建模与自回归解码的主流方案 |

---
## 参考
- [Neural Machine Translation by Jointly Learning to Align and Translate（Bahdanau et al., 2015，原文）](https://arxiv.org/abs/1409.0473)

---
## 具体案例
- [[Bahdanau 注意力 实战示例]](Bahdanau注意力_sample.py)：PyTorch 手写带加性注意力的 Seq2Seq，训练反转数字序列任务并打印对齐矩阵。
