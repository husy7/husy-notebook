---
title: "词嵌入与 Word2Vec / GloVe"
tags: [NLP, 词嵌入, Word2Vec, GloVe, Embedding]
date: 2026-08-29
---

# 词嵌入与 Word2Vec / GloVe

## 定义

词嵌入（Word Embedding）是一类表示学习方法：把每个离散 token（词或子词）映射为一个**低维稠密实数向量**，用向量在连续空间中的几何位置来承载语义。它解决的是 one-hot 编码的根本缺陷——one-hot 向量维度等于词表大小、极其稀疏，且任意两个词都正交、内积恒为 0，完全无法表达语义相似度；而词嵌入把"相似"转化为"距离近"，并能支持向量代数运算（如 `king − man + woman ≈ queen`）。

词嵌入的根基是**分布假说（Distributional Hypothesis）**：*语义相近的词，出现于相似的上下文中*。因此所有词嵌入方法本质上都是"用上下文统计来刻画语义"。

核心特征有三点：①低维稠密、可学习；②语义相似 → 向量空间距离近（常用余弦距离度量）；③按是否随语境变化分为**静态嵌入**（一词一向量，Word2Vec / GloVe / FastText）与**上下文嵌入**（每处出现动态生成，ELMo / BERT）。

适用范畴：静态词向量适合做相似度检索、聚类、特征提取与简单基线；在 LLM 时代，所有预训练模型的 **Input Embedding** 本质上仍是一张可学习的词表向量表（token 在训练中逐步学到语义），但语义的主力已转向依赖 Transformer 的上下文动态嵌入。

## 原理

Word2Vec 有 CBOW 与 Skip-gram 两种训练结构，两者都在一个小滑动窗口内，用"预测搭配"来迫使网络把搭配信息压缩进向量：

| 结构 | 任务 | 思想 |
|------|------|------|
| **CBOW** | 用上下文词预测中心词 | 周围词 → 中间词 |
| **Skip-gram** | 用中心词预测上下文词 | 中间词 → 周围词 |

窗口大小控制语义粒度：窗口越小，学到的语义越"局域"（句法/搭配），越大越"主题化"。

训练本质：以预测误差为目标做最小化，学到一个投影矩阵 W，其**每一行就是一个词的向量**（W 即查表用的 Embedding 矩阵）。由于词表巨大、直接对全词表做 Softmax 归一化代价太高，采用两类加速：
- **负采样（Negative Sampling）**：不更新全部词表，每步只随机采样少量"负样本"做二分类，大幅加速训练；
- **层级 Softmax（Hierarchical Softmax）**：用 Huffman 树把归一化拆成一系列二分类，把 O(V) 降到 O(log V)。

GloVe（Global Vectors）走另一条路：先在整份语料上统计**全局共现矩阵** X（X_ij 表示词 j 出现在词 i 上下文的次数），再用加权最小二乘拟合词向量的点积逼近共现次数（近似最小化 `(log X_ij − w_i·w̃_j − b_i − b̃_j)²`，按共现次数加权）。由于吃进全局统计而非局部窗口，GloVe 在类比任务上往往更稳，且可固定预计算、训练快。

演进路径：Word2Vec/GloVe（静态、整词）→ **FastText 子词嵌入**（把词拆成字符 n-gram 向量叠加，能处理未见词）→ **ELMo/BERT 上下文嵌入**（每处出现都由深层模型根据语境动态生成向量，解决一词多义）。

## 应用

典型使用场景：语义相似度检索与类比推理（`most_similar`、`king−man+woman`）、词聚类、把词向量作为特征喂给下游网络（分类/情感/序列标注）、需要快速出效果时的静态基线、以及中文/形态丰富语言的字级或子词表征。

快速上手步骤：①对语料做清洗与（必要时）分词，得到句子级 token 序列；②用 gensim 的 `Word2Vec` 在自建语料上训练，或直接加载预训练 GloVe / word2vec 权重；③通过 `model.wv` 查询近义词、做向量运算或取 `model.wv[token]` 作为下游输入；④相似度度量用余弦距离而非欧氏距离（向量已归一化后等价于内积）；⑤作为特征时须配合下游训练好的网络使用，单独"喂未训练的 Embedding 表"没有语义。

注意事项 / 常见坑：
- ❌ 静态嵌入**一词多义**（bank 银行/河岸）共用同一向量 → 语义混淆。✅ 改用上下文嵌入（BERT 等）。
- ❌ 用随机初始化的 Embedding 表直接预测语义 → 未训练时向量无意义。✅ 大规模预训练或加载预训练权重后再用。
- ❌ 直接把不同框架训练的向量混用（维度/词表不一致）。✅ 统一词表与维度，或重新对齐。
- ❌ 中文直接用 Word2Vec（需预分词）易出噪声。✅ 用子词/字级嵌入或中文字符 n-gram。
- 边界：词向量作为特征要配合下游学好的网络；相似度通常用余弦距离而非欧氏。

```python
import numpy as np
from gensim.models import Word2Vec

# ===== 案例详解 =====
# 1) 语料必须是"分词后的句子列表"：每句是一个词 token 组成的 list。
sentences = [["king", "walks", "with", "his", "crown"],
             ["prince", "walks", "in", "the", "palace"]]

# 2) 训练静态词向量（Skip-gram/CBOW 由 sg 参数决定，默认 sg=0 即 CBOW）。
#    vector_size=10 → 每个词映射为 10 维稠密向量（远小于 one-hot 的词表维度）；
#    window=2       → 滑动窗口半径，控制语义粒度（越小越局域，越大越主题化）；
#    min_count=1    → 词频低于该值的词丢弃（本例语料极小故设 1）。
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1)

# 3) 查询近义词：'king' 与 'prince' 都常与 walks/crown/palace 同现，
#    按分布假说语义相近 → 训练后向量距离较近，会被排在 topn 前列。
print(model.wv.most_similar("king", topn=2))

# ===== GloVe 向量代数示例 =====
# 假设已加载预训练 GloVe 向量（每词一个固定向量，示意值）：
king   = np.array([0.5, 0.1])   # vec("king")
man    = np.array([0.4, 0.2])   # vec("man")
woman  = np.array([0.4, 0.0])   # vec("woman")
# vec("king") - vec("man") + vec("woman") → 应逼近 vec("queen")，
# 体现静态词向量空间的"语义方向"性质（性别方向可平移复用）。
queen_approx = king - man + woman
print(queen_approx)
```

---
## 关联
- 前置：[[文本预处理与 Tokenizer]]（分词与 one-hot 稀疏表示是词嵌入的输入前提）
- 类似：[[Word2Vec-CBOW与Skipgram-note]]（区别是：该篇聚焦 CBOW 与 Skip-gram 两种结构的内部原理与推导细节，本篇站在 Word2Vec/GloVe 等静态嵌入整体做定义、选型与边界梳理）
- 进阶：[[注意力机制与 Self-Attention]]（从静态词向量走向 BERT/Transformer 的上下文动态嵌入）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：Word2Vec（CBOW/Skip-gram） | 局部窗口内做"预测搭配"任务，投影矩阵 W 的行即词向量；负采样/层级 Softmax 加速 | 中大规模语料快速训练静态向量、相似度检索、下游特征基线 |
| 替代方案：GloVe | 先统计全语料全局共现矩阵，再用加权最小二乘让向量点积逼近共现次数 | 全局统计充分、类比任务要求稳、需离线固定预训练向量 |
| 替代方案：FastText / 子词嵌入 | 词 = 字符 n-gram 向量叠加（含整词） | 罕见词/未见词多、形态学丰富的语言、中文免精细分词 |
| 替代方案：ELMo / BERT 上下文嵌入 | 每处出现的向量由深层模型按语境动态生成 | 一词多义、需要语境化语义、LLM/Transformer 下游微调 |

---
## 参考
- [Efficient Estimation of Word Representations in Vector Space（Word2Vec 原论文）](https://arxiv.org/abs/1301.3781)
- [GloVe: Global Vectors for Word Representation（Stanford 官方项目页）](https://nlp.stanford.edu/projects/glove/)

---
## 具体案例
- [[词嵌入与 Word2Vec 实战示例]](词嵌入与Word2Vec_sample.py)
