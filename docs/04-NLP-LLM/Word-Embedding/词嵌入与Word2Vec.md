---
title: "词嵌入与 Word2Vec / GloVe"
tags: [NLP, 词嵌入, Word2Vec, GloVe, Embedding]
date: 2026-08-29
---

# 词嵌入与 Word2Vec / GloVe

## 一、核心思想

One-hot 表示每个词是词表大小的稀疏向量，彼此正交、无法表达语义相似度。**词嵌入（Word Embedding）**把每个 token 映射为一个**低维稠密向量**，使得语义相近的词在向量空间中距离也更近，还能进行向量运算（如 `king − man + woman ≈ queen`）。

**分布假说**是这一切的根基：*语义相近的词，出现于相似的上下文中*。

## 二、Word2Vec

### 2.1 两种训练结构

| 结构 | 任务 | 思想 |
|------|------|------|
| **CBOW** | 用上下文词预测中心词 | 周围词 → 中间词 |
| **Skip-gram** | 用中心词预测上下文词 | 中间词 → 周围词 |

两者都在一个小滑动窗口内，用"预测搭配"来学习词向量。窗口越小语义越"局域"，越大越"主题化"。

### 2.2 原理与加速

本质是最小化预测误差，学到一个投影矩阵 W，其行即词向量。由于词表巨大，用：
- **负采样（Negative Sampling）**：不更新全部词表，只采样少量"负样本"做二分类，大幅加速。
- **层级 Softmax**：用 Huffman 树近似归一化。

```python
import numpy as np
from gensim.models import Word2Vec

sentences = [["king", "walks", "with", "his", "crown"],
             ["prince", "walks", "in", "the", "palace"]]
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1)

# 'king' 与 'prince' 距离较近（语义相近）
print(model.wv.most_similar("king", topn=2))
```

## 三、GloVe（Global Vectors）

### 3.1 与 Word2Vec 区别

- **Word2Vec**：局部窗口、在线逐样本学习。
- **GloVe**：先统计整语料的全局**共现矩阵**，再用矩阵分解/加权最小二乘拟合向量的点积来逼近共现次数。

GloVe 利用全局统计信息，向量质量在类比任务上往往更稳，且固定预计算、训练快。

### 3.2 例子

```python
# 假设已用预训练 GloVe 向量
king = np.array([0.5, 0.1])             # 示意
# vec("king") - vec("man") + vec("woman") → 逼近 vec("queen")，体现向量代数性质
```

## 四、从词向量到子词 / 上下文嵌入

| 方案 | 表示 | 能否处理未见词 | 是否上下文相关 |
|------|------|:---:|:---:|
| **Word2Vec/GloVe（静态）** | 每个词一个固定向量 | 否（用子词可缓解） | 否（一词义固定） |
| **FastText/子词嵌入** | 词拆成字符 n-gram 叠加 | ✅ | 否 |
| **ELMo/BERT（上下文嵌入）** | 每处出现都由模型动态生成 | ✅ | ✅ |

> LLM 时代主流是**上下文动态嵌入**（依赖 Transformer），静态 Word2Vec 常用于特征/simple 基线。当前所有预训练模型的 **Input Embedding** 即一张可学习的词表向量表，token 在训练中逐步学到语义。

## 五、边界与坑

- ❌ 静态嵌入**一词多义**（bank 银行/河岸）共用同一向量 → 语义混淆。✅ 用上下文嵌入（BERT 等）。
- ❌ 用随机初始化的 Embedding 表直接预测语义 → 未训练时无意义。✅ 需大规模预训练或加载预训练权重。
- ❌ 直接把不同框架训练的向量混用（维度/词表不同）。✅ 统一词表与维度，或重新对齐。
- ❌ 中文直接用 Word2Vec（需预分词）易出噪声。✅ 用子词/字级嵌入或中文字符 n-gram。
- 边界：作为特征要配合下游学好的网络；相似度通常用余弦距离而非欧氏。

## 六、关联

- 前置知识：分词/Tokenizer、one-hot。
- 同板块：[文本预处理与 Tokenizer](../Text-Preprocessing/文本预处理与Tokenizer.md)。
- 进阶：上下文化词嵌入 → BERT/Transformer 表示，详见 [注意力机制与 Self-Attention](../Seq2Seq-Attention/注意力机制与Self-Attention.md)。

## 七、参考

- Efficient Estimation of Word Representations (Word2Vec) — https://arxiv.org/abs/1301.3781
- GloVe: Global Vectors for Word Representation — https://nlp.stanford.edu/projects/glove/
