# -*- coding: utf-8 -*-
"""
词嵌入与 Word2Vec / GloVe —— 典型代码演示
==========================================
覆盖知识点：
  1. One-hot 表示 vs 稠密词嵌入（Dense Embedding）的对比
  2. 用 gensim 训练 Word2Vec（CBOW / Skip-gram），找近义词
  3. 向量代数演示（king - man + woman ≈ queen）
  4. 在 PyTorch 中查看预训练模型的 Input Embedding 查表机制

依赖：pip install gensim numpy torch
"""

import numpy as np
import torch
import torch.nn as nn

# =====================================================================
# 一、One-hot 与 Dense Embedding 对比
# =====================================================================
vocab = ["king", "man", "woman", "queen", "crown", "apple"]
word2id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
D = 8                                   # 嵌入维度

# One-hot：每个词是 V 维、大部分为 0 的稀疏向量
one_hot = np.eye(V)
print("[One-hot] 'king' vector(6维):", one_hot[word2id["king"]].astype(int))
print("[One-hot] 维度大(V) 且两两正交 → 无法表达语义相似度")

# Dense Embedding：每个词是 D 维稠密向量（通常 D << V）
# 创建嵌入层，等价于一张可学习的词表(索引)查找表 (V, D)
emb_layer = nn.Embedding(V, D)
king_id, queen_id = word2id["king"], word2id["queen"]
vec_king = emb_layer(torch.tensor(king_id))     # 查表得到 king 的嵌入
vec_queen = emb_layer(torch.tensor(queen_id))
print(f"\n[Dense Embedding] king 嵌入维度 = {tuple(vec_king.shape)}（远小于 V={V}）")
print("[Dense Embedding] 可学习嵌入层权重 shape =", tuple(emb_layer.weight.shape))

# =====================================================================
# 二、基于 gensim 训练 Word2Vec，并演示语义
# =====================================================================
from gensim.models import Word2Vec

# 构造一个小语料（真正的应用中会用大规模 wikipedia 语料）
sentences = [
    ["king", "rules", "his", "kingdom"],
    ["queen", "rules", "her", "kingdom"],
    ["man", "works", "in", "the", "city"],
    ["woman", "works", "in", "the", "city"],
    ["prince", "is", "the", "son", "of", "king"],
    ["princess", "is", "the", "daughter", "of", "queen"],
    ["apple", "fruit", "tree", "grows"],
    ["banana", "is", "a", "fruit"],
]

# 训练 Skip-gram（sg=1），vector_size 设小便于演示
model = Word2Vec(sentences,
                 vector_size=16,   # 嵌入维度
                 window=3,         # 上下文窗口
                 sg=1,             # 1=Skip-gram, 0=CBOW
                 min_count=1,      # 频次≥1 的词进入词表
                 epochs=50,
                 seed=42)

# 找 near synonyms（语义相近的词）
print("\n[Word2Vec] 与 'king' 最相似的词:")
for w, sim in model.wv.most_similar("king", topn=4):
    print(f"   {w}: {sim:.3f}")

# 向量代数：king - man + woman ≈ queen（理想情形）
def analogy(a, b, c, topn=3):
    """求 vec(a) - vec(b) + vec(c) 最接近的词。"""
    result = model.wv.most_similar(positive=[a, c], negative=[b], topn=topn)
    return [(w, round(s, 3)) for w, s in result]

print("\n[向量代数] 'king' - 'man' + 'woman' 最接近:", analogy("king", "man", "woman"))

# 余弦相似度：衡量两个向量的语义距离
sim = model.wv.similarity("king", "queen")
sim2 = model.wv.similarity("king", "banana")
print(f"\n[余弦相似度] king~queen  = {sim:.3f}（语义近）")
print(f"[余弦相似度] king~banana = {sim2:.3f}（语义远）")

# =====================================================================
# 三、静态嵌入 vs 上下文嵌入的局限演示
# =====================================================================
# 静态嵌入一个词只对应一个向量，一词多义无法区分
# 例："bank"（银行 / 河岸）在 Word2Vec 里只有一个均值向量
bank_vec = model.wv.get_vector("bank") if "bank" in model.wv else None
print("\n[静态嵌入局限] 'bank' 只有唯一向量 → '银行'与'河岸'两个义项混在一起")
print("[对比] BERT/Transformer 上下文嵌入会按语境动态生成不同向量")

# =====================================================================
# 四、在 PyTorch 中：token → id → embedding 查找（LLM 的输入端）
# =====================================================================
# 模拟一个极小词表 (V=6, D=8)
vocab_tokens = ["<PAD>", "<CLS>", "cat", "sat", "on", "mat"]
tok2id = {t: i for i, t in enumerate(vocab_tokens)}
emb = nn.Embedding(len(vocab_tokens), 8)

# 一句话 "cat sat on mat" → ids → embedding 序列
sentence_tokens = ["cat", "sat", "on", "mat"]
ids = torch.tensor([tok2id[t] for t in sentence_tokens])       # (4,)
emb_seq = emb(ids)                                              # (4, 8)
print("\n[demo] token ids:", ids.tolist())
print("[demo] 每个 token 的嵌入 shape =", tuple(emb_seq.shape),
      "→ (sequence_len, emb_dim)")

# =====================================================================
# 小结
# =====================================================================
# 嵌入本质 = 一张可训练的 (词表×维度) 查找表；
# Word2Vec/GloVe 给静态向量；现代 LLM 用上下文相邻的动态向量（基于 Transformer）。
