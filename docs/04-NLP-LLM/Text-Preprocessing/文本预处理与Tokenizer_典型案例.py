# -*- coding: utf-8 -*-
"""
文本预处理与 Tokenizer（BPE 分词）—— 典型代码演示
==================================================
覆盖知识点：
  1. 文本清洗流程：小写、去标点、停用词、N-gram
  2. 手工实现极简 BPE（理解算法本质）
  3. 使用 HuggingFace tokenizer（BERT / GPT-2 风格）实战
  4. padding / truncation 正确处理 batch

依赖：pip install transformers sentencepiece (可选项)
"""

import re

# =====================================================================
# 一、文本清洗：小写、分词、去标点、停用词、N-gram
# =====================================================================
raw_text = "Hello, World!  I'm learning Natural Language Processing. NLP is fun!"

# 1) 小写 + 去标点（简单方案）
def clean(text, remove_punct=True):
    text = text.lower()                         # 统一小写，减少词表冗余
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text)    # 非 单词/空格 的字符替换为空格
    return text

cleaned = clean(raw_text)
print("[清洗后] ", cleaned)

# 2) 空白分词
tokens = cleaned.split()
print("[空白分词]", tokens)

# 3) 去停用词（对主题/情感类任务有用）
STOPWORDS = {"i", "is", "the", "a", "an", "am", "to", "of", "and"}
filtered = [t for t in tokens if t not in STOPWORDS]
print("[去停用词]", filtered)

# 4) N-gram：连续 N 个词组成的片段
def ngrams(tokens, n=2):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

print("[二元 N-gram]", ngrams(tokens, 2))

# =====================================================================
# 二、手工实现极简 BPE（理解算法本质）
# =====================================================================
from collections import Counter

def get_pairs(word):
    """返回相邻字符对的列表。"""
    return [(word[i], word[i+1]) for i in range(len(word)-1)]

def manual_bpe(corpus, num_merges=8):
    """最简 BPE：
       1) 每个词拆成整词级别（这里以字符为例）
       2) 反复合并频率最高的相邻 pair，直到达到合并次数上限
    """
    # 词汇表：每词按"字符 空格分隔 + </w>"（词尾标记）拆分
    vocab = {}
    for w in corpus.split():
        # 用空格隔开字符并标记词尾，方便"优先合并词内/跨词"
        symbolize = " ".join(list(w)) + " </w>"
        vocab[symbolize] = vocab.get(symbolize, 0) + 1

    merges = []
    for _ in range(num_merges):
        # 统计所有相邻 pair 的频率
        pair_freq = Counter()
        for word, cnt in vocab.items():
            chars = word.split()
            for pair in get_pairs(chars):
                pair_freq[pair] += cnt
        if not pair_freq:
            break
        # 选出频率最高的一对
        best = max(pair_freq, key=pair_freq.get)
        merges.append(best)
        # 合并这对字符成新 token（用特殊连接符，避免歧义）
        new_token = "".join(best)
        new_vocab = {}
        for word, cnt in vocab.items():
            new_word = word.replace(best[0] + " " + best[1], new_token)
            new_vocab[new_word] = cnt
        vocab = new_vocab
    return merges


corpus = "low lower newest widest low low"
merges = manual_bpe(corpus, num_merges=5)
print("\n[手工BPE] 按频率依次合并出的子词:", merges)
print("[手工BPE] 含义：例如把 'low' 等常见词与其词尾组合保留为整体子词")

# =====================================================================
# 三、使用 HuggingFace AutoTokenizer（生产实践）
# =====================================================================
try:
    from transformers import AutoTokenizer
except ImportError:
    print("\n未安装 transformers，跳过 HuggingFace 部分。pip install transformers")
else:
    # 加载 BERT 的 tokenizer（WordPiece 风格，含 [CLS]/[SEP]/[UNK]/[PAD]）
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "Learning NLP is great"
    # 直接编码：自动加入 [CLS] 与 [SEP]
    enc = tok(text)
    ids = enc["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    print("\n[BERT tokenizer] ids:", ids)
    print("[BERT tokenizer] tokens:", toks)
    print("[注意] '##' 前缀表示词内片段：nl + ##p = 'nlp' 被拆开")

    # 处理一批句子 → 自动 padding + truncation 成等长矩阵
    sentences = ["I like dogs.", "机器学习和自然语言处理用于做智能应用。",
                 "short"]
    batch = tok(sentences, padding=True, truncation=True, max_length=12)
    print("\n[batch] 输入 shape:", list(batch["input_ids"].shape),
          "等长矩阵，方便模型一次性处理")
    print("[batch] 掩码(0=padding):", batch["attention_mask"].tolist())

    # GPT-2 风格（Byte-Level BPE，一个 token 对应一段字节，通用且无 OOV）
    tok_gpt = AutoTokenizer.from_pretrained("gpt2")
    print("\n[GPT-2 tokenizer] 特殊参数并不含 [CLS]，词表含字节 token")
    print("[GPT-2 tokenizer] 一段文本 ids:",
          tok_gpt("Hello, world!").input_ids)
