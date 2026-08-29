---
title: "文本预处理与 Tokenizer：BPE 分词"
tags: [NLP, 文本预处理, Tokenizer, BPE]
date: 2026-08-29
---

# 文本预处理与 Tokenizer：BPE 分词

## 一、核心思想

NLP 的第一步是把原始文本变成模型可计算的**数字序列**。预处理分两个层次：

1. **文本清洗**：分词、去停用词、归一化、去标点。
2. **Tokenizer（子词分词）**：把词拆成 **token**，再映射为整数 id（词汇表索引）——这是现代神经网络（尤其 LLM）的标准入口。

现代 LLM 几乎统一使用**子词（subword）分词**（如 BPE、WordPiece、SentencePiece），兼顾词表的精简与未登录词（OOV）的覆盖。

## 二、文本清洗（经典流程）

```python
import re

text = "Hello, World!  I'm learning NLP."
text = text.lower()                                   # 小写
text = re.sub(r'[^\w\s]', ' ', text)                  # 去标点（视任务可保留）
tokens = text.split()                                 # 简单空白分词
```

- **停用词（stopwords）**：如 the/is/的 —— 对情感/主题等任务往往可去除，但对语法/LLM 输入通常保留。
- **N-gram**：连续 N 个 token 的组合（字/词/字符），用于捕获局部语境或构建特征。

```python
def ngrams(tokens, n=2):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
print(ngrams(["I", "love", "NLP"], 2))   # [('I','love'), ('love','NLP')]
```

## 三、BPE 子词分词（Byte-Pair Encoding）

### 3.1 原理：不断合并最频繁的字节对

1. 把所有词拆成单个字符（初始词表最小）。
2. 统计相邻字符对的出现频率，**每次合并频率最高的一对**，成为一个新 token。
3. 重复直到词表达到目标大小或无可合并。
4. 训练完成后，新文本按"从最长词表 token 优先"的方式切分。

**效果**：
- 常见整词保留（如 "learning"）。
- 罕见词/新词被拆成已知子词（如 "unhappiness" → "un"+"happiness"）。
- 词表大小固定且远比词级别小，同时极少 OOV。

### 3.2 为什么 BPE 比"整词分词"好

| 方式 | 词表 | OOV | 语义 |
|------|------|-----|------|
| 词级分词 | 大，需覆盖海量词 | 有，新词无法处理 | 语义完整 |
| 字符级 | 极小 | 无 | 序列过长、信息碎片 |
| **子词(BPE)** | 适中（32k~256k） | 极少 | 兼顾 ^ |

> 现代 LLM 词表一般在 ~32k（LLaMA）到 ~100k+（GPT-4）个 token。

## 四、用 HuggingFace Tokenizer 实战

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")   # WordPiece，BERT

text = "Learning NLP is great"
ids  = tok(text)["input_ids"]
tokens = tok.convert_ids_to_tokens(ids)
print(ids)      # [101, 4083, 5650, 2003, 2363, 102]  （含 [CLS]/[SEP]）
print(tokens)   # ['[CLS]','learning','nl','##p','is','great','[SEP]']
```

> 观察 `##p`：BPE 用 `##` 前缀表示"接在前面的 token 后"，表示它是词内片段。

## 五、常见编码格式

| 方案 | 特点 | 使用 |
|------|------|------|
| **Byte-Level BPE**（GPT-2/4） | 以字节为最小单元，可编码任意 UTF-8 | 通用、无 OOV |
| **WordPiece**（BERT） | 不断二分合并，类似 BPE | BERT 家族 |
| **SentencePiece**（LLaMA/T5） | 原始文本直接训练，已内置 BPE/Unigram | 多语言、字节无关 |

## 六、边界与坑

- ❌ 直接 `text.split()` 处理中文 → 词边界不存在的语言（中文无空格）切分错误。✅ 用 SentencePiece 或中文分词器（如 jieba/transformers 内置）。
- ❌ 训练与推理用**不同的 tokenizer** 版本 → 词表 id 错位，预测崩坏。✅ 严格保存并复用同一个 tokenizer。
- ❌ 忽略 `add_special_tokens`、padding/truncation → batch 长度不一致或缺 [CLS]/[PAD]。✅ 用 `padding=True, truncation=True`。
- ❌ 用词级分词处理 OOV → 新词丢失。✅ 用子词分词消解 OOV。
- 边界：清洗策略高度依赖任务（保留标点 vs 去除）；停用词在上下文建模类任务里通常不该删。

## 七、关联

- 前置知识：正则、字符串处理。
- 同板块：[词嵌入](./../Word-Embedding/词嵌入与Word2Vec.md)。
- 进阶：Token 到 Embedding 的映射（词表行索引查找）、位置编码。

## 八、参考

- 论文 Neural Machine Translation of Rare Words with Subword Units (BPE) — https://arxiv.org/abs/1508.07909
- HuggingFace 预训练 Tokenizer 官方文档 — https://huggingface.co/docs/tokenizers/
