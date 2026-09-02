---
title: "文本预处理与 Tokenizer：BPE 分词"
tags: [NLP, 文本预处理, Tokenizer, BPE]
date: 2026-08-29
---

# 文本预处理与 Tokenizer：BPE 分词

## 定义

NLP/LLM 管线的第一步，是把原始自然语言文本变成模型可计算的**数字序列**。这一步整体叫"文本预处理"，内部又分两个层次：第一层是**文本清洗**（分词、去停用词、归一化、去标点等规则化操作），第二层是 **Tokenizer（子词分词）**——把词拆成 **token**，再映射为整数 id（词汇表行索引），这是现代神经网络（尤其 LLM）的标准输入入口。**BPE（Byte-Pair Encoding，字节对编码）**是其中占统治地位的一种子词分词算法：它把"数据压缩里反复合并最频繁字节对"的思路迁移到分词上，通过反复合并语料中出现频率最高的相邻字符/字节对，从最小字符集出发逐步生长出一个大小固定、可配置的词表（现代 LLM 一般在 ~32k，如 LLaMA，到 ~100k+，如 GPT-4）。它解决的问题是词级分词的两个死穴：词表爆炸（需覆盖海量整词）与未登录词 OOV（新词无法处理）；同时避免字符级分词的序列过长、信息碎片化。核心特征：常见整词被保留为单个 token，罕见词/新词被拆成已知子词（如 "unhappiness" → "un"+"happiness"），词表适中（32k~256k）且极少 OOV。适用范畴：现代预训练语言模型与 LLM 的事实标准入口——GPT 系列用 Byte-Level BPE，BERT 家族用近亲 WordPiece，LLaMA/T5 用 SentencePiece（内部可选 BPE/Unigram）。

## 原理

BPE 的本质是**在训练语料上做有监督的统计合并**，训练流程如下：

1. **初始化词表**：把语料中的每个词拆成最小单元（字符；Byte-Level BPE 用字节），此时词表最小。
2. **统计相邻对频率**：遍历语料，统计所有相邻单元对（字符对/字节对）的出现次数。
3. **合并最高频对**：每次把**频率最高**的相邻对合并成一个新子词 token，加入词表。
4. **迭代**：重复步骤 2~3，直到词表达到目标大小，或没有可再合并的对为止。

**推理（编码新文本）阶段**：按"从最长词表 token 优先匹配"的贪心规则切分输入——新文本中能整体命中的整词直接用单个 token 表示，拼不出来的部分退回到已学到的子词组合。

**为什么它比"整词分词"好**：合并过程让高频整词在多轮迭代后被整体保留（如 "learning" 成为单 token），低频/新词则退化为已知子词拼接（"unhappiness" → "un"+"happiness"）；词表大小是超参、可固定，远小于词级词表，同时几乎消灭 OOV（Byte-Level BPE 以字节为最小单元，可无损编码任意 UTF-8 文本，理论零 OOV）。代价与边界：合并统计完全由训练语料决定，因此**训练与推理必须使用同一版本、同一词表文件的 tokenizer**，否则词表 id 错位、预测直接崩坏；另外 token 的 id 只是词表行索引，本身没有语义，语义要靠后续 Embedding 层的查表与训练习得。

**各变体的机制差异**：WordPiece（BERT）同为"反复合并相邻对"，但合并判据不是纯频率，而是"合并后能使语言模型似然提升最大"的对，并用 `##` 前缀标记词内片段（如 `nl`+`##p` 是同一个单词被切出的两个子词）；SentencePiece（LLaMA/T5）直接在原始文本（甚至字节）上训练，**不依赖空格预切分**，内部可选用 BPE 或 Unigram，天然适配无空格语言；Byte-Level BPE（GPT-2/4）以字节为最小单元做 BPE 合并，因此任意 UTF-8 都能编码、无 OOV。

## 应用

**典型使用场景**：①文本清洗——先做 lower、去标点（是否保留视任务而定，如情感分析常保留 `!`）、空白分词；②特征/局部语境捕获——用 N-gram（连续 N 个 token 的滑动窗口组合）构建特征或观察上下文；③子词 Tokenizer 实战——直接加载预训练 tokenizer，把文本批量转成 `input_ids`/`attention_mask` 送入模型。

**快速上手步骤（HuggingFace 为例）**：1) `AutoTokenizer.from_pretrained("bert-base-uncased")` 加载预训练词表 + 合并规则；2) `tok(text, padding=True, truncation=True)` 编码——`padding=True` 让 batch 内序列等长、`truncation=True` 截断超长样本、默认加 `[CLS]`/`[SEP]` 等 special token；3) 用 `convert_ids_to_tokens` 把 id 还原成 token 观察切分结果；4) `save_pretrained` 保存、推理端 `from_pretrained` 加载**同一个** tokenizer。

**常见坑（易错点）**：
- ❌ 中文直接用 `text.split()` → 中文词之间没有空格，按空白切分必然错误；✅ 改用 SentencePiece 或中文分词器（jieba/transformers 内置）。
- ❌ 训练与推理用**不同的 tokenizer 版本** → 词表 id 错位、预测崩坏；✅ 严格保存并复用同一个 tokenizer 文件。
- ❌ 忽略 `add_special_tokens`、padding/truncation → batch 内序列长度不一致或缺 `[CLS]`/`[PAD]`；✅ 统一 `padding=True, truncation=True`。
- ❌ 用词级分词处理 OOV → 新词整词丢失；✅ 子词分词把新词拆成已知子词消解 OOV。
- ❌ 一刀切的清洗策略 → 清洗高度依赖任务：标点保留 vs 去除要按任务定；停用词（the/is/的）对情感/主题类任务常可去除，但对语法建模/LLM 输入通常**应保留**（它们携带语法信息）。

```python
# ============ 第 1 步：文本清洗 ============
import re

text = "Hello, World!  I'm learning NLP."
text = text.lower()                                   # 统一小写
text = re.sub(r'[^\w\s]', ' ', text)                  # 去标点（视任务可保留，如情感中的 "!"）
tokens = text.split()                                 # 简单空白分词；⚠️ 中文无空格，不能这样切
print(tokens)   # ['hello', 'world', 'i', 'm', 'learning', 'nlp']

# ============ 第 2 步：N-gram（可选：捕获局部语境 / 构建特征） ============
def ngrams(tokens, n=2):
    """连续取 n 个 token 的滑动窗口组合"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

print(ngrams(["I", "love", "NLP"], 2))   # [('I','love'), ('love','NLP')]

# ============ 第 3 步：子词 Tokenizer 实战（以 BERT 的 WordPiece 为例） ============
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")   # 加载预训练词表 + 合并规则

text = "Learning NLP is great"
enc  = tok(text, padding=True, truncation=True)   # 自动补 [CLS]/[SEP]，batch 内等长
ids    = enc["input_ids"]
tokens = tok.convert_ids_to_tokens(ids)

print(ids)      # [101, 4083, 5650, 2003, 2363, 102]  （101/102 = [CLS]/[SEP]）
print(tokens)   # ['[CLS]', 'learning', 'nl', '##p', 'is', 'great', '[SEP]']
# 输出解读：
# - '##' 是 WordPiece（BERT）的"词内片段"标记：'nl' + '##p' 是 NLP 被切出的两个子词，
#   表示 '##p' 必须紧接在前一个 token 之后才能还原词义（注意严格说 ## 是 WordPiece 约定，
#   不是 BPE 本身的标记；BPE 系常用其他标记表示续词/空格）。
# - 'learning'、'great' 这类常见整词被整体保留为单个 token —— 这正是子词分词的平衡效果：
#   常用词完整、生僻词退化为子词，词表保持精简且几乎无 OOV。
# - 'i' 与 'm' 是 "I'm" 去标点后拆出的独立词，各自有 token（bert-base-uncased 词表不含 "'"）。

# ============ 第 4 步：训练/推理必须复用同一 tokenizer ============
tok.save_pretrained("./my_tok")                        # 训练侧保存（词表 + 合并规则）
tok2 = AutoTokenizer.from_pretrained("./my_tok")       # 推理侧加载同一份，保证 id 对齐
```

---
## 关联
- 前置：[[正则表达式]]（`re.sub` 去标点、字符串清洗）
- 前置：[[词嵌入与Word2Vec]]（token 数字化的下游：查词表行索引得到向量）
- 类似：[[WordPiece 分词]]（区别是__合并判据不同：WordPiece 选"合并后语言模型似然提升最大"的对，BPE 只按相邻对出现频率；WordPiece 用 `##` 标记词内片段，是 BERT 家族默认__）
- 类似：[[SentencePiece 分词]]（区别是__直接在原始文本/字节上训练、不依赖空格预切分，内部可选用 BPE 或 Unigram，天然适配无空格语言；BPE 经典实现通常要先按空格做词级预切分__）
- 类似：[[词级分词]]（区别是__整词为最小单元：词表巨大、OOV 高；而 BPE 用子词折中——词表适中（32k~256k）、OOV 极少__）
- 进阶：[[位置编码]]（token 数字化后的下一步：注入词序信息）
- 进阶：Token → Embedding 的映射（词表行索引查找，参 [[词嵌入与Word2Vec]]）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 词级分词 | 每个整词一个 token，需词表覆盖海量词；语义完整但 OOV 无法处理 | 受限封闭领域、词表可穷举 |
| 字符级分词 | 字符为最小单元；词表极小、无 OOV，但序列过长、信息碎片 | 极少单独使用，多作回退机制 |
| **子词 BPE（本文方案）** | 反复合并语料中最频繁的相邻字符/字节对，直到词表达到目标大小 | 通用 LLM 入口：需适中词表（32k~256k）+ 极低 OOV + 任意文本可编码 |
| WordPiece（BERT） | 合并"使语言模型似然增益最大"的相邻对，`##` 标记词内片段 | BERT 家族预训练与微调 |
| SentencePiece（LLaMA/T5） | 原始文本直接训练（字节无关），内置 BPE/Unigram 两种算法 | 多语言、无空格语言（中文/日文等） |
| Byte-Level BPE（GPT-2/4） | 以字节为最小单元做 BPE，可编码任意 UTF-8 | 通用、要求零 OOV 的场景 |

---
## 参考
- [Neural Machine Translation of Rare Words with Subword Units（BPE 原论文）](https://arxiv.org/abs/1508.07909)
- [HuggingFace Tokenizers 官方文档](https://huggingface.co/docs/tokenizers/)

---
## 具体案例
- [[BPE与子词分词 实战示例]](BPE与子词分词_sample.py)
