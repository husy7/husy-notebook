---
title: "BPE 与子词分词"
tags: [NLP, Tokenizer, 子词分词]
date: 2026-08-30
---

# BPE 与子词分词

## 定义

子词分词（subword tokenization）= 在"整词"与"字符"之间，用数据统计自动学出一批**最常用的中间粒度符号**：整词级别保留高频完整词（语义完整），低频/新词退化成子词甚至单字符（永不 OOV），词表大小固定且远小于词级词表。现代 LLM 的输入输出都建立在"token（子词）→ id"之上。

它要解决的问题是词级与字符级两种极端粒度的各自缺陷（见下表）：词级词表膨胀到几十万~上百万、遇到新词/拼写变体直接丢失（OOV）；字符/字节级虽几乎无 OOV，但序列极长（每字 1~3 token）、语义碎片化、难以学到形态信息。子词方案取中间态：词表适中（LLM 通常 32k~151k）、极少 OOV、序列长度中等、形态+语义兼顾。

子词还能天然共享形态学信息：`un`、`able`、`ing` 等片段在不同词里复用 → 参数共享，低资源片段也能获得有效 embedding。BPE（Byte-Pair Encoding，原义"合并字节对"）是学这套中间符号的**一种训练算法**；本文深入 BPE 的训练/编码算法本身、几种子词流派的关系，以及"merge 顺序即 id 顺序"这类只有自己训 tokenizer 才会踩的坑（与同目录《文本预处理与 Tokenizer.md》互补——那一篇讲流水线定位与 HuggingFace 工程用法）。

| 粒度 | 词表 | OOV | 序列长 | 语义 |
|------|------|-----|--------|------|
| 词级 | 几十万~百万，爆炸 | 新词/拼写变体全丢 | 短 | 完整 |
| 字符/字节级 | 极小 | 几乎无 | 很长（每字 1~3 token） | 碎片化，难以学到形态 |
| **子词** | 适中（LLM 通常 32k~151k） | 极少 | 中 | 形态+语义兼顾 |

## 原理

**BPE 训练流程**：原始文本 → **预分词（pre-tokenize）**成词（通常按空白+标点正则，如 GPT-2 用 `'\p{L}*|\p{N}+|...'`）→ 统计词频。对每个词，把"词频 × 字符序列（含词尾标记 `</w>` 或开头空格标记）"作为训练单元。然后循环：① 统计所有**相邻符号对**的加权频次（权重 = 词频）；② 选出频次最高的一对 `(a, b)`，合并成新符号 `ab`，写入 merges 表；③ 重扫一遍语料应用该合并，回到 ①；直到达到目标词表大小 / 合并次数上限。

**为什么合并按频次而非别的标准？** 直觉：频次最高的相邻对代表语料里最常见的拼写片段，把它固化成 token 能最大化"每个 token 的信息量"、压缩序列长度。WordPiece（BERT）则改为"合并后使语言模型似然**下降最少/上升最多**"的一对；Unigram 模型更是给每个子词一个概率，用 EM 迭代 + 正则化删除低分符号。BPE 是贪心频率，WordPiece/Unigram 是带目标函数的优化——结果形状相似，判定标准不同。

**复杂度坑**：朴素做法每轮全量重扫 O(合并次数 × 语料长度)；实用实现只更新"上轮合并发生处"的相邻 pair（用堆/优先队列维护），或像 SentencePiece 用 `unigram` 的 Viterbi 解码 + 后缀数组。

**编码与解码**：编码 = 把词按字符拆开，然后**按 merges 的学习顺序**依次尝试合并（某个 merge 只在当前符号串仍存在该相邻对时生效）；等价地，训练收敛后 token 集合已固定，可用"最长优先"贪心匹配词表切分（WordPiece 的做法）。解码 = 把 token 序列拼接，去掉词尾/空格标记还原文本。**任何编码出来的序列都能无损解码**——子词流是"词→子词"的可逆展开，这正是它优于词级分词（OOV 直接丢弃）的关键性质。

**字节级 BPE（Byte-Level BPE，GPT-2 起 LLM 标配）**：标准 BPE 的原子单元是"字符"，遇到语料里没见过的 Unicode 字符还是会 OOV。字节级 BPE 把文本先按 **UTF-8 编成字节序列**，再在字节上跑 BPE：任意 Unicode 文本（含 emoji、生僻字）都可无损编码 → **真正零 OOV**；代价是 token 数略多（一个汉字 = 3 字节，常见整字通常仍会被合并成一个 token）。GPT-2/4、Llama、Qwen 等现代词表都建立在字节级思想之上。

**SentencePiece：免预分词的训练框架**。BPE 原论文仍需要"先切词"（空格分词），对中文/日文等**没有空格边界**的语言很尴尬。SentencePiece 直接把**原始字符串**当输入训练，把"空格"本身也当作普通符号（默认在词首加伪空格 `▁`，即 `add_dummy_prefix`）→ 任何语言一视同仁；内部可选 BPE / Unigram / word / char 四种模型（T5 用 Unigram，Llama 用 BPE 模式）；默认做文本归一化（NFKC 等，可关）；提供 `byte_fallback`：未知字符降级为字节 → 无 `<unk>`。

> 注意：SentencePiece 的 `▁` 是**可打印的空格占位**，与 BERT WordPiece 的 `##` 前缀、GPT 的空格即普通 token，是三种不同的"空格处理哲学"，别混着看代码输出。

## 应用

**典型使用场景**：自己训练 tokenizer（新语言/垂直领域、控制压缩率与序列长度）；理解并调试现有 LLM 词表（为何 emoji 只占 1 个 token、`▁`/`##` 输出长什么样）；跨语言模型（中文/日文无空格语言）分词。HuggingFace 工程侧的调用与 WordPiece 细节见 [[文本预处理与Tokenizer]]，本文聚焦算法与自训。

**快速上手步骤**：
1. 收集大语料（几十 GB 级），固定随机种子；
2. 预分词：有空格语言用正则（GPT-2 风格 `'\p{L}*|\p{N}+|...'`），无空格语言直接用 SentencePiece（免预分词，空格也当一个符号）；
3. 设定目标词表大小（常见 32k~151k）；
4. 训练得到 merges/vocab，把 `vocab.txt` + `merges.txt` 与预训练权重**整套保存并版本化**；
5. 训练与推理必须加载同一套 vocab/merges（含同一套归一化规则）；
6. 用目标语言的测试集评估 PPL 来定词表大小。

**最容易踩的坑（超出"工程用法"层面的算法坑）**：
- ❌ merge 并列时打破平局的方式不同 → 同语料两次训练词表**可能 id 序不同**；预训练权重与词表/merges 文件必须整套保存复用，不能只存一个 txt 再重训。
- ❌ 训练与推理用不同 merge 表（比如手改 vocab）→ id 错位，输出乱码。
- ❌ 预训练后往词表**追加新 token**：embedding 表要 `resize_token_embeddings`，新 token 是随机初始化，需继续训练才有意义。
- ❌ 忽略 SentencePiece 归一化差异：训练/推理两端 normalizer 不一致 → 同一文本编码不同。
- ❌ 把"字符级演示的 merges"直接当生产词表：小语料学出的 merge 不稳定，换语料顺序结果都变；生产词表要用几十 GB 语料 + 固定随机种子 + 版本化。
- ❌ 中文任务若不做任何分词直接 Byte-level BPE，可能把常用双字词拆散；反过来 SentencePiece 又可能把长词合并得过细过碎——用目标语言的测试集评估 PPL 来定词表大小。

**与模型/显存的关系**：词表行索引 = Embedding 表：`vocab × hidden × 2` 字节（fp16）≈ 显存。Llama-3 词表 128k、hidden 4096 → 仅 embedding 就约 1GB（权重共享可省一半）。token 长度直接决定 KV-Cache 与算力成本（见 [[LLM推理与KV-Cache]]）：词表大、token 碎 → 序列长 → 慢且贵。

```python
# 纯 Python 迷你 BPE：加权统计相邻对 → 贪心合并 → 按 merges 顺序编码 → 无损解码
# 教学简化：真实实现用堆/优先队列只更新"上轮合并发生处"的相邻对，避免每轮全量重扫（见"原理"复杂度坑）

from collections import Counter

freqs = Counter({"low": 5, "lower": 3, "lowest": 2,   # 预分词后的词频语料：训练单元 = 词频 × 字符序列
                 "newer": 2, "newest": 1})

def to_symbols(w):                                    # 词 → 字符序列，末尾追加独立词尾标记 </w>
    return list(w) + ["</w>"]                         # </w> 保留词边界信息，解码时去除

splits = {w: to_symbols(w) for w in freqs}            # 每个词的当前切分状态（训练过程不断更新）
vocab = {c for w in splits for c in splits[w]}        # 初始原子符号表（26 字母 + </w>）

def pair_stats():
    """统计所有相邻符号对的加权频次：权重 = 词频，这是 BPE 与朴素字符统计的关键差异"""
    stats = Counter()
    for w, syms in splits.items():
        for (a, b), c in Counter(zip(syms, syms[1:])).items():
            stats[(a, b)] += c * freqs[w]             # 加权：高频词的片段在合并竞争中更有分量
    return stats

def apply_merge(a, b):
    """把训练语料中所有相邻对 (a,b) 合并为新符号 ab（一次合并 = 词表 +1）"""
    for w in list(splits):
        syms, i, out = splits[w], 0, []
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                out.append(a + b); i += 2             # 命中 → 粘成 ab，跳过两个位置
            else:
                out.append(syms[i]); i += 1
        splits[w] = out

merges = []                                           # merges 表：写入顺序 = 词表新 token 的 id 顺序
for _ in range(8):                                    # 目标合并次数（= 词表增量，真实 LLM 达几十万级）
    (a, b), _ = pair_stats().most_common(1)[0]        # 贪心：取加权频次最高的相邻对
    merges.append((a, b)); vocab.add(a + b)
    apply_merge(a, b)

def encode(w):                                        # 编码：把词按 merges 的学习顺序逐条尝试合并
    syms = to_symbols(w)
    for a, b in merges:                               # 顺序必须与训练完全一致，否则 id 错位（见"应用"坑）
        i, out = 0, []
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(syms[i]); i += 1
        syms = out
    return syms

def decode(tokens):                                   # 解码：拼接 token 并去掉词尾标记 → 无损还原
    return "".join(tokens).replace("</w>", "")

print("merges =", merges)
for w in freqs:
    ts = encode(w)
    print(f"{w:7s} -> {ts}  ->  decode: {decode(ts)!r}")

# 案例详解：
# 1) 语料中 low/lower/lowest 共享前缀，首轮 (l,o) 的加权频次最高（5+3+2=10）必然胜出 → 合并为 "lo"；
#    之后 "lo"+"w"、"w"+"</w>"、"(e,r)" 等按同样规则依次合并，输出中的 merges 顺序即 token 的 id 顺序。
# 2) 打印结果若与"直觉"略有出入，往往来自并列 pair 的打破平局顺序——这正是"坑 1"：
#    小语料 + 不同平局策略 → 词表/merges 可能不同，所以生产词表必须整套版本化保存。
# 3) 观察 decode：任何 token 序列都能无损还原原文（"词→子词"是可逆展开），
#    这是子词优于词级分词（OOV 直接丢弃）的关键性质；字节级 BPE 只是把原子从字符换成 UTF-8 字节。
# 4) </w> 在编码期保留词边界信息（避免把词尾字符与下一词首字符错误合并），解码期删除即可还原文本。
```

---
## 关联
- 前置：[[文本预处理与Tokenizer]]（先建立"子词分词在预处理流水线中的定位、WordPiece 的 `##`、词表量级、HF API"的工程认知，再读本页的算法细节更顺）
- 类似：[[词嵌入与Word2Vec]]（区别是：Word2Vec 解决"词/子词 → 稠密向量"的表示学习，本文解决"文本 → 子词 token 序列"的分词；且词表行索引正对应 Embedding 表，两者在显存与参数共享上直接耦合）
- 进阶：[[LLM推理与KV-Cache]]（token 序列长度直接决定 KV-Cache 与算力成本：词表大、token 碎 → 序列长 → 慢且贵）
- 进阶：[[RNN与LSTM-note]]（token 序列上的序列建模与梯度，子词分词的输出即其输入）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：BPE（含字节级变体） | 贪心合并加权频次最高的相邻符号对，merges 顺序即 token id 顺序 | GPT-2/4、Llama、Qwen 等通用 LLM 词表；字节级对任意 Unicode（emoji/生僻字）真正零 OOV |
| 替代方案：WordPiece | 每次合并"使语料语言模型似然下降最少/上升最多"的相邻对（带目标函数） | BERT 系预训练；`##` 词内前缀风格，词首词内结构清晰 |
| 替代方案：Unigram | 给每个子词一个概率，EM 迭代 + 正则化剪枝删除低分符号 | SentencePiece 默认算法（T5）；对词表大小/压缩率敏感、需按概率打分剪枝的场景 |
| 替代方案：词级 / 纯字符级 | 整词入表，或不做合并直接用字符/UTF-8 字节序列 | 词级适合词表可控的小语料垂直领域；字符级适合把"零 OOV + 无损可逆"放第一位、不追求压缩的基线 |

---
## 参考
- [Neural Machine Translation of Rare Words with Subword Units (BPE 原论文)](https://arxiv.org/abs/1508.07909)
- [SentencePiece (google/sentencepiece)](https://github.com/google/sentencepiece)

---
## 具体案例
- [[BPE 与子词分词 实战示例]](BPE与子词分词_sample.py)：配套代码，纯 Python 从零实现 BPE 训练/编码/解码
