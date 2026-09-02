"""BPE（Byte-Pair Encoding）从零实现：训练 + 编码 + 解码（纯 Python，可直接运行）

演示经典例：low / lower / lowest / newest 一族词的 merge 过程。
本实现刻意写得直白（每轮全量重扫），教学优先；生产环境请用
sentencepiece / tokenizers 库（内部带优先队列与并发优化）。

配套笔记：BPE与子词分词-note.md
"""
from collections import Counter
from typing import Dict, List, Tuple

# 词尾标记：标记"符号位于词末尾"，解码时据此还原空格
_END = "</w>"


class BPE:
    """极简 BPE：merges 按学习顺序保存，编码时按同样顺序回放合并。"""

    def __init__(self, num_merges: int = 20):
        self.num_merges = num_merges          # 要学习的 merge 次数（词表大小 ≈ 初始字符数 + num_merges）
        self.merges: List[Tuple[str, str]] = []   # 依次学到的 (a, b) -> ab
        self.vocab: List[str] = []                 # 初始字符 + 所有合并产物（index 即 token id）
        self.splits: Dict[str, List[str]] = {}     # 每个词当前被切成的符号序列

    # ---------- 训练 ----------
    def train(self, corpus: List[str]) -> None:
        """corpus: 句子列表。内部做最简单的空白预分词 + 词频统计。"""
        # 1) 预分词：按空白切词，统计词频（真实场景会带正则/字节级预切分）
        word_freqs = Counter()
        for sent in corpus:
            for w in sent.split():
                word_freqs[w] += 1

        # 2) 初始符号：词切成单字符并保留词尾标记
        #    splits[w] = ['l','o','w','</w>'] 之类
        #    多字符词把最后一个字符带上词尾标记；单字符词直接 字符+标记
        for w in word_freqs:
            chars = list(w)
            self.splits[w] = chars[:-1] + [chars[-1] + _END] if len(chars) > 1 else [chars[0] + _END]

        # 3) 迭代合并：每轮找"频次最高的相邻符号对"
        for _ in range(self.num_merges):
            pair_freqs: Counter = Counter()
            for w, freq in word_freqs.items():
                syms = self.splits[w]
                for i in range(len(syms) - 1):
                    pair_freqs[(syms[i], syms[i + 1])] += freq   # 权重=词频
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)      # 贪心选最高频对
            self.merges.append(best_pair)
            # 4) 应用到所有词：把 (a,b) 就地合并成 ab
            a, b = best_pair
            merged = a + b
            for w in self.splits:
                syms = self.splits[w]
                new_syms, i = [], 0
                while i < len(syms):
                    if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                self.splits[w] = new_syms

        # 5) 汇总词表：所有"字符级原子符号"(含词尾标记变体) + merges 产物
        #    真实 BPE 的原子集合在训练前就固定(字符/字节)，未出现的字符合法文本
        #    也会被拆成原子符号 → 原子符号必须全在词表里
        chars = set()
        for w in word_freqs:
            chars.update(w)                       # 语料中出现过的所有字符
        symbols = sorted(chars) + sorted(c + _END for c in chars)   # 原子符号
        symbols += [a + b for a, b in self.merges]                  # 合并产物
        self.vocab, seen = [], set()
        for t in symbols:
            if t not in seen:
                seen.add(t)
                self.vocab.append(t)

    def _ensure_id(self, sym: str) -> int:
        """symbol -> id；遇到词表外符号(训练语料从未出现的字符)时兜底追加。
        真实系统不能这样动态扩词表(id 稳定性/embedding 表都要求固定词表，
        生产用字节级 BPE 保证原子集合封闭)。这里仅为保证演示可跑。"""
        if sym not in self.vocab:
            self.vocab.append(sym)
        return self.vocab.index(sym)

    # ---------- 编码 ----------
    def encode(self, text: str) -> List[int]:
        """把整句编码为 token id 列表：按空格切词，词内按 merges 顺序回放合并。"""
        ids: List[int] = []
        for word in text.split():
            # 词初始切成字符（最后一个字符带词尾标记）
            chars = list(word)
            syms = chars[:-1] + [chars[-1] + _END] if len(chars) > 1 else [chars[0] + _END]
            # 按学习顺序逐条 merge：当前串里仍存在的 pair 就合并
            for a, b in self.merges:
                merged = a + b
                out, i = [], 0
                while i < len(syms):
                    if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                syms = out
            ids.extend(self._ensure_id(t) for t in syms)
        return ids

    # ---------- 解码 ----------
    def decode(self, ids: List[int]) -> str:
        """把 token id 序列还原成文本：拼回字面、去掉词尾标记并还原空格。"""
        out = []
        for i in ids:
            tok = self.vocab[i]
            if tok.endswith(_END):          # 词尾标记 → 去掉并补一个空格
                out.append(tok[: -len(_END)] + " ")
            else:
                out.append(tok)
        return "".join(out).strip()


def main() -> None:
    corpus = [
        "low lower lowest newest",
        "new newer newest low",
        "lower low low",
        "new new newer lower",
    ]
    bpe = BPE(num_merges=15)
    bpe.train(corpus)

    print("== 学到的 merges（按学习顺序）==")
    for i, (a, b) in enumerate(bpe.merges):
        print(f"  merge#{i:02d}: {a!r} + {b!r} -> {a + b!r}")

    print("\n== 词表（id -> token）==")
    for i, t in enumerate(bpe.vocab):
        print(f"  {i:02d}: {t!r}")

    # 训练过的词 + 一个没见过的词（会退化为更细的子词，而不是 OOV）
    for w in ["lowest", "newer", "lowering", "lowestest"]:
        ids = bpe.encode(w)
        toks = [bpe.vocab[i] for i in ids]
        back = bpe.decode(ids)
        print(f"\n  encode({w!r}) -> {toks}  decode -> {back!r}  还原一致: {back == w}")

    # 整句往返验证：任何文本都可无损还原（子词流的可逆性）
    sent = "lower newest low"
    ids = bpe.encode(sent)
    print(f"\n整句往返: {sent!r} -> ids {ids} -> {bpe.decode(ids)!r}")


if __name__ == "__main__":
    main()

    # ---- 生产建议（可选演示，需要额外安装）----
    # pip install tokenizers
    # from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    # tok = Tokenizer(models.BPE(unk_token="<unk>"))
    # tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<unk>", "<s>", "</s>"])
    # tok.train_from_iterator([line for line in open("corpus.txt")], trainer)
    # tok.save("tokenizer.json")   # merge 表/词表随文件版本化保存
