"""Word2Vec CBOW 与 Skip-gram 的 gensim 对比示例（需 pip install gensim，可直接运行）

设计：
  1) 程序化生成"三族语义受控语料"：royal / vehicle / kitchen 三族各配专属语境词，
     族内词强共现、族间词几乎不共现；
  2) 同数据分别训 sg=0(CBOW) 与 sg=1(Skip-gram)，对比族内/族间相似度、训练耗时；
  3) 演示相似度、词向量保存加载与 OOV 处理。

配套笔记：Word2Vec-CBOW与Skipgram-note.md
"""
import logging
import os
import random
import time

logging.basicConfig(level=logging.ERROR)  # 压掉 gensim 训练日志

from gensim.models import Word2Vec, KeyedVectors

# ---- 语料生成：三族语义词 + 专属语境词；"the/of"作共享功能词 ----
FAMILIES = {
    "royal":   (["king", "queen", "prince", "royal"], ["throne", "crown", "kingdom", "noble"]),
    "vehicle": (["car", "bus", "truck"],              ["road", "traffic", "driver", "station"]),
    "kitchen": (["chef", "cook", "waiter"],            ["pan", "pot", "kitchen", "menu"]),
}

rng = random.Random(42)


def build_corpus(n_per_family: int = 250) -> list:
    """每句：族内取 2 个族词 + 1~2 个语境词（确保族内词同句强共现，族间零共现）。"""
    sentences = []
    for name, (words, ctx) in FAMILIES.items():
        for _ in range(n_per_family):
            pick = rng.sample(words, k=2)                       # 至少 2 个族词共现
            pick += rng.sample(ctx, k=rng.randint(1, 2))        # 加专属语境词
            rng.shuffle(pick)
            sentences.append(pick)
    return sentences


def train(sentences, sg: int, label: str) -> tuple:
    t0 = time.time()
    model = Word2Vec(sentences=sentences, vector_size=16, window=3,
                     sg=sg, negative=5, sample=0,          # 玩具语料关闭子采样(sample=0)
                     min_count=1,                           # 否则高频相对词会被大量丢弃
                     epochs=20, workers=1, seed=7)   # 固定种子保证可比
    dt = time.time() - t0
    print(f"  训练耗时 {dt:5.2f}s   vocab={len(model.wv)}")
    return model, dt


def family_stats(wv, label: str) -> None:
    """打印族内/族间平均余弦相似度：好词向量的标志是 族内 >> 族间。"""
    names = list(FAMILIES)
    words = {n: set(FAMILIES[n][0]) for n in names}          # 族词
    intra, inter = {}, []
    for n in names:
        ws = sorted(words[n])
        s = sum(wv.similarity(ws[i], ws[j])
                for i in range(len(ws)) for j in range(i + 1, len(ws)))
        intra[n] = s / max(1, len(ws) * (len(ws) - 1) / 2)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s = sum(wv.similarity(x, y) for x in words[a] for y in words[b])
            inter.append(s / (len(words[a]) * len(words[b])))
    avg_inter = sum(inter) / len(inter)
    print(f"  [{label}] 族内相似度: " + "  ".join(f"{k}={v:.3f}" for k, v in intra.items()))
    print(f"  [{label}] 族间平均相似度: {avg_inter:.3f}   (理想: 族内远大于族间)")


def main() -> None:
    corpus = build_corpus()
    print(f"语料: {len(corpus)} 句")

    print("\n== CBOW (sg=0): 上下文词 -> 预测中心词 ==")
    cbow, _ = train(corpus, 0, "CBOW")
    family_stats(cbow.wv, "CBOW")
    print("  most_similar('king'):", [(w, round(s, 3)) for w, s in cbow.wv.most_similar("king", topn=4)])
    print("  most_similar('car'): ", [(w, round(s, 3)) for w, s in cbow.wv.most_similar("car", topn=4)])

    print("\n== Skip-gram (sg=1): 中心词 -> 预测上下文词 ==")
    skip, _ = train(corpus, 1, "Skip-gram")
    family_stats(skip.wv, "Skip-gram")
    print("  most_similar('king'):", [(w, round(s, 3)) for w, s in skip.wv.most_similar("king", topn=4)])
    print("  most_similar('car'): ", [(w, round(s, 3)) for w, s in skip.wv.most_similar("car", topn=4)])

    # 语义类比写法（king - royal + crown 之类在玩具语料上不稳定，仅演示 API 形态）
    try:
        print("\n类比 car+road-bus ≈", skip.wv.most_similar(positive=["car", "road"], negative=["bus"], topn=1))
    except KeyError as e:
        print("类比缺词跳过:", e)

    # OOV 处理 + 保存/加载（KeyedVectors = 只存向量，部署/发布常用）
    probe = "spaceship"
    print(f"\n'{probe}' in vocab: {probe in skip.wv}  → 未见词需回退策略"
          f"(fastText 子词/向量聚合/默认向量)，直接查询会 KeyError")
    tmp = "w2v_demo.kv"
    skip.wv.save_word2vec_format(tmp)
    wv2 = KeyedVectors.load_word2vec_format(tmp)
    print(f"保存并重载 {len(wv2)} 词: king~queen={wv2.similarity('king','queen'):.3f}")
    os.remove(tmp)


if __name__ == "__main__":
    main()
