---
title: "向量数据库（Embedding 检索与选型）"
tags: [Embedding, ANN, 向量检索, 数据库]
date: 2026-08-30
---

# 向量数据库（Embedding 检索与选型）

## 定义

向量数据库（Vector Database）是一种以高维浮点向量（embedding）为核心数据模型、以「相似度 / 最近邻」为核心查询语义的数据库系统。它存储并索引由深度学习模型（text-embedding、CLIP、图像 encoder 等）把文本、图片、音频等非结构化数据映射出的稠密向量，支持按向量距离返回最相似的一批记录。

它解决的核心问题是：传统数据库的等值 / 范围 / 前缀匹配无法表达「语义相近」——如"如何退款"与"退费流程"字面不同但含义相近；而数据量达到百万乃至十亿级后，每次查询精确遍历全部向量做 KNN 不可接受，必须用 ANN（近似最近邻）索引把查询降到亚线性复杂度。

核心特征可归纳为五条：① embedding 的写入、更新与持久化；② 内置 ANN 索引（HNSW、IVF、PQ、DiskANN 等），用户无需自研暴力扫描或树结构；③ 提供余弦 / 内积 / L2 等相似度度量并支持按业务切换；④ 向量与标量元数据（tag、时间、权限）的混合过滤；⑤ 面向分布式的分片、复制与横向扩展。

适用范畴：RAG 检索增强生成、企业知识库问答、以图搜图、语义查重与聚类、推荐召回粗排、异常检测等一切「按内容而非按词匹配」的检索需求。它与全文检索（BM25）并非替代而是互补关系：向量检索管语义近似，全文检索管精确词与专有术语。

## 原理

为什么这样设计？语义检索的底层逻辑是「把语义相近编码成空间邻近」：embedding 模型把每个对象映射进同一 d 维空间，训练目标就是让语义相近的样本向量距离更近，于是检索被转成几何问题——给定查询向量 q，找出库中距离最近的前 k 个向量。

最常用的相似度度量是余弦相似度（文本检索默认）、内积（配合归一化向量）与 L2 距离（图像特征常用）：

$$
\text{cosine}(A, B)=\frac{A\cdot B}{\|A\|_2\cdot\|B\|_2},\qquad \|A\|_2=\sqrt{\sum_{i=1}^{d}A_i^2}
$$

把向量归一化成单位向量后 $\|A\|_2=\|B\|_2=1$，则内积恰好等于余弦相似度、距离可取 $1-\cos(A,B)$。这正是多数向量库默认「归一化 + 内积」组合的原因：打分退化为一次点积，计算极快。

精确 Top-K 的时间复杂度为 $O(N\cdot d)$（N 为库中向量总数），海量数据下无法满足毫秒级查询，因此向量库普遍采用 ANN——牺牲少量召回率换取亚线性查询。三种最主流的索引思路：

- HNSW：构建分层小世界图——底层含全部点、越往上越稀疏；插入时从顶层入口贪心逼近找到近邻并连接双向边，查询沿「高层粗定位 → 底层精搜」下探，效果近似 $O(\log N)$，是延迟与召回平衡最好的默认选择（论文见参考）。
- IVF（倒排文件）：用 k-means 把空间划分为 nlist 个桶，查询只扫描与 q 最近的 nprobe 个桶，复杂度约 $O(\frac{N}{\text{nlist}}\cdot \text{nprobe})$；内存占用小、适合超大规模，常与 PQ 组合成 IVF-PQ。
- PQ（乘积量化）：把向量切成 m 段、每段独立码本量化，压缩存储至 1/8～1/16，用查表近似算距离，用于内存受限的十亿级场景。

```mermaid
flowchart LR
    Q[用户查询文本] --> E["Embedding 模型 text-embedding-3 / bge-m3"] --> V[查询向量]
    V --> S{ANN 索引}
    S -->|HNSW 图搜索| C["候选 top_k × nprobe"]
    S -->|IVF 倒排扫描| C
    C --> M[元数据过滤] --> R[可选：精确重排] --> Out[Top-K 结果]
```
关键认知：① ANN 返回的是近似结果，删除 / 更新需要索引维护（HNSW 常以墓碑标记 + 定期重建实现）；② 候选集先放大再精确重排（rerank）是兼顾召回率与耗时的通用手段；③ 向量维度升高后距离趋于均匀（维度灾难），高维场景常配合降维、聚类或换度量使用。

## 应用

典型场景：① RAG——知识库先切块（chunk）向量化入库，提问时先语义召回最相关片段再交给大模型生成，缓解幻觉；② 企业知识库 / 客服问答，用户表述灵活、同义改写多；③ 以图搜图、以文搜图等多模态检索；④ 内容近重复检测、去重聚类、推荐粗排召回；⑤ Agent / Copilot 的长期记忆检索。

快速上手五步：1) 选定 embedding 模型并记录输出维度，写入与查询必须用同一模型版本；2) 预处理——按内容结构分块（200～1000 token 为宜，可加重叠）、清洗、批量生成向量；3) 入库并建 ANN 索引（HNSW 主要调 M、efConstruction，查询时调 efSearch）；4) 查询链路：query 向量化 → ANN 取 top_k×nprobe 候选 → 元数据过滤 → 精确重排；5) 离线评估 hit rate（召回率）与 P99 时延后再上线迭代。

若团队已有 PostgreSQL，可用 pgvector 零新组件起步，SQL 写法与要点如下：

```sql
-- pgvector：在 PostgreSQL 内新增向量类型与 HNSW 索引
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE docs (
  id        bigserial PRIMARY KEY,
  content   text,
  embedding vector(1536)               -- 维度必须与 embedding 模型输出一致
);

CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops);  -- 余弦度量 HNSW
-- 语义检索：<=> 是余弦距离，升序排列 = 最相似在前
SELECT id, content, 1 - (embedding <=> $1) AS similarity
FROM docs
ORDER BY embedding <=> $1
LIMIT 5;
```

常见坑：① 混用多个 embedding 模型会形成「同库不同空间」，检索失真且不可迁移，换模型必须全量重建索引；② 相似度阈值无普适值，需按自己数据的分布统计（如取 p95）后再过滤；③ HNSW 删除成本高、ANN 有漏召回，对账等要求 100% 精确的场景慎用近似索引；④ 元数据过滤顺序影响结果——先过滤再检索可能因桶太小漏召回，需权衡 pre-filter / post-filter；⑤ 中文切分敏感，chunk 过碎丢失语义、过大引入噪声，应结合标题与段落结构切分；⑥ Python 内置 hash() 每进程加盐不可复现，做向量哈希要用确定性算法（见下方示例）。

```python
"""自包含向量检索最小示例：确定性哈希词袋向量 + 余弦 Top-K。
真实项目应使用 FAISS / Milvus / pgvector 并接入真正的语义 embedding 模型。"""
import math
import re
import zlib
DIM = 512  # 向量维度（text-embedding-3-small 是 1536）

def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())  # 简单分词

def embed(text: str) -> list[float]:
    """文本 -> DIM 维单位向量。演示要点：词袋不识别同义词改写，
    这正是真实语义 embedding 的价值；此处仅示范"文本如何变向量"。"""
    vec = [0.0] * DIM
    for tok in tokenize(text):
        idx = zlib.crc32(tok.encode("utf-8")) % DIM  # 确定性哈希，可跨进程复现
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0  # 空文本时防除零
    return [v / norm for v in vec]  # 单位化后：内积 == 余弦相似度

class TinyVectorDB:
    """极简向量库：插入 + 精确余弦 Top-K（全量扫描版，复杂度 O(N*d)）。"""

    def __init__(self) -> None:
        self._rows: list[tuple[list[float], str]] = []  # (单位向量, 原文)

    def insert(self, doc: str) -> None:
        self._rows.append((embed(doc), doc))

    def search(self, query: str, top_k: int = 2) -> list[tuple[str, float]]:
        q = embed(query)
        scored = sorted(((sum(a * b for a, b in zip(vec, q)), doc)
                         for vec, doc in self._rows), reverse=True)
        return [(doc, round(s, 4)) for s, doc in scored[:top_k]]

if __name__ == "__main__":
    db = TinyVectorDB()
    corpus = ["vector database uses HNSW index for fast nearest neighbor search",
              "PostgreSQL can do vector search via the pgvector extension",
              "it is a nice day, good for jogging outside"]
    for doc in corpus:
        db.insert(doc)
    for q in ["database nearest neighbor search", "nice weather"]:
        print(f"query: {q}")
        for doc, score in db.search(q, top_k=2):
            print(f"  {score:.4f}  {doc}")
```

案例详解：`embed` 先分词，再用 zlib.crc32 对每个词做确定性哈希散列进 512 维（刻意不用内置 hash()——它按进程加盐，跨运行结果不同，是常见坑；维度够高可避免不同词哈希碰撞），最后做 L2 归一化；模长归一化为 1 后，`search` 中的一行点积 `sum(a * b ...)` 就直接等于余弦相似度，按分数降序取前 top_k 即结果。整段复杂度 O(N·d)，是「精确扫描」而非近似检索——这正是需要 HNSW / IVF 提速的原因。
输出预期：query 1（"database nearest neighbor search"）top-1 是第 1 条语料（≈0.63，四个查询词全部命中），第 2 条仅因共现 "search" 排第二（≈0.17）；query 2（"nice weather"）命中第 3 条语料（≈0.24，只有 "nice" 词面重合，"weather" 一字未中），其余两条为 0。这个结果恰好暴露词袋的局限：它只做字面共现计数、不理解语义——若把 query 换成同义改写（如 "lookup similar items by vectors"），词袋法可能一条都召不回，这正是语义 embedding + 向量数据库存在的理由。生产落地只需两步替换：把 `embed` 换成真实 embedding 模型输出、把 `search` 换成 FAISS `IndexHNSWFlat` 或上文 pgvector 的 SQL，整体流程与调用接口完全一致。

---
## 关联
- 前置：[[索引原理与B+树-note]]（理解索引如何用额外结构换查询速度；B+ 树面向精确 / 范围匹配，向量索引是把同一思路搬到高维空间的近似形态）
- 类似：[[事务与隔离级别-note]]（区别是：传统关系库以 ACID 事务与强一致为核心承诺，向量库多为最终一致、弱事务，把资源优先投向海量向量的低延迟召回）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 向量数据库 + ANN 索引（本文方案，如 Milvus / Qdrant / FAISS 自建） | 存 dense embedding，用 HNSW / IVF-PQ 做近似最近邻，支持向量 + 标量混合过滤与分布式 | 语义检索、RAG、以图搜图、百万级以上的向量召回 |
| 关键词全文检索（Elasticsearch BM25 / PostgreSQL 全文） | 倒排索引 + TF-IDF / BM25 词频打分，只按字面匹配 | 精确词与专有名词匹配、可解释性要求高、日志与文本搜索 |
| 关系型数据库 + pgvector | 复用既有 SQL 生态，新增 vector 类型与 HNSW / IVF 索引 | 已有 PostgreSQL 业务、数据量中小、需要 SQL 联合过滤与事务 |
| 纯暴力精确检索（numpy 全量点积） | 不建索引，每次查询遍历全部向量精确算分 | 数据量很小、必须 100% 精确召回、用于验证 ANN 召回率基准 |

---
## 参考
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs（HNSW 原始论文）](https://arxiv.org/abs/1603.09320)
- [Facebook AI Similarity Search (FAISS) 官方仓库与文档](https://github.com/facebookresearch/faiss)
- [pgvector 官方仓库（PostgreSQL 向量检索扩展）](https://github.com/pgvector/pgvector)
- [Milvus 官方文档（向量数据库架构与索引指南）](https://milvus.io/docs/)

---
## 具体案例
- [[向量数据库（Embedding 检索与选型） 实战示例]](向量数据库_sample.py)
