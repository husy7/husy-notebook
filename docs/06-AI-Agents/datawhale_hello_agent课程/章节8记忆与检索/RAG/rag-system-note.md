---
title: "RAG 检索增强生成系统"
tags: [RAG, 向量检索, 知识库, AI]
date: 2026-08-19
---

# RAG 检索增强生成系统

> 在生成回答前，先从外部知识库检索相关信息注入上下文，克服 LLM 知识时效性差和专业领域不足的局限。

## 核心原理和流程

> 简记：检-增-生（检索 -> 增强 -> 生成）

### RAG 三阶段演进

| 阶段 | 时期 | 检索方式 | 生成方式 |
|------|------|---------|---------|
| 朴素 RAG | 2020-2021 | TF-IDF/BM25 关键词匹配 | 文档直接拼接 |
| 高级 RAG | 2022-2023 | 稠密嵌入语义检索 | 查询重写、分块、重排序 |
| 模块化 RAG | 2023- | 混合检索、MQE、HyDE | 思维链、自我反思 |

### HelloAgents RAG 架构（五层七步）

```
用户层:   RAGTool 统一接口（add_text/add_document/search/ask/stats）
  ↓
应用层:   智能问答 / 搜索 / 知识库管理
  ↓
处理层:   MarkItDown 转换 -> Markdown 智能分块 -> 向量化
  ↓
存储层:   Qdrant 向量数据库（命名空间隔离）
  ↓
基础层:   嵌入模型（DashScope/Local/TFIDF 三级降级）
```

### 数据处理流水线

```python
# RAGTool 统一入口
rag_tool = RAGTool(knowledge_base_path="./knowledge_base",
                  collection_name="rag_knowledge_base",
                  rag_namespace="default")

# 添加文档（触发完整流水线）
rag_tool.execute("add_document", file_path="doc.pdf",
                 chunk_size=1000, chunk_overlap=200)

# 问答
answer = rag_tool.execute("ask", question="什么是大语言模型？",
                          enable_mqe=True, enable_hyde=True)
```

### 七步处理流程

```
任意格式文档 → MarkItDown转换 → Markdown文本 → 标题层次分块 → Token分块 → 向量化 → Qdrant存储
```

#### 1. 多模态文档载入（MarkItDown）

微软开源工具，将 PDF/Word/Excel/图片/音频统一转为 Markdown：

```python
def _convert_to_markdown(path):
    if ext == '.pdf':
        return _enhanced_pdf_processing(path)  # PDF 增强处理
    result = MarkItDown().convert(path)
    return result.text_content
```

#### 2. Markdown 结构感知分块

```python
# 标题层次解析（# / ## / ###）-> 段落语义分割 -> Token 计算 -> 重叠分块
def _split_paragraphs_with_headings(text):
    # 维护 heading_stack，遇到标题更新层级
    # 遇到空行 flush 当前段落缓冲
    # 每个 paragraph 记录 heading_path（如 "第一章 > 1.1 节"）

def _chunk_paragraphs(paragraphs, chunk_tokens=1000, overlap_tokens=200):
    # 按 token 预算累积段落，超限时切块
    # 重叠：保留尾部段落作为下块开头，保证上下文连续
```

#### 3. 中英文 Token 估算

```python
def _approx_token_len(text):
    cjk = sum(1 for ch in text if _is_cjk(ch))  # CJK 字符按 1 token
    non_cjk = len([t for t in text.split() if t])  # 其他按空格分词
    return cjk + non_cjk
```

#### 4. 统一嵌入与向量存储

```python
# 嵌入模型三级降级：DashScope API -> LocalTransformer -> TFIDF
embedder = get_text_embedder()  # 统一接口，内部自动降级
vecs = embedder.encode(texts)   # 批量编码
store.add_vectors(vectors=vecs, metadata=[...], ids=[...])
```

### 高级检索策略（MQE + HyDE）

```python
def search_vectors_expanded(query, top_k=8,
                            enable_mqe=True, enable_hyde=True,
                            candidate_pool_multiplier=4):
    expansions = [query]
    if enable_mqe:
        expansions.extend(_prompt_mqe(query, n=2))  # LLM 生成多样化查询
    if enable_hyde:
        expansions.append(_prompt_hyde(query))       # LLM 生成假设答案段落

    # 对每个扩展查询并行检索，合并去重，按分数排序
    pool = top_k * candidate_pool_multiplier  # 扩大候选池
    per = pool // len(expansions)
    agg = {}
    for q in expansions:
        hits = store.search_similar(query_vector=embed_query(q), limit=per)
        for h in hits:  # 去重取最高分
            mid = h["metadata"]["memory_id"]
            if mid not in agg or h["score"] > agg[mid]["score"]:
                agg[mid] = h
    return sorted(agg.values(), key=lambda x: x["score"], reverse=True)[:top_k]
```

**MQE（多查询扩展）**：LLM 生成语义等价的多样化查询（"如何学Python" -> "Python入门教程"/"Python学习方法"），解决用词差异导致的召回遗漏，提升召回率 30%-50%。

**HyDE（假设文档嵌入）**：LLM 先生成假设性答案段落，用答案去检索真实文档，缩小问题与文档间的语义鸿沟（问题是疑问句，文档是陈述句）。

## 易错点

> **分块过大/过小**：块太大 -> 嵌入语义模糊，检索不精确；块太小 -> 上下文断裂，信息碎片化。
> `chunk_size=1000, overlap=200` 是经验起点，需根据文档类型调整。

> **无标题文档分块失效**：小说/法律条文无明确标题结构 -> 标题层次分块退化。
> 用语义边界分块：按段落/句子切分，结合滑动窗口+重叠保证连续性（见习题2-1）。

> **嵌入模型不一致**：索引时用 DashScope，查询时用 LocalTransformer -> 向量维度/空间不匹配，检索全错。
> 索引和查询必须用同一嵌入模型；切换模型需重新索引全部文档。

> **MQE 生成查询偏离原意**：LLM 扩展查询语义漂移 -> 检索到不相关文档。
> 限制扩展数量（n=2-3），prompt 约束"语义等价"，合并时用原查询分数加权。

> **未设 min_score 过滤**：低相关度结果混入上下文 -> LLM 被噪声干扰，生成错误答案。
> 设 `min_score=0.1-0.3` 过滤低分结果。

## 练习

- Q1：（章末习题2-1）无标题文档如何优化分块？设计语义边界分块算法。
  A1：① 按段落（双换行）分割；② 段落内按句子（句号/问号/感叹号）切分；③ 递归合并短句至接近 chunk_size；④ 超长段落按句子边界滑动窗口切块+overlap。核心：以句子为最小语义单元，保证块内语义完整、块间有重叠。

- Q2：（章末习题2-2）基础检索 vs MQE vs HyDE 适用场景对比？
  A2：基础检索（单查询向量检索）：速度最快，适合简单明确查询（如"Python 定义"）。MQE：适合模糊/多义查询（如"学习"可能指多种含义），用词多样性提升召回。HyDE：适合专业领域查询（问题与文档用语差异大），假设答案含领域术语，桥接语义鸿沟。组合 MQE+HyDE 效果最佳但成本最高（多轮 LLM 调用）。

- Q3：（章末习题2-3）三种嵌入方案选型对比？
  A3：

| 方案 | 准确性 | 速度 | 成本 | 离线 | 最佳场景 |
|------|--------|------|------|------|---------|
| DashScope API | 高 | 中（网络延迟） | 按量付费 | 否 | 云端生产环境 |
| LocalTransformer | 中高 | 中（首次加载慢） | 免费（需GPU） | 是 | 本地部署、隐私敏感 |
| TFIDF | 低 | 快 | 免费 | 是 | 兜底/原型验证 |

选型建议：生产首选 DashScope，隐私敏感选 Local，资源受限用 TFIDF 兜底。同一系统必须保持索引和查询使用相同方案。

- Q4：RAG 与 MemoryTool 的关系？何时用 RAG，何时用 Memory？
  A4：RAG 检索外部知识库（文档、手册），Memory 检索内部交互历史（用户偏好、对话事件）。RAG 适合"文档里有答案"的客观知识查询；Memory 适合"之前聊过/用户说过"的主观上下文回忆。复杂任务两者结合（见 [[智能文档问答助手]]）。

## 知识关联

- 前置：[[智能体记忆系统]]、向量数据库基础、嵌入模型
- 横向：[[ReAct 智能体范式]]（RAG 可作为 ReAct 的工具）、LangChain RAG、LlamaIndex
- 进阶：[[智能文档问答助手]]（RAG+Memory 组合实战）、多模态 RAG、GraphRAG、Agentic RAG

## 对比与选型

| 方案 | 核心思想 | 检索质量 | 实现复杂度 | 最佳场景 |
|------|---------|---------|-----------|---------|
| 基础向量检索 | 单查询向量相似度 | 中 | 低 | 简单明确查询 |
| MQE 增强 | 多查询扩展+合并 | 高（召回+30-50%） | 中（需LLM） | 模糊/多义查询 |
| HyDE 增强 | 假设文档检索 | 高（精度提升） | 中（需LLM） | 专业领域查询 |
| MQE+HyDE | 双策略组合 | 最高 | 高 | 高质量要求场景 |

**选型速查**：简单查询用基础检索，模糊查询加 MQE，专业领域加 HyDE，追求最佳效果双开但注意成本。

## 执行意图

- If 用户问题涉及文档/手册中的客观知识，then 用 RAGTool 检索外部知识库，而非依赖 LLM 内置知识。
- If 查询模糊/多义/专业术语差异大，then 启用 MQE（生成多样化查询）和 HyDE（生成假设文档）提升召回。
- If 索引和查询嵌入模型不一致 / 切换嵌入方案，then 必须重新索引全部文档，否则向量空间不匹配。

## 费曼解释

> RAG 就像开卷考试：LLM 是考生（会做题但知识有限），RAG 是课本（答案在里面）。考试时先翻书找到相关页（检索），把内容抄在草稿纸上（增强），再答题（生成）。MQE 是"换几种方式翻目录"，HyDE 是"先自己想个大概答案再去找书里最像的那段"。

## 参考

- 框架：HelloAgents `hello_agents.memory.rag` 模块
- MarkItDown：https://github.com/microsoft/markitdown
- 论文：Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
