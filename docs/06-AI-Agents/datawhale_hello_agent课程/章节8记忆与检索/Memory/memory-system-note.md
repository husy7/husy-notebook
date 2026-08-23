---
title: "智能体记忆系统"
tags: [Memory, 认知科学, AI, Agents]
date: 2026-08-19
---

# 智能体记忆系统

> 借鉴人类记忆层次（感觉/工作/长期），为智能体构建四类可检索、可遗忘、可整合的记忆模块，解决 LLM 无状态对话遗忘问题。

## 核心原理和流程

> 简记：编-存-检-整-忘（对应人类记忆五阶段）

### 为何需要记忆

LLM 两大局限：① **无状态** -> 每次调用独立，跨会话遗忘对话历史、用户偏好；② **知识静态** -> 训练数据有截止点，无法获取最新/专业领域知识（后者由 RAG 解决，见 [[RAG 检索增强生成系统]](../RAG/rag-system-note.md)）。

### 人类记忆层次 -> 智能体映射

| 人类记忆   | 特征              | 智能体映射           | 存储          |
| ---------- | ----------------- | -------------------- | ------------- |
| 感觉记忆   | 0.5-3秒，容量巨大 | （映射为输入预处理） | -             |
| 工作记忆   | 15-30秒，7±2项    | WorkingMemory        | 纯内存+TTL    |
| 情景记忆   | 个人经历事件      | EpisodicMemory       | SQLite+Qdrant |
| 语义记忆   | 一般知识概念      | SemanticMemory       | Qdrant+Neo4j  |
| 程序性记忆 | 技能习惯          | （映射为工具/函数）  | -             |

### 系统架构（四层）

```
基础设施层:  MemoryManager / MemoryItem / MemoryConfig / BaseMemory
记忆类型层:  WorkingMemory / EpisodicMemory / SemanticMemory / PerceptualMemory
存储后端层:  QdrantVectorStore / Neo4jGraphStore / SQLiteDocumentStore
嵌入服务层:  DashScope / LocalTransformer / TFIDF（三级降级）
```

### 记忆生命周期（MemoryTool 统一接口）

```python
# MemoryTool 采用"统一入口，分发处理"
def execute(self, action, **kwargs):
    # add / search / summary / stats / update / remove / forget / consolidate / clear_all
    if action == "add":    return self._add_memory(**kwargs)
    if action == "search": return self._search_memory(**kwargs)
    if action == "forget": return self._forget(**kwargs)
    if action == "consolidate": return self._consolidate(**kwargs)
    # ...

# add：模拟编码 -> 存储
memory_tool.run("add", content="用户张三是Python开发者",
                memory_type="semantic", importance=0.8)

# search：语义检索 + 多因素评分
results = memory_tool.run("search", query="前端工程师", limit=3)

# forget：选择性遗忘（3种策略）
memory_tool.execute("forget", strategy="importance_based", threshold=0.2)

# consolidate：短期 -> 长期（模拟记忆固化）
memory_tool.execute("consolidate", from_type="working",
                    to_type="episodic", importance_threshold=0.7)
```

### 四种记忆评分公式对比

| 记忆类型         | 评分公式                                               | 设计理由                                 |
| ---------------- | ------------------------------------------------------ | ---------------------------------------- |
| WorkingMemory    | `(TF-IDF×0.7 + 关键词×0.3) × 时间衰减 × (0.8+imp×0.4)` | 容量小，需快检索+时间衰减自动淘汰        |
| EpisodicMemory   | `(向量×0.8 + 时间近因×0.2) × (0.8+imp×0.4)`            | 事件有明确时间，近因性帮助回溯近期经历   |
| SemanticMemory   | `(向量×0.7 + 图×0.3) × (0.8+imp×0.4)`                  | 知识需关系推理，图检索发现概念间隐含关联 |
| PerceptualMemory | `(向量×0.8 + 时间近因×0.2) × (0.8+imp×0.4)`            | 同模态向量匹配为主，时间衰减为辅         |

> 重要性权重范围统一为 [0.8, 1.2]，避免重要性过度影响相似度排序。

## 易错点

> **importance 滥用**：所有记忆都设 importance=0.9 -> 重要性失去区分度，遗忘和整合失效。
> 按真实重要性分级：日常对话 0.3-0.5，关键事件/知识 0.7-0.9。

> **工作记忆不持久化**：WorkingMemory 纯内存 -> 重启丢失。
> 重要工作记忆需 `consolidate` 到情景/语义记忆持久化。

> **嵌入服务未降级**：DashScope API 配额耗尽或网络故障 -> 嵌入失败，记忆系统瘫痪。
> 配置三级降级：DashScope -> LocalTransformer -> TFIDF。

> **感知记忆跨模态维度不匹配**：文本/图像/音频向量维度不同直接混合 -> 检索错误。
> 模态分离存储（独立 collection），跨模态检索用对齐的统一向量空间。

## 练习

- Q1：（章末习题1-1）情景记忆为何强调时间近因性（0.2），语义记忆为何强调图检索（0.3）？
  A1：情景记忆存储具体事件，事件天然有时间属性，近期事件更可能被回溯（如"上次对话"），故加权时间近因性。语义记忆存储抽象知识概念，知识价值在于关联推理而非时间，图检索能发现概念间隐含关联（如"张三->技能->Python"），故加权图检索。

- Q2：（章末习题1-2）"个人健康管理助手"如何组合四种记忆？
  A2：① 工作记忆：当前对话上下文（用户刚说了什么）；② 情景记忆：每日饮食/运动/睡眠事件记录（带时间戳）；③ 语义记忆：用户健康画像（偏好、过敏、慢病史）、健康知识规则；④ 感知记忆：用户上传的食物照片、运动手环数据截图。

- Q3：（章末习题1-3）重要工作记忆何时应整合为长期记忆？如何设计自动触发条件？
  A3：当工作记忆 importance 超阈值（如 0.7）且被多次检索访问时整合。自动触发条件：① importance ≥ 阈值；② 被检索次数 ≥ N 次（表明确有价值）；③ TTL 即将过期但 importance 高。三个条件满足任一即触发 consolidate。

- Q4：（章末习题3-1）设计"智能遗忘"策略。
  A4：加权评分 = importance×0.4 + 访问频率×0.3 + 时间近因×0.3；分数低于阈值者遗忘。访问频率用计数器记录，时间近因用指数衰减。比单因素策略更公平：高重要性但长期未访问、低重要性但近期高频访问的记忆都能被合理保留。

## 知识关联

- 前置：[[HelloAgents：自建智能体框架]]、[[ReAct 智能体范式]]、认知心理学基础
- 横向：[[RAG 检索增强生成系统]]（RAGTool，与 MemoryTool 并列的工具）、MemGPT、LangChain Memory
- 进阶：[[智能文档问答助手]]（MemoryTool+RAGTool 组合实战）、多智能体共享记忆、记忆压缩与摘要

## 对比与选型

| 方案               | 核心思想              | 持久化              | 检索能力         | 复杂度 | 最佳场景            |
| ------------------ | --------------------- | ------------------- | ---------------- | ------ | ------------------- |
| HelloAgents Memory | 四类型分层+多存储后端 | SQLite+Qdrant+Neo4j | 向量+图+词法混合 | 高     | 生产级长期记忆Agent |
| LangChain Memory   | 对话缓冲/摘要         | 可选                | 线性历史         | 低     | 快速原型、简单对话  |
| MemGPT             | 分页记忆管理          | 向量库              | 向量             | 中     | 超长上下文管理      |

**选型速查**：学习框架原理选 HelloAgents Memory，快速原型选 LangChain，超长上下文选 MemGPT。

## 执行意图

- If 智能体需要跨会话记住用户偏好/历史事件/知识，then 引入 MemoryTool 并按场景选择记忆类型（事件->episodic，知识->semantic）。
- If 工作记忆 importance 高且 TTL 将过期，then 自动触发 consolidate 整合为长期记忆，而非任其丢失。
- If 记忆数据库积累过多，then 启用 forget 策略（importance_based/time_based/capacity_based）清理低价值记忆。

## 费曼解释

> 智能体的记忆就像人的记忆：工作记忆是"现在脑子里想的事"（容量小，转眼忘）；情景记忆是"昨天发生了什么"（带时间地点）；语义记忆是"地球绕太阳转"这类知识（长期不变）；感知记忆是"看过的照片听过的歌"。重要的短期记忆睡前"复习"变成长期记忆（consolidate），不重要的慢慢忘掉（forget）。

## 参考

- 论文：Atkinson & Shiffrin (1968). "Human memory: A proposed system and its control processes."
- 框架：HelloAgents `hello_agents.memory` 模块
