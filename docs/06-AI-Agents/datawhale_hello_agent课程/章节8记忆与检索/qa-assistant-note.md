---
title: "智能文档问答助手"
tags: [RAG, Memory, 实战, Gradio]
date: 2026-08-19
---

# 智能文档问答助手

> 将 MemoryTool 和 RAGTool 组合封装为 PDFLearningAssistant，实现文档加载、智能问答、笔记记录、学习报告的完整闭环。

## 核心原理和流程

> 简记：载-问-记-顾-报（五步闭环）

### 五步执行流程

```
1. 加载文档（RAGTool: PDF->Markdown->分块->向量化）+ MemoryTool 记录事件
2. 智能问答（RAGTool: MQE+HyDE 检索）+ MemoryTool 记录到工作/情景记忆
3. 笔记记录（MemoryTool: 存入语义记忆）
4. 学习回顾（MemoryTool: 检索情景记忆）
5. 报告生成（汇总 Memory+RAG 统计）
```

### 核心类设计

```python
class PDFLearningAssistant:
    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化两个工具（用户级隔离）
        self.memory_tool = MemoryTool(user_id=user_id)
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")

        self.stats = {"session_start": datetime.now(),
                      "documents_loaded": 0, "questions_asked": 0,
                      "concepts_learned": 0}
        self.current_document = None
```

**关键设计决策**：
- `user_id` 隔离：不同用户记忆独立（MemoryTool 层）
- `rag_namespace=f"pdf_{user_id}"`：不同用户知识库独立（RAGTool 层 Qdrant 命名空间）
- `session_id`：追踪单次学习会话，便于学习历程回顾

### 五步核心实现

#### 步骤1：加载文档

```python
def load_document(self, pdf_path):
    # RAGTool 处理 PDF（一行触发完整流水线）
    result = self.rag_tool.execute("add_document", file_path=pdf_path,
                                   chunk_size=1000, chunk_overlap=200)
    if result["success"]:
        self.current_document = os.path.basename(pdf_path)
        self.stats["documents_loaded"] += 1
        # MemoryTool 记录到情景记忆（事件类型=文档加载）
        self.memory_tool.execute("add",
            content=f"加载了文档《{self.current_document}》",
            memory_type="episodic", importance=0.9,
            event_type="document_loaded", session_id=self.session_id)
```

#### 步骤2：智能问答

```python
def ask(self, question, use_advanced_search=True):
    # 记录问题到工作记忆
    self.memory_tool.execute("add", content=f"提问: {question}",
        memory_type="working", importance=0.6, session_id=self.session_id)

    # RAGTool 高级检索问答（MQE+HyDE）
    answer = self.rag_tool.execute("ask", question=question, limit=5,
        enable_advanced_search=use_advanced_search,
        enable_mqe=use_advanced_search, enable_hyde=use_advanced_search)

    # 记录问答到情景记忆
    self.memory_tool.execute("add", content=f"关于'{question}'的学习",
        memory_type="episodic", importance=0.7,
        event_type="qa_interaction", session_id=self.session_id)
    self.stats["questions_asked"] += 1
    return answer
```

#### 步骤3-5：笔记、回顾、报告

```python
def add_note(self, content, concept=None):
    self.memory_tool.execute("add", content=content,
        memory_type="semantic", importance=0.8,   # 笔记是知识 -> 语义记忆
        concept=concept, session_id=self.session_id)
    self.stats["concepts_learned"] += 1

def recall(self, query, limit=5):
    return self.memory_tool.execute("search", query=query, limit=limit)

def generate_report(self, save_to_file=True):
    report = {"session_info": {...}, "learning_metrics": self.stats,
              "memory_summary": self.memory_tool.execute("summary"),
              "rag_status": self.rag_tool.execute("stats")}
    # 保存为 JSON 文件
    json.dump(report, f, ensure_ascii=False, indent=2)
```

### 记忆类型选择逻辑

| 操作 | 记忆类型 | 理由 |
|------|---------|------|
| 加载文档 | episodic (imp=0.9) | 具体事件，高重要性 |
| 用户提问 | working (imp=0.6) | 当前对话上下文，临时 |
| 问答交互 | episodic (imp=0.7) | 学习事件记录 |
| 笔记 | semantic (imp=0.8) | 知识概念，长期保存 |

## 易错点

> **用户隔离不彻底**：只隔离了 MemoryTool 的 user_id，但 RAGTool 仍共享同一 collection -> 用户间知识泄漏。
> RAGTool 用 `rag_namespace=f"pdf_{user_id}"` 隔离 Qdrant 命名空间，MemoryTool 用 user_id 过滤。

> **RAG vs Memory 路由缺失**：所有问题都走 RAG 检索 -> 用户问"我之前学过什么"时，RAG 答不了（不在文档里）。
> 设计智能路由：客观知识问题走 RAG，个人学习历史走 Memory（见习题5-1）。

> **报告只是统计罗列**：当前报告只有数字（提问次数/文档数），无智能分析。
> 扩展为分析学习轨迹、识别知识盲点、推荐下一步内容（见习题5-2）。

> **会话状态不持久**：`session_id` 和 `stats` 在内存中 -> 重启丢失。
> 关键统计信息应持久化到情景记忆或数据库。

## 练习

- Q1：（章末习题5-1）设计 RAG vs Memory 智能路由机制。
  A1：路由决策树：
  ```
  用户问题 -> LLM 意图分类
  ├─ 客观知识类（"什么是X"/"文档里怎么写的"）-> RAGTool 检索知识库
  ├─ 个人历史类（"我之前学过"/"我上周问了什么"）-> MemoryTool 检索情景记忆
  ├─ 用户偏好类（"我喜欢什么风格"）-> MemoryTool 检索语义记忆
  └─ 混合类 -> 双路检索，RAG 提供知识 + Memory 提供个性化上下文，合并注入提示词
  ```
  实现方式：用 LLM 做意图分类（zero-shot），或基于关键词规则（"我/之前/上次"->Memory）。

- Q2：（章末习题5-2）扩展学习报告：识别知识盲点+推荐学习内容。
  A2：① 从情景记忆检索所有 qa_interaction 事件，分析用户提问分布；② 对比文档目录结构，找出未覆盖章节（知识盲点）；③ 从语义记忆提取已学概念，生成知识图谱，识别关联但未学习的节点；④ 推荐：未读章节+关联未学概念。需用：episodic（提问历史）+ semantic（已学概念）+ RAG stats（文档覆盖）。

- Q3：（章末习题5-3）多用户 Web 服务数据隔离方案。
  A3：① Qdrant：用 `rag_namespace` + payload 的 `user_id` 字段双重隔离，检索时 `where={"user_id": uid, "rag_namespace": ns}`；② Neo4j：节点加 `user_id` 属性，查询时 WHERE 条件过滤；③ SQLite：表加 `user_id` 列+索引；④ 性能优化：为高频用户建独立 collection，冷数据共享 collection 用 payload 过滤。

- Q4：（章末习题4-3）敏感信息遗忘是否删除数据库即可？
  A4：不充分。向量数据库：删除向量点后，向量索引可能仍缓存（需 `force_rebuild` 或等待合并）；图数据库：删除节点后关系边可能残留，且 Neo4j 事务日志（WAL）可能仍含数据；备份：定期备份的快照仍含旧数据。彻底清除需：删除数据+重建索引+清理事务日志+更新备份策略。

## 知识关联

- 前置：[[智能体记忆系统]]、[[RAG 检索增强生成系统]]、[[HelloAgents：自建智能体框架]]
- 横向：[[Reflection 机制]]（可用于报告生成时的自我审视）、LangChain RAG+Memory
- 进阶：多模态文档助手、Agentic RAG（智能体主动决策检索策略）、多用户 SaaS 部署

## 执行意图

- If 要构建文档问答/学习助手，then 封装 RAGTool+MemoryTool 为统一助手类，用 user_id+namespace 做用户隔离。
- If 用户问题涉及个人历史/偏好，then 路由到 MemoryTool 而非 RAGTool 检索。
- If 要生成智能学习报告，then 分析情景记忆（提问历史）+ 语义记忆（已学概念）+ RAG 文档覆盖度，识别盲点并推荐。

## 费曼解释

> 就像一个有记性的图书管理员：你给他一本 PDF（加载文档），他读完记住内容（RAG）。你问问题他翻书找答案（检索），同时记下你问了什么（Memory 情景记忆）。你做笔记他帮你存好（Memory 语义记忆）。最后他能告诉你"今天学了啥、哪些没看懂、明天该学啥"（报告生成）。

## 参考

- 代码：`code/chapter8/11_Q&A_Assistant.py`
- 框架：HelloAgents `PDFLearningAssistant` 类
- Gradio：https://gradio.app/
