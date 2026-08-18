---
title: "LangGraph：图结构工作流框架"
tags: [LangGraph, 状态机, 有向图, 框架]
date: 2026-08-14
---

# LangGraph：图结构工作流框架

> 把智能体流程建模为"状态机 + 有向图"，节点干活、边定跳转，原生支持循环。

## 核心原理和流程

> 简记：**状态（State）为中心，节点（Node）干活，边（Edge）定跳转**。

三大基本构成要素：

**① 全局状态（State）**：所有节点共享的 `TypedDict`，可含对话历史、中间结果、步数等。

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str       # 理解后的需求
    search_query: str     # 优化后的搜索关键词
    search_results: str   # 搜索结果
    final_answer: str     # 最终答案
    step: str             # 当前步骤标记
```

**② 节点（Nodes）**：接收状态、返回更新字段的 Python 函数。

```python
def understand_query_node(state: SearchState) -> dict:
    user_msg = state["messages"][-1].content
    # ... LLM 理解意图 + 生成搜索词 ...
    return {"user_query": ..., "search_query": ..., "step": "understood"}
```

**③ 边（Edges）**：常规边固定跳转；**条件边（Conditional Edges）**根据状态动态路由 → 实现循环/分支的关键。

```python
def should_continue(state: SearchState) -> str:
    if len(state["messages"]) < 3:
        return "continue"   # 回到上游节点
    return "end"            # 结束

workflow.add_conditional_edges("executor", should_continue,
    {"continue": "planner", "end": END})
```

**三步组装**（理解→搜索→回答的线性案例）：

```python
from langgraph.graph import StateGraph, START, END

wf = StateGraph(SearchState)
wf.add_node("understand", understand_query_node)
wf.add_node("search", tavily_search_node)
wf.add_node("answer", generate_answer_node)

wf.add_edge(START, "understand")     # 线性连接
wf.add_edge("understand", "search")
wf.add_edge("search", "answer")
wf.add_edge("answer", END)

app = wf.compile(checkpointer=InMemorySaver())  # 编译成可执行应用
for event in app.stream({"messages": [HumanMessage(content="...")]}):
    print(event)
```

## 易错点

> **简单任务过度建模**：单轮问答也要定义状态+节点+边，前期代码量大。
> 简单任务用链式（LangChain）或 ReAct 即可，LangGraph 适合多步/带循环的流程。

> **状态字段在节点间被意外覆盖**：节点返回的 dict 会合并进全局 State，字段名撞了就覆盖。
> 用 `Annotated[list, add_messages]` 这类 reducer 做追加而非覆盖；状态 Schema 要清晰。

> **条件边的返回值与路由映射不匹配**：`should_continue` 返回的字符串在 `add_conditional_edges` 的映射 dict 中找不到 → 报错。
> 路由映射的 key 必须覆盖条件函数所有可能返回值。

> **调试难度随节点数上升**：错误可能在节点逻辑、状态异变、边跳转条件任一处。
> 用 `app.stream()` 逐步打印每个节点的输出；对全局状态有完整理解。

## 练习

- Q1：LangGraph 的三大构成要素是什么？
  A1：全局状态（State，共享 TypedDict）、节点（Node，执行具体工作的 Python 函数）、边（Edge，常规边固定跳转 + 条件边动态路由）。

- Q2：条件边（Conditional Edges）为什么重要？
  A2：它根据当前状态动态决定跳转目标，是实现循环（反思-修正）、分支逻辑和自我修正工作流的关键。没有条件边就只能做线性流程。

- Q3：LangGraph 相比基于对话的框架（AutoGen/CAMEL）的核心优势？
  A3：高度可控性与可预测性。流程显式定义、原生支持循环、节点模块化、易插入人工审核节点（Human-in-the-loop），适合需高可靠性/可审计性的生产级应用。

## 知识关联

- 前置：[[ReAct 智能体范式]]、状态机、LangChain 基础
- 横向：[[AutoGen 对话驱动多智能体框架]]、[[AgentScope 工程化多智能体平台]]、[[CAMEL 角色扮演协作框架]]
- 进阶：Human-in-the-loop、反思循环（Reflexion）、Checkpointing 持久化、LangSmith 可观测性

## 对比与选型

| 方案 | 核心思想 | 灵活性 | 稳定性 | 最佳场景 |
|------|---------|--------|--------|----------|
| LangGraph | 状态机 + 有向图 | 中 | 高 | 精确流程控制 + 循环反思 |
| AutoGen | 群聊对话驱动 | 高 | 中 | 流程化多角色协作 |
| AgentScope | 消息驱动 + 分布式 | 中 | 高 | 高并发/分布式系统 |
| CAMEL | 角色扮演 + 引导提示 | 高 | 中 | 双专家深度协作 |

**选型速查**：要精确流程控制/循环反思选 LangGraph；要对话协作选 AutoGen；要轻量双专家选 CAMEL；要分布式高并发选 AgentScope。

## 执行意图

- If 我要构建**流程明确、需循环反思/分支跳转/人工审核**的智能体（如风控审批、代码生成-测试-修复），then 选 LangGraph 的图结构。
- If 我准备用 LangGraph 建模一个简单单轮问答，then 停下来，前期状态/节点/边代码量过大，用更轻量方案。

## 参考

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- 代码示例见仓库 `code/` 目录下的三步问答助手案例
