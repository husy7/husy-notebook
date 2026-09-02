---
title: "A2A协议笔记"
tags: [A2A, 智能体通信, 协作, 协议]
date: 2026-08-29
---
## 1. A2A 协议基础

A2A（Agent-to-Agent Protocol）由 Google 提出，核心设计理念是**实现智能体之间的点对点通信与协作**。与 MCP 关注“智能体-工具”不同，A2A 关注“智能体-智能体”之间的对话、协商和任务委托。设计哲学是“对等通信”，每个智能体既是服务提供者也是消费者，避免中心化协调器的单点故障和性能瓶颈。

### 1.1 核心概念

| 概念 | 说明 |
|------|------|
| **Task** | 智能体之间传递的任务单元，具有生命周期（创建、协商、执行、完成、失败等） |
| **Artifact** | 任务执行产生的工件（如文档、代码、图像） |

### 1.2 与传统中心化架构的对比

| 问题 | 中心化（星型拓扑） | A2A（网状拓扑） |
|------|-------------------|-----------------|
| 单点故障 | 协调器失效则系统瘫痪 | 无中心节点，鲁棒性强 |
| 性能瓶颈 | 所有通信经过中心 | 直接通信，并发能力强 |
| 扩展性 | 增加智能体需改动中心逻辑 | 即插即用，动态扩展 |

---

## 2. 使用 A2A 协议实战

HelloAgents 基于 a2a-sdk 提供了简化的 A2A 实现。

### 2.1 创建 A2A 智能体

```python
from hello_agents.protocols.a2a.implementation import A2AServer

calculator = A2AServer(
    name="calculator-agent",
    description="数学计算智能体",
    version="1.0.0",
    capabilities={"math": ["addition", "multiplication"]}
)

@calculator.skill("add")
def add_numbers(query: str) -> str:
    # 解析 "计算 5 + 3" 并返回结果
    ...
```

### 2.2 自定义 A2A 智能体

通过 `@agent.skill(name)` 装饰器添加技能，每个技能是一个函数，接收文本输入并返回文本结果。

---

## 3. 使用 HelloAgents A2A 工具

### 3.1 创建 A2A 服务端

```python
researcher = A2AServer(name="researcher", description="研究员")

@researcher.skill("research")
def handle_research(text: str) -> str:
    # 解析 topic 并返回研究结果
    ...

researcher.run(host="localhost", port=5000)  # 后台线程运行
```

### 3.2 创建 A2A 客户端

```python
from hello_agents.protocols import A2AClient

client = A2AClient("http://localhost:5000")
response = client.execute_skill("research", "research AI在医疗领域的应用")
```

### 3.3 构建多智能体协作网络

研究员、撰写员、编辑分别运行在不同端口，客户端依次调用完成“研究 → 撰写 → 编辑”的流水线。

---

## 4. 在智能体中使用 A2A 工具

通过 `A2ATool` 包装器将远程 A2A 智能体作为工具添加到 HelloAgents 智能体中。

```python
from hello_agents.tools import A2ATool

researcher_tool = A2ATool(
    name="researcher",
    description="研究员Agent，可以搜索和分析资料",
    agent_url="http://localhost:5000"
)
coordinator.add_tool(researcher_tool)
```

### 4.1 实战案例：智能客服系统

- **接待员**（SimpleAgent）分析问题类型，转发给技术专家或销售顾问
- **技术专家** / **销售顾问** 作为 A2A 服务运行
- 接待员通过 `A2ATool` 调用对应专家，整理回答返回客户

### 4.2 高级用法：Agent 间协商

A2A 支持协商机制，两个 Agent 可以通过 `propose` / `negotiate` 技能交换提案和反提案，实现任务条件协商（如截止日期）。

---

## 5. 相关习题及参考答案（A2A 部分）

### 5.1 为什么 A2A 强调“对话式协作”？与 MCP 的“上下文共享”有何不同？

**参考答案**：  
A2A 目标是智能体之间的协作，需要多轮交互、任务协商、状态同步。对话式协作意味着智能体可以像人类一样通过消息传递意图、反馈、修正，支持复杂的协作流程。而 MCP 的上下文共享侧重于工具调用时提供丰富的上下文，属于一次性请求-响应模式。A2A 需要维护任务状态、支持流式消息，MCP 更偏向无状态的工具调用。

### 5.2 在“研究团队”案例中添加“审稿人”智能体，设计三个智能体的协作流程并实现

**参考答案**：  
流程：研究员（research）→ 撰写员（write）→ 审稿人（review）→ 若需修改则返回撰写员。审稿人技能 `review` 接收文章内容，返回评审意见和是否通过。代码示例：

```python
reviewer = A2AServer(name="reviewer")
@reviewer.skill("review")
def review_article(text: str) -> str:
    # 解析 article，返回 {"approved": bool, "feedback": str}
    ...
```

协作时，客户端调用 reviewer 的 review 技能，若未通过则再次调用 writer 修改。

### 5.3 A2A 定义了 task、task_result 等消息类型，如何设计冲突解决机制（如 negotiation、voting）？

**参考答案**：  
在 A2A 中扩展消息类型：增加 `negotiation_request`、`negotiation_response`、`vote_request`、`vote_response`。协商：双方交换提案，直到达成一致或超时。投票：多个智能体对方案投票，根据多数或权重决定。实现上可在 A2A Server 中定义相应技能，如 `propose`、`vote`，并设计状态机管理。

### 5.4 对比 A2A 与 AutoGen、CAMEL 等多智能体框架，它们能否互相替代？如何让 A2A 智能体与 AutoGen 智能体通信？

**参考答案**：  
A2A 是通信协议标准，AutoGen/CAMEL 是框架/库，提供多智能体编排、对话管理等高层抽象。它们不是同一层面，不能互相替代，但可以结合：AutoGen 可以作为 A2A 的客户端或服务端实现，或者通过适配器将 A2A 消息转换为 AutoGen 的对话格式。方案：在 AutoGen 智能体外部封装 A2A 接口，使其作为 A2A Server 运行，其他 A2A 客户端可调用；或让 AutoGen 智能体通过 A2A Client 调用外部 A2A 服务。

### 5.5 A2A 通信中可能包含敏感信息，设计端到端加密和身份认证方案

**参考答案**：  
同 MCP 安全方案，使用 TLS + 应用层加密，DID 或证书认证，消息签名，访问控制列表。可参考 ANP 的 DID 机制：每个智能体拥有公私钥对，通信时使用私钥签名，接收方验证签名和身份。

### 5.6 如何防止恶意智能体发送虚假信息或发起拒绝服务攻击？设计信任评估系统

**参考答案**：  
信任评估系统记录每个智能体的历史行为（成功任务数、失败数、被投诉次数、响应时间等），计算信任分数；根据分数决定是否接受其消息、路由优先级、资源配额。对于 DoS 攻击，可设置速率限制、挑战-应答验证、黑名单。结合 ANP 的信任机制，实现跨网络信任传递。

---

## 6. 跨协议综合题及参考答案

（同 MCP 笔记中的综合题，此处从略，可参考 MCP 笔记第 8 节。）

---