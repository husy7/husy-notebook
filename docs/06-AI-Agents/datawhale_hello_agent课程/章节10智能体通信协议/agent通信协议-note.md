---
title: "[智能体通信协议]"
tags: [agents, protocols, MCP, A2A, ANP]
date: 2026-08-27
---


# 智能体通信协议学习笔记

本章解决的核心问题：**如何让智能体与外部世界高效交互，以及多个智能体之间如何协作**。HelloAgents 引入三种协议：MCP、A2A、ANP，分别对应工具访问、智能体协作、大规模网络管理。

---

## 一、为什么需要通信协议？

单个 ReAct 智能体虽然能推理、调用工具，但面临三个限制：

- **工具集成困境**：每接一个新服务（GitHub、数据库、天气 API）都要手写适配器，代码重复、难维护、无法复用。
- **能力扩展瓶颈**：智能体能力被固定在预定义工具集内，无法动态发现新服务。
- **协作缺失**：多智能体协作只能手动编排，没有标准化对话机制。

通信协议的价值：像 TCP/IP 统一设备通信一样，统一智能体与工具、智能体与智能体之间的接口。

---

## 二、三种协议对比速览

| 协议 | 全称 | 解决什么问题 | 设计理念 | 应用场景 |
|------|------|--------------|----------|----------|
| MCP  | Model Context Protocol | 智能体 ↔ 工具的标准化通信 | 上下文共享 | 访问文件、数据库、API 等外部服务 |
| A2A  | Agent-to-Agent Protocol | 智能体 ↔ 智能体的点对点协作 | 对等通信 | 多智能体任务分配、协商 |
| ANP  | Agent Network Protocol | 大规模智能体网络的服务发现与路由 | 去中心化服务发现 | 动态扩展的智能体生态系统 |

选择原则：
- 访问外部工具/资源 → **MCP**
- 多个智能体直接协作 → **A2A**
- 构建大规模开放网络 → **ANP**

---

## 三、MCP 协议实战

### 1. MCP 是什么？

MCP 就像智能体的“USB-C”，统一了工具接入方式。无论底层是文件系统、GitHub 还是数据库，智能体都能用相同的方式访问。

**架构**：Host（用户界面）→ Client（协议客户端）→ Server（实际执行者）  
**核心能力**：
- **Tools**：执行操作（如读文件、查天气）
- **Resources**：提供数据（如文件内容、数据库记录）
- **Prompts**：提供模板（如代码审查提示）

**与 Function Calling 的区别**：Function Calling 是模型能力，绑定具体提供商；MCP 是模型无关的标准化协议，工具定义一次，所有支持 MCP 的模型都能用。

### 2. 使用 MCP 客户端

HelloAgents 基于 FastMCP 2.0 实现，支持五种传输方式：

| 传输方式 | 适用场景 |
|----------|----------|
| Memory   | 单元测试、快速原型 |
| Stdio    | 本地开发、Python 脚本服务器 |
| HTTP     | 生产环境、远程服务 |
| SSE      | 实时流式通信 |
| Streamable HTTP | 双向流式 HTTP |

最常用的是 **Stdio**，通过命令启动本地服务器进程：

```python
from hello_agents.tools import MCPTool

# 连接社区文件系统服务器
fs_tool = MCPTool(
    name="fs",
    server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
)
```

### 3. 在智能体中使用 MCP 工具

`MCPTool` 有个重要特性：**自动展开**。添加 MCP 工具到 Agent 时，服务器上的所有工具会自动注册为独立工具，Agent 可以直接调用。

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="助手", llm=HelloAgentsLLM())
mcp_tool = MCPTool(name="calculator")  # 内置演示服务器
agent.add_tool(mcp_tool)
# 自动展开为 calculator_add, calculator_multiply 等工具

response = agent.run("计算 25 乘以 16")
```

**注意**：多个 MCPTool 必须指定不同的 `name`，否则展开的工具名会冲突。

---

## 四、A2A 协议实战

### 1. 设计动机

MCP 解决工具访问，A2A 解决智能体之间的协作。传统中心化协调器存在单点故障、性能瓶颈、扩展困难的问题。A2A 采用**点对点架构**，智能体可以直接通信。

核心概念：
- **Task（任务）**：智能体之间传递的工作单元，有完整生命周期（创建、协商、执行中、完成、失败等）。
- **Artifact（工件）**：任务产生的输出结果。

### 2. 使用示例

创建 A2A 服务器并定义技能：

```python
from hello_agents.protocols import A2AServer

researcher = A2AServer(name="researcher", description="研究员")

@researcher.skill("research")
def handle_research(text: str) -> str:
    # 处理研究请求
    return "研究结果..."
```

客户端调用：

```python
from hello_agents.protocols import A2AClient

client = A2AClient("http://localhost:5000")
response = client.execute_skill("research", "research AI 在医疗领域的应用")
```

### 3. 在智能体中使用 A2A 工具

通过 `A2ATool` 包装器将远程 Agent 作为工具集成到本地智能体：

```python
from hello_agents.tools import A2ATool

coordinator = SimpleAgent(name="协调者", llm=llm)
researcher_tool = A2ATool(
    name="researcher",
    agent_url="http://localhost:5000",
    description="研究员 Agent"
)
coordinator.add_tool(researcher_tool)
```

---

## 五、ANP 协议实战

### 1. 协议目标

ANP 解决大规模、开放网络中智能体的三个问题：
- **服务发现**：如何找到能处理任务的智能体？
- **智能路由**：多个候选时如何选择最优？
- **动态扩展**：新智能体如何被网络发现？

### 2. 核心概念

| 概念 | 说明 |
|------|------|
| 服务注册 | 智能体向发现中心注册自己的能力和端点 |
| 服务发现 | 按类型、能力、负载等条件查找服务 |
| 路由 | 根据策略选择最优服务 |

### 3. 使用示例

```python
from hello_agents.protocols import ANPDiscovery, register_service

discovery = ANPDiscovery()
register_service(
    discovery,
    service_id="nlp_agent_1",
    service_type="nlp",
    endpoint="http://localhost:8001",
    metadata={"load": 0.3}
)

# 发现并选择负载最低的服务
services = discovery.discover_services(service_type="nlp")
best = min(services, key=lambda s: s.metadata["load"])
```

---

## 六、构建自定义 MCP 服务器

虽然可以直接用社区 MCP 服务器，但封装自有业务逻辑时需自定义。

**步骤**：
1. 创建 `MCPServer` 实例
2. 定义工具函数并注册
3. 运行服务器

示例（天气查询服务器）：

```python
from hello_agents.protocols import MCPServer

server = MCPServer(name="weather-server")

@server.add_tool
def get_weather(city: str) -> str:
    # 调用天气 API 并返回 JSON
    return '{"city": "北京", "temperature": 10}'

if __name__ == "__main__":
    server.run()
```

测试服务器：

```python
client = MCPClient(["python", "weather_server.py"])
async with client:
    result = await client.call_tool("get_weather", {"city": "北京"})
```

发布到 Smithery 平台可以让全球开发者使用你的服务器。

---

## 七、总结与建议

- **MCP 生态最成熟**，优先使用社区现成服务器，减少重复开发。
- **A2A 适合小团队紧密协作**，用 `A2ATool` 可将远程 Agent 无缝集成到本地智能体。
- **ANP 用于大规模网络**，关注服务发现和负载均衡。
- **协议可以组合使用**：例如用 MCP 访问数据库，用 A2A 让多个智能体协作，用 ANP 管理整个智能体集群。

**实践建议**：
1. 先掌握 MCP，因为它是工具访问的基础。
2. 尝试构建自己的 MCP 服务器，加深理解。
3. 再学习 A2A 和 ANP，根据项目规模选择合适的协议组合。

---

**关键点回顾**：
- MCP = 工具访问标准
- A2A = 智能体对话标准
- ANP = 智能体网络标准
- 三者统一抽象为 `Tool` 接口，在 HelloAgents 中无缝集成。

## 参考资料

### 一、MCP 官方资源

**官方组织与规范**

- **MCP 官方 GitHub 组织**：https://github.com/modelcontextprotocol —— 包含协议规范、各语言 SDK 及参考服务器的官方仓库
- **协议规范与文档**：https://github.com/modelcontextprotocol/modelcontextprotocol —— MCP 协议规范和白皮书，由 Linux 基金会托管

**官方 SDK（多语言支持）**

- **Python SDK**：https://github.com/modelcontextprotocol/python-sdk —— 官方 Python 实现，v2.0.0 已稳定发布，支持 2026-07-28 协议修订版
- **TypeScript SDK**：https://github.com/modelcontextprotocol/typescript-sdk —— 官方 TypeScript 实现，支持 Node.js、Bun 和 Deno
- **其他语言 SDK**：Go、Java、Kotlin、C#、PHP、Ruby、Rust、Swift 等官方 SDK 均在 MCP 组织下维护

**官方参考服务器**

- **servers 仓库**：https://github.com/modelcontextprotocol/servers —— MCP 参考实现集合，包含 Everything（测试服务器）、Fetch（网页抓取）、Filesystem（安全文件操作）、Git（仓库操作）、Memory（知识图谱记忆）、Sequential Thinking（序列化思考）、Time（时间转换）等

**MCP 官方 Registry**

- **MCP Registry**：https://registry.modelcontextprotocol.io/ —— 官方 MCP 服务器目录，可浏览已发布的社区服务器

---

### 二、MCP 社区资源

**社区整理的服务器列表**

- **awesome-mcp-servers**：https://github.com/mctrinh/awesome-mcp-servers —— 精选 MCP 服务器列表，涵盖 A2A 桥接、Airtable、AWS、GitHub、Google Maps、Slack 等数百个服务器
- **awesome-mcp**：https://github.com/shaneholloman/awesome-mcp —— 按领域分类的 MCP 服务器合集，包含聚合器、数据库、浏览器自动化、云平台等类别
- **awesome-mcp（abordage）**：https://github.com/abordage/awesome-mcp —— 每日自动更新的 MCP 服务器、客户端和框架列表

**社区目录与生态**

- **MCP-Directory**：https://github.com/girishlade111/MCP-Directory —— 最全面的社区驱动 MCP 服务器目录，收录 100+ 服务器，支持 Claude Code、Gemini CLI、Cursor 等工具
- **mcp-servers-microsoft-ecosystem**：https://github.com/ppiova/mcp-servers-microsoft-ecosystem —— 微软生态 MCP 服务器目录，涵盖 Azure、Microsoft 365、Fabric、GitHub、Copilot Studio

**知名 MCP 服务器**

- **GitHub MCP Server**：GitHub 官方的 MCP 服务器实现，使 AI 智能体能够通过 MCP 协议与 GitHub API 无缝交互，访问仓库、Issue 和 PR
- **Brave Search MCP Server**：https://github.com/brave/brave-search-mcp-server —— Brave 官方搜索 MCP 服务器

---

### 三、A2A 与 ANP 资源

**A2A（Agent-to-Agent Protocol）**

- **A2A 官方项目**：https://a2aproject.github.io/A2A —— Google 发起的开放协议，支持不同供应商和框架的 AI 智能体安全通信与协作
- **A2A 参考实现**：https://github.com/SURESHBEEKHANI/Agent2Agent-A2A-protocol —— 包含 Greeting Agent 入门示例和多智能体羽毛球调度系统（使用 Google ADK、LangChain、CrewAI）

**ANP（Agent Network Protocol）**

- **ANP 官方仓库**：https://github.com/agent-network-protocol/AgentNetworkProtocol —— 开源智能体通信协议，致力于成为“Agentic Web 时代的 HTTP”
- **AgentConnect 实现**：https://github.com/agent-network-protocol/AgentConnect —— ANP 协议的开源实现
- **mcp2anp 桥接服务**：https://github.com/agent-network-protocol/mcp2anp —— MCP ↔ ANP 桥接服务，让支持 MCP 的客户端像调用本地工具一样使用 ANP 智能体

---

### 四、HelloAgents 项目

- **HelloAgents GitHub**：https://github.com/datawhalechina/hello-agents —— 本笔记所属的开源项目，包含完整的智能体学习教程和配套代码
- **第 10 章源码**：hello-agents/docs/chapter10/Chapter10-Agent-Communication-Protocols.md —— 智能体通信协议（MCP、A2A、ANP）章节的完整实现