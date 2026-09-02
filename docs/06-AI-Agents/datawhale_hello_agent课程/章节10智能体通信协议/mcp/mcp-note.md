---
title: "MCP协议笔记"
tags: [MCP, 智能体通信, 工具调用, 协议]
date: 2026-08-29
---

## 1. MCP 协议基础

MCP（Model Context Protocol）由 Anthropic 提出，核心设计理念是**标准化智能体与外部工具/资源的通信方式**。它像“USB-C”一样统一了智能体与外部工具的连接，让不同服务都能以相同方式被访问。MCP 的设计哲学是“上下文共享”，不仅是一个 RPC 协议，还允许智能体和工具之间共享丰富的上下文信息。

### 1.1 MCP 架构

MCP 采用 **Host–Client–Server** 三层架构：

- **Host（宿主层）**：用户直接交互的界面，管理对话流程（如 Claude Desktop）。
- **Client（客户端层）**：负责与 MCP Server 建立连接、发送请求和接收响应。
- **Server（服务器层）**：执行具体功能实现（如文件系统、数据库、GitHub 等）。

### 1.2 MCP 核心能力

| 能力 | 说明 | 角色 |
|------|------|------|
| **Tools** | 可执行的函数，智能体主动调用 | 主动（执行操作） |
| **Resources** | 提供数据，供智能体读取 | 被动（提供数据） |
| **Prompts** | 预定义的提示模板 | 指导性（提供模板） |

### 1.3 MCP 与 Function Calling 的关系

Function Calling 是 LLM 的内置能力（何时调用函数、生成参数），而 MCP 是工程层面的基础设施协议（工具如何与模型连接）。二者互补：Function Calling 相当于“会打电话”，MCP 相当于“全球电话通信标准”。

---

## 2. 使用 MCP 客户端

HelloAgents 基于 FastMCP 2.0 实现了完整的 MCP 客户端，支持异步和同步 API。

### 2.1 连接到 MCP 服务器

```python
import asyncio
from hello_agents.protocols import MCPClient

async def connect_to_server():
    client = MCPClient([
        "npx", "-y",
        "@modelcontextprotocol/server-filesystem",
        "."
    ])
    async with client:
        tools = await client.list_tools()
        print(f"可用工具: {[t['name'] for t in tools]}")

asyncio.run(connect_to_server())
```

### 2.2 发现与调用工具

```python
# 列出工具及参数
tools = await client.list_tools()
for tool in tools:
    print(tool['name'], tool.get('description'), tool.get('inputSchema'))

# 调用工具
result = await client.call_tool("read_file", {"path": "README.md"})
```

### 2.3 访问资源与提示

```python
# 资源
resources = client.list_resources()
content = client.read_resource("file:///path/to/file")

# 提示模板
prompts = client.list_prompts()
prompt = client.get_prompt("code_review", {"language": "python"})
```

### 2.4 完整示例：GitHub MCP 服务

```python
from hello_agents.tools import MCPTool

github_tool = MCPTool(
    server_command=["npx", "-y", "@modelcontextprotocol/server-github"]
)

# 列出工具
result = github_tool.run({"action": "list_tools"})

# 搜索仓库
result = github_tool.run({
    "action": "call_tool",
    "tool_name": "search_repositories",
    "arguments": {"query": "AI agents language:python", "page": 1, "perPage": 3}
})
```

---

## 3. MCP 传输方式

MCP 协议传输层无关，HelloAgents 支持五种传输方式：

| 传输方式 | 适用场景 | 特点 |
|----------|----------|------|
| **Memory** | 单元测试、快速原型 | 进程内通信，无网络开销 |
| **Stdio** | 本地开发、脚本服务器 | 通过标准输入输出，简单可靠 |
| **HTTP** | 生产环境、远程服务 | 标准 HTTP 请求，可跨网络 |
| **SSE** | 实时通信、流式处理 | 服务器推送事件，适合长连接 |
| **Streamable HTTP** | 双向流式通信 | 支持流式请求和响应 |

示例：

```python
# Memory（内置演示服务器）
mcp_tool = MCPTool()

# Stdio（自定义或社区服务器）
mcp_tool = MCPTool(server_command=["python", "my_mcp_server.py"])
mcp_tool = MCPTool(server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

# HTTP / SSE / Streamable HTTP 需使用 MCPClient 并指定 URL 与 transport_type
client = MCPClient("http://localhost:8080/mcp", transport_type="streamable_http")
```

---

## 4. 在智能体中使用 MCP 工具

HelloAgents 提供 `MCPTool` 包装器，可自动将 MCP 服务器提供的所有工具展开为独立工具。

### 4.1 自动展开机制

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="助手", llm=HelloAgentsLLM())

# 内置演示服务器
mcp_tool = MCPTool(name="calculator")
agent.add_tool(mcp_tool)
# 展开为 calculator_add, calculator_subtract, ... 等

response = agent.run("计算 25 乘以 16")
```

使用外部服务器时需指定唯一 `name` 作为前缀，避免冲突：

```python
fs_tool = MCPTool(
    name="fs",
    description="访问本地文件系统",
    server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
)
agent.add_tool(fs_tool)
# 展开为 fs_read_file, fs_write_file 等
```

### 4.2 实战案例：智能文档助手

使用两个 SimpleAgent 协作：GitHub 搜索专家（MCP GitHub 工具）+ 文档生成专家（MCP 文件系统工具）。搜索专家调用 `gh_search_repositories` 获取项目，文档专家根据结果生成 Markdown 报告并保存。

---

## 5. MCP 社区生态

丰富的社区服务器可直接使用：

- **官方服务器**：filesystem、github、postgres、slack、google-drive 等
- **社区热门**：playwright（浏览器自动化）、obsidian（笔记）、jira（项目管理）、youtube（视频字幕）等

资源库：
- Awesome MCP Servers
- MCP Servers Website
- Official MCP Servers

---

## 6. 构建自定义 MCP 服务器

### 6.1 创建 MCP 服务器

```python
from hello_agents.protocols import MCPServer

weather_server = MCPServer(name="weather-server", description="天气查询服务")

def get_weather(city: str) -> str:
    # 实现天气查询逻辑
    return json.dumps({...})

weather_server.add_tool(get_weather)

if __name__ == "__main__":
    weather_server.run()
```

### 6.2 测试与使用

通过 `MCPClient` 连接自定义服务器进行测试，或在 Agent 中通过 `MCPTool` 加载。

### 6.3 发布到 Smithery

Smithery 是 MCP 服务器的发布平台。需要准备 `smithery.yaml`、`pyproject.toml`、`Dockerfile` 等文件，然后在 smithery.ai 提交 GitHub 仓库。发布后可通过 `smithery run` 命令或直接配置使用。

---

## 7. 相关习题及参考答案（MCP 部分）

### 7.1 为什么 MCP 强调“上下文共享”？这一设计理念解决了什么核心问题？

**参考答案**：  
MCP 的核心目标是为智能体提供统一的外部工具访问接口。传统工具调用只传递输入参数和返回结果，缺乏对工具使用场景、依赖关系、数据模式等上下文信息的共享。MCP 强调“上下文共享”，意味着工具不仅返回结果，还能提供与结果相关的元数据、资源链接、提示模板等，使智能体更好地理解工具的能力和限制，从而做出更智能的决策。例如，访问代码仓库时，MCP 服务器可提供文件内容、代码结构、依赖关系等，帮助智能体进行代码分析。这解决了工具集成中的“信息孤岛”问题，提升了智能体利用工具的效率和准确性。

### 7.2 扩展 MCP 服务器，添加数据库查询、数据可视化、报表生成工具，并实现协作

**参考答案**：  
设计三个 MCP 工具：`query_database(sql)`、`visualize_data(data, chart_type)`、`generate_report(data, format)`。工具之间协作：智能体调用 `query_database` 获取原始数据，然后调用 `visualize_data` 生成图表，最后调用 `generate_report` 生成包含图表的报告。可设计一个 MCP 服务器，内部维护状态（如数据缓存），使得后续工具能够使用前一个工具的结果。关键代码结构：

```python
from hello_agents.protocols import MCPServer
import json

server = MCPServer(name="data-analysis-server")

@server.add_tool
def query_database(sql: str) -> str:
    # 模拟查询，返回 JSON 数据
    data = [{"x": 1, "y": 10}, {"x": 2, "y": 20}]
    return json.dumps(data)

@server.add_tool
def visualize_data(data_json: str, chart_type: str = "bar") -> str:
    # 生成图表文件或返回 base64
    return f"chart_{chart_type}.png"

@server.add_tool
def generate_report(data_json: str, chart_file: str, format: str = "pdf") -> str:
    return "report.pdf"
```

协作流程：Agent 依次调用三个工具，传递中间结果。

### 7.3 查阅 MCP 官方文档，设计一个同时利用 Tools、Resources、Prompts 的智能体应用

**参考答案**：  
应用场景：代码审查助手。  
- **Tools**：提供 `list_files`、`read_file`、`run_linter` 等操作。  
- **Resources**：提供项目文件树、代码规范文档等只读资源。  
- **Prompts**：提供“代码审查报告模板”、“安全漏洞检查提示”等。  
智能体可以使用 Resources 获取规范，使用 Tools 执行检查，使用 Prompts 生成结构化报告。例如，用户要求审查某个仓库，Agent 先通过 Resource 读取编码规范，然后用 Tool 运行静态分析，最后用 Prompt 生成报告。

### 7.4 分析 MCP 使用 JSON-RPC 2.0 和 stdio 的优势与局限性，如何扩展支持远程 HTTP/WebSocket？

**参考答案**：  
**优势**：JSON-RPC 2.0 简单、跨语言、易于实现；stdio 无需网络配置，适合本地工具，安全隔离。  
**局限性**：stdio 只支持本地进程，无法远程访问；JSON-RPC 本身无状态，需要额外管理会话；性能上 stdio 可能不如长连接。  
**扩展**：使用 HTTP/SSE/WebSocket 作为传输层，保持 JSON-RPC 消息格式不变，仅更换传输通道。HelloAgents 的 `MCPClient` 支持通过 URL 和 `transport_type` 连接远程服务器，这就是扩展方式。

### 7.5 MCP 客户端可调用任意工具，存在哪些安全风险？如何设计权限控制？

**参考答案**：  
**风险**：恶意工具可能删除文件、执行系统命令、窃取数据、发起网络攻击。  
**权限控制方案**：
- 在 MCP 服务器端实施白名单/黑名单，限制可执行操作；
- 客户端在调用前显示工具描述并请求用户确认；
- 引入权限令牌，工具需要授权才能执行敏感操作；
- 对工具输入进行校验，防止注入；
- 记录日志便于审计。

### 7.6 设计一个端到端加密方案，确保 MCP 通信安全

**参考答案**：  
使用 TLS 加密传输层（如 HTTPS/WSS）；在应用层使用非对称加密交换对称密钥，然后用对称加密（AES）加密消息体；对消息进行签名防篡改；使用证书或 DID 进行身份认证。对于 stdio 本地通信，风险较小，可依赖操作系统权限隔离。

---

## 8. 跨协议综合题及参考答案

### 8.1 为什么 MCP 强调“上下文共享”，A2A 强调“对话式协作”，ANP 强调“网络拓扑”？分别解决什么核心问题？

**参考答案**：  
MCP 上下文共享解决工具集成中信息孤岛问题，让智能体更理解工具能力和数据；A2A 对话式协作解决智能体之间任务协商、状态同步和复杂协作问题；ANP 网络拓扑解决大规模网络中服务发现、路由和扩展性问题。三者分别对应“智能体-工具”、“智能体-智能体”、“智能体-网络”三个层面的标准化需求。

### 8.2 假设你要构建一个“智能客服系统”，需要以下功能：（1）访问客户数据库和订单系统；（2）多个专业客服智能体协作处理复杂问题；（3）支持大规模并发用户请求。请为每个功能选择最合适的协议，并说明理由。

**参考答案**：  
- 功能（1）访问客户数据库和订单系统 → **MCP**：这是智能体与外部数据源的交互，MCP 提供标准化工具接口，可方便地封装数据库查询、订单操作等。  
- 功能（2）多个专业客服智能体协作 → **A2A**：需要智能体间对话、任务委托、结果汇总，A2A 的点对点通信和任务生命周期管理正好适用。  
- 功能（3）大规模并发用户请求 → **ANP**：需要服务发现、负载均衡和动态扩展，ANP 提供去中心化的服务注册与发现机制，适合大规模网络。

### 8.3 三种协议是否可以组合使用？请设计一个实际应用场景，展示如何同时使用 MCP、A2A 和 ANP 来构建一个完整的智能体系统。画出系统架构图并说明各协议的职责。

**参考答案**：  
可以组合使用。例如构建一个“智能研究平台”：  
- **ANP** 作为底层网络基础设施，提供服务注册与发现，让用户能够找到研究、写作、数据分析等不同类型的智能体。  
- **A2A** 用于智能体之间的协作，比如研究员智能体与撰写员智能体通过 A2A 协议交换研究结果和文章草稿。  
- **MCP** 用于每个智能体访问外部工具，例如研究员通过 MCP 调用学术数据库检索工具，撰写员通过 MCP 调用文档处理工具。  
架构图（文字描述）：顶层是 ANP 服务发现中心，中间是多个 A2A 智能体节点（研究员、撰写员、审稿人等），每个节点内部通过 MCP 客户端连接各自的 MCP 服务器（数据库、文件系统、API 等）。各协议职责清晰，共同构成完整系统。

---

