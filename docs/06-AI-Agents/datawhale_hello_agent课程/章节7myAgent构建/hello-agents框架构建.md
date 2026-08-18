---
title: "HelloAgents：自建智能体框架"
tags: [Agent框架, 框架设计, 工具系统, Python]
date: 2026-08-15
---

# HelloAgents：自建智能体框架

> 从零构建轻量级 Agent 框架：分层解耦 + 万物皆工具，为学习而生的框架。

## 核心原理和流程

> 简记：**分层解耦、职责单一、接口统一**；三大层：core（基座）→ agents（范式）→ tools（能力）。

```text
hello_agents/
├── core/      # 基座：Agent基类(ABC) / HelloAgentsLLM / Message / Config / exceptions
├── agents/    # 范式：Simple / ReAct / Reflection / PlanAndSolve / FunctionCall
└── tools/     # 能力：Tool基类 / ToolRegistry / ToolChain / AsyncToolExecutor / builtin
```

四大设计理念：① 轻量级教学友好（仅依赖 OpenAI SDK）② 基于 OpenAI 标准 API（行业事实标准）③ 渐进式版本迭代（每章一个 pip 版本）④ **万物皆工具**（Memory/RAG/MCP 统一抽象为 Tool）。

**核心接口骨架**：

```python
class Agent(ABC):                      # 抽象基类，统一执行入口
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str: ...
    # 自带：add_message / get_history / clear_history（历史记录管理）

class Tool(ABC):                       # 工具基类：自描述 + 统一执行
    @abstractmethod
    def run(self, parameters: Dict) -> str: ...
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]: ...
    def to_openai_schema(self) -> Dict: ...   # 适配原生 Function Calling

class ToolRegistry:                    # 管理中枢，两种注册方式
    register_tool(tool)                # Tool对象：复杂工具，含参数校验
    register_function(name, desc, func)  # 函数直接注册：简单工具快速集成
```

**LLM 多供应商自动检测**（优先级从高到低）：

```python
def _auto_detect_provider(self, api_key, base_url) -> str:
    # 1. 特定env变量：MODELSCOPE_API_KEY > OPENAI_API_KEY > ZHIPU_API_KEY（命中即返回）
    # 2. base_url：域名特征(api-inference.modelscope.cn) / 端口(:11434→ollama, :8000→vllm)
    # 3. LLM_API_KEY 格式前缀辅助判断（如 "ms-" → modelscope）
    # 4. 兜底返回 "auto"
```

确定 provider 后由 `_resolve_credentials` 查找对应 env 并填默认 base_url，云端与本地（VLLM/Ollama 均 expose OpenAI 兼容 API）零代码切换。

**Agent 范式实现要点**（均继承 Agent 基类，支持 custom_prompt 定制）：

```python
# SimpleAgent：可选工具调用，[TOOL_CALL:name:params] 文本协议 + 正则解析，多轮循环(max_tool_iterations)
# ReActAgent：Thought→Action→Observation 循环，max_steps 防死循环，Finish[答案] 终止
# ReflectionAgent：initial→reflect→refine 三段提示词，custom_prompts 字典注入
# PlanAndSolve：planner 强制输出 python 列表 + executor 逐步执行（异常处理保证稳定）
# FunctionCallAgent：走 OpenAI 原生 tools 参数，比 prompt 约束更鲁棒

# 工具链：模板变量串联多工具（借鉴第六章图的概念）
chain.add_step("search", "{input}", output_key="search_result")
chain.add_step("calculator", "根据:{search_result}计算", output_key="calc")
tool_input = input_template.format(**context)   # 上一步输出注入下一步输入
```

## 易错点

> **自动检测优先级陷阱**：同时设置 `OPENAI_API_KEY` 和 `LLM_BASE_URL=localhost:11434` -> 检测为 openai 而非 ollama（特定 env 变量优先级最高），调用本地服务失败。  
> 用本地模型时应显式传 `provider="ollama"`，或只留 `LLM_BASE_URL` + `LLM_API_KEY` 通用变量。

> **文本协议解析脆弱**：`[TOOL_CALL:...]` 靠 LLM 自觉输出格式 + 正则解析，参数含 `:` 或 `]` 时误解析。  
> 生产环境用 FunctionCallAgent（原生 function calling），prompt 约束仅适合教学/不支持 FC 的模型。

> **循环无兜底**：ReAct 达到 max_steps、SimpleAgent 耗尽 max_tool_iterations 后直接返回失败/跳过最终回答。  
> 必须在循环外补一次"基于工具结果给出最终回答"的 LLM 调用兜底。

> **ast 求值裸 except 吞错**：计算器工具 `except: return "计算失败"` 掩盖真实原因，且白名单不全（如缺 `ast.USub`）导致 `-5` 解析失败返回 None。  
> 捕获具体异常并记录；白名单补齐一元运算符、Pow、比较运算。

> **工具链模板变量 KeyError**：`input_template.format(**context)` 引用了不存在的 output_key 直接崩溃。  
> 先注册完整链再执行；format 前 try/except KeyError 给出明确提示。

## 练习

- Q1：为何自建框架？主流框架的四大局限性是什么？  
  A1：① 过度抽象复杂（LangChain 概念陡峭）② 快速迭代不稳定（API 频繁变更）③ 黑盒实现难定制 ④ 依赖庞大易冲突。自建 = 从使用者到构建者，深度理解原理 + 完全控制权 + 培养系统设计能力。

- Q2：习题原题——同时设 `OPENAI_API_KEY` 和 `LLM_BASE_URL=http://localhost:11434/v1`，框架选哪个 provider？合理吗？  
  A2：选 **openai**（特定 env 变量最高优先级，命中即返回，不再看 base_url）。不合理之处：用户明确配置的 base_url 是更强的意图信号，却被遗留的 env 变量覆盖，导致请求发往 openai 官方地址 + 错误的 key。改进：显式传入的 base_url 参数应优先于 env 推断。

- Q3："万物皆工具"（Memory/RAG/MCP 皆抽象为 Tool）的优势与局限？  
  A3：优势——消除抽象层、学习成本低、ToolRegistry 统一管理、与 ReAct 天然集成。局限——① 工具接口是"字符串进字符串出"，结构化数据（向量、检索打分）被迫序列化，性能损耗 ② Memory 需要 Agent 每轮主动调用才生效，框架无法自动注入，容易"忘记记忆" ③ 缺乏类型安全，复杂 RAG 管道用工具抽象表达力不足。

- Q4：ToolRegistry 的两种注册方式分别适用什么场景？  
  A4：`register_tool`（Tool 对象）适合复杂工具：有状态、需参数校验、可导出 openai schema；`register_function` 适合简单函数快速集成，但无参数自描述能力，LLM 只能靠 description 猜参数格式。

- Q5：AsyncToolExecutor 用线程池并行执行工具，何时有收益？  
  A5：工具为 **IO 密集型**（网络搜索、API 调用）且相互独立时收益最大（等待时间重叠）；CPU 密集型受 GIL 限制无效；工具间有依赖（后一个需要前一个结果）则不能并行。

## 知识关联

- 前置：[[ReAct 智能体范式]]、Plan-and-Solve / Reflection 范式（第四章）、OpenAI API、抽象类（ABC）
- 横向：[[AgentScope：工程化多智能体平台]]、LangChain/LangGraph（第六章框架对比）、VLLM/Ollama 本地部署
- 进阶：Memory 与 RAG 系统（第八章）、上下文工程（第九章）、MCP 智能体协议（第十章）

## 对比与选型

| 方案 | 核心思想 | 学习曲线 | 可控性 | 依赖量 | 最佳场景 |
|------|---------|---------|--------|--------|----------|
| HelloAgents 自建 | 分层+万物皆工具 | 低 | 完全 | 极少 | 学习原理、轻量定制 |
| LangChain | 链式抽象生态 | 陡 | 中 | 庞大 | 快速原型、生态复用 |
| AgentScope | 消息驱动分布式 | 中 | 中 | 中 | 生产级多智能体系统 |

**工具调用方式选型**：prompt 文本协议（`[TOOL_CALL:]`）适合教学与不支持 FC 的本地小模型；原生 Function Calling（FunctionCallAgent + `to_openai_schema`）鲁棒性更强，生产首选。

**选型速查**：要理解原理/深度定制选自建；要生态和速度选 LangChain；要分布式生产级选 AgentScope。

## 执行意图

- If 我遇到框架黑盒难调试、版本升级 API 破坏性变更、依赖冲突，then 评估用自建轻量框架（HelloAgents 模式）替换，只依赖 OpenAI 标准 API。
- If 我要接入新的 LLM 供应商，then 通过**继承 HelloAgentsLLM 重写 `__init__`** 拦截 provider 分支，其余 super() 交还父类——绝不改库源码。
- If 我准备靠正则解析 LLM 输出来做工具调用上生产，then 停下来，改用原生 Function Calling + `to_openai_schema()`。

## 参考

- 框架源码：https://github.com/jjyaoao/helloagents （pip install hello-agents==0.1.1）
- 本书第四章：三种经典 Agent 范式从零实现
