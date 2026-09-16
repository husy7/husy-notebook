---
title: "第一个智能体实战：5 分钟智能旅行助手"
tags: [ai, agent, hello-agents, 动手实践, 函数调用]
date: 2026-08-14
---

# 第一个智能体实战：5 分钟智能旅行助手

> 用 Python + OpenAI 兼容客户端 + 两个工具函数，亲手实现 `Thought-Action-Observation` 主循环，让真实 LLM 自主"查天气→推荐景点"。

## 目标与整体步骤

任务输入："你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。" 智能体须先查天气，再据结果推荐景点。整体 5 步：

```
① 安装依赖 (requests / tavily-python / openai)
② 写系统提示词 + 两个工具函数 + 工具字典注册
③ 用 OpenAICompatibleClient 接入 LLM
④ 写主循环 (构建Prompt → 调LLM → 截断/解析Action → 执行工具 → 记录Observation)
⑤ 运行并解读三轮循环
```

## ① 准备工作（1.3.1）

```bash
pip install requests tavily-python openai
```

- `requests`：HTTP 库，访问网络 API。
- `tavily-python`：AI 搜索 API 客户端，获取实时网络搜索结果（需官网注册获取 `TAVILY_API_KEY`）。
- `openai`：OpenAI 官方 Python SDK，调用 GPT 等 LLM 服务。

## 系统提示词：工具与输出格式约定（1.3.1）

提示词是智能体的"说明书"，作为 `system_prompt` 传给 LLM，规定三件事：

| 约定 | 内容 | 作用 |
|------|------|------|
| 角色 | "你是一个智能旅行助手…" | 设定行为框架 |
| 可用工具 | `get_weather(city)` / `get_attraction(city, weather)` 及签名 | 让 LLM 知道能调什么、参数是什么 |
| 输出格式 | 每次只输出一对 `Thought:`+`Action:`；Action 只能是 `func(arg="val")` 或 `Finish[最终答案]`，且 Action 必须单行 | 让输出可被正则稳定解析 |

关键规则（原文强调）：

- 每次只输出一对 Thought-Action，Action 不换行。
- 收集到足够信息时必须用 `Action: Finish[最终答案]` 结束。

> 作用本质：把 LLM 的自由文本约束成**可解析协议**，是"提示工程 + 工具注册"的工程精髓。

## 两个工具函数 + 字典注册（1.3.1）

**工具 1 · get_weather**（真实天气，`wttr.in` 免费服务，`format=j1` 返回 JSON）关键骨架：

```python
def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url); response.raise_for_status()  # 请求 + 状态码校验
    data = response.json()                                     # 解析 JSON
    cc = data['current_condition'][0]
    desc, temp = cc['weatherDesc'][0]['value'], cc['temp_C']   # 提取天气/气温
    return f"{city}当前天气:{desc}，气温{temp}摄氏度"           # 自然语言返回
    # 用 try/except 分别兜底：RequestException（网络） 与 KeyError/IndexError（解析失败）
```

**工具 2 · get_attraction**（Tavily 搜索推荐，`include_answer=True` 取综合回答）：

```python
def get_attraction(city: str, weather: str) -> str:
    key = os.environ.get("TAVILY_API_KEY")          # 未配置则返回错误提示
    tavily = TavilyClient(api_key=key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    resp = tavily.search(query=query, search_depth="basic", include_answer=True)
    if resp.get("answer"): return resp["answer"]    # 优先用总结性回答
    # 否则拼接 results 的 title/content；为空则返回"没有找到相关推荐"
```

**工具字典注册**：函数名字符串 → 函数对象，主循环据此路由：

```python
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
```

## 接入 LLM：OpenAI 兼容客户端（1.3.2）

OpenAI、Azure、Ollama、vLLM 等大多遵循 OpenAI 接口规范，可写一个通用客户端：

```python
from openai import OpenAI

class OpenAICompatibleClient:
    def __init__(self, model, api_key, base_url):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)  # base_url 决定连哪家

    def generate(self, prompt, system_prompt):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(model=self.model,
                                                   messages=messages, stream=False)
        return resp.choices[0].message.content   # 异常 try/except 返回错误串
```

实例化需三件套：`API_KEY`、`BASE_URL`、`MODEL_ID`（取值取决于服务商：OpenAI 官方 / Azure / Ollama 本地等；无渠道可参考 Extra07-环境配置）。

## 主循环：Agent Loop 代码骨架（1.3.3，关键片段+注释）

```python
prompt_history = [f"用户请求: {user_prompt}"]            # 记忆 = 历史列表

for i in range(5):                                      # 最大循环次数，防死循环
    full_prompt = "\n".join(prompt_history)             # 3.1 拼接历史构造 Prompt
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)  # 3.2 思考
    # 正则截断模型可能多输出的 Thought-Action 对，只保留第一对
    prompt_history.append(llm_output)

    m = re.search(r"Action: (.*)", llm_output, re.DOTALL)   # 3.3 解析 Action
    if not m:                                           # 没解析到 → 记错误 Observation 继续
        prompt_history.append("Observation: 错误:未能解析到 Action 字段…"); continue
    action_str = m.group(1).strip()

    if action_str.startswith("Finish"):                 # 终止条件分支
        final = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print("任务完成，最终答案:", final); break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)   # 提取函数名
    args_str = re.search(r"\((.*)\)", action_str).group(1)   # 提取实参串
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))  # 解析 k="v" 参数
    obs = available_tools[tool_name](**kwargs) if tool_name in available_tools \
          else f"错误:未定义的工具 '{tool_name}'"             # 查字典执行
    prompt_history.append(f"Observation: {obs}")        # 3.4 记录观察，进入下一轮
```

## 典型运行案例解读（1.3.4）

成功流程是三轮循环：

| 轮次 | Thought（思考） | Action（行动） | Observation（观察） |
|------|----------------|---------------|--------------------|
| 1 | 先获取北京今天天气 | `get_weather(city="北京")` | 北京 Sunny，26℃ |
| 2 | 天气晴朗温适中，据此推荐 | `get_attraction(city="北京", weather="Sunny")` | 颐和园（湖景古建）、长城（壮观历史） |
| 3 | 已获建议，可答复用户 | `Finish[最终答案…]` | 结束 |

三轮循环演示了四项基本能力：**任务分解**（先天气后景点）、**工具调用**（调两个 API）、**上下文理解**（把天气结果作为下一步依据）、**结果合成**（整合成完整答复）。

## 常见报错

| 报错/现象 | 原因 | 处理 |
|----------|------|------|
| 调用 LLM API 报鉴权错误 | `API_KEY` 未填/错误 | 检查 `API_KEY` 与 `BASE_URL` 是否匹配同一服务商 |
| `model not found` | `MODEL_ID` 写错或服务商不支持 | 确认服务商支持的确切 `MODEL_ID` |
| 连不上/超时 | `BASE_URL` 错误或本地服务未启动 | 核对 `base_url`（Ollama 本地需先起服务） |
| Tavily 报"未配置 TAVILY_API_KEY" | 环境变量未设 | 设置 `TAVILY_API_KEY` 环境变量或代码里直接赋值 |
| "错误:未定义的工具" | LLM 生成了字典外函数名/拼写错 | 检查提示词工具名与字典 key 一致 |
| 无限循环/重复调用 | 无 `Finish` 且无次数上限 | 保留 `for i in range(5)` 类上限 |

## 易错点

> **Action 换行导致解析失败**：系统提示明确要求 Action 单行；主循环正则从 `Action:` 一直吃到下一对 Thought/Observation 之前，行内换行会破坏匹配。

> **模型一次吐多对 Thought-Action**：主循环需用正则截断，只保留第一对，否则一次循环执行多步动作、状态错乱。

> **工具环境变量位置**：`TAVILY_API_KEY` 须先设置（代码演示了直接 `os.environ[...] = ...` 赋值），漏设时 `get_attraction` 返回错误串而非崩溃，学会读错误串。

## 练习

- Q1：系统提示词规定了哪三类内容？  
  A1：角色设定、可用工具及签名、输出格式约束（Thought/Action 单行、函数调用或 Finish）。
- Q2：工具为什么放进 `available_tools` 字典？  
  A2：让主循环按 `Action` 里的函数名字符串查表调用，实现 LLM 文本 → 真实函数的安全路由。
- Q3：主循环的记忆机制是什么？  
  A3：`prompt_history` 列表，累计用户请求、每次 Thought-Action、每次 Observation，拼接成下轮 Prompt。
- Q4：`Finish` 与普通工具调用的分叉在哪判定？  
  A4：解析出 `action_str` 后先判 `startswith("Finish")`，是则提取最终答案 break；否则按工具名查字典执行。

## 知识关联

- 前置：[智能体构成与运行原理](智能体构成与运行原理-note.md)、[[提示工程]]、[[Thought-Action-Observation]]、[[工具调用]]
- 后续：[协作模式与Workflow对比](协作模式与Workflow对比-note.md)（本实战是"Agent"侧的实例）
- 横向：[ReAct 智能体范式](../章节4经典范式构建/ReAct/react-note.md)；主流框架 LangChain/LlamaIndex 的"工具+提示工程"精髓同源

## 对比与选型（速查）

| 环节 | 选择 |
|------|------|
| 天气数据 | 免费 `wttr.in`（无需 key） |
| 实时搜索 | Tavily Search API |
| LLM 接入 | OpenAI 兼容客户端（OpenAI/Azure/Ollama/vLLM 通用） |
| 工具路由 | 函数名字符串 → 字典查表 |
| 终止控制 | `Finish[...]` + 最大循环次数 |

## 参考

- 源文：第一章 初识智能体 · 1.3（hello-agents docs/chapter1）