## 一、Agent 核心公式

```
Agent = LLM + Tools + Loop
         ↓      ↓       ↓
       决策   执行   反复直到完成
```

**最小闭环**：LLM 输出「要调用什么工具」→ 代码执行工具 → 结果塞回 LLM → 再问 LLM → 直到 LLM 说"完成"。

---

## 二、必记的最小 Agent 骨架（背这个）

```python
import json, httpx, os

TOOLS = {
    "get_weather": lambda city: f"{city} 25°C 晴",
    "calculator": lambda expr: str(eval(expr)),   # 生产别用 eval
}

def call_llm(messages):
    resp = httpx.post(
        os.getenv("LLM_BASE_URL"),
        headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY')}"},
        json={"model": os.getenv("LLM_MODEL"), "messages": messages},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]

def agent(user_input, max_steps=5):
    messages = [
        {"role": "system", "content": "你是 Agent。需要工具时输出 JSON: {\"tool\":\"名字\",\"args\":{...}}"},
        {"role": "user", "content": user_input},
    ]
    for _ in range(max_steps):              # ← 循环上限，防死循环
        msg = call_llm(messages)
        content = msg["content"]
        messages.append(msg)

        try:
            action = json.loads(content)    # ← 尝试解析工具调用
        except json.JSONDecodeError:
            return content                  # ← 不是 JSON = 最终答案

        tool = TOOLS.get(action["tool"])
        result = tool(**action["args"]) if tool else "工具不存在"
        messages.append({"role": "user", "content": f"工具结果: {result}"})

    return "达到最大步数"
```

**必记四步**：`问 LLM` → `解析动作` → `执行工具` → `结果回填`，循环往复。

---

## 三、必记的 5 个核心概念

### 1. Tool（工具定义）
```python
# OpenAI function calling 格式（现代主流）
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询天气",      # ← 描述决定 LLM 会不会用
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
```
> **工具描述写得好 = Agent 聪明的关键**，比模型本身还重要。

### 2. 工具调用返回格式（记死路径）
```python
# LLM 返回时
msg["tool_calls"][0]["function"]["name"]      # 工具名
msg["tool_calls"][0]["function"]["arguments"] # JSON 字符串参数
msg["tool_calls"][0]["id"]                    # 调用 ID

# 回填结果时必须带 tool_call_id
{"role": "tool", "tool_call_id": call_id, "content": result}
```

### 3. ReAct 循环（Agent 的思维模式）
```
Thought: 我需要查天气
Action: get_weather({"city": "北京"})
Observation: 北京 25°C
Thought: 已知答案
Answer: 北京今天 25°C 晴
```
> 现代 Agent 用 **function calling** 替代手写 ReAct 文本格式，本质一样。

### 4. Memory（记忆）
| 类型 | 实现 | 用途 |
|---|---|---|
| **短期** | `messages` 列表 | 当前对话 |
| **长期** | 向量库（Chroma/FAISS） | 跨会话检索 |
| **摘要** | 超长时用 LLM 压缩历史 | 省 token |

```python
# 最简短期记忆：就是 messages 列表
messages.append({"role": "user", "content": "..."})
messages.append({"role": "assistant", "content": "..."})
```

### 5. 停止条件（防死循环）
```python
# 三选一或组合
if step >= max_steps: break              # 步数上限
if not msg.get("tool_calls"): break      # LLM 不再调工具
if msg["content"] == "DONE": break       # 约定结束词
```

---

## 四、必记的 3 个「防坑」模式

### ① 循环上限 + 超时（必须）
```python
for step in range(max_steps):
    if time.time() - start > 60: raise TimeoutError
```
> **不设上限的 Agent 会烧光你的 API 余额。**

### ② 工具异常兜底（必须）
```python
try:
    result = tool(**args)
except Exception as e:
    result = f"工具执行失败: {e}"    # ← 把错误喂回 LLM，让它自己纠正
```

### ③ 参数校验
```python
# LLM 给的 arguments 是字符串，要 json.loads
args = json.loads(call["function"]["arguments"])
# 可能缺字段 → 用 .get() 或 Pydantic 校验
```

---

## 五、必记的 Agent 架构模式

| 模式 | 说明 | 记忆点 |
|---|---|---|
| **ReAct** | 边想边做，单 Agent | 最基础 |
| **Plan-and-Execute** | 先列计划再执行 | 复杂任务 |
| **Multi-Agent** | 多个 Agent 分工 | 角色扮演 |
| **Reflection** | 做完自我批评再改 | 提升质量 |
| **RAG Agent** | 工具里带检索 | 知识问答 |

---

## 六、必记的框架选择

| 框架 | 特点 | 何时用 |
|---|---|---|
| **裸写** | 无依赖，最透明 | 学习 / 简单场景 |
| **OpenAI SDK** | 官方，function calling 友好 | 直接用官方 API |
| **LangChain** | 生态全，但抽象重 | 快速搭原型 |
| **LangGraph** | 图状编排，状态机 | 复杂多步 Agent |
| **LlamaIndex** | RAG 强 | 知识库 Agent |

> **建议**：先裸写一遍（理解原理），再用框架（提效）。

---

## 七、常见坑速查

| 现象 | 原因 |
|---|---|
| Agent 死循环 | 没设 `max_steps` |
| 工具没被调用 | 工具 `description` 写太烂 |
| 参数解析失败 | `arguments` 是字符串，忘了 `json.loads` |
| 回填报错 | `tool` 消息缺 `tool_call_id` |
| 越调越蠢 | 历史太长没压缩，上下文被挤爆 |
| 烧钱 | 没有 max_tokens / 步数限制 |
| 结果不稳 | temperature 没设 0 |

---

## 🎯 一句话总结必记核心

> **`messages 循环` → `LLM 决策` → `解析 tool_calls` → `执行工具` → `回填 tool 消息` → `直到不调工具`；四要素：工具定义要清晰、循环要有上限、错误要喂回、记忆要管理。**

---

## 附：现代 function calling 版（比 JSON 解析更稳）

```python
def agent(user_input, max_steps=5):
    messages = [{"role": "user", "content": user_input}]
    for _ in range(max_steps):
        resp = httpx.post(base_url, headers=headers, json={
            "model": model, "messages": messages, "tools": tools,
        })
        msg = resp.json()["choices"][0]["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):     # ← 关键：没工具调用 = 结束
            return msg["content"]

        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            result = TOOLS[name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],   # ← 必须带
                "content": str(result),
            })
    return "超过最大步数"
```

**这份和前面最小骨架的区别**：用官方 `tools` 参数 + `tool_calls` 字段，**结构化、不会解析失败**，是生产环境的标准写法。