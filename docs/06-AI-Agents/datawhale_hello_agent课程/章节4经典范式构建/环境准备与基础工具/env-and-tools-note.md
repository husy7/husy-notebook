---
title: "智能体基础设施：LLM 客户端与工具执行器"
tags: [LLM, Tools, AI, Agents]
date: 2026-08-19
---

# 智能体基础设施：LLM 客户端与工具执行器

> 封装 LLM 调用与工具管理的通用组件，为 ReAct/Plan-and-Solve/Reflection 三种范式提供可复用底座。

## 核心原理和流程

> 简记：配-封-注-执

本节构建三个可独立复用的组件，是后续所有范式的"积木"：

### 1. 环境配置（.env + python-dotenv）

```bash
# .env 文件（项目根目录）
LLM_API_KEY="YOUR-API-KEY"
LLM_MODEL_ID="YOUR-MODEL"
LLM_BASE_URL="YOUR-URL"
SERPAPI_API_KEY="YOUR_SERPAPI_API_KEY"   # 搜索工具用
```

```bash
pip install openai python-dotenv google-search-results
```

统一配置在环境变量中，代码通过 `load_dotenv()` 自动加载，避免硬编码密钥。

### 2. HelloAgentsLLM 客户端

```python
class HelloAgentsLLM:
    def __init__(self, model=None, apiKey=None, baseUrl=None, timeout=None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        # ...校验非空...
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages, temperature=0) -> str:
        # 默认流式响应，逐块打印并拼接
        response = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, stream=True)
        collected = []
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            collected.append(content)
        return "".join(collected)
```

设计要点：① 参数优先传入，缺省从环境变量加载；② 默认 `temperature=0` 保证输出确定性；③ 流式响应提升用户体验。

### 3. 工具三要素与 search 实现

一个良好定义的工具包含：**名称**（供 Action 调用）、**描述**（LLM 据此判断何时用）、**执行逻辑**。

```python
def search(query: str) -> str:
    """基于 SerpApi 的网页搜索，智能解析优先返回答案摘要/知识图谱"""
    params = {"engine": "google", "q": query, "api_key": os.getenv("SERPAPI_API_KEY"),
              "gl": "cn", "hl": "zh-cn"}
    results = SerpApiClient(params).get_dict()
    # 智能解析：answer_box > knowledge_graph > organic_results[:3]
    if "answer_box" in results and "answer" in results["answer_box"]:
        return results["answer_box"]["answer"]
    if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
        return results["knowledge_graph"]["description"]
    if "organic_results" in results:
        return "\n\n".join(f"[{i+1}] {r.get('title','')}\n{r.get('snippet','')}"
                           for i, r in enumerate(results["organic_results"][:3]))
    return f"未找到关于 '{query}' 的信息。"
```

### 4. ToolExecutor 通用工具管理器

```python
class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    def registerTool(self, name, description, func):  # 注册工具
        self.tools[name] = {"description": description, "func": func}
    def getTool(self, name) -> callable:               # 按名取函数
        return self.tools.get(name, {}).get("func")
    def getAvailableTools(self) -> str:                # 格式化描述（注入提示词）
        return "\n".join(f"- {n}: {info['description']}" for n, info in self.tools.items())
```

`getAvailableTools()` 输出的描述字符串会被注入 ReAct 提示词的 `{tools}` 占位符，LLM 据此决定调用哪个工具。

## 易错点

> **工具描述模糊 -> LLM 误调**：描述写"搜索工具"太泛 -> LLM 不知何时该用。
> 描述需含触发条件："当你需要回答关于时事、事实以及在你的知识库中找不到的信息时使用"。

> **工具数量爆炸 -> 提示词过长**：50-100 个工具全列在提示词中 -> Token 浪费、LLM 选择困难。
> 用 RAG 检索工具描述（按语义相似度筛选 top-k），或分层分类管理（见习题3-3）。

> **`os.getenv` 在模块加载时为空**：`.env` 未加载或路径错误 -> `apiKey=None` 导致初始化失败。
> 确保在导入客户端前调用 `load_dotenv()`，并用 `if not all([...])` 明确报错。

> **流式响应未拼接完整**：`chunk.choices` 可能为空 -> 直接取 `delta.content` 报错。
> 检查 `if not chunk.choices: continue`，并处理 `content` 为 `None` 的情况（`or ""`）。

## 练习

- Q1：一个良好定义的工具包含哪三个核心要素？哪个最关键？
  A1：名称（标识符）、描述（自然语言说明用途）、执行逻辑（函数）。**描述最关键**--LLM 完全依赖它判断何时用哪个工具。

- Q2：（章末习题3-2）设计"工具选择失败"的处理机制。
  A2：记录连续失败次数；超过阈值（如 3 次）时，在提示词中追加"你已多次调用错误工具，可用工具如下：[重新列出]"并给出参数格式示例；若仍失败，返回默认提示或转人工。

- Q3：（章末习题3-3）工具数量增至 50-100 个时如何优化？
  A3：当前全量列描述不可行。优化方案：① 用向量检索按 query 语义相似度筛 top-k 工具注入提示词；② 按领域分层分类，先选类别再选具体工具；③ 用 function calling 原生支持替代正则解析。

- Q4：HelloAgentsLLM 为什么默认用 `temperature=0`？
  A4：Agent 范式要求输出格式稳定（如 ReAct 的 `Thought:/Action:` 结构、Plan 的 Python 列表），`temperature=0` 降随机性，提高格式遵循率和结果可复现性。

## 知识关联

- 前置：Python 面向对象、OpenAI API、环境变量管理
- 横向：[[ReAct 智能体范式]]（直接使用本节组件）、Function Calling、LangChain Tools
- 进阶：[[Plan-and-Solve 范式]]、[[Reflection 机制]]、RAG 检索工具、多智能体工具共享

## 执行意图

- If 我要构建一个新智能体范式，then 先抽离 LLM 客户端和工具执行器为独立可复用组件，避免每种范式重复造轮子。
- If 工具数量超过 20 个，then 停止全量列描述，改用语义检索或分层分类管理工具。

## 费曼解释

> 像组装电脑：LLM 客户端是 CPU（负责思考），工具是外设（键盘、鼠标、显示器），ToolExecutor 是主板（统一管理和调度外设）。先把主板和 CPU 装好，后面不管装什么外设都能即插即用。

## 参考

- 代码：[llm_client.py](../ReAct/llm_client.py)、[tools.py](../ReAct/tools.py)、[ReAct_agent.py](../ReAct/ReAct_agent.py)
- SerpApi：https://serpapi.com/
- OpenAI Python SDK：https://github.com/openai/openai-python
