
## 一、LangGraph 核心公式

```
LangGraph = State（共享状态） + Node（干活函数） + Edge（决定下一步）
             ↓                    ↓                  ↓
        所有节点读写同一份     普通 Python 函数    固定边 / 条件边
        数据，通过 reducer    (state) -> dict     决定路由到哪个节点
        控制如何合并
```

**与裸写 Agent 的最大区别**：状态是**显式共享**的，流程是**图结构**的，天然支持循环、条件分支和持久化。

---

## 二、必记的最小骨架（背这个）

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    text: str

def node_a(state: State) -> dict:
    return {"text": state["text"] + "a"}   # ← 返回增量，不是完整 state

def node_b(state: State) -> dict:
    return {"text": state["text"] + "b"}

graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_edge(START, "node_a")           # ← 入口
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)             # ← 出口

app = graph.compile()
print(app.invoke({"text": ""}))           # {'text': 'ab'}
```

**必记四步**：`定义 State` → `add_node` → `add_edge` → `compile()`。


## 三、必记的 5 个核心概念

### 1. State（状态定义）
```python
from typing_extensions import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # ← reducer：追加而非覆盖
    query: str                               # ← 无 reducer：直接覆盖
    count: int
```
> **关键规则**：Node 返回的是**部分更新**（`Partial<State>`），不是完整 state。有 `Annotated` reducer 的字段按 reducer 合并，没有的直接**覆盖**。

**必记 reducer**：
| Reducer | 效果 |
|---|---|
| `operator.add` | 列表追加 |
| `add_messages` | 消息列表智能追加（LangChain 消息专用） |
| 无 | 直接覆盖（默认） |

### 2. Node（节点）
```python
def my_node(state: State) -> dict:
    # 读 state，做计算，返回要更新的字段
    return {"text": "new value"}   # ✅ 返回字典
    # return state                 # ❌ 不要返回完整 state
```
> 节点就是普通 Python 函数，可以是 LLM 调用、工具执行、任意代码。

### 3. Edge（边）
```python
# 固定边：A → B
graph.add_edge("node_a", "node_b")

# 条件边：根据 state 决定去哪
def should_continue(state: State) -> str:
    if state["count"] < 3:
        return "increment"
    return END

graph.add_conditional_edges("increment", should_continue)
```
> `should_continue` 返回的是**目标节点名**（字符串），不是状态更新。

### 4. START / END（虚拟节点）
```python
from langgraph.graph import START, END
graph.add_edge(START, "first_node")   # 入口
graph.add_edge("last_node", END)      # 出口
```

### 5. 编译（compile）
```python
app = graph.compile()                             # 基础编译
app = graph.compile(checkpointer=MemorySaver())   # 带记忆
```
> 编译会做结构检查（如孤立节点），并绑定运行时参数。


## 四、必记的循环模式（Agent 核心）

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # ← 必记：消息自动追加

def agent(state: AgentState) -> dict:
    # 调用 LLM
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:          # LLM 要调工具
        return "tools"
    return END                   # 没有工具调用 = 结束

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))       # 预构建工具节点
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")               # 工具执行后回到 agent → 形成循环

app = graph.compile()
```

**循环本质**：`agent → tools → agent → ... → END`，直到 LLM 不再调工具。


## 五、必记的持久化（Checkpointer + thread_id）

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver   # 开发用
# from langgraph.checkpoint.postgres import PostgresSaver  # 生产用

checkpointer = MemorySaver()                          # 内存版
app = graph.compile(checkpointer=checkpointer)

# 调用时必须传 thread_id
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [("user", "你好")]}, config)
result2 = app.invoke({"messages": [("user", "刚才说了啥")]}, config)  # ← 自动记得历史
```

**Checkpointer vs Store（记死区别）**：

| | Checkpointer | Store |
|---|---|---|
| 持久化对象 | 图状态快照 | 应用自定义 KV 数据 |
| 作用域 | 单个 thread | 跨 thread |
| 记忆类型 | 短期（对话连续性） | 长期（用户偏好、事实） |
| 访问方式 | `config` 传 `thread_id` | 节点内读写 |

> `MemorySaver` **重启就丢**，生产必须用 `SqliteSaver` 或 `PostgresSaver`。


## 六、必记的预构建组件

### ① ToolNode（工具执行节点）
```python
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools=[search, calculator])
graph.add_node("tools", tool_node)

# tools_condition：自动判断 LLM 是否要调工具
graph.add_conditional_edges("agent", tools_condition)
```
> `ToolNode` 自动解析 LLM 返回的 `tool_calls` 并执行，省去手写解析逻辑。

### ② create_react_agent（一行建 Agent）
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools, checkpointer=checkpointer)
result = agent.invoke({"messages": [("user", "北京天气")]}, config)
```
> 等价于上面手写的 `agent + tools + 循环`，适合快速原型。

### ③ create_supervisor（多 Agent 编排）
```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

research_agent = create_react_agent(model, tools=[search_tool], name="research")
writer_agent = create_react_agent(model, tools=[write_tool], name="writer")

workflow = create_supervisor(
    agents=[research_agent, writer_agent],
    model=model,
    prompt="你是主管，负责把任务分给 research 或 writer",
)
app = workflow.compile()
```
> **Supervisor 模式**：一个中心主管负责路由任务到专职 Agent。这是 LangGraph 多 Agent 最常用的模式。


## 七、必记的 3 个「防坑」模式

### ① 状态不更新的根因
```python
# ❌ 错误：直接修改 state 并返回
def bad_node(state: State) -> State:
    state["text"] = "new"    # ← 修改了引用，LangGraph 检测不到变化
    return state

# ✅ 正确：返回要更新的字段字典
def good_node(state: State) -> dict:
    return {"text": "new"}
```
> 这是最常见的坑：**节点必须返回字典，不能返回修改后的 state**。

### ② 并发更新冲突
```python
# 多个节点同时返回同一个 key 且没有 reducer → 抛 INVALID_CONCURRENT_GRAPH_UPDATE
# 解决：给该字段加 reducer
class State(TypedDict):
    results: Annotated[list, operator.add]   # ← 支持合并
```
> 扇出（一个节点连多个节点）时，多个节点同时写同一 key 会报错，除非有 reducer。

### ③ 循环要有出口
```python
# 条件边必须有路径能走到 END，否则图永远不会停止
def should_continue(state) -> str:
    if state["count"] >= max_steps:
        return END          # ← 必须有这个出口
    return "increment"
```


## 八、必记的人机交互（interrupt）

```python
from langgraph.types import interrupt, Command

def review_node(state):
    # 暂停，向人类展示内容
    human_input = interrupt({"question": "是否批准？", "data": state["draft"]})
    if human_input == "yes":
        return {"approved": True}
    return {"approved": False}

# 恢复执行
app.invoke(Command(resume="yes"), config)   # ← 用 Command 恢复
```
> `interrupt()` 类似 Python 的 `input()`，暂停图并向客户端展示信息，用 `Command(resume=...)` 恢复。**恢复时 thread_id 必须和中断时一致**。


## 九、必记的流式输出

```python
# 流式查看每一步
for event in app.stream({"messages": [("user", "你好")]}, config):
    print(event)   # 每个节点执行后的状态更新

# 只流式输出 token
for event in app.stream(input, config, stream_mode="messages"):
    print(event)
```


## 十、常见坑速查

| 现象 | 原因 | 解决 |
|---|---|---|
| State 不更新 | 节点返回了完整 state 而非 dict | 返回 `{"key": value}` |
| `INVALID_CONCURRENT_GRAPH_UPDATE` | 多节点并发写同一 key | 加 `Annotated` reducer |
| 图不停止 | 条件边没有到 `END` 的路径 | 加终止条件 |
| `MemorySaver` 重启丢数据 | 内存版不持久化 | 换 `SqliteSaver`/`PostgresSaver` |
| 找不到历史 | 没传 `thread_id` | `config={"configurable": {"thread_id": "..."}}` |
| `ImportError: TypeAdapter` | Pydantic v1/v2 冲突 | `pip install --upgrade pydantic>=2.5.0` |
| 节点名冲突 | 多 Agent 重名 | 每个 Agent 用唯一 `name` |
| `interrupt` 后 state 不变 | 中断时节点返回值不合并 | 恢复后在节点内重新计算 |
| TypedDict 不生效 | 用了 `typing.TypedDict` | 用 `typing_extensions.TypedDict` |


## 🎯 一句话总结必记核心

> **`StateGraph(State)` → `add_node(函数)` → `add_edge/add_conditional_edges` → `compile(checkpointer=...)`；节点返回字典增量，条件边返回节点名，循环靠条件边回到上游，持久化靠 checkpointer + thread_id。**

---

## 附：与前面 Agent 清单的对应关系

| 裸写 Agent | LangGraph 对应 |
|---|---|
| `messages` 列表手动管理 | `State` + `add_messages` reducer |
| 手写 for 循环 | 图结构 + 条件边自动循环 |
| 手写 `tool_calls` 解析 | `ToolNode` + `tools_condition` |
| 手写 if/else 路由 | `add_conditional_edges` |
| 手动存历史 | `Checkpointer` + `thread_id` |
| 手写多 Agent 调度 | `create_supervisor` |

**建议学习路径**：先用裸写 Agent 理解原理 → 再用 LangGraph 重构 → 最后用 `create_react_agent` / `create_supervisor` 提效。