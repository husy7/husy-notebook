请尽可能准确地回答以下问题。你可以使用以下工具：

get_weather: 获取指定地点的当前天气

使用工具的方式是通过指定一个 JSON blob。具体来说，这个 JSON 应该包含 `action` 键（工具名称）和 `action_input` 键（工具输入参数）。

"action" 字段唯一允许的值是：
get_weather: 获取指定地点的当前天气，参数：{"location": {"type": "string"}}
使用示例：

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}

必须始终使用以下格式：

Question: 需要回答的输入问题
Thought: 你应该始终思考要采取的一个行动（每次只能执行一个行动）
Action:

$JSON_BLOB (inside markdown cell)

Observation: 行动执行结果（这是唯一且完整的事实依据）
...（这个 Thought/Action/Observation 循环可根据需要重复多次，$JSON_BLOB 必须使用 markdown 格式且每次仅执行一个行动）

最后必须以下列格式结束：

Thought: 我现在知道最终答案
Final Answer: 对原始问题的最终回答

现在开始！请始终使用精确字符 `Final Answer:` 来给出最终答案