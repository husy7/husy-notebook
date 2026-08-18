import json
from openai import OpenAI
from typing import List, Dict, Any, Callable, Union
from dotenv import load_dotenv
load_dotenv()
import os
class ToolCallingAgent:
    def __init__(
        self,
        tools: List[Dict[str, Any]],
        tool_functions: Dict[str, Callable],
        model: str = os.getenv("OPENAI_MODEL"),
        base_url: str = "https://api.openai.com/v1",
        api_key: str = None,
        tool_choice: Union[str, Dict] = "auto",
        max_iterations: int = 5,
        **client_kwargs
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key, **client_kwargs)
        self.model = model
        self.tools = tools
        self.tool_functions = tool_functions
        self.tool_choice = tool_choice
        self.max_iterations = max_iterations

    def _call_llm(self, messages):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )

    def run(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        iteration = 0

        while iteration < self.max_iterations:
            response = self._call_llm(messages)
            response_message = response.choices[0].message
            messages.append(response_message.model_dump())

            if not response_message.tool_calls:
                return response_message.content or "（无内容）"

            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                func = self.tool_functions.get(func_name)
                if func is None:
                    result = f"未找到函数 '{func_name}'"
                else:
                    try:
                        result = func(**func_args)
                    except Exception as e:
                        result = f"执行出错：{e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

            iteration += 1

        return messages[-1].get("content", "超过最大迭代次数")


# 1. 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名，如 'Beijing'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

# 2. 实现工具函数
def get_current_weather(location: str, unit: str = "celsius") -> str:
    # 模拟返回，实际可调用天气 API
    return f"{location} 当前天气：25°C，晴"

tool_funcs = {"get_current_weather": get_current_weather}

# 3. 创建 Agent（替换您的 API KEY 和 BASE URL）
agent = ToolCallingAgent(
    tools=tools,
    tool_functions=tool_funcs,
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com/v1",   # 如果是 OpenAI 官方，则用默认值
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 4. 运行
answer = agent.run("北京今天天气怎么样？")
print(answer)