# ============================================================
# 一个能调用工具的 AI 助手（最小示例）
#
# 整体流程：
#   用户提问 → AI 判断要不要用工具 → 用工具执行 → 把结果告诉 AI
#   → AI 给出最终回答
# 这个「问-做-答」循环会一直重复，直到 AI 不再需要工具为止。
# ============================================================

# ---------- 1. 导入需要的工具包 ----------

from openai import OpenAI                      # OpenAI 官方 SDK，用来和模型对话
from dotenv import load_dotenv                 # 从 .env 文件读取密钥等配置
from openai.types.chat import (                # 从 SDK 里导入类型定义（仅用于类型提示，帮助写代码时自动补全）
    ChatCompletion,                            # 「一次模型回复」的类型
    ChatCompletionMessageFunctionToolCall,     # 「一次工具调用」的类型
    ChatCompletionMessageParam,                # 「一条对话消息」的类型
    ChatCompletionToolParam,                   # 「一个工具定义」的类型
)
from typing import List, cast                  # List 用于类型提示；cast 用于告诉类型检查器「我确定这是这个类型」

load_dotenv()                                  # 读取项目根目录的 .env 文件，把里面的配置加载到环境变量

import os                                      # 标准库，用来读环境变量
import json                                    # 标准库，用来把 JSON 字符串解析成 Python 字典


# ---------- 2. 创建与模型的连接 ----------

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),          # 从环境变量拿 API 密钥（不硬编码在代码里，更安全）
    base_url=os.getenv("LLM_BASE_URL"),        # 从环境变量拿服务地址（可以指向 OpenAI、也可以指向本地模型）
)


# ---------- 3. 给 AI 的「人设」设定 ----------

SYS_PROMPT = """你是一个可以调用工具的助手"""      # 这段文字会先发给模型，告诉它自己是什么角色


# ---------- 4. 定义「工具」：告诉 AI 它能做什么 ----------
#
# 这里的结构分三层：
#   最外层 {"type": "function", "function": {...}} —— OpenAI 要求的固定格式
#   function 里装名字、描述、参数说明
#   parameters 用 JSON Schema 描述「参数长什么样」
#
# 就像给 AI 一份「菜单」，告诉它：
#   - 有个工具叫 bash（名字）
#   - 它能执行命令行（描述）
#   - 需要传入一个叫 command 的字符串（参数）

TOOLS: list[ChatCompletionToolParam] = [{
    "type": "function",                        # 固定值：表示这是一个「函数工具」
    "function": {                              # 具体的函数定义
        "name": "bash",                        # 工具名，AI 会用它来「点名」调用
        "description": "执行一条 bash 命令，返回 stdout 与 stderr。",  # 告诉 AI 这个工具用来干嘛
        "parameters": {                        # 描述这个工具需要什么参数
            "type": "object",                  # 参数整体是一个「对象」（键值对）
            "properties": {                    # 列出每个参数的细节
                "command": {                   # 参数名叫 command
                    "type": "string",          # 它是字符串
                    "description": "要执行的命令"  # 说明这个参数该填什么
                }
            },
            "required": ["command"]            # command 是必填的，不能省略
        }
    }
}]


# ---------- 5. 工具的真正实现：把命令交给操作系统去跑 ----------

def run_bash(command: str) -> str:
    """执行一条 bash 命令，把结果当作字符串返回。"""

    import subprocess                          # 标准库：用来启动外部程序（这里是 shell 命令）

    try:
        r = subprocess.run(
            command,                           # 要执行的命令，例如 "dir G:\\" 或 "ls"
            shell=True,                        # 通过 shell 执行（这样才能用管道、通配符等）
            capture_output=True,               # 把命令的输出「抓」到变量里，而不是直接打印到屏幕
            timeout=60,                        # 最多跑 60 秒，超时就掐掉（防止卡死）
            encoding="utf-8",                  # 用 UTF-8 解码输出（避免中文乱码）
            errors="replace",                  # 遇到无法解码的字节用 � 替代，绝不抛异常
        )
        # r.stdout 是标准输出，r.stderr 是错误输出；用 or "" 兜底防止 None
        out = (r.stdout or "") + (r.stderr or "")
        return out or "(no output)"            # 如果什么都没输出，返回一句提示

    except subprocess.TimeoutExpired:          # 命令超过 60 秒没跑完
        return "[error] command timed out after 60s"

    except Exception as e:                     # 其它任何错误（命令不存在、权限不足等）
        return f"[error] {type(e).__name__}: {e}"


# ---------- 6. 核心：AI 与工具之间的「对话循环」 ----------

def agent_loop() -> str:
    """反复问 AI，直到它给出最终答案，再返回答案。"""

    # messages 是整个对话的历史记录，模型每轮都会看到完整历史
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "你是助手"},          # 第一句：给 AI 的人设
        {"role": "user",   "content": "列出当前目录"},    # 第二句：用户的问题
    ]

    while 1:                                    # 无限循环，直到 AI 说「我不需要工具了」才退出
        # ------ 6.1 把当前对话历史 + 可用工具一起发给模型 ------
        response = client.chat.completions.create(
            model="qwen3.5:4b",                 # 用哪个模型
            messages=messages,                  # 对话历史
            tools=TOOLS,                        # 告诉模型它有哪些工具
            tool_choice="auto",                 # 让模型自己决定要不要用工具
            reasoning_effort=None,              # 该参数部分模型支持，这里传 None 表示不启用
        )

        # ------ 6.2 取出模型这一轮说的内容 ------
        # 有两种情况：
        #   A) 模型直接给出文字回答（content 有内容，tool_calls 为空）
        #   B) 模型请求调用工具（content 为空，tool_calls 有内容）
        message = response.choices[0].message

        # 把模型这条消息存进历史（含 tool_calls 信息，下一轮模型才能看到）
        # exclude_none=True 表示去掉值为 None 的字段，让消息更干净
        messages.append(
            cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True))
        )

        # ------ 6.3 如果模型没有请求调用工具 → 它就是最终答案 ------
        if not message.tool_calls:
            return message.content or ""        # 返回文字内容，空则返回空串

        # ------ 6.4 否则，逐个执行模型请求的工具调用 ------
        for tool_call in message.tool_calls:
            # 类型检查：只处理「函数工具调用」，其它类型跳过（防御性编程）
            if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                continue

            # 模型给出的参数是 JSON 字符串，解析成 Python 字典，例如 {"command": "dir G:\\"}
            args = json.loads(tool_call.function.arguments)

            # 真正执行命令，得到输出字符串
            output = run_bash(args["command"])

            # 把工具的执行结果作为一条新消息追加到历史里
            # tool_call_id 是「回执单」——告诉模型：这次的结果对应你刚才那次调用
            messages.append(cast(ChatCompletionMessageParam, {
                "role": "tool",                 # 角色是「工具」，表示这不是用户也不是 AI 说的
                "tool_call_id": tool_call.id,   # 对应哪一次工具调用
                "content": output,              # 工具执行的结果
            }))

        #print(messages)

        # 循环回到 6.1：把「工具执行结果」再发给模型
        # 模型看到结果后，要么继续调工具，要么给出最终回答


# ---------- 7. 程序入口 ----------

if __name__ == "__main__":                      # 只有直接运行这个文件时才执行
    print(agent_loop())                               # 启动对话循环


# ============================================================
#
#   这个程序把「用户的问题」和「AI 能用的工具清单」一起交给 AI，
#   AI 说「我想执行一条命令」，
#   程序就真的去执行，
#   把结果告诉 AI，
#   AI 再决定：还要不要继续做？还是可以直接回答用户了？
#   如此反复，直到 AI 给出最终答案。
# ============================================================