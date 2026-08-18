---
title: "AutoGen：对话驱动的多智能体框架"
tags: [AutoGen, 智能体, 框架]
date: 2026-08-10
---

# AutoGen：对话驱动的多智能体框架

> 以"群聊对话"驱动多智能体协作，把复杂任务拆成不同角色的自动化对话。

## 核心原理和流程

> 简记：**定义角色 → 组建 Team → 群聊自治 → 终止条件收尾**。

AutoGen 架构核心：
- **autogen-core**：底层异步运行时（消息传递、模型交互）。
- **autogen-agentchat**：高级对话接口，封装常用智能体与 Team。
- **异步优先**：全程 `async/await`，等待 LLM 时不阻塞。

```python
#
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

#核心运行结构
async def run_software_development_team():
    """运行软件开发团队协作"""
    
    # 1. 初始化模型客户端
    model_client = create_openai_model_client()
    
    # 2. 创建所有智能体
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    user_proxy = create_user_proxy(model_client)
    
    # 3. 创建团队
    team = RoundRobinGroupChat(
        participants=[
            product_manager,
            engineer,
            code_reviewer,
            user_proxy
        ],
        termination_condition=TextMentionTermination("TERMINATE"),
        max_turns=20,
    )
    
    # 4. 定义任务
    task = """我们需要开发一个比特币价格显示应用，具体要求如下：

            核心功能：
            - 实时显示比特币当前价格（USD）
            - 显示24小时价格变化趋势（涨跌幅和涨跌额）
            - 提供价格刷新功能

            技术要求：
            - 使用 Streamlit 框架创建 Web 应用
            - 界面简洁美观，用户友好
            - 添加适当的错误处理和加载状态

            请团队协作完成这个任务，从需求分析到最终实现。"""
    
    # 5. 运行团队协作
    
    result = await Console(team.run_stream(task=task))
    
    # 6. 关闭模型客户端（释放资源）
    await model_client.close()
    
    return result

#成员定义参考
def create_user_proxy(model_client):
    """创建用户代理智能体"""
    system_message = """你是一个用户代理，代表用户与开发团队交互。

                        你的职责：
                        1. 提出用户的开发需求和期望
                        2. 对产品经理的方案提供用户视角的反馈
                        3. 对工程师的代码提供用户角度的建议
                        4. 最终确认功能是否符合预期

                        当你认为所有需求都满足，且代码质量达标时，请回复 "TERMINATE" 来结束整个开发流程。"""

    return AssistantAgent(
        name="UserProxy",
        model_client=model_client,
        system_message=system_message,
    )
```

**两类核心智能体**：
- `AssistantAgent`：由 LLM 驱动，负责"思考"（规划、写码、审查）。
- `UserProxyAgent`：不依赖 LLM，作为用户代言人 + 代码/工具执行器，并发 TERMINATE 终止信号。

## 易错点

> **角色发言顺序写错**：`participants` 列表顺序即发言顺序，放错会导致流程错乱（审查员先于工程师发言）。
> 严格按业务流程（需求→编码→审查→测试）排列。

> **无终止条件导致死循环**：只用 `max_turns` 兜底会烧 Token。
> 始终配 `TextMentionTermination`，让 `UserProxyAgent` 在验收后发 "TERMINATE"。

> **非 OpenAI 模型漏配 `model_info`**：用 DeepSeek/通义等模型时只传了 `api_key`/`base_url`，AutoGen 不知模型能力边界。
> 必须传 `model_info` 字典（function_calling、context_length、json_output 等）。

> **对话偏离主题难调试**：LLM 回复不确定，得到的是一长串对话而非错误堆栈。
> 用强约束系统消息 + 终止条件；关键节点加人工确认（Human-in-the-loop）。

## 练习

- Q1：`AssistantAgent` 和 `UserProxyAgent` 的核心分工是什么？
  A1：`AssistantAgent` 依赖 LLM 负责"思考"（规划/写码/审查）；`UserProxyAgent` 不依赖 LLM，代表用户发起任务、执行代码/工具并发 TERMINATE 终止信号。

- Q2：`RoundRobinGroupChat` 中 `participants` 列表顺序有何影响？
  A2：列表顺序就是发言顺序。轮询群聊会按顺序依次激活每个智能体，顺序错误会破坏业务流程。

- Q3：使用 DeepSeek 等非 OpenAI 模型时需要额外配置什么？
  A3：必须传入 `model_info` 字典，声明 function_calling、max_tokens、context_length、vision、json_output、family 等能力，框架才能正确适配。

## 知识关联

- 前置：[[ReAct 智能体范式]]、提示工程、async/await 异步编程
- 横向：[[AgentScope 工程化多智能体平台]]、[[CAMEL 角色扮演协作框架]]、[[LangGraph 图结构工作流框架]]、CrewAI
- 进阶：SelectorGroupChat（动态选人）、多智能体调试与可观测性

## 对比与选型

| 方案 | 核心思想 | 灵活性 | 稳定性 | 最佳场景 |
|------|---------|--------|--------|----------|
| AutoGen | 群聊对话驱动 | 高 | 中 | 流程化多角色协作（如软件开发团队） |
| AgentScope | 消息驱动 + 分布式 | 中 | 高 | 高并发/分布式生产级系统 |
| CAMEL | 角色扮演 + 引导提示 | 高 | 中 | 双专家深度协作 |
| LangGraph | 状态机 + 有向图 | 中 | 高 | 精确流程控制 + 循环反思 |

**选型速查**：要自然对话协作选 AutoGen，要分布式高并发选 AgentScope，要轻量双智能体协作选 CAMEL，要精确流程控制选 LangGraph。

## 执行意图

- If 我要构建**流程固定、角色明确**的多智能体协作（如软件开发流水线），then 选 AutoGen 的 `RoundRobinGroupChat` + 强约束系统消息。
- If 我准备不加 `max_turns` 和终止条件就跑群聊，then 停下来先加安全阀，防止死循环烧 Token。

## 参考

- [AutoGen 官方文档](https://microsoft.github.io/autogen/)
- 代码示例见仓库 `code/` 目录下的 AutoGen 案例
