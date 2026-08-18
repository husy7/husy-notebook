---
title: "CAMEL：角色扮演协作框架"
tags: [CAMEL, 多智能体, 角色扮演, 框架]
date: 2026-08-14
---

# CAMEL：角色扮演协作框架

> 用"角色扮演 + 引导性提示"让两个 AI 专家自主对话协作，轻架构重提示。

## 核心原理和流程

> 简记：**互补角色 + 引导提示 = 自主协作**。

两大核心概念：

**① 角色扮演（Role-Playing）**：一个任务由两个互补角色协作完成：
- **AI User**（需求方/推动者）：提需求、下指令、构思步骤。
- **AI Assistant**（执行方/方案提供方）：根据指令执行操作、给方案。

> 例：交易员（不懂编程）+ 程序员（不懂交易）协作开发交易工具。

**② 引导性提示（Inception Prompting）**：对话前注入的结构化初始指令，含四要素：
1. 明确自身角色
2. 告知协作者角色
3. 定义共同目标
4. **设定行为约束与沟通协议**（如"一次只提一个步骤"、`<SOLUTION>` 标志）

```python
from camel.societies import RolePlaying

# 一行初始化双智能体协作"社会"
session = RolePlaying(
    assistant_role_name="心理学家",   # 执行方：提供专业知识
    user_role_name="作家",           # 需求方：规划结构、提写作要求
    task_prompt="创作拖延症心理学科普电子书...",
    model=model,
    with_task_specify=False,         # 是否让 AI 细化任务
)

# 循环驱动协作
input_msg = session.init_chat()     # 自动生成开场白
while n < chat_turn_limit:
    assistant_resp, user_resp = session.step(input_msg)
    if "<CAMEL_TASK_DONE>" in user_resp.msg.content \
       or "<CAMEL_TASK_DONE>" in assistant_resp.msg.content:
        break
    input_msg = assistant_resp.msg   # 环环相扣
```

**协作四阶段自然涌现**：
1. 框架搭建与目标对齐（1-5 轮）
2. 核心内容生成与知识转译（6-20 轮）
3. 迭代优化与质量保证（21-25 轮）
4. 总结与升华（收尾）

## 易错点

> **引导性提示写得太松**：没设"一次只提一个步骤""完成标志"等约束 → 对话跑题或陷入循环。
> Inception Prompting 的行为约束是协作成功的关键，必须明确沟通协议。

> **角色分配搞反**：把"方案提供方"放成 `user_role_name`。
> `user` 是对话推动者/需求方，`assistant` 是执行者/方案方，反了会变成被需求驱动。

> **双智能体意见分歧无法终止**：一方想结束一方不想，卡死。
> 设计冲突解决机制：设定轮次上限强制终止，或引入第三方仲裁。

> **协作规模超出双智能体设计**：CAMEL 原生为双智能体优化，大规模多智能体缺乏路由/状态同步。
> 大规模场景用 workforce 模块或转 AutoGen/AgentScope。

## 练习

- Q1：CAMEL 的"引导性提示"包含哪四个关键部分？
  A1：①明确自身角色 ②告知协作者角色 ③定义共同目标 ④设定行为约束和沟通协议（如一次一个步骤、`<SOLUTION>` 完成标志）。

- Q2：`user` 角色和 `assistant` 角色的分工是什么？
  A2：`user`（AI User）是需求方/推动者，提需求下指令；`assistant`（AI Assistant）是执行方/方案提供方，根据指令执行操作。

- Q3：CAMEL 最大的局限性是什么？
  A3：高度依赖提示工程质量（提示差则协作差）；双智能体设计，大规模协作缺路由/状态同步/冲突仲裁机制。

## 知识关联

- 前置：[[ReAct 智能体范式]]、提示工程、[[ReAct 智能体范式|思维链 CoT]]
- 横向：[[AutoGen 对话驱动多智能体框架]]、[[AgentScope 工程化多智能体平台]]、[[LangGraph 图结构工作流框架]]、CrewAI
- 进阶：CAMEL `workforce` 多智能体模块、多模态协作、生态联动（LangChain 互操作）

## 对比与选型

| 方案 | 核心思想 | 灵活性 | 稳定性 | 最佳场景 |
|------|---------|--------|--------|----------|
| CAMEL | 角色扮演 + 引导提示 | 高 | 中 | 双专家深度协作（创作/研究） |
| AutoGen | 群聊对话驱动 | 高 | 中 | 流程化多角色协作 |
| AgentScope | 消息驱动 + 分布式 | 中 | 高 | 高并发/分布式系统 |
| LangGraph | 状态机 + 有向图 | 中 | 高 | 精确流程控制 |

**选型速查**：要轻量双专家深度协作选 CAMEL；要多角色流程协作选 AutoGen；要分布式高并发选 AgentScope；要精确流程控制选 LangGraph。

## 执行意图

- If 我要构建**两个跨领域专家深度协作**的创意/研究任务（如写书、方案设计），then 选 CAMEL 的角色扮演范式，重点打磨引导性提示。
- If 我准备做大规模多智能体协作却只用 CAMEL 双智能体，then 停下来，考虑 workforce 或换框架。

## 参考

- [CAMEL 官方文档](https://docs.camel-ai.org/)
- [CAMEL 论文 (NeurIPS 2023)](https://arxiv.org/abs/2303.17760)
- 代码示例见仓库 `code/` 目录下的 AI 科普电子书案例
