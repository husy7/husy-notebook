---
title: "ReAct 智能体范式"
tags: [ReAct, AI, Agents]
date: 2026-08-13
---

# ReAct 智能体范式
 
> 通过“思考-行动-观察”循环，让 LLM 交替推理与调用外部工具，动态解决复杂多步任务。

## 核心原理和流程

> 简记：循-析-行-观：

```python
while current_step < max_steps:
    # ① 构造 Prompt
    # ② LLM 思考（生成 Thought）
    # ③ 解析输出（得到 Action）
    # ④ 执行动作（调用工具）
    # ⑤ 记录观察（Observation 反馈）
```

本质：**推理与行动交替，依赖环境反馈闭环**。解决了纯推理模型易幻觉、纯行动模型缺乏规划的问题。

```python
# 最小可运行骨架
while step < max_steps:
    prompt = format_prompt(history)
    thought, action = parse(llm(prompt))
    if action == "Finish": return answer
    observation = tool(action)
    history.append(observation)
```

## 易错点

>  **无限循环死锁**：工具返回空或无效结果，未设最大步数/终止条件 → 反复执行同一动作。  
   设置 `max_steps`，对 `Observation` 做有效性校验，空结果强制终止或切换策略。

>  **上下文 Token 爆炸**：历史记录无限增长，超出 LLM 窗口。  
   限制保留最近 N 步，或对历史做摘要/裁剪。

>  **早期错误累加放大**：单路径线性推理，无回溯纠错，一步错步步错。  
   在关键步骤加入人工确认或反思机制（如 Reflexion）。

>  **响应延迟高**：串行多步，每步都要 LLM 调用。  
   对简单任务直接单次推理，复杂任务才用 ReAct。

## 练习

- Q1：ReAct 三步循环核心是什么？  
  A1：Thought（思考）→ Action（行动）→ Observation（观察）。
- Q2：ReAct 与 CoT 最关键区别？  
  A2：CoT 闭卷纯推理，ReAct 开卷交互，可调用外部工具获取实时信息。
- Q3：ReAct 最常见致命陷阱？  
  A3：无限循环死锁、Token 爆炸、错误累加、高延迟。

## 知识关联

- 前置：LLM 基础、思维链（CoT）、工具调用/函数调用、提示工程
- 横向：CoT、Toolformer、Plan-and-Execute、Reflexion、LLMCompiler
- 进阶：Reflexion（自我纠错）、多智能体协作、强化学习、思维树（ToT）、MCTS

## 对比与选型

| 方案 | 核心思想 | 灵活性 | 稳定性 | Token消耗 | 最佳场景 |
|------|---------|--------|--------|-----------|----------|
| ReAct | 边思考边行动 | 高 | 中 | 低 | 探索型、不确定任务 |
| Plan-and-Execute | 先规划后执行 | 中 | 高 | 中 | 流程型、清晰 SOP |
| Reflexion | 反思+纠错 | 高 | 高 | 高 | 高精度要求任务 |
| LLMCompiler | 并行调度 | 中 | 中 | 低 | 可并行子任务、追求速度 |

**选型速查**：要灵活选 ReAct，要稳定选 Plan-and-Execute，要质量选 Reflexion，要速度选 LLMCompiler。

## 执行意图

- If 遇到多步推理失败 / 工具调用无结果 / 开放域路径不明确，then 使用 ReAct 三要素循环，并设置最大步数与兜底终止条件。
- If 准备一次性批量调用所有工具 / 闭卷纯推理不检索 / 无最大步数限制循环，then 停下来检查是否违背“推理与行动交替、依赖环境反馈”原则。

## 参考

- 流程图：[react.png](react.png)
- 代码示例：[ReAct_agent.py](ReAct_agent.py)
