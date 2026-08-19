---
title: "Plan-and-Solve 范式"
tags: [Plan-and-Solve, AI, Agents]
date: 2026-08-19
---

# Plan-and-Solve 范式

> 先让 LLM 生成完整行动计划，再逐步执行--"三思而后行"，适合结构化、可分解的复杂任务。

## 核心原理和流程

> 简记：谋-行（先蓝图后施工）

本质：**规划与执行解耦**。ReAct 边想边干，Plan-and-Solve 先出完整计划再严格执行，保证全局目标一致性，避免中间步骤迷失。

```
规划阶段：P = π_plan(q)                          # 一次性生成 n 步计划列表
执行阶段：s_i = π_solve(q, P, (s_1,...,s_{i-1}))  # 逐步执行，依赖历史结果
最终答案：s_n
```

```python
# 最小可运行骨架
class PlanAndSolveAgent:
    def run(self, question):
        plan = self.planner.plan(question)      # ① 规划：LLM 输出 Python 列表
        if not plan: return
        return self.executor.execute(question, plan)  # ② 执行：逐步求解

class Executor:
    def execute(self, question, plan):
        history = ""
        for i, step in enumerate(plan):
            prompt = EXECUTOR_PROMPT.format(
                question=question, plan=plan,
                history=history or "无", current_step=step
            )
            result = llm(prompt)               # 每步独立 LLM 调用
            history += f"步骤{i+1}: {step}\n结果: {result}\n\n"  # 状态传递
        return result                           # 最后一步即最终答案
```

### 规划器提示词（关键设计）

```python
PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。
请严格按照以下格式输出,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", ...]
```
"""
# 解析：plan = ast.literal_eval(response.split("```python")[1].split("```")[0])
```

强制输出 Python 列表格式 -> 用 `ast.literal_eval` 安全解析，比解析自然语言稳定得多。

### 执行器提示词

包含四要素：原始问题（全局目标）+ 完整计划（定位当前步）+ 历史结果（上下文传递）+ 当前步骤（聚焦执行）。提示词要求"仅输出当前步骤答案"，防跑题。

## 易错点

> **静态计划无法适应变化**：计划一次生成不可改 -> 某步失败或结果不符预期，全盘崩溃。
> 设计"动态重规划"：执行中检测步骤失败时，将已完成步骤+失败原因回传 LLM，重新生成剩余计划（见习题4-1）。

> **规划解析失败**：LLM 未输出 ` ```python ` 标记或列表格式错误 -> `ast.literal_eval` 抛异常。
> try-except 捕获 `ValueError/SyntaxError/IndexError`，打印原始响应用于调试，返回空列表降级。

> **步骤间状态丢失**：执行器未将历史结果传入下一步 -> 每步独立计算，结果断裂。
> 维护 `history` 字符串，每步将"步骤+结果"追加，作为下一步的上下文。

> **粒度过粗或过细**：计划步骤太少 -> 仍是复杂问题；太多 -> 冗余调用、延迟增加。
> 规划提示词中约束"独立、可执行的子任务"，必要时用分层规划（高层粗粒度+每步再细拆）。

## 练习

- Q1：Plan-and-Solve 与 ReAct 在"思考与行动组织方式"上的本质区别？
  A1：ReAct 交错式（每步都依赖外部反馈，动态调整）；Plan-and-Solve 串行分离式（先全量规划再执行，执行时不思考）。前者灵活但不稳定，后者稳定但不适应变化。

- Q2：（章末习题4-2）预订"北京->上海商务旅行（机票+酒店+租车）"，选哪种范式？
  A2：Plan-and-Solve 更合适。该任务结构清晰、可分解（机票/酒店/租车是独立子任务），需要全局规划保证行程一致性。但若子任务中需实时查询（如比价），可在执行阶段内嵌 ReAct。

- Q3：（章末习题4-3）分层规划的优势？
  A3：先生成高层次抽象计划，再对每步生成详细子计划。优势：① 复杂任务分层降维，每层聚焦适合粒度；② 高层计划稳定，底层可灵活调整；③ 模拟人类"粗到细"思维，减少单次 LLM 认知负荷。

- Q4：（章末习题4-1）如何设计"动态重规划"机制？
  A4：执行器检测步骤失败（结果校验不通过/工具异常）时，暂停执行，将已完成步骤结果+失败原因+原始计划回传规划器，生成修订后的剩余计划，从失败点继续执行。

## 知识关联

- 前置：[[ReAct 智能体范式]]、思维链（CoT）、提示工程
- 横向：Plan-and-Execute（HuggingGPT）、CoT、ReWOO、LLMCompiler
- 进阶：[[Reflection 机制]]（可叠加在执行后做迭代优化）、动态重规划、分层任务网络（HTN）

## 对比与选型

| 方案 | 核心思想 | 灵活性 | 稳定性 | Token消耗 | 最佳场景 |
|------|---------|--------|--------|-----------|----------|
| Plan-and-Solve | 先规划后执行 | 中 | 高 | 中 | 结构化、可分解任务 |
| ReAct | 边思考边行动 | 高 | 中 | 低 | 探索型、不确定任务 |
| Reflection | 执行-反思-优化 | 高 | 高 | 高 | 高质量要求任务 |

**选型速查**：要稳定选 Plan-and-Solve，要灵活选 ReAct，要质量加 Reflection 叠层。复杂任务可混合：Plan 拆任务 -> ReAct 执行 -> Reflection 优化。

## 执行意图

- If 遇到结构清晰、可分解的多步任务（数学题、报告撰写、代码生成），then 优先用 Plan-and-Solve 先规划后执行。
- If 执行中发现某步失败 / 计划与现实不符，then 触发动态重规划，而非盲目继续执行静态计划。

## 费曼解释

> 像盖房子：先画完整蓝图（规划阶段），再按图纸一步步施工（执行阶段）。不会盖到一半发现图纸错了就傻眼--所以重要项目要留"改图纸"的余地（动态重规划）。

## 参考

- 论文：Wang L, et al. "Plan-and-Solve Prompting." arXiv:2305.04091, 2023.
