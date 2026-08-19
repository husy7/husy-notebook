---
title: "Reflection 机制"
tags: [Reflection, AI, Agents]
date: 2026-08-19
---

# Reflection 机制

> 通过"执行-反思-优化"迭代循环，让智能体审视自身产出并自我修正，持续提升结果质量。

## 核心原理和流程

> 简记：执-反-优（→ 再循环）

> [核心流程图](reflection.png)

本质：**事后自检 + 迭代逼近**。前两种范式产出即终止，Reflection 引入"评审员"角色形成内部纠错回路，不依赖外部工具反馈即可修正高层逻辑错误。

```python
# 最小可运行骨架
def run(task):
    code = llm(initial_prompt(task))          # ① 执行：生成初稿
    memory.add("execution", code)
    for i in range(max_iterations):
        feedback = llm(reflect_prompt(code))  # ② 反思：评审员批判
        memory.add("reflection", feedback)
        if "无需改进" in feedback: break       # 终止条件
        code = llm(refine_prompt(code, feedback))  # ③ 优化：按反馈修正
        memory.add("execution", code)
    return code
```

形式化：`F_i = π_reflect(Task, O_i)`；`O_{i+1} = π_refine(Task, O_i, F_i)`，循环至收敛。

### Memory 模块：迭代的前提

Reflection 的核心是迭代，迭代的前提是**记住历史轨迹**。Memory 类负责存储每次执行-反思记录，并序列化为提示词上下文：

```python
class Memory:
    def __init__(self):
        self.records = []                     # 按序存储 execution/reflection
    def add_record(self, type, content):
        self.records.append({"type": type, "content": content})
    def get_trajectory(self) -> str:          # 序列化为文本注入提示词
        # 将所有记录拼成 "--- 上一轮尝试 ---\n... \n\n --- 评审员反馈 ---\n..."
    def get_last_execution(self) -> str:      # 取最新初稿供反思
        ...
```

### 三套提示词协同

| 提示词 | 角色 | 目标 |
|--------|------|------|
| INITIAL_PROMPT | 资深程序员 | 首次生成代码（初稿） |
| REFLECT_PROMPT | 极其严格的代码评审专家 | 找算法效率瓶颈，给改进建议 |
| REFINE_PROMPT | 资深程序员 | 按反馈修正代码 |

反思提示词的角色设定（"极其严格""对性能有极致要求"）直接决定反思深度，是 Reflection 成败关键。

## 易错点

> **反思沦为"走过场"**：反思提示词角色设定太温和 -> 评审员只说"代码看起来不错"，迭代空转。
> 用强烈角色设定（"极其严格""专注算法效率"）+ 明确反思维度（时间复杂度/逻辑漏洞/边界情况）。

> **无终止条件死循环**：未设 `max_iterations` 或"无需改进"判断 -> 成本无上限。
> 设置 `max_iterations`（如 3 轮），并在反馈含"无需改进"时提前 break。

> **串行延迟高**：每轮需 2 次 LLM 调用（反思+优化），N 轮 = 2N+1 次调用。
> 仅用于对质量要求高、对实时性宽松的场景；快速响应场景选 ReAct/Plan-and-Solve。

> **反思用同一模型易"自我确认偏误"**：执行和反思用同一模型 -> 自己审自己，难发现盲点。
> 用更强的模型做反思、更快的模型做执行（见习题5）。

## 练习

- Q1：Reflection 三步循环是什么？与前两种范式的本质区别？
  A1：执行→反思→优化。区别：ReAct/Plan-and-Solve 产出即终止，Reflection 引入事后自检迭代回路，可修正高层逻辑错误，并形成"短期记忆"轨迹。

- Q2：（章末习题5改编）执行和反思用同一模型有什么风险？用不同模型会怎样？
  A2：同模型有"自我确认偏误"，难发现自己的盲点。用更强的模型做反思能发现更深问题，用更快的模型做执行可降低成本——但需权衡模型间风格差异导致的反馈可执行性。

- Q3：（章末习题5）当前终止条件（"无需改进"或最大迭代次数）是否合理？如何改进？
  A3：基本合理但有局限："无需改进"依赖模型自判，可能过早收敛或误判。可改进：引入量化指标（如测试通过率、复杂度数值），设最小迭代轮数防过早停止，或用外部验证器（如实际运行代码）做客观终止判断。

- Q4：（章末习题5）如何为"学术论文写作助手"设计多维度 Reflection？
  A4：拆分多个并行反思器，各负责一个维度（段落逻辑性/方法创新性/语言表达/引用规范），分别产出反馈；优化阶段加权合并所有反馈统一修订。或串行多轮，每轮只聚焦一个维度深入优化。

## 知识关联

- 前置：[[ReAct 智能体范式]]、[[Plan-and-Solve 范式]]、提示工程、LLM 基础
- 横向：Reflexion 框架（Shinn 2023）、Self-Refine、CoT、Tree of Thoughts
- 进阶：多智能体辩论（Multi-Agent Debate）、多模态 Reflection、强化学习反馈

## 对比与选型

| 方案 | 核心思想 | 灵活性 | 稳定性 | Token消耗 | 最佳场景 |
|------|---------|--------|--------|-----------|----------|
| Reflection | 执行-反思-优化迭代 | 高 | 高 | 高（2N+1次调用） | 高质量代码/报告、逻辑推演 |
| ReAct | 边思考边行动 | 高 | 中 | 低 | 探索型、不确定任务 |
| Plan-and-Solve | 先规划后执行 | 中 | 高 | 中 | 流程型、清晰 SOP |

**选型速查**：要质量选 Reflection，要灵活选 ReAct，要稳定选 Plan-and-Solve。三者可组合：Plan 拆任务 → ReAct 执行 → Reflection 优化。

## 成本收益分析

| 维度 | 成本 | 收益 |
|------|------|------|
| 调用开销 | 每轮 +2 次调用，N 轮 = 2N+1 次 | — |
| 延迟 | 串行，总耗时 ≈ 单次 ×(2N+1) | — |
| 提示工程 | 需为执行/反思/优化分别设计提示词 | — |
| 质量 | — | 合格初稿 → 优秀终稿的阶梯跃迁 |
| 可靠性 | — | 自我纠错修复逻辑漏洞、边界情况 |

**适用**：关键业务代码、技术报告、科学推演、决策支持——对质量要求高、实时性宽松。
**不适用**：快速响应、"大致正确"即可的场景——用 ReAct/Plan-and-Solve 更具性价比。

## 执行意图

- If 遇到代码生成质量不达标 / 初始方案有明显性能或逻辑缺陷 / 需要高可靠性产出，then 启用 Reflection 循环，配强角色设定的反思提示词。
- If 准备让同一模型自审自身输出而不设独立反思角色，then 停下来检查是否会产生"自我确认偏误"，考虑换更强模型做反思。

## 费曼解释

> 就像写作文：先写初稿（执行），再请一位很严格老师批改（反思），根据老师意见改一版（优化），然后再请老师看……直到老师说"挺好的不用改了"为止。每次改完都比上一版更好。

## 参考

- 流程图：[reflection.png](reflection.png)
- 代码：[reflection_.py](reflection_.py)、[memory.py](memory.py)、[_prompt.py](_prompt.py)
- 论文：Shinn N, et al. "Reflexion: Language agents with verbal reinforcement learning." NeurIPS 2023.
