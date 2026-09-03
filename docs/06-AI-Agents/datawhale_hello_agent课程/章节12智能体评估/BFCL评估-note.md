---
title: "BFCL 工具调用能力评估笔记"
tags: [BFCL, 工具调用, 函数调用, AST匹配, 评估基准]
date: 2026-09-03
---

# BFCL 工具调用能力评估笔记

## 1. 概述
- **全称**：Berkeley Function Calling Leaderboard，由 UC Berkeley 推出。
- **评估目标**：智能体的**函数调用（工具调用）能力**。
- **核心任务**：
  - 从自然语言请求中提取关键信息
  - 从可用工具集中选择正确工具
  - 正确构造函数调用（函数名 + 参数）
  - 处理多函数调用、并行调用、无关调用等复杂场景

## 2. 数据集结构
- **来源**：官方 GitHub 仓库 `ShishirPatil/gorilla` 下的 `berkeley-function-call-leaderboard`。
- **样本字段**：
  - `id`：样本唯一标识
  - `question`：用户自然语言问题
  - `function`：可用函数列表（含函数名、描述、参数 schema）
  - `ground_truth`：标准答案（期望的函数调用，含函数名和参数）
- **四个评估类别**（难度递增）：
  1. **Simple**：单函数调用
  2. **Multiple**：顺序调用多个函数
  3. **Parallel**：并行调用多个函数
  4. **Irrelevance**：判断是否需要调用函数（可能不需要）

## 3. 核心评估算法：AST 匹配
- **为什么不用字符串匹配**？字符串匹配过于严格，无法处理参数顺序、等价表达式、格式差异。
- **AST 匹配原理**：
  - 将预测的函数调用和标准答案分别解析为抽象语法树（AST）
  - 比较两棵语法树的结构和节点值是否等价
- **等价条件**：
  - 函数名完全一致（精确匹配）
  - 参数键值对集合相等（忽略顺序）
  - 参数值语义等价（如 `2+3` 等价于 `5`，`"hello"` 等价于 `'hello'`）
- **多函数调用**：要求调用相同数量的函数，每个函数调用匹配，但顺序可不同（集合匹配）。

## 4. 评估指标
- **准确率（Accuracy）**：AST 匹配成功的样本比例，最核心指标。
- **AST 匹配率**：与准确率相同，强调使用 AST 算法。
- **分类准确率（Category-wise Accuracy）**：每个类别单独计算准确率。
- **加权准确率（Weighted Accuracy）**：按类别权重加权平均。
- **错误率（Error Rate）**：1 - Accuracy。

## 5. 在 HelloAgents 中使用 BFCL 评估
提供三种使用方式：

### 方式一：使用 `BFCLEvaluationTool`（推荐，一键评估）
```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import BFCLEvaluationTool

llm = HelloAgentsLLM()
agent = SimpleAgent(name="TestAgent", llm=llm)
bfcl_tool = BFCLEvaluationTool()

results = bfcl_tool.run(
    agent=agent,
    category="simple_python",   # 评估类别
    max_samples=5               # 样本数，0表示全部
)
print(f"准确率: {results['overall_accuracy']:.2%}")
```
- 自动完成：加载数据 → 运行评估 → 导出官方格式 → 运行官方评估 → 生成报告。

### 方式二：命令行脚本
```bash
python chapter12/04_run_bfcl_evaluation.py --category simple_python --samples 10 --model-name "Qwen/Qwen3-8B"
```

### 方式三：直接使用 `BFCLDataset` 和 `BFCLEvaluator`
```python
from hello_agents.evaluation import BFCLDataset, BFCLEvaluator

dataset = BFCLDataset(bfcl_data_dir="./.../bfcl_eval/data", category="simple_python")
data = dataset.load()
evaluator = BFCLEvaluator(dataset=dataset, category="simple_python")
results = evaluator.evaluate(agent, max_samples=10)
```

## 6. 核心组件实现要点
- **BFCLDataset**：负责加载数据，支持本地路径和 HuggingFace。
- **BFCLEvaluator**：
  - `_build_prompt()`：构造包含问题和函数定义的提示词
  - `_extract_function_calls()`：从模型响应中提取函数调用，支持 JSON、代码块、纯文本格式
  - `_compare_calls()`：调用 AST 匹配进行对比
- **BFCLMetrics**：计算准确率、AST匹配率、参数准确率、F1 等。
- **AST 匹配实现**：将参数字典转为虚拟函数调用代码，用 `ast.parse` 解析，再 `ast.dump` 比较。

## 7. 扩展与优化建议
- **当前局限**：SimpleAgent 使用自定义工具调用格式 `[TOOL_CALL:...]`，在复杂场景可能不如原生 Function Calling。
- **提升方向**：
  - 使用支持原生函数调用的 LLM（如 GPT-4、Claude）
  - 优化提示词，让模型更准确理解工具调用格式
  - 针对不同类别设计不同策略（multiple/parallel/irrelevance）
- **实践建议**：
  - 渐进式评估：从小样本开始，逐步增加
  - 多类别评估，分析薄弱环节
  - 对比不同配置（提示词、模型）
  - 若结果优秀可提交至 BFCL 官方排行榜
```


