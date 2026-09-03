---
title: "智能体性能评估笔记"
tags: [智能体评估, BFCL, GAIA, LLM Judge, Win Rate, 评估基准]
date: 2026-09-03
---

## 一、智能体评估基础

### 1.1 为什么需要评估？

- **核心价值**：量化智能体表现、客观比较不同设计方案、发现弱点、证明可靠性。
- **挑战**：
  - 输出不确定性：同一问题多个正确答案。
  - 评估标准多样性：不同任务需不同评估方法（如工具调用检查函数签名、问答评估语义相似度）。
  - 成本高昂：大量API调用，成本可达数百元。

### 1.2 主流评估基准概览

| 类别 | 基准 | 特点 |
|------|------|------|
| **工具调用** | BFCL | UC Berkeley，1120+样本，四类别，AST匹配 |
| | ToolBench | 清华，16000+真实API场景 |
| | API-Bank | 微软，53个API，专注文档理解 |
| **通用能力** | GAIA | Meta & HuggingFace，466个真实问题，3级难度，准精确匹配 |
| | AgentBench | 清华，8领域任务 |
| | WebArena | CMU，真实网页环境 |
| **多智能体协作** | ChatEval, SOTOPIA | 评估对话系统、社交互动 |
| **常用指标** | 准确性、效率、鲁棒性、协作 | 具体指标如Accuracy、Exact Match、F1、Response Time等 |

### 1.3 HelloAgents 评估体系

- **评估模块**：`hello_agents/evaluation/benchmarks/`
  - `bfcl/`：数据集加载、AST匹配、指标
  - `gaia/`：数据集加载、准精确匹配、指标
  - `data_generation/`：LLM Judge、Win Rate
- **内置工具**：`BFCLEvaluationTool`、`GAIAEvaluationTool`、`LLMJudgeTool`、`WinRateTool`

## 二、BFCL：工具调用能力评估

### 2.1 BFCL 基准介绍

- **四个评估类别**：Simple、Multiple、Parallel、Irrelevance
- **AST匹配**：将函数调用解析为语法树，比较结构和节点值，允许参数顺序不同、等价表达式、字符串表示差异。
- **评估指标**：准确率、AST匹配率、分类准确率、加权准确率、错误率。

### 2.2 获取 BFCL 数据集

```bash
git clone https://github.com/ShishirPatil/gorilla.git temp_gorilla
cd temp_gorilla/berkeley-function-call-leaderboard
```

使用 HelloAgents 加载：

```python
from hello_agents.evaluation import BFCLDataset

dataset = BFCLDataset(
    bfcl_data_dir="./temp_gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
    category="simple_python"
)
data = dataset.load()
```

### 2.3 在 HelloAgents 中实现 BFCL 评估

**方式一：使用 `BFCLEvaluationTool`（推荐）**

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import BFCLEvaluationTool

llm = HelloAgentsLLM()
agent = SimpleAgent(name="TestAgent", llm=llm)
bfcl_tool = BFCLEvaluationTool()

results = bfcl_tool.run(
    agent=agent,
    category="simple_python",
    max_samples=5
)
print(f"准确率: {results['overall_accuracy']:.2%}")
```

**方式二：命令行脚本**

```bash
python chapter12/04_run_bfcl_evaluation.py --category simple_python --samples 10
```

**方式三：直接使用 Dataset 和 Evaluator**

```python
from hello_agents.evaluation import BFCLDataset, BFCLEvaluator

dataset = BFCLDataset(bfcl_data_dir="...", category="simple_python")
evaluator = BFCLEvaluator(dataset=dataset, category="simple_python")
results = evaluator.evaluate(agent, max_samples=10)
```

### 2.4 BFCL 官方评估工具集成

`BFCLEvaluationTool` 自动完成：
- 导出结果到 `evaluation_results/bfcl_official/`
- 复制到 `result/{model_name}/`
- 运行官方评估命令
- 生成 Markdown 报告

### 2.5 核心组件实现

- **BFCLDataset**：加载数据集，支持本地和 HuggingFace。
- **BFCLEvaluator**：构造提示词、提取函数调用、AST 对比。
- **BFCLMetrics**：计算准确率、F1等。
- **AST 匹配**：将参数字典转为 AST 节点，使用 `ast.dump` 比较。

### 2.6 扩展与优化建议

- 当前 `SimpleAgent` 使用自定义工具调用格式，可考虑原生 Function Calling。
- 渐进式评估、多类别评估、对比评估。
- 提交到 BFCL 官方排行榜。

## 三、GAIA：通用 AI 助手能力评估

### 3.1 GAIA 基准介绍

- **设计理念**：真实世界任务，需要多步推理、知识运用、多模态理解、网页浏览等。
- **数据集**：466 个问题，分 Level 1/2/3。
- **准精确匹配**：对答案归一化后精确匹配。归一化规则：数字移除逗号、单位；字符串小写、去冠词、去多余空格；列表排序。
- **指标**：精确匹配率、分级准确率、难度递进下降率、平均推理步骤数。
- **官方系统提示词**：要求输出 `FINAL ANSWER: [答案]`。

### 3.2 获取 GAIA 数据集

- 受限数据集，需在 HuggingFace 申请权限，设置 `HF_TOKEN`。
- 使用 `GAIADataset` 自动下载到 `./data/gaia/`。

### 3.3 在 HelloAgents 中实现 GAIA 评估

**方式一：使用 `GAIAEvaluationTool` 一键评估**

```python
from hello_agents.tools import GAIAEvaluationTool

GAIA_SYSTEM_PROMPT = "..."  # 官方提示词
agent = SimpleAgent(name="TestAgent", llm=llm, system_prompt=GAIA_SYSTEM_PROMPT)
gaia_tool = GAIAEvaluationTool()

results = gaia_tool.run(
    agent=agent,
    level=1,
    max_samples=5,
    export_results=True,
    generate_report=True
)
```

**方式二：直接使用 Dataset + Evaluator**

```python
from hello_agents.evaluation import GAIADataset, GAIAEvaluator

dataset = GAIADataset(level=1)
evaluator = GAIAEvaluator(dataset=dataset, level=1)
results = evaluator.evaluate(agent, max_samples=5)
```

### 3.4 提交结果到 GAIA 官方排行榜

- 评估后生成 `gaia_level1_result_*.jsonl` 和提交说明，上传至 [HuggingFace leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)。

### 3.5 核心组件实现

- `GAIADataset`：从 HuggingFace 下载，支持多模态附件。
- `GAIAEvaluator`：提取答案（`FINAL ANSWER:` 格式）、归一化、准精确匹配。
- `GAIAEvaluationTool`：一键评估并生成报告。

## 四、数据生成质量评估（以 AIME 题目生成为例）

### 4.1 评估方法概述

- **LLM Judge**：使用 LLM 作为评委，从正确性、清晰度、难度匹配、完整性四个维度评分（1-5分）。
  - 指标：平均分、及格率（≥3.5）、优秀率（≥4.5）。
- **Win Rate**：成对对比生成题目与真题，统计胜率、败率、平局率。理想胜率≈50%。
- **人工验证**：Gradio 界面，人工评分、标注状态（approved/rejected/needs_revision）。

### 4.2 系统架构

- `AIMEGenerator`：使用 `SimpleAgent` 生成题目，支持延迟、进度条、LaTeX 处理。
- `LLMJudgeTool`、`WinRateTool`：封装评估流程。
- `HumanVerificationUI`：Gradio 界面。
- `run_complete_evaluation.py`：完整流程。

### 4.3 AIME 题目生成器

- 从 `TianHongZXY/aime-1983-2025` 加载参考题目，随机选择作为示例。
- 生成提示词要求输出 JSON（problem、answer、solution、topic）。
- 处理 LaTeX 转义问题。

### 4.4 LLM Judge 评估工具

- 评估提示词要求返回 JSON 格式的四个维度评分和评语。
- 生成报告包含总体评分、维度评分和可视化。

### 4.5 Win Rate 评估工具

- 对比提示词要求输出 winner（A/B/Tie）和 reason。
- 报告包含胜率统计和结论。

### 4.6 人工验证界面

- Gradio 应用，显示题目、答案、解答，可打分、选状态、加评论。
- 结果保存为 JSON。

### 4.7 完整评估流程

1. 生成题目（`AIMEGenerator`）
2. LLM Judge 评估
3. Win Rate 评估
4. 生成综合报告（`comprehensive_report.md`）

### 4.8 综合评估报告

- 包含基本信息、主题分布、LLM Judge 结果、Win Rate 结果、综合结论、改进建议。

## 五、本章小结

- **三层评估体系**：工具调用（BFCL）、通用能力（GAIA）、数据生成质量（AIME）。
- **关键技术**：AST 匹配、准精确匹配、LLM Judge、Win Rate。
- **扩展方向**：添加新基准、自定义指标、集成 CI/CD、扩展数据评估。

## 习题要点

1. 对比 BFCL 和 GAIA：评估对象、匹配算法、优缺点。
2. 为智能客服设计评估指标和方法。
3. 应对评估挑战的方案。
4. AST 匹配的优缺点及改进。
5. 设计 BFCL 边界测试样本。
6. 扩展 BFCL 评估器（执行顺序、效率、错误分析）。
7. 分析 GAIA 级别差异，设计 Level 4。
8. 设计更智能的答案匹配算法。
9. 构建自定义 GAIA 评估集。
10. 分析 LLM Judge 的偏见和局限性。
11. 为不同场景设计 LLM Judge 评分标准。
12. 设计多评委评估系统。
13. 分层评估策略。
14. 持续评估系统设计。
15. 面向不同受众的评估报告生成。