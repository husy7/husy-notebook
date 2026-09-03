---
title: "GAIA 通用AI助手能力评估笔记"
tags: [GAIA, 通用能力, 准精确匹配, 评估基准, 多模态]
date: 2026-09-03
---

# GAIA 通用 AI 助手能力评估笔记

## 1. 概述
- **全称**：General AI Assistants，由 Meta AI 和 Hugging Face 联合推出。
- **评估目标**：智能体的**通用问题解决能力**，强调真实世界任务的综合表现。
- **核心能力**：多步推理、知识运用、多模态理解、网页浏览、文件操作等。

## 2. 数据集结构
- **来源**：HuggingFace 受限数据集 `gaia-benchmark/GAIA`（需申请访问权限）。
- **规模**：466 个真实世界问题，分三个难度级别（Level 1/2/3）。
- **样本字段**：
  - `task_id`：任务唯一标识
  - `Question`：问题描述
  - `Level`：难度级别（1~3）
  - `Final answer`：标准答案（数字、文本或文件）
  - `file_name` / `file_path`：附件文件（如图片、PDF）
  - `Annotator Metadata`：标注者元数据（推理步骤、所需工具等）

## 3. 核心评估算法：准精确匹配（Quasi Exact Match）
- **思想**：先对答案进行归一化，再进行精确匹配。
- **归一化规则**：
  - **数字**：移除逗号（1,000→1000）、单位符号（$100→100，50%→50）
  - **字符串**：转小写、移除冠词（the/a/an）、移除多余空格、移除末尾标点
  - **列表**：按逗号分隔，对每个元素应用字符串归一化，按字母顺序排序后重新连接
- **公式**：`Quasi_Exact_Match(A_pred, A_true) = 1 if N(A_pred) == N(A_true) else 0`

## 4. 评估指标
- **精确匹配率（Exact Match Rate）**：核心指标，准精确匹配成功的样本比例。
- **分级准确率（Level-wise Accuracy）**：每个难度级别单独计算准确率。
- **难度递进下降率（Drop Rate）**：从低级别到高级别准确率的相对下降幅度，衡量能力衰减。
- **平均推理步骤数（Average Reasoning Steps）**：正确样本的平均步骤数。

## 5. 官方系统提示词
GAIA 要求模型输出以 `FINAL ANSWER: [答案]` 结尾，且答案格式严格：
- 数字：不使用逗号、单位符号
- 字符串：不使用冠词、缩写，数字用英文单词
- 列表：逗号分隔，按字母顺序

## 6. 在 HelloAgents 中使用 GAIA 评估

### 方式一：使用 `GAIAEvaluationTool`（一键评估）
```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import GAIAEvaluationTool

GAIA_SYSTEM_PROMPT = "..."  # 官方提示词
llm = HelloAgentsLLM()
agent = SimpleAgent(name="TestAgent", llm=llm, system_prompt=GAIA_SYSTEM_PROMPT)
gaia_tool = GAIAEvaluationTool()

results = gaia_tool.run(
    agent=agent,
    level=1,                # 难度级别
    max_samples=5,
    export_results=True,    # 导出官方格式
    generate_report=True    # 生成报告
)
print(f"精确匹配率: {results['exact_match_rate']:.2%}")
```
- 自动完成：下载数据集 → 评估 → 导出 JSONL 结果 → 生成提交说明 → 生成评估报告。

### 方式二：直接使用 `GAIADataset` 和 `GAIAEvaluator`
```python
from hello_agents.evaluation import GAIADataset, GAIAEvaluator

dataset = GAIADataset(level=1)
items = dataset.load()
evaluator = GAIAEvaluator(dataset=dataset, level=1)
results = evaluator.evaluate(agent, max_samples=5)
```

## 7. 核心组件实现要点
- **GAIADataset**：
  - 使用 `snapshot_download` 从 HuggingFace 下载数据集（需 `HF_TOKEN`）
  - 支持多模态附件（图片、PDF 等）
- **GAIAEvaluator**：
  - `_build_prompt()`：将问题和附件信息组装成提示词
  - `_extract_answer()`：从响应中提取 `FINAL ANSWER:` 后的内容，支持多种备用模式
  - `_normalize_answer()`：实现 GAIA 官方归一化规则
  - `_normalize_single_answer()`：处理单个答案（无逗号）
- **GAIAEvaluationTool**：封装完整流程，自动生成报告和提交说明。

## 8. 提交到 GAIA 官方排行榜
- 评估后生成 `gaia_level1_result_*.jsonl` 和 `SUBMISSION_GUIDE_*.md`
- 访问 [HuggingFace GAIA Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard) 提交结果
- 提交前可手动检查 JSONL 文件内容

## 9. 注意事项
- GAIA 是受限数据集，需在 HuggingFace 申请权限并设置 `HF_TOKEN`
- 当前 SimpleAgent 工具调用能力有限，Level 1 准确率可能不高，属正常现象
- 评估结果与官方排行榜一致需使用官方评估流程
```

