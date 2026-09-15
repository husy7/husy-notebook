---
title: "调用开源大语言模型与模型选型"
tags: [ai, llm, hello-agents, transformers, 模型选型]
date: 2026-09-14
---

# 调用开源大语言模型与模型选型

> API 快但受制于人，本地开源可控但吃硬件。用 Hugging Face Transformers 完成「加载 → 格式化 → 编码 → 生成 → 解码」五步，即可在个人电脑跑通一个开源对话模型；选模型则是性能/成本/速度/隐私之间的权衡。

## 一、3.2.3 调用开源大语言模型

### 1.1 环境与模型选择

实践选用 `Qwen/Qwen1.5-0.5B-Chat`（阿里达摩院开源，约 5 亿参数，体积小、性能优异，适合入门与本地部署）。

```python
pip install transformers torch
```

### 1.2 加载模型 / 分词器 / 设备

`AutoModelForCausalLM` 与 `AutoTokenizer` 自动匹配权重与分词器；设备优先 GPU、回退 CPU。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen1.5-0.5B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"  # 设备选择
tokenizer = AutoTokenizer.from_pretrained(model_id)        # 加载分词器
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)  # 加载并搬移模型
```

### 1.3 chat template 格式化（`apply_chat_template`）

不同对话模型有各自的对话模板；Qwen1.5-Chat 需按 role 组织消息，再用分词器的模板渲染成文本。

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "你好，请介绍你自己。"},
]
# tokenize=False 只渲染成字符串；add_generation_prompt=True 补上"助手"开头提示
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)  # 编码为 token id 张量
```

编码后得到 `{'input_ids': tensor(...), 'attention_mask': tensor(...)}`，即数字化的 token 序列。

### 1.4 编码 → 生成 → 只解码新增 token

三步完整链路：

```python
# ① 生成：max_new_tokens 控制最多新生成多少个 token
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)

# ② 截掉输入部分，只保留"新生成"的 token id
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# ③ 解码回文本（跳过特殊 token）
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

关键点：

- `max_new_tokens` 限制**新增** token 数（而非整个序列长度），防止无限生成；
- 用 `output_ids[len(input_ids):]` 切片"只解新增部分"，避免误解码输入；
- `skip_special_tokens=True` 清理掉模板里的特殊控制符。

### 1.5 生成解码策略

源文在生成环节只展示了 `max_new_tokens`。采样策略（温度 Temperature、Top-k、Top-p，`T=0`/`Top-k=1` 退化为贪心）在 3.2.1 提示工程小节完整讲解，见[提示工程与分词](提示工程与分词-note.md) 1.1 节；二者配合即可控制"生多长"与"怎么随机"。

### 1.6 本地部署硬件提示

- 本地部署成本体现在**硬件（GPU、内存）与运维**，而非按 token 计费；
- 选小规模模型（如 0.5B）能在普通个人电脑运行，是入门/离线/敏感数据场景的务实选择；
- 源文未展开更细的显存估算与量化技术，需要时按 Llama/Qwen 生态工具链自行补充。

## 二、3.2.4 模型的选择

### 2.1 选型八维度

| 维度 | 关注点 |
|------|--------|
| 性能与能力 | 逻辑推理/代码/创意/多语言各有侧重，可参考 LMSys Chatbot Arena 等榜单 |
| 成本 | 闭源按 Token 计价；开源在硬件 + 运维 |
| 速度（延迟） | 实时交互（客服、NPC）要求低延迟，轻量/优化模型更优 |
| 上下文窗口 | 长文档/代码库/长对话记忆需 128K 等大窗口 |
| 部署方式 | API 简单但数据外发；本地部署保隐私、高自主、门槛高 |
| 生态与工具链 | LangChain/LlamaIndex/Transformers 支持、社区活跃度 |
| 可微调性 | 领域定制能力，开源通常更灵活 |
| 安全与伦理 | 偏见/毒性/幻觉表现、负责任 AI 投入 |

### 2.2 闭源模型概览（开箱即用）

- **OpenAI GPT 系**：GPT-3 开启大模型时代 → ChatGPT（RLHF 对齐意图）→ GPT-4 多模态 → GPT-5 统一文本/音频/图像，实时语音突出。
- **Google Gemini 系**：原生多模态 + 超长上下文；Ultra（最强）/Pro（通用高效）/Nano（端侧）；Gemini 2.5 Pro/Flash 继续强化推理与成本效益。
- **Anthropic Claude 系**：以 AI 安全、长文档、遵循指令可靠著称；Opus（最强）/Sonnet（平衡）/Haiku（最快）。
- **国内主流**：文心一言、腾讯混元、华为盘古、讯飞星火、月之暗面等，中文处理有天然优势。

### 2.3 开源模型概览（随心定制）

- **Meta Llama 系**：开源里程碑，Llama 4 首次采用 MoE 架构；Scout（千万 token 长窗/移动端）、Maverick（多模态/编码/推理）、Behemoth（最强）。
- **Mistral AI 系**：法国，"小尺寸、高性能"；Mistral Medium 3.1（2025.08）代码/STEM 推理强，原生多模态 + 语调适配层。
- **国内开源**：阿里通义千问（Qwen）、清华×智谱 ChatGLM 提供强大中文能力与活跃社区。

## 三、易错点

> **忘记 `add_generation_prompt=True`**：Chat 模板少补"助手"开头，模型可能把应生成的内容当作用户消息继续，输出错乱。

> **直接 `batch_decode` 整个 output**：会把输入 prompt 也一起解出来；必须 `output_ids[len(input_ids):]` 只切新增段。

> **把 `max_new_tokens` 当序列总长**：它是"新增 token 上限"，输入长时实际总长度 = 输入长度 + max_new_tokens，需守上下文窗口。

> **加载模型不 `.to(device)`**：默认留在 CPU/原 device，与已搬移的 input 不在一处会报 device mismatch。

> **用同一 prompt 套所有模型**：不同模型对话模板不同，务必走各自 tokenizer 的 `apply_chat_template`。

> **只看榜单选最大模型**：不是"最大最强"就好，是性能/成本/速度/部署/隐私的综合权衡。

## 四、练习

- Q1：`apply_chat_template` 的两个关键参数作用？  
  A1：`tokenize=False` 只渲染文本不编码；`add_generation_prompt=True` 补上助手回复的引导前缀。
- Q2：生成后为何要先切片再解码？  
  A2：`generate` 返回输入 + 新增的全部 token，切片只留新增部分，避免把 prompt 也解读进回答。
- Q3：从性能、成本、可控性、隐私比较闭源与开源？  
  A3：闭源通常性能前沿、按 token 计费、部署省事但数据外发、可控受条款限；开源可本地部署保隐私、完全可控可微调，但硬件运维成本高。
- Q4：给客服智能体选模型考虑什么？  
  A4：实时低延迟、中文/多语言能力、上下文窗口（历史会话）、成本、隐私合规、生态工具链与安全伦理。

## 五、知识关联

- 前置：[Transformer 架构](Transformer架构-note.md)（Decoder-Only 生成原理与词汇/Token）
- 横向：[[Hugging Face Transformers]]、[[chat template]]、[[采样参数]]
- 进阶：[提示工程与分词](提示工程与分词-note.md)（采样策略、Token 计费） · [缩放法则与幻觉](缩放法则与幻觉-note.md)（模型规模与选型依据）

## 六、对比与选型

| 维度 | 闭源 API | 开源本地 |
|------|----------|----------|
| 上手速度 | 快（注册即用） | 慢（部署 + 硬件） |
| 数据隐私 | 需外发 | 本地自主可控 |
| 成本结构 | 按 Token | GPU/内存 + 运维 |
| 定制 | 受条款限制 | 完全可控、可微调 |
| 前沿能力 | 通常领先 | 追赶中、易自研 |

**选型速查**：要快速上线、拿最前沿能力 → 闭源 API；要处理敏感数据、离线、可控成本、深度定制 → 开源本地（Qwen/ChatGLM 中文优先，Llama 生态最广）。

## 参考

- 源文：第三章 大语言模型基础 · 3.2.3~3.2.4（hello-agents docs/chapter3）