---
title: "LLM 微调：SFT / LoRA / RLHF"
tags: [LLM, 微调, SFT, LoRA, RLHF]
date: 2026-08-29
---

# LLM 微调：SFT / LoRA / RLHF

## 一、核心思想

预训练 LLM 学会的是"广泛语言与知识"，要让它**对齐到具体任务/风格/指令**，需要微调（Fine-tuning）。微调 ≠ 从零训练：在大模型上做小步调整，把通用能力对齐到目标分布。

主要路径：
- **SFT（有监督微调）**：用标注好的 `(指令, 期望回答)` 数据继续训练。
- **LoRA / PEFT（参数高效微调）**：只训练少量低秩参数，大幅省显存与算力。
- **RLHF（人类反馈强化学习）**：用人类偏好让模型学会"说什么好、什么有害"，提升对齐与有用性。

## 二、SFT（Supervised Fine-Tuning）

- 使用 chat 格式的 instruction-response 数据，标准交叉熵损失训练。
- 让模型学会"指令 → 高质量回答"的映射（指令跟随能力）。
- **关键**：数据质量 > 数据量；需清洗、去重、去有害内容。

```text
[INST] 请用一句话总结本文 [/INST] <目标输出>
→ 最大化 target tokens 的条件概率
```

## 三、LoRA（Low-Rank Adaptation）— 省算力之王

### 3.1 原理

假设权重更新是**低秩**的：$\Delta W \approx B A$，其中 $A,B$ 是两个很小的低秩矩阵。训练时**冻结原权重**，只更新低秩矩阵 $A,B$：

$$ h = W_0 x + \tfrac{\alpha}{r} B A x $$

- 训练参数量骤降（通常只训 0.1%~1% 的参数）。
- 新增参数可随时合并/卸载，方便一个基座做多个下游适配。
- 结合量化（QLoRA）可在消费级 GPU 上微调 7B 级模型。

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("base-model")
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, cfg)         # 冻结原权重，仅训练 LoRA 适配器
model.print_trainable_parameters()         # 可训练参数很少
```

## 四、RLHF（Reinforcement Learning from Human Feedback）

三阶段：
1. **SFT**：先做基础监督微调。
2. **奖励模型 RM**：收集人类对多个回答的偏好排序，训练一个打分模型奖励模型。
3. **策略优化**：用 PPO/DPO 等把 SFT 模型的策略拉向"高奖励"方向，同时用 KL 惩罚防止偏离原模型太远。

- **PPO**：经典 RL 算法，稳定但工程复杂。
- **DPO（Direct Preference Optimization）**：把偏好学习改写为直接优化目标的监督式目标，无需单独训练 RM/采样 rollout，更轻量（当前主流替代）。

```text
RLHF 目标:
  max E[奖励模型打分]  -  β·KL(新策略 || 原SFT策略)   # 既要好又要不跑偏
```

## 五、微调路线选型

| 方案 | 训练成本/显存 | 改变什么 | 适用 |
|------|:---:|------|------|
| **全量微调** | 高 | 所有权重 | 强大到可承受、任务跨度大 |
| **LoRA/QLoRA** | 低 | 少量低秩权重 | 单 GPU 微调、多数场景首选 |
| **Prompt/In-Context（不微调）** | 几乎 0 | 不改变权重 | 快速原型 |
| **RLHF (PPO/DPO)** | 高（PPO 需采样+RM） | 对齐与效用 | 追求对齐/安全 |

> 💡 很多下游任务：先用 **In-context/提示词** 验证，需要再上 **LoRA**，最后按需做对齐（RLHF）。

## 六、边界与坑

- ❌ 微调数据污染/重复 → 过度拟合、知识记忆强化而非泛化。✅ 高质量去重清洗。
- ❌ LoRA rank（r）开太高 → 训练更贵且可能过拟合；太低 → 表达能力不足。✅ 常用 r=8~64，先小后大。
- ❌ 全量 SFT 学坏了基础能力（catastrophic forgetting）。✅ 混入少量通用语料 / 用 LoRA / 控制步数。
- ❌ 直接给模型标了偏见/有害数据 → 放大有害内容。✅ 严格数据治理与红队测试。
- ❌ PPO 训练不稳定（奖励黑客）。✅ 用 DPO 或加 KL 惩罚、reward 归一化。
- 边界：微调提升的是**行为风格/知识领域**，无法凭空增加模型没有学到的基础推理力——那是预训练负责的。

## 七、关联

- 前置知识：Transformer、反向传播、损失函数、采样。
- 同板块：[LLM 推理：KV-Cache 与采样策略](LLM推理与KV-Cache.md)。
- 进阶：多模态微调、上下文蒸馏、结构化微调数据（火种 SFT）。

## 八、参考

- LoRA: Low-Rank Adaptation (Hu et al.) — https://arxiv.org/abs/2106.09685
- DPO: Direct Preference Optimization — https://arxiv.org/abs/2305.18290
- HuggingFace PEFT 官方文档 — https://huggingface.co/docs/peft/
