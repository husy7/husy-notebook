---
title: "LLM 微调：SFT / LoRA / RLHF"
tags: [LLM, 微调, SFT, LoRA, RLHF]
date: 2026-08-29
---

# LLM 微调：SFT / LoRA / RLHF

## 定义

LLM 微调（Fine-tuning）指在预训练大语言模型的基础上，用特定任务/风格/指令的数据继续训练，把模型的**通用能力对齐到目标分布**的过程。预训练让模型学会的是"广泛语言与知识"，却缺少对"指令 → 高质量回答"这类具体行为的控制，微调正是为解决这个对齐问题而生。

- **微调 ≠ 从零训练**：权重从预训练结果初始化，只做小步调整，保留绝大部分通用能力，把目标分布"拉"到任务上。
- 三条主要路径：
  - **SFT（Supervised Fine-Tuning，有监督微调）**：用标注好的 `(指令, 期望回答)` 数据继续训练，学会指令跟随。
  - **LoRA / PEFT（参数高效微调）**：只训练少量低秩参数，大幅节省显存与算力。
  - **RLHF（Reinforcement Learning from Human Feedback）**：用人类偏好教会模型"什么回答好、什么有害"，提升对齐与有用性。
- **核心特征**：数据质量 > 数据量；适配器可随时合并/卸载，一个基座支持多个下游适配；结合量化（QLoRA）可下放到消费级 GPU 微调 7B 级模型。
- **适用边界**：微调提升的是**行为风格/知识领域**，无法凭空增加模型没有学到的基础推理力——那是预训练负责的。

## 原理

**SFT 的原理**：把训练数据组织成 chat 格式的 instruction-response 样本，用标准交叉熵损失训练；损失只在 target tokens（期望回答部分）上计算，即最大化其条件概率，从而让模型学会"指令 → 高质量回答"的映射（指令跟随能力）：

```text
[INST] 请用一句话总结本文 [/INST] <目标输出>
→ 最大化 target tokens 的条件概率
```

**LoRA 的原理**：假设预训练权重在下游任务上的更新量是**低秩**的，即 $\Delta W \approx B A$，其中 $A, B$ 是两个很小的低秩矩阵。训练时**冻结原权重 $W_0$**，只更新 $A, B$，前向计算变为：

$$ h = W_0 x + \tfrac{\alpha}{r} B A x $$

其中 $r$ 为秩（rank）、$\alpha$ 为缩放系数。由此训练参数量骤降（通常只训 0.1%~1% 的参数）；新增的低秩参数可随时合并进 $W_0$ 或卸载，方便一个基座做多个下游适配；再结合量化（QLoRA）即可在消费级 GPU 上微调 7B 级模型。

**RLHF 的原理**（三阶段）：
1. **SFT**：先做基础监督微调，得到策略起点；
2. **奖励模型 RM**：收集人类对多个回答的偏好排序，训练一个打分模型（奖励模型）近似人类偏好；
3. **策略优化**：用 PPO/DPO 等把 SFT 模型的策略拉向"高奖励"方向，同时用 KL 惩罚防止偏离原模型太远：

```text
RLHF 目标:
  max E[奖励模型打分]  -  β·KL(新策略 || 原SFT策略)   # 既要好又要不跑偏
```

- **PPO**：经典 RL 算法，稳定但工程复杂（需在线采样 + RM）。
- **DPO（Direct Preference Optimization）**：把偏好学习改写为直接优化目标的监督式目标，无需单独训练 RM、无需采样 rollout，更轻量（当前主流替代）。

## 应用

典型使用场景：指令跟随/对话风格对齐、垂直领域知识适配、单张消费级 GPU 上微调 7B 级模型（QLoRA）、一个基座衍生多个下游适配器。

快速上手步骤：
1. **先验证再训练**：很多下游任务先用 **In-context/提示词** 验证可行性，确认需要再上 **LoRA**，最后按需做对齐（RLHF）。
2. **数据治理**：清洗、去重、去除有害/偏见内容——数据质量 > 数据量。
3. **配置 LoRA**：确定秩 `r`（常用 8~64，先小后大）、`lora_alpha`（缩放系数，常取约 2r）、`target_modules`（自回归模型常用 `q_proj, v_proj`，可扩展到更多线性层）。
4. **训练与评估**：训练前用 `model.print_trainable_parameters()` 确认可训练参数占比（0.1%~1%）；训练中控制步数、混入少量通用语料以对抗灾难性遗忘。

注意事项/常见坑：
- ❌ 微调数据污染/重复 → 过拟合、知识记忆强化而非泛化。✅ 高质量去重清洗。
- ❌ LoRA rank（r）开太高 → 训练更贵且可能过拟合；太低 → 表达能力不足。✅ 常用 r=8~64，先小后大。
- ❌ 全量 SFT 学坏了基础能力（catastrophic forgetting）。✅ 混入少量通用语料 / 用 LoRA / 控制步数。
- ❌ 直接喂偏见/有害数据 → 放大有害内容。✅ 严格数据治理与红队测试。
- ❌ PPO 训练不稳定（奖励黑客）。✅ 用 DPO 或加 KL 惩罚、reward 归一化。

```python
# LoRA 参数高效微调示例（HuggingFace PEFT + transformers）
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 1. 加载预训练基座（配合 bitsandbytes 4bit 量化即 QLoRA，消费级 GPU 可微调 7B）
model = AutoModelForCausalLM.from_pretrained("base-model")

# 2. 配置 LoRA 适配器
#    r=8：低秩维度，控制可训练参数量与表达能力（8~64，先小后大）
#    lora_alpha=16：缩放系数，实际缩放为 alpha/r = 2
#    target_modules：对哪些线性层注入低秩旁路（自回归常用 q_proj/v_proj，可扩展）
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])

# 3. 冻结原权重 W0，仅训练 LoRA 低秩矩阵 A、B（前向等价于 h = W0·x + (alpha/r)·B·A·x）
model = get_peft_model(model, cfg)
model.print_trainable_parameters()   # 可训练参数通常只占 0.1%~1%

# 案例详解：
# - get_peft_model 会把原模型全部参数 requires_grad=False，只给 target_modules
#   对应的层挂上 A/B 两个低秩旁路并置为可训练，因此显存与优化器开销极小。
# - 训练结束后适配器可合并回 W0（model.merge_and_unload()），也可单独保存成
#   几 MB ~ 几十 MB 的 adapter 文件，随时热插拔到同一基座上做多任务适配。
# - 流程主线：清洗 SFT 数据（(指令, 期望回答)）→ 交叉熵损失只算在回答 token 上 →
#   训练 LoRA → 评测；效果不足再考虑加大 r、换 target_modules 或上 RLHF/DPO 对齐。
```

---

## 关联

- 前置：[[Transformer]]、[[交叉熵与反向传播]]、[[KV-Cache 与采样策略]]（推理与评测基础）
- 类似：[[Prompt 工程与 In-Context Learning]]（区别是它完全不训练、不改变任何权重，成本几乎为 0、适合快速原型验证；而 LoRA 需要训练少量低秩参数，用于真正落地的任务适配）
- 进阶：[[RLHF 与 DPO 详解]]、多模态微调、上下文蒸馏、结构化微调数据（火种 SFT）

---

## 对比选型

| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| **本文方案：SFT + LoRA/QLoRA** | 冻结原权重，仅训练低秩旁路（0.1%~1% 参数），低成本改变行为风格与知识领域 | 单 GPU/消费级显卡微调、多数下游任务首选；一个基座挂多个适配器 |
| 全量微调 | 更新所有权重，成本/显存高，任务跨度大但易灾难性遗忘 | 算力充足可承受、任务跨度大、追求极限效果 |
| RLHF（PPO/DPO） | 用人类偏好（RM 打分）拉高奖励并加 KL 约束；PPO 稳定但工程复杂，DPO 轻量免 RM/rollout | 追求对齐与安全；PPO 需采样 + RM，DPO 为当前主流替代 |
| Prompt/In-Context（不微调） | 不改变任何权重，靠上下文示例即时引导 | 快速原型验证（成本几乎为 0），先验证再决定是否微调 |

> 💡 很多下游任务：先用 **In-context/提示词** 验证，需要再上 **LoRA**，最后按需做对齐（RLHF）。

---

## 参考

- [LoRA: Low-Rank Adaptation of Large Language Models（Hu et al.）](https://arxiv.org/abs/2106.09685)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model（Rafailov et al.）](https://arxiv.org/abs/2305.18290)
- [HuggingFace PEFT 官方文档](https://huggingface.co/docs/peft/)

---

## 具体案例

- [[LLM微调SFT与LoRA 实战示例]](LLM微调SFT与LoRA_sample.py)
