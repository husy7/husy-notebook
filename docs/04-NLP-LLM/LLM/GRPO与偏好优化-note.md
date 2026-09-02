---
title: "GRPO 与偏好优化：从 RLHF/PPO 到 DPO、GRPO 的对齐算法族"
tags: [LLM, 偏好优化, 对齐]
date: 2026-08-30
---

# GRPO 与偏好优化：从 RLHF/PPO 到 DPO、GRPO 的对齐算法族

## 定义

大语言模型的后训练分两步：先做 **SFT（监督微调）** 学会"像人话一样回答"，再做**偏好优化（Preference Optimization）**，学会"回答得更好、更符合人类/任务偏好"。偏好优化要解决的核心问题是：仅靠 SFT 的交叉熵无法区分"通顺但平庸"与"真正符合人类/任务偏好"的回答——有用性、无害性、数学/代码正确性这类目标很难直接用 teacher-forcing 建模。

偏好优化家族的共同骨架是：给模型一个**参考策略约束**——优化目标里带 `−β·KL(πθ ‖ π_ref)`（π_ref 常取 SFT 模型），在提升奖励的同时限制与参考策略的距离，防止模型为了刷奖励而跑飞（**reward hacking / Goodhart 效应**）。不同算法只在三个维度上分岔：**谁来打分**（训练奖励模型 RM / 静态偏好对 / 规则·可验证奖励）、**怎么算优势**（GAE 价值函数 / 无 / 组内标准化）、**要不要在线采样**（PPO、GRPO 在线，DPO 完全离线）。

本知识点涵盖三条主流路线：经典 **RLHF/PPO**（显式"奖励模型 RM + 在线 RL"）、**DPO**（把奖励闭式消掉、离线直接优化策略）、**GRPO**（在线 RL 但砍掉 critic，用组内相对比较当优势；DeepSeekMath 提出、DeepSeek-R1 用它做推理强化）。适用范畴覆盖：通用对齐、偏好数据现成的便宜稳定微调，以及数学/代码/推理等"可客观打分"场景下的推理模型（reasoning model）训练。

> 注：本篇与 `模型微调/GRPO-note.md` 互补——那篇是 GRPO 算法的**单点深讲**（动机、公式、PPO 对比、优缺点）；本篇从 LLM 根目录视角讲**整个偏好优化谱系**，GRPO 只做简式回顾并指向深读链接。SFT 相关内容见 `LLM微调SFT与LoRA.md`。

## 原理

**统一的优化骨架。** 家族共享的目标形如：最大化 `E[r(x,y)] − β·KL(πθ ‖ π_ref)`。其中 π_ref 通常冻结为 SFT 模型，β 控制"别离题太远"：太大 → 学不动（贴着 π_ref）；太小 → 过优化、多样性崩、输出退化重复。关键点是**参考模型必须冻结保存**——DPO/GRPO 全程依赖 π_ref，π_ref 一变化就等于目标漂移。

**经典 RLHF（PPO 路线）：显式的"奖励模型 + RL"。** 三步走：① 训练奖励模型 RM：用人工标注的偏好对 (y_w 好, y_l 差)，学 `r(x,y)` 使 Bradley-Terry 偏好概率最大化：`P(y_w > y_l) = σ(r(x,y_w) − r(x,y_l))`；② 用 PPO 让策略最大化 `E[r(x,y)] − β·KL(πθ ‖ π_ref)`；③ PPO 需要**价值网络 critic** 做优势估计（GAE）→ 等于多养一个与策略同规模的大模型，显存/训练开销大、调参敏感。

**DPO：把"奖励"消掉，直接优化策略（离线、简单）。** 关键观察：RLHF 的目标有闭式解
`π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)`，反解出 `r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + const`，**代回 Bradley-Terry** 后奖励模型和 RL 循环统统消失，只剩对策略的监督式损失：

```
L_DPO = − E_{(x, y_w, y_l)} log σ( β · ( log πθ(y_w|x)/π_ref(y_w|x)
                                       − log πθ(y_l|x)/π_ref(y_l|x) ) )
```

直观含义：让"好回答相对参考模型的概率提升"大于"坏回答的相对提升"，实现就是一次二元分类损失。优点：**完全离线**（静态偏好数据集即可）、无 RM、无在线采样、训练稳定。代价：学不到 RM 才有的新知识；数据一旦固定，策略只在数据分布内优化；对 π_ref 与 β 敏感。改进分支：IPO/KTO/ORPO（去掉参考模型的 DPO 式隐式奖励）等。

**GRPO：在线 RL 但砍掉 critic，用"组内相对比较"当优势。** 每个问题 q 用当前策略采样 **G 个回答**；奖励打分可用 RM，更常用**规则/可验证奖励 RLVR**（数学答案对错、代码用例通过）；优势 = 组内标准化：`A_i = (r_i − mean(r_1..r_G)) / std(...)`——**组内相对**而非绝对好坏，天然去掉奖励模型的常数偏移，也因此不需要价值网络（省一个同规模模型）；更新目标含 PPO 式 clip 的概率比 + `−β·KL(πθ‖π_ref)`。

**现实里三者并不互斥**：常见流水线 = SFT → (可选 RLHF/PPO) → DPO 精修；而"可验证奖励 + GRPO"在数学/代码/推理上往往显著优于纯 DPO，是当前训练推理模型的主流路线（DeepSeek-R1：规则奖励 + 输出长度惩罚）。

## 应用

**典型场景选型**：能客观打分（答案对错、代码用例、格式约束）→ 规则/可验证奖励 + GRPO（RLVR 路线）；只有成对偏好标注、想便宜稳定 → DPO 离线精修；奖励复杂主观且算力预算充足 → RM + PPO。

**快速上手 GRPO 闭环**：每个问题采样 G 个回答 → 用规则/可验证奖励打分 → 组内标准化得到优势 A_i（不需要 critic）→ 按带 clip 的概率比与 `−β·KL(πθ‖π_ref)` 做策略更新 → 循环采样新 batch（在线）。最小可运行实现见下方代码与 `GRPO与偏好优化_sample.py`。

**快速上手 DPO**：收集/复用静态偏好对 (x, y_w, y_l) → 冻结 π_ref → 最小化二元分类损失 `L_DPO`；想提高上限可先 SFT 再 DPO，或用 DPO 精修 RLHF/PPO 之后的模型。

**坑（跨算法通用）**：
- ❌ **reward hacking / Goodhart**：奖励定义太粗，模型找到"刷分捷径"（如 R1 早期答非所问但格式正确）；✅ 奖励可验证化 + 输出长度/格式惩罚 + KL 约束。
- ❌ β/KL 系数拍脑袋：太大 → 学不动（贴着 π_ref）；太小 → 过优化、多样性崩、输出退化重复。✅ 观察 πθ 与 π_ref 的 KL 与奖励曲线做权衡。
- ❌ 忽略参考模型：DPO/GRPO 全程依赖 π_ref（π_ref 变化 = 目标漂移），**冻结并保存好**。
- ❌ 偏好数据质量差：**数据质量 > 算法**——选/拒标签错、长度偏差（偏好长答案）、位置偏差都会直接教坏策略；标注前做清洗与校准。
- ❌ 用测试集分布外的奖励做在线训练却不评估分布漂移；每个训练阶段固定评测集监控。
- ❌ 把 DPO 当"万能对齐"：它不探索、不生成新样本；需要探索/新行为时回到在线 RL。

```python
# GRPO 与偏好优化：最小实现骨架（torch 玩具版示意；可运行的完整版见同目录 GRPO与偏好优化_sample.py）
# 覆盖两条主线：① GRPO = 可验证奖励 + 组内优势 + KL 约束；② DPO = 静态偏好对上的二元分类损失
import torch
import torch.nn.functional as F

def grpo_loss(logps, logps_ref, rewards, beta=0.04, eps=0.2):
    """GRPO 一步更新。
    logps/logps_ref: (G,) —— 当前策略与参考策略对 G 个采样回答的 log 概率；
    rewards: (G,) —— 规则/可验证奖励（如数学答案对错、代码用例通过数）。
    """
    # 1) 组内相对优势：去均值、除标准差 → 不需要 critic/价值网络，
    #    天然消除奖励函数的常数偏移（只看"这组里谁相对更好"）
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
    # 2) 概率比 πθ/π_ref，配合 PPO 式 clip 限制单步更新幅度，防止策略跑飞
    ratio = torch.exp(logps - logps_ref.detach())
    pg_loss = -torch.min(ratio * adv,
                         torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
    # 3) KL 正则 −β·KL(πθ‖π_ref)：约束策略别离 SFT 参考模型太远（防 reward hacking）
    kl = (logps_ref.detach() - logps).mean()
    return pg_loss + beta * kl

def dpo_loss(logps_w, logps_l, logps_w_ref, logps_l_ref, beta=0.1):
    """DPO 一步更新（完全离线，无 RM、无采样）。
    输入: (x, y_w 好, y_l 差) 静态偏好对在各策略下的 log 概率。
    推导来源: RLHF 目标闭式解 r(x,y)=β·log(π/π_ref)+const 代回 Bradley-Terry，
    奖励模型与 RL 循环被消掉，只剩一个 sigmoid 二元分类损失。
    """
    # 好回答的相对提升 (log πθ(y_w) − log π_ref(y_w))
    # 要大于坏回答的相对提升 (log πθ(y_l) − log π_ref(y_l))
    log_ratio = (logps_w - logps_w_ref.detach()) - (logps_l - logps_l_ref.detach())
    return -F.logsigmoid(beta * log_ratio).mean()

# 案例详解（训练闭环示意）：
# for batch in rl_loop:                        # GRPO：在线采样
#     G = 8                                    # 每个问题采 8 个回答
#     logps = policy(q, [y_1..y_G]);  ref = ref_model(q, [y_1..y_G])
#     r = rule_reward(q, [y_1..y_G])           # 可验证奖励：对错/用例/长度惩罚
#     loss = grpo_loss(logps, ref, r)          # 组内标准化优势 + clip + KL
# for (x, y_w, y_l) in pref_loader:            # DPO：离线偏好对
#     loss = dpo_loss(logps_w, logps_l, logps_w_ref, logps_l_ref)
```

---
## 关联

- 前置：[[LLM微调SFT与LoRA]]（SFT 是偏好优化的前提：先学会"像人话一样回答"，再谈"回答得更好"）
- 类似：[[模型微调/GRPO-note]]（区别是____那篇把 GRPO 当单点深讲——动机、完整公式、与 PPO 的逐项对比、优缺点；本篇是 RLHF/PPO、DPO、GRPO 整个谱系的横向综述与选型，GRPO 只留骨架）
- 进阶：沿"参考"中的 DPO / DeepSeekMath / DeepSeek-R1 原论文继续深入，再研究 IPO、KTO、ORPO 等去参考模型的 DPO 改进分支，以及"推理强化 + RLVR"在智能体（Agentic RL）上的迁移

---
## 对比选型

| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（GRPO 路线） | 在线采样 G 个回答，规则/可验证奖励打分后做**组内标准化**当优势（无需 critic），配 PPO 式 clip + `−β·KL(πθ‖π_ref)` | 数学/代码/推理等可客观打分任务，训练推理模型（DeepSeek-R1 主流路线） |
| 替代方案：DPO | 由 RLHF 目标闭式解反解出隐式奖励并代回 Bradley-Terry，**消去 RM 与 RL**，只剩静态偏好对上的二元分类损失，完全离线 | 偏好数据现成、要便宜稳定，SFT → DPO 精修流水线（Zephyr/Llama 系） |
| 替代方案：RLHF/PPO | 显式训练 RM（Bradley-Terry 偏好概率）+ 在线 PPO 策略梯度，靠 critic/GAE 估优势 | 通用对齐、奖励复杂且难以规则化、算力预算充足（InstructGPT 路线） |

速选细节对照（原笔记保留）：

| 维度 | PPO(RLHF) | DPO | GRPO |
|------|-----------|-----|------|
| 奖励来源 | 需训练 RM | 静态偏好对(隐含奖励) | RM 或规则/可验证奖励 |
| 数据 | 在线采样 | 离线固定集 | 在线采样(G 个/问) |
| 需 critic 价值网络 | 是 | 否 | **否**(组内基线) |
| 资源/复杂度 | 高 | 低 | 中(采样 G 个开销换省 critic) |
| 优势估计 | GAE(价值函数) | — | 组内标准化 |
| 适用 | 通用对齐、复杂奖励 | 偏好数据现成、要便宜稳定 | 可客观打分任务、推理模型 |
| 代表 | InstructGPT | Zephyr/Llama 系偏好微调 | DeepSeek-R1 |

> 现实里三者不是互斥：常见流水线 = SFT → (可选 RLHF/PPO) → DPO 精修；而"可验证奖励 + GRPO"在数学/代码/推理上往往显著优于纯 DPO。

---
## 参考

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model（DPO 原始论文）](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO 提出论文）](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（GRPO 推理强化实践）](https://arxiv.org/abs/2501.12948)

---
## 具体案例
- [[GRPO 与偏好优化：对齐算法族实战]](GRPO与偏好优化_sample.py)（torch 玩具版：可验证奖励 + 组内优势 + KL 的 GRPO 训练闭环，外加 DPO 损失的最小演示）
