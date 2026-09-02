---
title: "GRPO群组相对策略优化"
tags: [GRPO, LLM, 强化学习]
date: 2026-08-30
---

# GRPO群组相对策略优化

## 定义

群组相对策略优化（Group Relative Policy Optimization，**GRPO**）是一种用于大语言模型（LLM）强化学习（RL）后训练阶段的在线策略优化算法，由 DeepSeek 团队在 **DeepSeekMath** 论文中首次提出，并在 **DeepSeek-R1** 等推理模型中得到大规模验证与应用。

它要解决的问题是：传统 RLHF 采用 PPO 时需要额外维护一个与策略网络（Actor）规模相当的价值网络（Critic）来估计状态价值，这会带来显存与计算开销大、Critic 训练不稳定、对单次生成任务价值估计方差大等问题。GRPO 的核心思路是**放弃单独的价值网络**，改为对同一问题采样多个回答、在组内进行相对比较来估计优势函数，从而在显著降低训练资源消耗的同时保持甚至提升模型性能。

核心特征可概括为四点：① 无 Critic，只需维护策略模型与参考模型，工程复杂度与资源开销低；② 用组内（group-wise）奖励标准化代替绝对价值估计，天然降低奖励模型绝对偏差的影响并压缩方差；③ 使用带裁剪（clip）的代理目标并叠加对参考模型的 KL 惩罚，训练更稳定；④ 特别适合可用规则客观打分的数学、代码等推理任务。

适用范畴：大规模 LLM 的 RLHF / 后训练对齐阶段，尤其是奖励可明确判定对错的场景；也可作为 PPO 的通用替代用于在线策略优化（OpenRLHF、TRL 等开源库已内置支持）。

与 SFT 的关系：**SFT 教会模型"怎么回答"，而 GRPO 则通过强化学习让模型"回答得更好、更符合人类偏好"**。

## 原理

### 为什么不用价值网络（设计动机）

PPO 的价值网络带来三类问题：

- 价值网络通常与策略网络（Actor）规模相当，显著增加显存和计算开销；
- 价值网络训练不稳定、难以收敛；
- 对"只生成一次回答"的任务，价值估计的方差较大。

GRPO 的洞察：优势的作用只是对回答排序，既然无需绝对价值，就可以**对同一问题采样一组回答，在组内做相对比较**来判断"谁相对更好"。组内标准化等价于一种自适应基线，从而整个去掉 Critic，这就是"相对策略优化"的由来。

### 核心流程与公式

对每一个输入问题 $q$：

1. 使用当前策略模型（Actor）采样一组 $G$ 个回答：
$$o_1, o_2, \dots, o_G \sim \pi_{\theta_{\text{old}}}( \cdot \mid q )$$

2. 用奖励模型（Reward Model）对这 $G$ 个回答分别打分，得到奖励：
$$r_1, r_2, \dots, r_G$$

3. 在**组内**对这 $G$ 个奖励进行标准化，得到每个回答的优势估计（实际实现常加入小常数防止除零）：
$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$

4. 使用类似 PPO 的 clipped 目标函数更新策略模型，最大化以下目标：
$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \rho_i(\theta) A_i, \; \operatorname{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

其中：
- $\rho_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$ 是新旧策略的概率比；
- $\epsilon$ 是裁剪范围，通常取 $0.2$，防止单步更新过猛；
- $\beta$ 控制 KL 惩罚强度，防止策略偏离参考模型（通常是 SFT 后的模型）太远。

### 组内优势估计的意义

- 不需要价值网络就能知道哪些回答"相对更好"；
- 比较的是组内相对好坏，天然削弱奖励模型绝对分数的偏差影响；
- 组内标准化相当于自适应基线，减少了方差。

### 与 PPO 的完整特性对比（保留原文表）

| 特性 | PPO | GRPO |
|------|-----|------|
| 是否需要价值网络（Critic） | 是，需要额外训练一个与策略模型规模相当的价值网络 | 否，完全移除价值网络 |
| 优势估计方式 | 使用 GAE（广义优势估计），依赖价值函数 | 组内相对标准化，无需价值函数 |
| 显存与计算开销 | 高（需要存储和训练 Critic） | 低（省去 Critic，可训练更大模型或增加采样数） |
| 采样数量 | 每个 prompt 通常只采样 1 个回答（或少量） | 每个 prompt 采样一组 $G$ 个回答（如 4~16） |
| 稳定性 | 训练可能不稳定，Critic 和 Actor 互相影响 | 更稳定，因为没有 Critic 的干扰 |
| 适用场景 | 传统 RLHF | 大规模 LLM 的高效对齐，尤其适合数学、推理等可客观打分的任务 |

## 应用

### 典型使用场景

- **DeepSeekMath**：GRPO 首次提出的落地场景，用于数学推理任务的强化学习，显著提升数学能力；
- **DeepSeek-R1**：推理模型训练中大量使用 GRPO，结合规则奖励（如答案正确性、格式检查），实现了接近 OpenAI o1 的推理性能；
- **开源生态**：OpenRLHF、TRL（如 `GRPOTrainer`）等库已支持 GRPO，社区中也有不少模型采用 GRPO 进行对齐。

此类场景中选择 GRPO 的收益：① 显存和计算大幅降低——省去价值网络，相同硬件可训练更大的策略模型，或使用更大的 batch size 和采样组数；② 训练更稳定——避免 Critic 训练不收敛或与 Actor 不协调的问题；③ 适合规则化奖励的任务——数学、代码等可客观判断对错的任务中组内比较非常有效；④ 实现简单——只需维护策略模型和参考模型两个大模型，工程复杂度降低。

### 快速上手步骤

1. 准备基座：SFT 后的策略模型，同时作为参考模型 $\pi_{\text{ref}}$ 的初始权重；
2. 定义奖励：规则验证器（答案正确性、格式检查）或训练好的奖励模型（Reward Model）；
3. 数据流：对每个 prompt 用当前策略采样一组 $G$（4~16）个回答；
4. 计算优势：对 $G$ 个回答分别打分 → 组内标准化得到 $A_i$（加小常数防除零）；
5. 更新策略：按 clipped 目标 + KL 惩罚项做梯度更新，循环迭代至收敛（生产环境直接用 TRL / OpenRLHF 的现成实现）。

### 注意事项 / 常见坑

- **依赖多次采样**：每个 prompt 需要采样多个回答，推理成本增加；不过省去 Critic 后总体成本通常仍低于 PPO。
- **奖励模型的质量仍然重要**：组内标准化只降低对绝对分数偏差的敏感度，若奖励模型无法区分组内回答的相对好坏，训练效果会受限。
- **可能偏向"相对更好"而非"绝对正确"**：若一组回答全部很差，组内标准化仍会给相对较好的回答正优势，可能导致策略学到低质量的模式——通常需要配合 KL 惩罚或其他正则化手段。
- **对开放式任务挑战更大**：主观性强、没有明确对错的任务（如创意写作）中，组内相对比较可能不够稳定，需要更精细的奖励设计。
- **实现细节**：组内标准化必须做防除零处理（标准差上加大约 $10^{-4}$ 的小常数）。

```python
# GRPO 核心计算示意（教育用最小实现）
# 工程上建议直接用 TRL 的 GRPOTrainer / OpenRLHF 等成熟实现
import torch

# --- 1) 采样与打分 --------------------------------------------------
# 对同一个问题 q，当前策略采样出 G 个回答，RM/规则验证器给出奖励 r_1..r_G
G = 8
rewards = torch.tensor([0.9, 0.4, 1.0, -0.2, 0.7, 0.1, 0.6, -0.5])

# --- 2) 组内相对标准化 -> 优势估计 A_i ------------------------------
mean_r = rewards.mean()
std_r = rewards.std(unbiased=False) + 1e-4   # +小常数：防除零
A = (rewards - mean_r) / std_r               # 相对组内均值的"好坏"
print("优势 A:", A)                          # 正 = 组内相对更好；负 = 相对更差

# --- 3) 概率比与 clipped 代理目标（示意张量） ------------------------
new_logp = torch.tensor([-1.1, -1.3, -0.9, -1.5, -1.2, -1.4, -1.0, -1.6])
old_logp = torch.tensor([-1.0, -1.2, -1.0, -1.4, -1.3, -1.3, -1.1, -1.5])
ratio = torch.exp(new_logp - old_logp)              # ρ_i(θ) = π_θ / π_θ_old
eps = 0.2                                          # 裁剪范围，通常取 0.2
surr1 = ratio * A
surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * A   # clip(ρ, 1-ε, 1+ε) * A
policy_loss = -torch.min(surr1, surr2).mean()      # 取 min 防止单步更新过猛；负号=梯度下降

# --- 4) KL 惩罚：约束 π_θ 不偏离参考模型 π_ref（通常是 SFT 模型） ----
# kl 实际由前向计算得到，这里用示意值；beta 控制惩罚强度
beta = 0.04
kl = 0.35
loss = policy_loss + beta * kl
print("policy_loss:", round(policy_loss.item(), 4), "| total loss:", round(loss.item(), 4))
```

---
## 关联
- 前置：[[SFT 监督微调]]（GRPO 以 SFT 模型作为初始策略与参考模型）、[[RLHF]]
- 类似：[[PPO]]（区别是 PPO 需要额外训练价值网络并用 GAE 估计优势，GRPO 完全移除 Critic、改为组内相对标准化估计优势）
- 进阶：[[DeepSeek-R1]]（GRPO 的大规模成功应用）、[[DeepSeekMath]]（GRPO 首次提出的论文）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| GRPO（本文方案） | 去掉价值网络，对同一问题采样 $G$ 个回答做组内相对标准化估计优势，再用 clipped 目标 + KL 惩罚更新策略 | 大规模 LLM 后训练对齐；数学、代码等可用规则/奖励模型客观打分的推理任务；硬件资源受限又想训练更大模型 |
| PPO（替代方案） | 保留价值网络，用 GAE（广义优势估计）估计优势，配合 clipped 目标做策略更新 | 传统 RLHF；需要连续状态价值估计、或每个 prompt 只能采样少量回答的通用任务 |

---
## 参考
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO 原始论文）](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [TRL 官方文档：Group Relative Policy Optimization (GRPO)](https://huggingface.co/docs/trl/main/en/grpo)
- [PPO 原始论文：Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---
## 具体案例
- [[GRPO群组相对策略优化 实战示例]](GRPO群组相对策略优化_sample.py)
