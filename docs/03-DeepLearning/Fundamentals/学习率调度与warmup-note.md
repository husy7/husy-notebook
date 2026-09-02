---
title: "学习率调度与 Warmup"
tags: [深度学习, 优化器, 学习率, warmup]
date: 2026-08-29
---

# 学习率调度与 Warmup

## 定义

学习率（learning rate，lr）是深度学习训练中控制"每一步参数更新幅度"的最重要超参数：它决定权重沿梯度方向走多远。**固定 lr 常常两头不讨好**——lr 太小，前期参数离最优远却爬不动、收敛极慢甚至困在次优解；lr 太大，后期参数已接近最优点却因步长过大而在谷底来回震荡，甚至发散为 loss=NaN。

**学习率调度（learning rate scheduling）**：按训练进度（以 epoch 或 iteration 为单位）动态调整 lr 的训练机制。核心思想是"不同阶段给不同步长"：前期参数远离最优，给大步快走；中后期靠近最优，逐步缩小步长以精细收敛。经验上"前期平缓上升或保持、中后期逐渐下降"的 lr 曲线普遍优于全程固定。

**Warmup（预热）**：调度的一种前奏——在最开始的 W 个 step/epoch 内，把 lr 从接近 0 **线性（或指数）提升到目标 lr**，而不是开局就上满火力。它相当于给优化器与 BN 统计一个"预热期"。

两者常组合成 **linear warmup + cosine decay（线性预热 + 余弦退火）**，是大模型/Transformer（BERT/GPT 类）训练与微调的标配。

适用范畴：几乎所有基于梯度下降的监督/自监督训练；尤其适配 Adam/AdamW 等自适应优化器、大 batch、含 BatchNorm 的网络以及长训练周期。PyTorch 中由 `torch.optim.lr_scheduler` 统一提供实现。

## 原理

**为什么需要"先大后小"的步长曲线**：训练的不同阶段对 lr 的需求不同——前期参数远离最优点、曲面相对平坦，需要大步长快速穿越；后期靠近非凸曲面的谷底、曲面狭窄陡峭，大步长会震荡甚至跳出。lr 过大 → loss 发散/NaN；lr 过小 → 收敛极慢或困在次优。因此理想的 lr 曲线是"前期平缓上升或保持、中后期逐渐下降"。

**Cosine 衰减机制**：CosineAnnealingLR 在 T_max 个周期内把 lr 从 η_max 平滑降到 η_min，核心公式：

```
lr = η_min + ½(η_max − η_min)(1 + cos(π·t / T_max))
```

曲线在开头与结尾斜率都趋缓（平滑无突变）、中段自然过渡，因此是"默认稳妥之选"。

**Warmup 为什么必要（动机，以 Adam 类自适应优化器为主）**：

1. Adam 维护的一阶/二阶矩估计在训练初期（样本少）不可靠，一上来就大步长会放大噪声；
2. 模型刚随机初始化，大 lr 一步可能把权重踢进糟糕区域（尤其配合 BN、大 batch 时统计还不稳定）；
3. warmup 相当于给优化器与 BN 统计一个"预热期"，之后再用目标 lr 训练。

**通用配方**：`warmup 1%~10% 步数 + cosine 衰减`（BERT/GPT 类训练几乎都用）。

**与 optimizer 的配合节奏（机制层面的关键，最易踩坑处）**：

- 标准顺序：`optimizer.step()`（更新参数）→ `scheduler.step()`（推进调度进度）；
- 按 epoch 调度（StepLR/CosineAnnealingLR/…）：每个 epoch 结束 step 一次；
- 按 iteration 调度（OneCycleLR、以及把 cosine 换算成总 step 数）：每个 batch 后 step 一次；
- 不要混用：`CosineAnnealingLR(T_max=总epoch数)` 却在 batch 循环里 step → 一个 epoch 内就把 lr 衰减完。

**主流调度思想（torch.optim.lr_scheduler）**：

| 调度器 | 公式/行为 | 典型用途 |
| --- | --- | --- |
| StepLR | 每 `step_size` 个 epoch ×gamma（阶梯式） | 简单基线 |
| MultiStepLR | 只在指定里程碑（如 [30,60,90]）×gamma | ResNet 训练惯例 |
| ExponentialLR | 每个 epoch ×gamma | 平滑指数衰减 |
| CosineAnnealingLR | `lr = η_min + ½(η_max−η_min)(1+cos(π·t/T_max))`，T_max 内从 η_max 平滑降到 η_min | 默认稳妥之选 |
| CosineAnnealingWarmRestarts | 余弦衰减 + 周期重启（回升再降） | 跳出局部最优/超参搜索 |
| OneCycleLR | 先升后降的"单周期"（内含 warmup） | 快速收敛（超参敏感） |
| ReduceLROnPlateau | 监控验证指标，指标停滞才降 lr（0.1×） | 不知道衰减时机时的自适应兜底 |

PyTorch 还提供 `SequentialLR([s1, s2], milestones=[...])` 串联两段调度（如先 LinearLR warmup 再 cosine），以及 `LambdaLR` 自定义任意 lr 函数。

## 应用

**典型使用场景**：

- 大模型/Transformer 预训练与微调：linear warmup + cosine decay，几乎必用；
- 传统 CNN 训练（如 ResNet）：常用 MultiStepLR（在指定里程碑 ×gamma）；
- 迁移学习/微调：小 lr + linear warmup + cosine；
- 不知道衰减时机：ReduceLROnPlateau 监控验证指标自适应兜底；追求快速收敛可试 OneCycleLR（内含 warmup，但超参敏感）。

**快速上手步骤**：

1. 先小规模扫 lr（log 刻度），找到"能让 loss 下降的最大 lr"作为目标值，再套 warmup + cosine；
2. 定总步数 total_steps 与 warmup 步数 W（取总步数的 1%~10%）；
3. 构建优化器（大模型默认 AdamW）→ `LinearLR(start_factor=1/W, total_iters=W)` + `CosineAnnealingLR(T_max=total_steps−W)` → 用 `SequentialLR(milestones=[W])` 串联；
4. 每个 batch 后按 `optimizer.step()` → `scheduler.step()` 的顺序推进（iteration 级调度）；
5. 用 `scheduler.get_last_lr()` 观测当前 lr（PyTorch ≥1.10），断点续训时把 `optimizer.state_dict()` 与 `scheduler.state_dict()` 一起存、一起载。

**常见坑 ❌✅**：

- ❌ 忘记 `scheduler.step()`，lr 永远不衰减。
- ❌ epoch 级调度器在 batch 循环里 step / iteration 级调度器在 epoch 循环里 step → 衰减节奏错乱。
- ❌ 自己改 `optimizer.param_groups[0]['lr']` 又同时用调度器 → 两边打架（调度器基于自身计数覆盖你的值）。
- ✅ 想手动控 lr 就放弃 scheduler，或统一用 scheduler（`set_lr` 类方法如 `CosineAnnealingWarmRestarts`）。
- ❌ warmup 从 0 开始瞬间就把 lr 设为目标值 → 相当于没做 warmup。
- ✅ warmup 里乘 `(step+1)/W` 的线性斜坡，或直接 `LinearLR(start_factor=1/W)`。
- ❌ 恢复训练（checkpoint）时 lr 被重置 → 可能"重新从大 lr 跑"打乱进度。
- ✅ `optimizer.state_dict()` 与 `scheduler.state_dict()` 一起存/一起载（scheduler 也记录 last_epoch）。
- ❌ 所有任务无脑照搬一个大 lr。
- ✅ 先小规模扫 lr（log 刻度），找到"能下降的最大 lr"，再套 warmup+cosine。
- ✅ 观测用 `scheduler.get_last_lr()`（PyTorch ≥1.10），调试时每 epoch 打一行。

```python
# 案例详解：用 PyTorch 实现大模型训练标配的 "linear warmup + cosine decay"，
# 并演示正确的推进节奏（iteration 级：每个 batch 后 step 一次）与断点续训。
import torch
from torch import nn, optim

# ---------- 1. 模型与优化器 ----------
model = nn.Linear(64, 10)                              # 仅占位，换成你的网络即可
base_lr = 1e-3                                         # 目标/峰值 lr（先小规模扫出来）
optimizer = optim.AdamW(model.parameters(), lr=base_lr)  # 大模型默认 AdamW

# ---------- 2. 定步数并串联 warmup + cosine 两个调度器 ----------
total_steps = 10_000          # 计划总 step = epochs × steps_per_epoch
W = int(total_steps * 0.06)   # warmup 步数：经验上占总步数 1%~10%

warmup = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1 / W, total_iters=W  # lr 从 base_lr/W≈0 线性升到 base_lr
)
cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - W, eta_min=0   # 剩余步数内按余弦曲线平滑降到 0
)
# SequentialLR：前 W 步自动跑 warmup，到里程碑后无缝切换到 cosine
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[W]
)

# ---------- 3. 训练主循环 ----------
start_step = 0                                        # 断点续训时改为 checkpoint 里的 step
for step in range(start_step, total_steps):
    # ... 前向/反向占位：loss = criterion(model(x), y); loss.backward() ...
    optimizer.step()      # ① 先更新参数（顺序固定，别颠倒）
    scheduler.step()      # ② 再推进调度器（别漏！漏了 lr 永远不衰减）
    if step % 1000 == 0:
        # 观测当前 lr（PyTorch ≥1.10 提供 get_last_lr()），调试时每 epoch 打一行
        print(f"step {step}: lr = {scheduler.get_last_lr()[0]:.2e}")

# ---------- 4. 断点续训：optimizer 与 scheduler 的 state_dict 一起存 / 一起载 ----------
# 保存（scheduler 内部也记录 last_epoch，lazy 类还会记录 last_lr 等状态）：
torch.save({"model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "step": step + 1}, "ckpt.pt")
# 恢复：加载后从 start_step 继续，lr 进度不重置；否则会"重新从大 lr 跑"打乱进度。
```

---
## 关联

- 前置：[[优化器（SGD/Adam/AdamW）]]（scheduler 挂在 optimizer 之上、基于其 param_groups 缩放 lr，先理解两者配合节奏；AdamW 是大模型默认）
- 类似：[[梯度消失与梯度爆炸]]（区别是：梯度消失/爆炸源于网络内部信号尺度失衡，靠初始化、归一化、残差结构缓解；而学习率调度是更新步长的时序策略，解决"前期爬不动、后期震荡"的步长矛盾——两者都会表现为训练不稳定，但成因与手段不同）
- 进阶：[[迁移学习与微调]]（微调惯用小 lr + linear warmup + cosine，可配合冻结/解冻策略）

---
## 对比选型

| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：linear warmup + cosine decay | 先线性预热（适配 Adam 初期矩估计不稳、BN 统计预热），再余弦平滑衰减到 η_min | Transformer/BERT/GPT 大模型预训练与微调、长训练默认稳妥之选 |
| 固定 lr（无调度） | 全程一个步长，前期爬不动、后期在谷底震荡 | 极小规模玩具任务、快速验证代码通路 |
| StepLR / MultiStepLR | 每 step_size 或指定里程碑 ×gamma 阶梯式衰减 | ResNet 训练惯例、简单基线 |
| ReduceLROnPlateau | 监控验证指标，指标停滞才降 lr（0.1×） | 不知道合适衰减时机时的自适应兜底 |
| OneCycleLR | 先升后降的"单周期"（内含 warmup），大 lr 冲高再回落 | 追求快速收敛（超参敏感，需谨慎调参） |

---
## 参考

- [PyTorch 官方文档：torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html)
- [Goyal et al., Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour（linear warmup 出处）](https://arxiv.org/abs/1706.02677)
- [Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts（余弦退火出处）](https://arxiv.org/abs/1608.03983)
- [Smith & Topin, Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates（OneCycleLR 出处）](https://arxiv.org/abs/1708.07120)

---
## 具体案例

- [[学习率调度与Warmup 实战示例]](学习率调度与warmup_sample.py)
