---
title: "梯度消失与梯度爆炸"
tags: [深度学习, 反向传播, 初始化, 归一化]
date: 2026-08-29
---

# 梯度消失与梯度爆炸

## 定义

一句话本质：反向传播的梯度是"跨层连乘"的产物——每穿过一层就要乘一次权重与激活导数的 Jacobian；平均放大系数 < 1 时梯度随深度指数衰减（消失），> 1 时指数放大（爆炸）。两者是同一机制的两面，区别只在放大系数是否大于 1。

梯度消失（Vanishing Gradient）与梯度爆炸（Exploding Gradient）是深层神经网络训练中最常见的一类失败模式，核心问题是"网络为什么一深就训不动"：

- 消失时：深层（靠近输出）梯度正常，但越往输入侧梯度越小、趋近于 0，浅层几乎拿不到梯度——**浅层学不动**，表现为 loss 长时间不动、只有最后一层在变化；
- 爆炸时：梯度指数放大，参数更新剧烈震荡、很快溢出为 NaN/Inf（发散）。

核心特征：失效程度随层数**指数**变化（≈ ρ^L）、与激活函数的饱和区间强相关（Sigmoid 饱和时导数趋近 0）、与初始化尺度强相关、并与损失面不平滑相互耦合。

适用范畴覆盖几乎所有深层结构：深层前馈网络（MLP/CNN）、循环网络（RNN 中同一权重矩阵被反复相乘，最易爆炸）、Transformer 类注意力网络；相应的对策体系（激活-初始化配对、残差连接、归一化、梯度裁剪、门控结构）是深度学习工程的基础技能。

## 原理

**1. 链式法则视角：梯度是跨层矩阵连乘。** 对第 l 层参数的梯度依赖其后每一层，中间是一串矩阵连乘：

```
∂L/∂y_l = (∏_{k=l+1..L} W_kᵀ · diag(f'(z_k))) · ∂L/∂y_L
```

每穿过一层就要乘一次"权重转置 × 激活导数对角阵"（即该层 Jacobian），数值大小由连乘项的"平均放大系数"决定，与方向无关、只问尺度。

**2. 数量级直觉：L 层之后 ≈ ρ^L。** 若每层平均放大系数约 ρ：

- ρ < 1 → 指数衰减，深层梯度先消失，浅层几乎拿不到梯度（浅层学不动）；
- ρ > 1 → 指数放大，很快溢出为 NaN/Inf（爆炸）。

所以问题的根源不在某一步，而在**每一步放大系数是否恰好 ≈ 1**——这正是初始化（把放大系数拨回 ≈1）和归一化要解决的问题。

**3. Sigmoid 是消失的经典源头。** `σ'(z) = σ(z)(1-σ(z)) ≤ 0.25`，|z| 稍大就趋近 0。即使权重取值正常，激活一旦饱和就会把每层放大系数压得远小于 1，连乘后梯度迅速归零。

**4. 爆炸更常见于三类场景：** RNN（同一权重矩阵被反复相乘，等效于 ρ^t 随时间步累积）、初始化过大、深层无归一化网络的训练早期阶段。

**5. 诊断（先分清是哪一种病，再下药）：**

- 挂 backward hook 逐层打印梯度范数：从输出侧走向输入侧**单调掉到 ~0 → 消失**；出现天文数字/NaN → 爆炸。
- loss 长时间不动、只有最后一层在变化 → 大概率消失。
- 看 `||梯度|| / ||权重||` 与 lr 的乘积：异常大 → 更新震荡/发散（爆炸倾向）；异常小 → 学不动（消失倾向）。

## 应用

**快速上手（先诊断、后下药）：**

1. **诊断**：按上文挂 hook 看梯度范数走势，分清消失还是爆炸。
2. **按激活配对初始化**：Tanh/Sigmoid → Xavier(Glorot)；ReLU 系 → He(Kaiming)。初始化的作用就是把"每层放大系数"拨回 ≈1。
3. **选对激活**：隐藏层用 ReLU 族 / GELU；Sigmoid 只放在二分类输出层（概率语义）。
4. **要更深时加结构**：残差 `y = x + F(x)` 给梯度一条"不衰减"的高速通路；BatchNorm 拉平层输入、平滑损失面；RNN 换 LSTM/GRU（门控提供"闸门"路径）；Transformer 用 LayerNorm + 残差 + 缩放初始化。
5. **训练兜底**：出现爆炸/NaN 时在 `step()` 前梯度裁剪（这只是保险丝，不是消失的解药）。

**常见坑与对策 ❌✅**

- ❌ 深网隐藏层直接用 Sigmoid/Tanh 且不做任何补偿 → 几乎必然消失。
- ✅ 隐藏层用 ReLU 族 / GELU；Sigmoid 只放在二分类输出层（概率语义）。
- ❌ 用默认初始化（PyTorch Linear 默认 kaiming uniform）去初始化以 Sigmoid/Tanh 为主干的网络。
- ✅ 按激活配对初始化：Tanh/Sigmoid → Xavier(Glorot)；ReLU 系 → He(Kaiming)。
- ❌ 为了"更深"盲目堆层，却无残差、无归一化。
- ✅ 残差连接 `y = x + F(x)`：恒等捷径给梯度一条"不衰减"的高速通路，深到几百层仍可训练。
- ✅ 批归一化：把层输入拉回稳定区间、平滑损失面，顺带允许更大学习率（注意训练/推理的统计行为差异、与小 batch 的相互作用）。
- ❌ 以为"梯度裁剪能治消失"。
- ✅ `clip_grad_norm_`（按整体范数）/ `clip_grad_value_`（按元素阈值）只能压爆炸和 NaN，是兜底保险丝，不影响消失。
- ✅ RNN 场景换 LSTM/GRU：门控给梯度提供"闸门"路径，是 RNN 对抗消失的主流设计。
- ✅ 注意力/Transformer 场景：LayerNorm + 残差 + 初始化缩放（如 `xavier_uniform_(gain=1/sqrt(2))` / DeepNorm 类方案）是标配。

```python
# 梯度消失与爆炸：诊断 + 对策 演示
import torch
import torch.nn as nn

# ===== 1) 诊断工具：backward hook 逐层记录"输出侧梯度范数" =====
# backward() 之后读 norms：范数随深度单调掉到 ~0 → 消失；天文数字/NaN → 爆炸
def watch(model):
    norms = {}
    def make_hook(name):
        def hook(module, grad_in, grad_out):
            norms[name] = grad_out[0].norm().item()   # 该层输出侧梯度范数
        return hook
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.register_full_backward_hook(make_hook(name))
    return norms

def make_net(act_cls, depth=20):
    """隐藏层 = Linear + 激活 交替；输出层不加激活（回归任务）。"""
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(64, 64), act_cls()]
    layers.append(nn.Linear(64, 1))
    return nn.Sequential(*layers)

def he_init(m):
    """He(Kaiming) 初始化：与 ReLU 族配对，把每层放大系数拨回 ≈1。"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0)   # 配 ReLU；Tanh/Sigmoid 应换 Xavier
        nn.init.zeros_(m.bias)

def grad_profile(model, x, t):
    """返回 (最深层层号梯度范数, 最浅层层号梯度范数)。"""
    norms = watch(model)
    nn.functional.mse_loss(model(x), t).backward()
    vals = sorted(norms.items(), key=lambda kv: int(kv[0]))  # 按层号排序
    return vals[-1][1], vals[0][1]

x, t = torch.randn(16, 64), torch.randn(16, 1)

# —— 反面案例：Sigmoid 深网（默认初始化）→ 浅层梯度 ≈ 0（消失）——
out_g, in_g = grad_profile(make_net(nn.Sigmoid, depth=20), x, t)
print(f"Sigmoid×20:  输出侧 {out_g:.4f} → 输入侧 {in_g:.2e}  ← 消失（浅层学不动）")

# —— 对策 A：ReLU + He 初始化，100 层梯度仍能传到输入侧 ——
relu_net = make_net(nn.ReLU, depth=100).apply(he_init)
out_g, in_g = grad_profile(relu_net, x, t)
print(f"ReLU×100+He: 输出侧 {out_g:.4f} → 输入侧 {in_g:.4f}  ← 梯度健康")

# —— 对策 B：梯度裁剪 = 爆炸/NaN 的"保险丝"（对消失无效！）——
opt = torch.optim.SGD(relu_net.parameters(), lr=0.1)
for step in range(200):
    opt.zero_grad()
    loss = nn.functional.mse_loss(relu_net(x), t)
    loss.backward()
    # 每步 step() 之前调用；按整体范数裁剪到 1.0
    # （clip_grad_value_ 则按元素阈值裁剪）
    nn.utils.clip_grad_norm_(relu_net.parameters(), max_norm=1.0)
    opt.step()
print(f"训练 200 步后 loss: {loss.item():.4f}")

# 案例详解：
# - Sigmoid 网络 loss 虽能正常反传，但最浅层梯度 ≈ 1e-30 量级 → 消失：对应
#   "隐藏层用 Sigmoid + 默认初始化"的典型坑，表现为浅层学不动、只有最后一层在变。
# - 同一结构换成 ReLU + He 后 100 层梯度仍能传到输入侧 → 对应"按激活配对初始化"。
# - clip_grad_norm_ 只在 step() 前压住整体梯度范数，是爆炸/NaN 的兜底保险丝，
#   对消失无效（消失要靠 ReLU / 残差 / BatchNorm / LSTM 门控去治）。
```

---
## 关联
- 前置：[[反向传播]]（梯度跨层连乘的推导起点）
- 类似：[[激活函数对比]]（区别是____激活函数对比聚焦"饱和区间导数趋近 0"这一消失的单一源头；本笔记从跨层连乘的放大系数出发，统一解释消失与爆炸两种现象，并覆盖初始化/残差/归一化/裁剪全套对策）
- 进阶：[[残差网络 ResNet]]、[[批归一化 BatchNorm]]、[[LSTM 与门控机制]]、[[Transformer 的 LayerNorm 与初始化缩放]]

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：ReLU 族 + He 初始化（进阶加残差/BN） | 把每层放大系数拨回 ≈1，让梯度能跨几十上百层传递 | 深层 MLP/CNN/Transformer 主干 |
| Sigmoid/Tanh + Xavier 初始化 | 对饱和激活按 fan_in/fan_out 缩放权重、维持信号方差 | 浅层网络、二分类输出层 |
| 残差连接 y = x + F(x) | 恒等捷径给梯度一条"不衰减"的高速通路 | 超深网络（几百层仍可训练） |
| 梯度裁剪（clip_grad_norm_ / clip_grad_value_） | 限制整体范数/元素阈值，压住爆炸与 NaN | RNN、大 lr、爆炸/NaN 兜底（治不了消失） |
| LSTM/GRU 门控、LayerNorm+残差 | 门控/归一化为梯度提供稳定通道 | RNN 序列建模、Transformer |

---
## 参考
- [PyTorch 官方文档：torch.nn.utils.clip_grad_norm_ / clip_grad_value_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [PyTorch 官方文档：torch.nn.init（Xavier / Kaiming / gain 参数）](https://pytorch.org/docs/stable/nn.init.html)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification（He 初始化）](https://arxiv.org/abs/1502.01852)
- [Deep Residual Learning for Image Recognition（ResNet，残差连接）](https://arxiv.org/abs/1512.03385)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

---
## 具体案例
- [[梯度消失与爆炸 实战示例]](梯度消失与爆炸_sample.py)
