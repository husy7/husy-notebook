---
title: "激活函数对比：Sigmoid / Tanh / ReLU / LeakyReLU / GELU"
tags: [深度学习, 激活函数, 基础]
date: 2026-08-29
---

# 激活函数对比：Sigmoid / Tanh / ReLU / LeakyReLU / GELU

> 一句话：激活函数决定网络的"表达能力 + 梯度路径"。选择核心看三件事：是否**零中心**（影响收敛）、是否**两端饱和**（决定会不会消失）、负半轴**梯度是否归零**（决定会不会"死亡"）。

## 定义

激活函数（Activation Function）是神经网络中施加在每个神经元线性输出 z = Wx + b 之上的逐元素非线性变换 f(z)。它解决的核心问题有两个：其一，没有非线性时，任意多层线性变换 (W2(W1x)) 都能合并成单个线性变换，网络深度完全失去意义；其二，引入非线性后网络才可能逼近任意函数（万能逼近定理），同时激活函数的导数参与反向传播链式法则，直接决定梯度能否健康地流过深层网络。核心特征可用三个维度刻画：**是否零中心**——影响收敛速度与梯度更新方向是否单一；**是否两端饱和**——导数趋近 0 的区域决定是否出现梯度消失；**负半轴梯度是否归零**——决定是否出现"死亡神经元"。适用范畴必须区分两层逻辑：隐藏层负责提供非线性与健康的梯度路径（默认 ReLU 族 / GELU）；输出层按任务类型决定语义（二分类 Sigmoid、多分类 Softmax、回归不加激活），不套用隐藏层逻辑。本笔记覆盖的主流成员包括 Sigmoid、Tanh、ReLU、LeakyReLU、GELU，以及同属"自门控"一族的 SiLU/Swish。

## 原理

**为什么需要非线性**：对线性层做堆叠 f(x) = W2(W1x + b1) + b2，恒有 W2W1x + (W2b1 + b2)，结果仍是线性变换——层数再多也只是参数合并，表达力不增。非线性激活打破这种合并，使深层网络能表示任意复杂映射；其导数则进入反向传播链式法则 ∂L/∂x_l = ∂L/∂x_{l+1} · W · f'(z_l)，因此**饱和区（导数≈0）与负半轴零梯度区会直接掐断梯度路径**——这就是"激活函数决定梯度路径"的机制根源。

主流激活对照表（公式 + 梯度机制）：

| 激活 | 公式 | 输出范围 | 导数 | 关键特性 | 主要风险 |
| --- | --- | --- | --- | --- | --- |
| Sigmoid | 1/(1+e⁻ˣ) | (0,1) | f(1−f) ≤ 0.25 | 光滑、有概率语义 | 非零中心；两端饱和 → 消失 |
| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | (−1,1) | 1−f² ≤ 1 | 零中心，比 Sigmoid 收敛快 | 仍两端饱和（深层照样消失） |
| ReLU | max(0, x) | [0, +∞) | 1{x>0} | 计算极简、正半轴导数恒 1（无饱和） | 负半轴导数恒 0 → 神经元死亡 |
| LeakyReLU | x>0? x : αx（默认 α=0.01） | (−∞,∞) | 1 或 α | 给负半轴留小梯度，缓解死亡 | α 是超参；PReLU 可把它学出来 |
| GELU | x·Φ(x)（Φ 标准正态 CDF） | (−∞,∞) | Φ(x)+x·φ(x) | 平滑、Transformer 主流、近似 ReLU 的"软门控" | 计算略贵；注意 erf 精确版与 tanh 近似版 |

**逐项机制解读**：Sigmoid 导数最大仅 0.25，深层连乘后梯度指数级收缩，且输出恒正 (0,1) 使上层梯度符号单一（zigzag 式更新）；Tanh 把输出搬到 (−1,1) 实现零中心，收敛快于 Sigmoid，但两端导数同样趋 0，深度足够时照样消失；ReLU 正半轴导数恒 1，为深层梯度提供"高速公路"，但负半轴导数恒 0，一旦权重把输入推入负区，该神经元输出恒 0 且梯度恒 0、**无法自愈**（死亡 ReLU）；LeakyReLU 在负半轴保留小斜率 α（默认 0.01），让负区也有梯度回流，α 作为超参可由 PReLU 端到端学习；GELU = x·Φ(x) 用标准正态累积分布做"软门控"——按输入大小概率性保留自身，处处可微、平滑且保留少量负梯度，PyTorch `F.gelu(x)` 默认精确 erf 版，`approximate='tanh'` 是更快的 tanh 近似。**SiLU/Swish = x·σ(x)** 与 GELU 同属"自门控"一族，形态接近、常互相替换。另需注意：Softmax 不是逐元素激活，而是把 logits 归一化为概率分布；手写 exp(x)/Σexp(x) 有数值溢出风险，标准做法是做 log-sum-exp 稳定（`log_softmax` / `CrossEntropyLoss` 内部已实现）。

## 应用

**选型指南（快速上手）**：隐藏层默认起点选 **ReLU**（快、稳、好调）；发现 ReLU 大面积死亡时，升级到 LeakyReLU / PReLU / ELU 一档，但**先检查学习率是否过大、初始化是否不当（配 He 初始化）**，不要盲目换函数；深层 CNN / Transformer 主干用 **GELU**（或 SiLU），平滑性对 BN/LN + 深网络的梯度更友好；输出层按任务定，不套隐藏层逻辑：二分类 → Sigmoid，多分类 → Softmax（归一化分布，不是逐元素激活），回归 → 无激活（Linear）；隐藏层一般不放 Sigmoid/Tanh，它们只出现在"需要输出有界 + 可微"的特殊中间层（如门控、归一化后的重构）。

**常见坑 ❌✅**：
- ❌ Sigmoid/Tanh 堆进深网 → 饱和 + 梯度消失（见《梯度消失与爆炸》）。
- ❌ ReLU 配很大的学习率/负偏置初始化 → 大量神经元对任何输入都输出 0（**死亡 ReLU**），且死亡后该分支梯度恒 0、无法自愈。
- ✅ 先检查初始化（He）与 lr；仍大面积死亡再换 LeakyReLU/PReLU。
- ❌ 把 Softmax 当"元素级激活"逐点套用，或手写 `exp(x)/sum(exp(x))` 不做数值稳定。
- ✅ 用 `log_softmax` / `CrossEntropyLoss`（内部已做 log-sum-exp 稳定）。
- ❌ 忽略 GELU 有两种近似实现。
- ✅ PyTorch `F.gelu(x)` 默认精确(erf)版，`approximate='tanh'` 是更快的 tanh 近似；两者误差极小，但**加载预训练模型时必须与原模型的实现保持一致**。

```python
# 激活函数对比：公式、导数、死亡 ReLU 与 GELU 用法验证
# 配套笔记：《激活函数对比：Sigmoid / Tanh / ReLU / LeakyReLU / GELU》
# 环境：numpy + math，可直接运行；GELU 工程用法附 torch 写法（见第 4 步注释）

import math
import numpy as np

def sigmoid(x):
    # 数值稳定写法：x>=0 用 1/(1+e^-x)；x<0 改用 e^x/(1+e^x)，避免 exp(-x) 溢出
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

x = np.linspace(-6, 6, 1001)

# ---- 1) Sigmoid / Tanh：导数峰值对照 ----
# 关键结论：Sigmoid 导数 = f(1-f)，峰值仅 0.25 → 深层连乘极易梯度消失
#           Tanh   导数 = 1-f^2，峰值 1 且零中心收敛更快，但两端仍饱和（导数→0）
f_s = sigmoid(x); d_s = f_s * (1 - f_s)
f_t = np.tanh(x); d_t = 1 - f_t ** 2
print("Sigmoid 导数峰值:", round(d_s.max(), 3), "| Tanh 导数峰值:", round(d_t.max(), 3))

# ---- 2) 死亡 ReLU 演示 ----
# ReLU = max(0, x)：正半轴导数恒 1，负半轴导数恒 0。
# 若学习率过大 / 负偏置初始化，线性部分恒为负 → 输出恒 0 且梯度恒 0，分支无法自愈。
def relu(x):
    return np.maximum(0.0, x)

W, b = -1.0, -0.5                 # 示意一组"坏"参数（负权重 + 负偏置）
X = np.array([0.2, -0.3, 1.0, -0.8])
z = W * X + b
print("线性输入 z:", z)
print("死亡 ReLU 输出:", relu(z))   # 全 0 → 该分支梯度恒 0，不再更新

# ---- 3) 补救方案：LeakyReLU 给负半轴留小梯度 α ----
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

print("LeakyReLU 输出:", leaky_relu(z))   # 负区保留 α 斜率的小梯度，神经元可"复活"

# ---- 4) GELU：软门控（Transformer FFN 标配）----
# 精确版 = x·Φ(x)，Φ 为标准正态 CDF（用误差函数 erf 实现）
def gelu_erf(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))

# 工程上直接用 PyTorch，勿手写：
#   import torch
#   torch.nn.functional.gelu(x)                        # 默认精确 erf 版
#   torch.nn.functional.gelu(x, approximate='tanh')    # 更快的 tanh 近似
# 注意：两者误差极小，但加载预训练模型时实现必须与原模型保持一致。

# ---- 5) 选型速查 ----
# 隐藏层默认 ReLU → 大面积死亡换 LeakyReLU/PReLU → 深主干/Transformer 用 GELU/SiLU
# 输出层：二分类 Sigmoid；多分类 Softmax（归一化分布，非逐元素激活）；回归 Linear（无激活）
# 多分类损失直接用 log_softmax / CrossEntropyLoss，内部已做 log-sum-exp 数值稳定
```

---
## 关联
- 前置：[[梯度消失与爆炸]]（两端饱和 ⇔ 梯度消失，是理解本笔记选型逻辑的前提）
- 类似：[[Softmax 与交叉熵]]（区别是：Softmax 是输出层的归一化分布变换、非逐元素激活，且手写需 log-sum-exp 数值稳定；Sigmoid 才是可逐元素使用的二分类输出激活）
- 进阶：[[Transformer 架构]]（观察 FFN 中间层 GELU 与 LayerNorm、残差连接的配合方式）
- 相关：[[初始化 Xavier/He]]、[[BatchNorm]]、[[ResNet 残差]]（ReLU 配 He 初始化；死亡/消失排查时与激活函数协同调整）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（隐藏层默认 ReLU） | 正半轴导数恒 1、无饱和，计算极简，配 He 初始化收敛快 | 大多数 CNN / MLP 隐藏层的默认起点 |
| 替代方案 1：GELU / SiLU（自门控平滑族） | x·Φ(x) 或 x·σ(x) 软门控：处处可微、保留少量负梯度，对深网络更友好 | Transformer FFN、深层 CNN 主干（配合 BN/LN） |
| 替代方案 2：Sigmoid / Tanh | 输出有界 (0,1) / (−1,1)，具备概率或门控语义；Tanh 零中心、收敛更快 | 二分类输出层（Sigmoid）、门控等"有界+可微"中间层；隐藏层慎用（饱和消失） |
| 替代方案 3：LeakyReLU / PReLU | 负半轴保留小斜率 α（PReLU 可学习 α），缓解死亡 ReLU | ReLU 大面积死亡、且已排除 lr 与初始化问题之后 |

---
## 参考
- [PyTorch 官方文档：Non-linear Activations（Sigmoid/Tanh/ReLU/LeakyReLU/GELU）](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [PyTorch 官方文档：torch.nn.GELU（erf 精确版与 approximate='tanh' 说明）](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [Gaussian Error Linear Units (GELU)，Hendrycks & Gimpel, 2016（arXiv:1606.08415）](https://arxiv.org/abs/1606.08415)

---
## 具体案例
- [[激活函数对比 现实案例]](激活函数对比_sample.py)
