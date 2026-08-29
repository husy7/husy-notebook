# -*- coding: utf-8 -*-
"""
反向传播与激活函数 —— 典型代码演示
==================================
覆盖知识点：
  1. PyTorch autograd 自动求导：forward 与 backward
  2. 常用激活函数：Sigmoid / Tanh / ReLU / LeakyReLU / Softmax
  3. 手写一个"单隐藏层网络 + 梯度下降"复现反向传播原理
  4. 观察梯度消失（Sigmoid vs LeakyReLU 在深层网络中的梯度）

依赖：pip install torch numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 一、autograd：自动求梯度
# =====================================================================
# 创建带梯度的张量（leaf）
x = torch.tensor([2.0, 3.0], requires_grad=True)
# 前向计算：y = x^2 的均值，这是一个标量损失
y = x.pow(2).mean()
print("y =", y.item())

y.backward()                 # 反向传播，自动算出 dy/dx
print("x.grad =", x.grad, "  理论值 = 2x·(1/n) =", 2 * x * 0.5)

# 关键细节 1：backward 后梯度会积累
try:
    y2 = x.pow(2).mean()
    y2.backward()
    print("再次 backward 后梯度累加 =", x.grad, "（说明需在训练中清零）")
except RuntimeError as e:
    print("重复 backward 抛错（默认 retain_graph=False）：", str(e)[:40])

# 关键细节 2：用 no_grad 让某段不建计算图（推理/评估省显存）
with torch.no_grad():
    z = torch.randn(100, 16)
    y_nograd = (z * 0.5).mean()
    print("no_grad 下张量 requires_grad =", y_nograd.requires_grad)

# =====================================================================
# 二、激活函数对比实验
# =====================================================================
xvals = torch.linspace(-6, 6, 200)          # 采样输入区间

# 逐个计算激活输出，观察取值范围与形态
print("\nsigmoid 输出范围 =", (torch.sigmoid(xvals).min().item(),
                             torch.sigmoid(xvals).max().item()))
print("tanh 输出范围    =", (torch.tanh(xvals).min().item(),
                             torch.tanh(xvals).max().item()))
print("relu 最小输出    =", F.relu(xvals).min().item(),
      " 最大 =", F.relu(xvals).max().item())
print("leaky_relu(负区)=", F.leaky_relu(xvals, 0.01).min().item())

# 观察：Sigmoid/Tanh 在 |x| 很大处"饱和"（导数趋 0 → 梯度消失）
# 用自动微分看 sigmoid 在 x 很大处的梯度
x_sig = torch.tensor([-5.0, 0.0, 5.0], requires_grad=True)
out = torch.sigmoid(x_sig).sum()
out.backward()
print("\nsigmoid 在 x=[-5,0,5] 处的梯度 =", x_sig.grad,
      "（两端接近 0 → 梯度消失）")

# =====================================================================
# 三、手写单隐藏层网络验证反向传播原理
# =====================================================================
torch.manual_seed(0)

# 生成一个可分的小数据集（4 个样本，2 特征，线性可分）
X = torch.tensor([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0], [-2.0, -1.0]])
y = torch.tensor([[1.0], [1.0], [0.0], [0.0]])

in_dim, hidden, out_dim = 2, 8, 1
# 用 Xavier 均匀初始化保证梯度尺度合理（避免梯度消失/爆炸）
W1 = torch.empty(in_dim, hidden); nn.init.xavier_uniform_(W1)
b1 = torch.zeros(hidden)
W2 = torch.empty(hidden, out_dim); nn.init.xavier_uniform_(W2)
b2 = torch.zeros(out_dim)
W1, b1, W2, b2 = [p.requires_grad_(True) for p in [W1, b1, W2, b2]]

lr = 0.5
losses = []
for step in range(200):
    # ---- 前向 ----
    z1 = X @ W1 + b1                      # 隐层线性
    a1 = torch.tanh(z1)                   # 激活
    z2 = a1 @ W2 + b2                     # 输出层线性
    pred = torch.sigmoid(z2)              # 输出概率
    loss = F.binary_cross_entropy(pred, y)  # 二分类交叉熵
    losses.append(loss.item())

    # ---- 反向传播（PyTorch 自动完成链式法则）----
    loss.backward()

    # ---- 更新权重（一步梯度下降）----
    with torch.no_grad():
        for p in [W1, b1, W2, b2]:
            p -= lr * p.grad               # theta = theta - lr * grad
            p.grad.zero_()                 # 清零，避免累计

    if step % 40 == 0:
        acc = ((pred > 0.5).long() == y.long()).float().mean().item()
        print(f"step {step:3d}: loss={loss.item():.4f} acc={acc:.2f}")

print("\n手动网络训练完成，最终 loss =", round(losses[-1], 4))
print("预测概率 =", torch.round(pred.squeeze(), decimals=2))

# =====================================================================
# 四、Softmax：多分类概率输出
# =====================================================================
logits = torch.tensor([[2.0, 1.0, 0.1], [5.0, 1.0, 1.0]])
probs = F.softmax(logits, dim=-1)     # 对最后一维归一化 → 每行和为 1
print("\nsoftmax 每行概率:\n", probs.numpy())
print("每行概率和 =", probs.sum(dim=-1).numpy(), "→ 归一化")

# 更好的做法：用 CrossEntropyLoss（内部合并 softmax + log，数值更稳定）
import torch.nn as nn
my_model = nn.Linear(4, 3)            # 输出概率用 logits，不要手动 softmax
crit = nn.CrossEntropyLoss()
logit_out = my_model(torch.randn(8, 4))
targets = torch.tensor([0, 1, 2, 0, 1, 0, 2, 1])
print("CrossEntropyLoss =", crit(logit_out, targets).item())

# =====================================================================
# 小结
# =====================================================================
# 反向传播 = 链式法则自动求导；激活函数直接决定梯度传播好坏（饱和→消失）；
# 隐藏层优先 ReLU 族，输出层分类用 softmax/交叉熵。
