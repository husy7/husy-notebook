---
title: "PyTorch Tensor 与 autograd"
tags: [深度学习, PyTorch, Tensor, autograd]
date: 2026-08-29
---

# PyTorch Tensor 与 autograd

## 一、核心思想

PyTorch 的核心是 **Tensor（张量）**——多维数组，是所有计算的基本单位；配合 **autograd（自动微分）**，只要让需要求梯度的量带 `requires_grad=True`，一次 `backward()` 就能自动算出所有相关参数的梯度，**把复杂的手动反向传播变成一次调用**。

## 二、Tensor 基础

### 2.1 创建

```python
import torch

torch.zeros(2, 3)              # 全 0
torch.ones(2, 3)               # 全 1
torch.randn(2, 3)              # 标准正态随机
torch.tensor([[1, 2], [3, 4]]) # 从列表
x = torch.arange(5)            # [0,1,2,3,4]
```

### 2.2 属性与设备

```python
x = torch.randn(2, 3)
print(x.shape, x.dtype, x.device)   # torch.Size([2, 3]) torch.float32 cpu

y = x.to("cuda")              # 移到 GPU
z = x.cpu()                   # 移回 CPU
```

### 2.3 运算

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

a + b, a * b          # 逐元素
a @ b                 # 矩阵乘法（a.matmul(b)）
a.sum(), a.mean()     # 归约
a.view(1, 4)          # 变形（视图，共享内存）
a.reshape(4, 1)       # 变形（可能复制）
```

## 三、autograd 自动微分

### 3.1 使用流程

```python
x = torch.randn(3, requires_grad=True)   # 需要求梯度
y = x.pow(2).mean()                      # 前向计算
y.backward()                             # 反向自动求导
print(x.grad)                            # dy/dx，等于 2x·(1/n)
```

- 只有 `requires_grad=True` 的叶子张量会积累 `.grad`。
- `backward()` 只能调用一次（默认），除非 `retain_graph=True`。

### 3.2 中断梯度跟踪

```python
with torch.no_grad():      # 推理/评估时省显存，不建计算图
    pred = model(x)

z = y.detach()             # 分离，得到同值但 requires_grad=False 的张量
```

### 3.3 求导到明确目标

```python
x = torch.randn(2, requires_grad=True)
y = x ** 3
y.sum().backward()         # 对标量损失 backward
print(x.grad)              # 3*x^2
```

## 四、常见坑

- ❌ `a * b` 以为是矩阵乘，其实是**逐元素乘**。✅ 矩阵乘用 `a @ b` 或 `a.matmul(b)`。
- ❌ `loss.backward()` 前忘了 `optimizer.zero_grad()` → 梯度会**累加**到上次。✅ 每个 batch 前清零。
- ❌ 对整批数据做 `.sum()` 后再 backward 未除 batch size，学习率效果被放大。✅ 用 `.mean()` 归一。
- ❌ 在 `torch.no_grad()` 之外做**验证推理**却叠加过多层 → 占显存（仍建计算图）。✅ 推理包 `no_grad`。
- ❌ `view` 与 `reshape/transpose` 混用导致非连续内存坑。✅ 用 `.contiguous()` 后再 view。

## 五、关联

- 前置知识：神经网络、矩阵运算、链式法则。
- 同板块：[反向传播与激活函数](..\Fundamentals\反向传播与激活函数.md)、[训练循环模板](..\PyTorch\训练循环模板.md)。
- 类似概念：NumPy 数组 vs Tensor（后者自动微分 + GPU）；`view` vs `reshape`。

## 六、参考

- PyTorch 快速入门 — https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html
- autograd 教程 — https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
