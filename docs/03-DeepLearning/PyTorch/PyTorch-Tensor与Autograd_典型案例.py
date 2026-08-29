# -*- coding: utf-8 -*-
"""
PyTorch Tensor 与 autograd —— 典型代码演示
==========================================
覆盖知识点：
  1. Tensor 的创建、属性、设备迁移（CPU/GPU）
  2. 常用运算：逐元素、矩阵乘法、归约、变形
  3. autograd 自动微分：requires_grad / backward / no_grad / detach
  4. 常见坑：就地操作(in-place)、view vs reshape、梯度累加

依赖：pip install torch
"""

import torch

# =====================================================================
# 一、Tensor 创建与基本属性
# =====================================================================
z = torch.zeros(2, 3)            # 全 0
o = torch.ones(2, 3)             # 全 1
r = torch.randn(2, 3)            # 标准正态
t = torch.tensor([[1, 2], [3, 4]])  # 从列表创建
a = torch.arange(5)              # [0,1,2,3,4]

print("zeros:\n", z.numpy())
print("shape:", z.shape, " dtype:", z.dtype, " device:", z.device)

# 设备迁移（有 CUDA 才用 GPU）
if torch.cuda.is_available():
    g = r.to("cuda")             # 移到 GPU
    print("GPU 上:", g.device)
else:
    print("当前无 GPU，张量保持在 CPU")

# =====================================================================
# 二、常用运算
# =====================================================================
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print("\nA + B =\n", A + B)            # 逐元素加法
print("A * B =（逐元素乘，不是矩阵乘！）\n", A * B)
print("A @ B =(矩阵乘法)\n", A @ B)    # 或 A.matmul(B)

print("A.sum() =", A.sum().item(),
      " A.mean() =", A.mean().item(),
      " A.max() =", A.max().item())

# view 与 reshape
print("\nA.view(1, 4) =", A.view(1, 4))       # 视图（共享内存）
print("A.reshape(4, 1) =", A.reshape(4, 1))   # 可能复制
# transpose 后是非连续内存，不能直接 view
try:
    A.t().view(-1)
except RuntimeError as e:
    print("\ntranspose 后直接 view 报错（非连续）:", str(e)[:45])
    print("解决：加 .contiguous() →", A.t().contiguous().view(-1))

# =====================================================================
# 三、autograd：自动求导全流程
# =====================================================================
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)   # 需要求梯度
y = x.pow(2).sum()            # 标量损失
y.backward()                  # 反向自动求导
print("\ny = sum(x^2), dy/dx = 2x →", x.grad)

# grad_fn 记录计算图
print("y 的计算来源:", y.grad_fn)          # 记录了"sum"节点
print("x 是叶子(requires_grad):", x.requires_grad)

# =====================================================================
# 四、中断梯度跟踪：no_grad 与 detach
# =====================================================================
a_ = torch.randn(3, 3, requires_grad=True)

# 推理/验证：不建计算图，省显存
with torch.no_grad():
    out_nograd = (a_ * 2).sum()
    print("\nno_grad 输出 requires_grad =", out_nograd.requires_grad,
          "（用于评估省显存）")

# detach：分离出同值但不再需要梯度的张量
out_detached = (a_ * 2).detach()
print("detach 后 requires_grad =", out_detached.requires_grad)

# =====================================================================
# 五、常见坑演示
# =====================================================================
# 坑1：就地操作破坏计算图（需对 requires_grad 的叶子做 in-place 会报错）
leaf = torch.ones(2, 2, requires_grad=True)
try:
    leaf += 1                    # 对需要梯度的张量就地 + 会破坏 autograd
except RuntimeError as e:
    print("\n[坑] 对 requires_grad 张量就地运算报错:", str(e)[:50])

# 坑2：梯度累加——backward 后 grad 会累加而非覆盖
g = torch.tensor([1.0], requires_grad=True)
(g * 2).backward()
print("[坑] 第一次 backward grad =", g.grad.item())
# 若不清零再 backward，会累加
