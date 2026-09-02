---
title: "PyTorch Tensor 与 autograd"
tags: [深度学习, PyTorch, Tensor, autograd]
date: 2026-08-29
---

# PyTorch Tensor 与 autograd

## 定义

Tensor（张量）是 PyTorch 的核心数据结构，本质是一个**多维数组**（对标 NumPy 的 ndarray），是所有数值计算、模型参数与中间激活值的基本单位；任何深度学习模型的前向计算本质上都是在做 Tensor 的运算组合。

autograd（自动微分）是 PyTorch 内置的**自动求导引擎**：只要把需要求梯度的量标记为 `requires_grad=True`，所有基于它产生的运算都会被记录，最后对损失调用一次 `backward()`，就能自动沿着计算图用链式法则算出所有相关参数的梯度。

它要解决的问题是：手写反向传播极其繁琐且易错——每个算子都要自己推导梯度公式并逐层手动回传。autograd 把**复杂的手动反向传播压缩成一次函数调用**，让研究者只需关心前向逻辑，梯度由框架自动完成。

核心特征：动态计算图（边运行边建图，支持 if/for 等任意 Python 控制流）、叶子张量梯度自动积累、`torch.no_grad()` / `detach()` 可随时切断梯度流、原生支持 CPU/GPU 无感切换。

适用范畴：深度学习模型的训练（loss.backward() + 优化器更新）、推理/评估时关闭梯度以省显存、以及任何需要"可微编程"的场景（如自定义损失、元学习、物理信息神经网络）。

## 原理

**动态计算图**：PyTorch 在前向传播过程中"边执行边建图"——每做一次张量运算，autograd 就记录一个节点（算子）与其输入输出，输出张量会挂上对应的 `grad_fn`（如 `PowBackward0`、`MeanBackward0`）。计算图不是预先编译好的静态结构，因此 `if/for` 等运行时控制流天然支持。

**链式法则驱动反向传播**：`backward()` 从标量损失出发，沿计算图反向逐节点求导。设损失 `L`，中间量 `y_j`，输入 `x_i`，则 `∂L/∂x_i = Σ_j (∂L/∂y_j) · (∂y_j/∂x_i)`；框架按拓扑序把上游梯度逐层回传相乘，最终写到叶子张量的 `.grad`。

**梯度积累规则**：只有 `requires_grad=True` 的**叶子张量**（由用户直接创建、非运算产物）会积累 `.grad`；非叶子的中间张量梯度默认用完即释放。默认情况下一次 `backward()` 后计算图被释放，再次调用会报错，除非传 `retain_graph=True`（多次 backward 的常见写法）。

**标量约束**：`backward()` 只接受标量输出，因此非标量要先 `.sum()`/`.mean()` 归约。以 `x = randn(n, requires_grad=True)`、`y = x.pow(2).mean()` 为例：`y = (1/n)·Σxᵢ²`，则 `∂y/∂xᵢ = (1/n)·2xᵢ = 2xᵢ/n`，所以 `backward()` 后 `x.grad == 2x/n`。再如 `y = x**3` 再 `y.sum().backward()`，`∂(Σxᵢ³)/∂xᵢ = 3xᵢ²`。

**中断梯度跟踪**：`torch.no_grad()` 上下文内不建计算图、不记录 grad_fn（推理/评估省显存）；`detach()` 返回与原张量共享数据但 `requires_grad=False` 的新张量，常用于取出数值、拼接不可导分支。

**梯度累加陷阱**：`backward()` 是把新梯度**累加**到 `.grad` 上而非覆盖，所以每个 batch 训练前必须 `optimizer.zero_grad()`（或 `model.zero_grad()`），否则梯度跨 batch 叠加导致更新方向错误。同理，归约用 `.mean()` 而非 `.sum()`，否则等价于学习率被 batch size 放大。

**内存布局关联**：`view` 要求底层内存连续且与原张量共享存储；`transpose` 等操作会产生非连续张量，此时直接 `view` 报错——需先 `.contiguous()` 或改用 `reshape`（不满足连续时自动复制）。

## 应用

典型使用场景：① 训练神经网络——创建 `requires_grad=True` 的参数 → 前向算 loss → `loss.backward()` 自动求梯度 → 优化器 `step()` 更新，每 batch 前先 `zero_grad()`；② 推理/验证——用 `torch.no_grad()` 包裹，避免为不用的梯度建图而白白占用显存；③ 求导数/敏感性分析——对任意可微表达式标量化后 `backward()` 取出 `x.grad`。

快速上手步骤：先用 `torch.zeros/ones/randn/tensor/arange` 建张量，确认 `shape/dtype/device`；需要微分的量创建时加 `requires_grad=True`（或之后 `.requires_grad_(True)`）；前向计算得到标量 loss；`loss.backward()` 后读取 `param.grad`；训练循环中每步 `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`；训练结束或验证阶段用 `no_grad()`/`detach()` 切断梯度。

**常见坑（易错点）**：

- ❌ 把 `a * b` 当作矩阵乘——实际是**逐元素乘**。✅ 矩阵乘用 `a @ b` 或 `a.matmul(b)`。
- ❌ `loss.backward()` 前忘了 `optimizer.zero_grad()` → 梯度**累加**到上次。✅ 每个 batch 前清零。
- ❌ 对整批数据做 `.sum()` 后再 backward 却未除 batch size → 学习率效果被放大。✅ 用 `.mean()` 归一。
- ❌ 在 `torch.no_grad()` 之外做**验证推理**且层叠过多 → 仍建计算图、白占显存。✅ 推理包进 `no_grad`。
- ❌ `view` 与 `reshape/transpose` 混用导致非连续内存报错。✅ 先 `.contiguous()` 再 `view`，或直接用 `reshape`。
- ❌ `backward()` 重复调用不传 `retain_graph=True` → 报"尝试二次反向"错误；`loss` 非标量直接 `backward()` → 报维度错误，需先 `.sum()`/`.mean()`。

```python
import torch

# ---------- 1. Tensor 创建 ----------
torch.zeros(2, 3)               # 全 0 张量，形状 (2, 3)
torch.ones(2, 3)                # 全 1
torch.randn(2, 3)               # 标准正态分布随机
torch.tensor([[1, 2], [3, 4]])  # 从 Python 列表直接创建
x = torch.arange(5)             # [0, 1, 2, 3, 4]

# ---------- 2. 属性与设备 ----------
x = torch.randn(2, 3)
print(x.shape, x.dtype, x.device)   # torch.Size([2, 3]) torch.float32 cpu
y = x.to("cuda")                    # 移到 GPU（无 GPU 环境会报错）
z = x.cpu()                         # 移回 CPU

# ---------- 3. 运算：区分逐元素与矩阵乘 ----------
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
a + b, a * b        # 逐元素加 / 逐元素乘（* 不是矩阵乘！）
a @ b               # 矩阵乘法，等价于 a.matmul(b)
a.sum(), a.mean()   # 归约：求和 / 均值
a.view(1, 4)        # 变形（视图，与 a 共享底层内存）
a.reshape(4, 1)     # 变形（可能复制；不满足连续性时自动兜底）

# ---------- 4. autograd 基本流程 ----------
x = torch.randn(3, requires_grad=True)  # 标记叶子张量：需要梯度
y = x.pow(2).mean()                     # 前向：y = (1/3)·Σxᵢ²，同时建计算图
y.backward()                            # 反向自动求导，一次调用完成 BP
print(x.grad)                           # dy/dxᵢ = 2xᵢ/3
# 注意：backward() 默认只能调用一次；再次调用需 retain_graph=True
# 只有 requires_grad=True 的叶子张量会积累 .grad

# ---------- 5. 中断梯度跟踪 ----------
with torch.no_grad():   # 推理/评估：不建计算图，省显存（示意）
    pred = model(x)

z = y.detach()          # 分离：同值但 requires_grad=False，不再挂计算图

# ---------- 6. 求导到明确目标 ----------
x = torch.randn(2, requires_grad=True)
y = x ** 3
y.sum().backward()      # loss 必须是标量，先 .sum() 归约再 backward
print(x.grad)           # ∂(Σxᵢ³)/∂xᵢ = 3·xᵢ²
```

**案例详解**：示例 4 中若 `x = [x₁, x₂, x₃]`，则 `y = (x₁²+x₂²+x₃²)/3`，对每个分量求偏导得 `∂y/∂xᵢ = 2xᵢ/3`，故打印的 `x.grad` 就是 `2x/3`——这正是训练时把 `loss` 写成 `.mean()` 的原因：梯度被自动归一，学习率语义稳定。示例 6 演示"标量化"套路：非标量 `y` 不能直接 `backward()`，先 `y.sum()` 把目标合成一个标量，梯度即逐分量的导数 `3xᵢ²`。示例 5 是训练/推理切换的关键：训练走 `backward()` 更新权重，验证/测试则包在 `no_grad()` 里只算前向，显存占用与耗时都会显著下降。

---
## 关联
- 前置：[[链式法则]]、[[矩阵运算]]、[[神经网络]]（autograd 反向传播的数学基础；梯度正是按链式法则沿计算图逐层回传）
- 类似：[[NumPy 数组]]（区别是 Tensor 挂接 autograd 自动求导、可 `x.to("cuda")` 上 GPU，NumPy 需手写梯度与设备搬运）；`view` vs `reshape`（区别是 view 要求内存连续且与原张量共享存储，reshape 不满足连续时会自动复制）
- 进阶：[[反向传播与激活函数]]（同板块，理解每层梯度如何传递）、[[训练循环模板]]（把 zero_grad → backward → step 串进完整训练循环）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| PyTorch Tensor + autograd（本文） | 动态计算图：前向逐算子建图，`requires_grad` 标记 + 一次 `backward()` 完成链式求导 | 深度学习训练与研究原型、含 if/for 的动态结构模型、快速迭代 |
| NumPy + 手动反向传播 | 用 ndarray 手写前向，并人工实现每个算子的梯度公式与回传 | 教学演示、无框架依赖的极简实现（无自动微分、无 GPU 加速） |
| TensorFlow（GradientTape / 静态图） | 计算图自动微分：Eager 模式用 `tf.GradientTape` 记录梯度，或 XLA 静态图编译优化 | 生产部署、大规模分布式训练、需要图优化与移动端导出的场景 |

---
## 参考
- PyTorch 快速入门（Tensor 教程）— https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html
- autograd 教程 — https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

---
## 具体案例
- [[PyTorch Tensor 与 autograd 实战示例]](PyTorch Tensor 与 autograd_sample.py)
