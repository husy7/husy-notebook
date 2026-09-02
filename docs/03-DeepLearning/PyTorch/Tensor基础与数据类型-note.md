---
title: "Tensor 基础与数据类型"
tags: [PyTorch, Tensor, dtype, 基础]
date: 2026-08-29
---

# Tensor 基础与数据类型

## 定义

Tensor（张量）是 PyTorch 的数据核心与计算基本单位：数据集经 Dataset/DataLoader 数据管道包装后默认产出 torch.Tensor；模型的输入、参数（权重）、中间激活与梯度也全部是 Tensor，"前向/反向传播"本质就是一系列张量运算。它与 NumPy 的 ndarray 结构同源、可零拷贝互转，但额外多了两层能力：**device**（可驻留 CPU/GPU/MPS，参与异构计算）与 **autograd**（自动构建计算图求梯度）。

一个张量由**四要素**共同决定：

1. **shape（形状）**：维度结构，如 `(2,3)`；`ndim` = 维度数，`numel()` = 元素总数。
2. **dtype（元素数据类型）**：常用 `float32`（默认）、`float64`、`float16`/`bfloat16`（半精度训练）、`int64`（索引/标签默认）、`int32`、`uint8`（图像 0-255）、`bool`。
3. **device（设备）**：`cpu` / `cuda:N` / `mps`，通过 `tensor.to(device)`、`cuda()` 切换。
4. **内存布局（storage + contiguity）**：底层是否按行主序连续存放；`transpose`/`permute` 只改"视图"不改内存 → 得到**非连续**张量。

核心特征：① 形状化批量运算，支持 broadcast 自动对齐 shape；② 类型化存储，同一张量内元素类型一致，算子按 dtype 分派；③ 视图（view）机制，多数改形操作 O(1) 只改元数据、共享数据内存；④ 可微分，`requires_grad=True` 的张量接入 autograd 计算图。

适用范畴：深度学习训练/推理、混合精度（float16/bfloat16 + AMP）、图像与序列数据预处理、NumPy 互操作数据管线、设备间搬运等一切"批量、带类型、可微分"的数值计算场景。**绝大多数 PyTorch "奇怪报错"（尺寸不匹配、dtype/device 不一致、view 兼容性、in-place 与 autograd 冲突）都能归因到四要素之一没对齐。**

## 原理

**存储模型**：Tensor = header 元数据（shape / stride / dtype / device）+ storage 数据块。改形操作分两类——O(1) 的**视图操作**（`view`/`transpose`/`permute`/`squeeze`/`expand`）只改写 shape/stride 等元数据、不搬数据，因此快且省内存；**拷贝操作**（`.clone()`、非连续时的 `.contiguous()`、`reshape` 的兜底路径）则要重新分配并搬运内存。代价是视图与源张量**共享同一块数据内存**：对其中一个做原地修改，另一个同步变化；需要独立副本时必须显式 `.clone()`。

**contiguity（连续性）**：元素在底层按行主序连续存放才算 contiguous。`transpose`/`permute` 只是交换了 stride（"元数据换轴"），内存未动 → 产生非连续张量。`view` 依赖步长推导内存排布，因此**要求连续**（可用 `is_contiguous()` 判断）；`reshape` 在连续时等价于 view，不连续时自动 `.contiguous()`（触发一次真实拷贝）兜底 → 经验法则：**能用 view 语义看清就 view，不确定就 reshape。**

**dtype/device 运算约束**：两个不同 device 的张量**不能直接运算**，须先 `.to(device)` 统一；`.to()` 本身不会打断梯度（只是数据搬运节点），但会留在计算图上。dtype 混算有隐式类型升级规则（如 float + int），而 `bool` 与数值混算、int64 索引张量传错 dtype 常产生静默的错误语义。

**广播（broadcast）机制**：`a + b` 从尾部维度开始对齐 shape，大小为 1 的维可与任意长度对齐，实质是"虚拟扩展"而非复制内存，这也是 shape 差一维却能直接相加的原因。

**autograd 与 in-place 的冲突**：带下划线的就地方法（`add_`/`mul_`/`zero_`/`copy_`）直接改写共享存储，省内存但会破坏视图共享与 autograd 的版本计数；对 `requires_grad=True` 的**叶子**张量就地操作会抛 `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`。安全更新叶子数据的方式：`.data` 上改、`with torch.no_grad()` 里改、或用 `copy_` 写到非叶子张量（前两种在 torch ≥2.x 实测均有效）。

**与 NumPy 的互操作原理**：`torch.from_numpy(np)` / `t.numpy()` 是**零拷贝共享同一块内存**（"相同的内存描述、不同的对象壳"）→ 改一边另一边跟着变；要独立需 `.clone()`/`np.copy()`。

## 应用

**典型使用场景**：神经网络训练与推理（输入/权重/梯度张量）；图像预处理（`uint8` → `float32` 后除以 255 归一化）；索引与标签（统一 `int64`）；GPU 并行（`.to('cuda')`）；半精度/混合精度训练（AMP `torch.autocast` + `GradScaler`，或 `bfloat16`）；与 NumPy 生态互转做数据清洗与可视化。

**快速上手四步曲**：
1. **创建**：`torch.tensor(...)`/`zeros`/`ones`/`randn`/`arange`/`full`/`eye`；要随机数明确用 `rand/randn` 或 `empty`+初始化。
2. **对齐类型**：先 `.dtype`/`.shape`/`.device` "三连查"，需要时 `.float()`/`.long()`/`.to(dtype)`。
3. **搬运设备**：`t.to('cuda')`/`.cpu()`，跨设备运算前先统一 device。
4. **改形换轴**：优先 `view`（看清连续语义时），不确定用 `reshape`；`transpose`/`permute` 任意换轴后常需 `.contiguous()` 再接 `view`。

**常用操作速查**（源笔记保留）：

| 目的 | 写法 | 备注 |
| --- | --- | --- |
| 创建 | `torch.tensor(...)`/`zeros`/`ones`/`randn`/`arange`/`full`/`eye` | `torch.Tensor(3)` 是"空存储"坑，慎用；要随机数用 `rand/randn` 或 `empty`+初始化 |
| 转类型 | `t.float()`/`.long()`/`.to(dtype)` | 索引转 int64、图像 uint8→float32 再归一化 |
| 换设备 | `t.to('cuda')`/`.cpu()` | 梯度图里 `.to()` 会打断吗？不会，但会留在图上 |
| 改形状 | `t.view(s)` / `t.reshape(s)` / `t.flatten()` | view 需连续 |
| 换轴 | `t.transpose(0,1)`/`t.permute(1,2,0)`/`t.contiguous()` | permute 任意换轴后常需 contiguous 再接 view |
| 增删轴 | `t.unsqueeze(0)`/`t.squeeze()` | 广播与 batch 维常用 |
| 就地运算 | `t.add_(1)`/`t.mul_(2)`/`t.zero_()`/`t.copy_(src)` | 下划线=就地，省内存但破坏视图共享/autograd |
| 取数值 | `t.item()`(标量)/`t.tolist()`/`t.numpy()` | item 只能用于单元素 |
| 拼接 | `torch.cat([...], dim)` / `torch.stack([...], dim)` | cat 沿已有维拼；stack 新建一维 |
| 广播 | `a + b`（shape 自动对齐） | 从尾部对齐：1 可与任意维对齐 |
| 与 NumPy | `torch.from_numpy(np)` / `t.numpy()` | **共享内存**，改一边另一边变；需要独立用 `.clone()`/`np.copy` |

**常见坑 ❌✅**：

- ❌ `torch.Tensor(3)`：得到未初始化 3 个元素的空张量（不是 `[3]`！）。想要 3 元素随机数用 `torch.randn(3)`。
- ❌ `view` 报 "view size is not compatible with input tensor's size" / 对 transpose 后的张量直接 view。
- ✅ 先 `.contiguous()`（会拷贝）或直接用 `reshape`。
- ❌ 忘掉就地运算与 autograd 的交互：对 `requires_grad=True` 的**叶子**张量做 `add_` 会报 `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`。
- ✅ 需要更新参数/数据：`.data` 改、`with torch.no_grad()` 里改、或用 `copy_` 到非叶子。
- ❌ `from_numpy`/`.numpy()` 后没意识到两边共享内存 → 数据被悄悄互相污染。
- ✅ 明确要共享（数据搬移省拷贝）就刻意用；要独立就 clone/np.copy。
- ❌ 类型不匹配硬算：float 张量 + int 张量能隐式转，但 `bool` 与数值混算、int64 索引张量传错 dtype 常出隐晦错误。
- ✅ 养成先 `.dtype`、`.shape`、`.device` 三连查的习惯。
- ❌ 半精度训练直接 `model.half()` 全换 → 数值/BN 精度问题。
- ✅ 用 AMP `torch.autocast` + `GradScaler`（混合精度），或 bfloat16；主权重仍留 float32。

```python
import torch
import numpy as np

# ===== 1. 创建与四要素 =====
x = torch.randn(2, 3)   # 默认 float32 + cpu + 连续
print(x.dtype, x.device, x.shape, x.is_contiguous())
# torch.float32 cpu torch.Size([2, 3]) True
y = torch.zeros(2, 3, dtype=torch.float64)   # 显式指定 dtype

# 坑：torch.Tensor(3) 是"3 个元素的未初始化空存储"，不是 torch.Size([3])！
# bad = torch.Tensor(3)          # ❌ 未初始化，值不可控
good = torch.randn(3)            # ✅ 要 3 元素随机数用 randn/rand

# ===== 2. dtype 转换：图像 uint8(0-255) -> float32 归一化 =====
img_u8 = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
img_f = img_u8.float() / 255.0            # 索引/标签默认用 int64
labels = torch.tensor([1, 0, 2]).long()

# ===== 3. 换设备：不同 device 的张量不能直接运算 =====
if torch.cuda.is_available():
    x = x.to('cuda')          # .to() 不打断梯度，但会留在计算图上

# ===== 4. 视图 vs reshape / contiguity（核心机制演示） =====
t = torch.arange(6).reshape(2, 3)   # 连续
t2 = t.transpose(0, 1)              # 视图：只换 stride、内存未动 -> 非连续
print(t2.is_contiguous())           # False
# t2.view(3, 2)
# ❌ RuntimeError: view size is not compatible with input tensor's size and stride
t3 = t2.reshape(3, 2)               # ✅ reshape 自动 .contiguous() 拷贝兜底
t4 = t2.contiguous().view(3, 2)     # 等价写法：先拷贝再 view

# ===== 5. 共享存储的两面性：视图 / .numpy() 都只改元数据 =====
a = torch.arange(4)
b = a.view(2, 2)          # 视图共享内存
a.add_(1)                 # 就地修改 -> b 也跟着变（除非 .clone()）
n = a.numpy()             # 与 NumPy 共享内存：改 a 则 n 同步变
c = a.clone()             # ✅ 需要独立副本时 clone / np.copy

# ===== 6. 广播：从尾部对齐，size=1 的维可与任意长度对齐 =====
m = torch.ones(2, 3)
v = torch.ones(3)
s = m + v                 # (2,3) + (3,) -> 广播成 (2,3)，虚拟扩展不复制内存

# ===== 7. autograd：叶子张量禁止直接 in-place =====
w = torch.randn(3, requires_grad=True)
# w.add_(1)
# ❌ RuntimeError: a leaf Variable that requires grad is being used in an in-place operation
w.data.add_(1)            # ✅ 方案一：.data 上改（绕过 autograd 版本检查）
# with torch.no_grad():   # ✅ 方案二：no_grad 上下文里改（实测均有效）
#     w.add_(1)
out = (w * 2).sum()
out.backward()            # 正常反向传播，不影响梯度计算
print(w.grad)             # tensor([2., 2., 2.])

# ===== 8. 取数值：item 仅限单元素 =====
scalar = x.sum()          # 0-dim 张量
print(scalar.item())      # 多元素用 .tolist()
```

---
## 关联
- 前置：[[NumPy ndarray]]（Tensor 与 ndarray 共享内存、零拷贝互转，dtype 与底层存储概念一脉相承）
- 类似：[[广播 Broadcasting 与 einsum]]（区别是：广播/einsum 描述的是"不同 shape 张量运算时如何对齐收缩"的规则，而本文四要素与视图机制描述的是"单个张量的构成与高效改形"）
- 进阶：[[自动微分 autograd]]、[[设备与显存管理 CUDA]]、[[混合精度训练 AMP]]、[[Dataset 与 DataLoader]]（数据管道里默认把样本包装成 torch.Tensor）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| torch.Tensor（本文方案） | 四要素（shape/dtype/device/storage）显式建模 + O(1) 视图改形 + 原生 autograd 与 GPU/MPS 加速 | 深度学习训练/推理、需要自动微分与异构计算的可微数值运算 |
| NumPy ndarray | 单一 dtype 的同构多维数组 + 向量化算子，纯 CPU，无 device/计算图概念 | CPU 科学计算、数据预处理/后处理/可视化；与 Tensor 靠 from_numpy/.numpy() 零拷贝互转 |

---
## 参考
- [PyTorch Tensors 官方文档](https://pytorch.org/docs/stable/tensors.html)
- [torch.Tensor.view 官方文档](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)
- [torch.reshape 官方文档](https://pytorch.org/docs/stable/generated/torch.reshape.html)
- [torch.autocast 混合精度官方文档](https://pytorch.org/docs/stable/amp.html)

---
## 具体案例
- [[Tensor基础与数据类型 实战示例]](Tensor基础与数据类型_sample.py)
