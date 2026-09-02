---
title: "Dataset 与 DataLoader"
tags: [PyTorch, Dataset, DataLoader, 数据管道]
date: 2026-08-29
---

# Dataset 与 DataLoader

## 定义

`Dataset` 与 `DataLoader` 是 PyTorch 数据管道（`torch.utils.data`）中一对分工明确的核心组件。

- **Dataset 回答"一条样本长什么样、怎么取"**：它是一个抽象类，使用者只需实现两个方法——`__len__`（返回样本总数）和 `__getitem__(idx)`（按整数索引返回单条样本，常见为 `(x, y)` 元组或 dict）。
- **DataLoader 回答"怎么把样本们高效地拼成批次喂给模型"**：它在 Dataset 之上统一处理**采样（shuffle）、分组（batch）、并行（worker 子进程预取）、拼装（collate）**四个环节，训练循环直接 `for x, y in loader:` 即可拿到一个批次的 tensor。
- **解决什么问题**：模型训练最常见循环是"随机抽 N 条 → 对齐成 tensor → 前向反向"。若把"取样本"与"组批次"耦合写死，换数据源、改采样策略、加多进程并行都要重写整段训练循环；二者解耦后每个环节各自独立、可替换、可测试。
- **核心特征 / 使用原则**：**自己只写 Dataset，批量逻辑全部交给 DataLoader。** Dataset 只管单样本，因此容易单测、缓存、叠加 transform；DataLoader 统一管批量逻辑，做到"换数据不换管道"。
- **适用范畴**：几乎覆盖所有以 mini-batch 梯度下降训练的监督/自监督任务（图像、文本、表格、多模态等），只要数据可按索引随机访问；流式或不可随机访问的数据源则改用 iterable-style 形态。
- **两种风格**：map-style（最常用）按整数索引取样本，`len` 决定 epoch 长度；iterable-style（`IterableDataset`）语义不同——无 `shuffle`、无索引采样，跨 worker 需自己分片。

## 原理

- **为什么这样设计（分工解耦）**：训练循环可拆成"取样本（Dataset）"与"组批次（DataLoader）"两段。Dataset 只对**单样本**负责，便于测试、缓存、做 transform；DataLoader 统一处理采样/分组/并行/拼装。收益是换数据（如从图片换成文本）时训练管道不用重写，只换 Dataset + 必要时换 `collate_fn`。
- **map-style 内部机制（数据如何变成批次）**：DataLoader 内部是一条流水线——① **Sampler** 产出索引序列：`shuffle=True` 时每 epoch 开始用 `RandomSampler` 重排一次，保证 epoch 间随机、epoch 内每样本恰好一次；② **BatchSampler** 把索引按 `batch_size` 切成一组一组；③ 索引被分发给 **worker 子进程**（`num_workers` 控制进程数），各 worker 调用 `__getitem__(idx)` 取样本并按 `prefetch_factor` 预取若干批，使磁盘读/解码与 GPU 计算重叠；④ **`collate_fn`** 把 batch 个 `__getitem__` 结果拼成一个批次——默认走 `torch.stack` 语义，因此样本结构必须能 stack；变长序列或自定义结构必须自写 `collate_fn`。
- **IterableDataset 的语义差异**：它不依赖索引（无 `__len__` 语义），DataLoader 无法 shuffle、无法按索引采样；多 worker 时每个 worker 默认各自从头迭代同一数据源，**必须自行分片**（如按 worker id 切数据段），否则数据会被重复。
- **效率机制**：多进程 + 预取让 IO 与计算重叠；`pin_memory=True` 用锁页内存缩短 CPU→GPU 拷贝；`persistent_workers=True` 跨 epoch 复用 worker 子进程，省去反复启动/销毁的开销。
- **可复现机制**：shuffle 的随机序列由 DataLoader 的 `generator` 控制，配合全局 `torch.manual_seed` 可复现；worker 进程内的 RNG 由 `worker_init_fn` 决定——若在 `__getitem__` 里做有状态随机而不固定种子，多进程下结果不可复现。
- **关键前提**：`__len__` 决定 epoch 长度（iterable 除外）；collate 拼出的批次必须能被模型消费（形状/类型一致或由 `collate_fn` 统一）。

## 应用

**快速上手 4 步（map-style 标准流程）**：

1. 写 `class MyDataset(Dataset)`，实现 `__len__`（缺了会报 "not implemented"/长度未知）和 `__getitem__(idx)`（返回 `(x, y)` 元组或 dict）。**耗时重活（如磁盘读 + 解码）尽量别放在 `__getitem__` 热路径上且不做缓存，否则会成为训练瓶颈**。
2. 增强/预处理：单样本级在 `__getitem__` 里套 transform（如 torchvision 的 `Compose([Resize, ToTensor, Normalize])`）；批级处理（如变长 padding）则在 DataLoader 传 `collate_fn`。
3. 构造 `DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, ...)`。
4. 训练循环 `for x, y in loader:`（需要步数时用 `enumerate(loader)` 拿 epoch 内步数）。

**DataLoader 关键参数**：

| 参数 | 作用 | 注意 |
| --- | --- | --- |
| `batch_size` | 每个批次样本数 | 与显存/内存匹配 |
| `shuffle=True` | 每 epoch 开始重排（走 RandomSampler） | 训练开、验证/测试关 |
| `sampler` / `batch_sampler` | 自定义采样（加权、子集、分布不均） | 与 `shuffle` 互斥 |
| `num_workers` | 子进程数，预取下一批 | **Windows 必须 `if __name__=='__main__'` 保护**，否则无限递归 |
| `collate_fn` | 把 batch 个 `__getitem__` 结果拼成一批 | 默认自动 stack；变长/自定义结构时必写 |
| `drop_last` | 末尾不满 batch 是否丢弃 | 与 BN/步数稳定相关 |
| `pin_memory` | 锁页内存，CPU→GPU 拷贝更快 | 配 CUDA 用 |
| `persistent_workers` | 跨 epoch 复用 worker | 省重启开销，显存类问题小心 |
| `prefetch_factor` | 每 worker 预取批数 | 调 IO 吞吐 |

**常见坑 ❌✅**：

- ❌ Dataset 忘写 `__len__` / `__getitem__` 拼错（如把 `len` 写成方法、返回 None）。
- ✅ 单测阶段 `print(len(ds)); print(ds[0])` 跑通再进训练。
- ❌ Windows + `num_workers>0` 不加 main 保护 → 进程无限派生崩溃。
- ✅ `if __name__ == "__main__":` 包住训练代码；或先用 `num_workers=0` 排错。
- ❌ 在 `__getitem__` 里改全局/做有状态随机而不用固定种子 → 多进程下不可复现。
- ✅ 复现：`torch.manual_seed` + DataLoader 传 `generator`/`worker_init_fn`。
- ❌ 变长文本/序列直接返回不同长度 → 默认 collate 的 stack 报错。
- ✅ 写 `collate_fn` 做 pad（返回的 batch 结构由你定义）。
- ❌ 训练集做增强、验证集也做增强（或相反）→ 指标失真。
- ✅ 增强只给训练集；验证/测试只做标准化（Normalize/ToTensor）。
- ❌ 把 `shuffle=True` 用到验证集 → 指标波动不可比。
- ✅ 固定评估顺序：`shuffle=False`（或用自己的 sampler）。
- ❌ 每 epoch 重建 Dataset 去"打乱"文件列表 → 不如用 DataLoader 的 shuffle + sampler 语义。
- ❌ 小 batch + `num_workers` 过多 → 通信开销反而更慢；IO 密集型才值得多 worker。
- ✅ `collate_fn` 里少做逐样本 Python 循环，尽量向量化。
- ⚠️ 大 epoch/断点续训：把 `epoch`、sampler 状态、RNG 一并存档恢复。

```python
# Dataset 与 DataLoader 最小可运行示例（变长文本 + padding collate + 可复现 + Windows main 保护）
import torch
from torch.utils.data import Dataset, DataLoader

# 1) 自定义 Dataset：只回答"一条样本长什么样、怎么取"
class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts            # 每句话词数不一（3~8 个词），长度可变
        self.labels = labels
        # 词 -> id 的极简词典；真实项目请用预训练 tokenizer，而非手写词表
        self.vocab = {w: i + 1 for i, w in enumerate({w for t in texts for w in t})}

    def __len__(self):
        return len(self.texts)        # 决定 epoch 长度；缺了会报 "not implemented"

    def __getitem__(self, idx):
        x = [self.vocab[w] for w in self.texts[idx]]   # 单样本：id 序列
        return x, self.labels[idx]    # 返回变长 list——stack 交给 collate_fn 处理

# 2) 批级处理：默认 collate 只能 stack 等长张量，变长序列必须自写 collate_fn 做 padding
def pad_collate(batch):
    xs, ys = zip(*batch)                        # 拆开一批的输入与标签
    max_len = max(len(x) for x in xs)
    xs_pad = [x + [0] * (max_len - len(x)) for x in xs]   # 0 作为 pad token 补齐
    return torch.tensor(xs_pad), torch.tensor(ys)         # batch 结构由你定义

# 3) Windows + num_workers>0 必须加 main 保护，否则子进程无限递归派生崩溃
if __name__ == "__main__":
    torch.manual_seed(0)                          # 配合下方 generator 保证可复现
    texts  = [["我", "爱", "深度学习"],
              ["PyTorch", "真好用"],
              ["数据", "管道", "很", "关键"]]
    labels = [1, 1, 0]

    ds = MyDataset(texts, labels)
    # 单测先行：print(len(ds)); print(ds[0]) 跑通再进训练

    loader = DataLoader(ds, batch_size=2, shuffle=True,
                        num_workers=0,            # 排错阶段先关多进程，调通再开
                        collate_fn=pad_collate)   # 变长输入必传

    # 4) 训练循环：loader 产出 (x, y) 批次张量
    for epoch in range(2):
        for step, (x, y) in enumerate(loader):
            print(f"epoch={epoch} step={step} x={x.tolist()} y={y.tolist()}")
            # ... 此处接 forward / loss / backward

# 案例详解：
# - 第 1 步只写 Dataset（__len__ + __getitem__），与批量逻辑完全解耦；
# - 变长文本不做 padding 直接返回 → 默认 collate 的 stack 必然报错，故第 2 步自写
#   pad_collate，把一批 pad 到 batch 内最大长度，并保证返回结构一致；
# - 第 3 步 main 保护是 Windows 多进程的硬性要求（先 num_workers=0 排错）；
# - 第 4 步每个 epoch 因 shuffle=True 看到不同样本顺序（随机序列由 torch.manual_seed 固定）。
```

---
## 关联
- 前置：[[Tensor 基础]]（collate 结果必须能 stack 成 batch，形状/类型不一致会直接报错）；[[torchvision.transforms]]（transform 管道与数据增强策略，属 torchvision 生态）
- 类似：[[IterableDataset]]（区别是面向流式/不可随机访问数据：无 `__len__` 索引语义、DataLoader 无法 shuffle、跨 worker 需自行分片）
- 进阶：[[DistributedSampler]]（分布式训练下按 rank 分片并保持每 epoch 的 shuffle 语义）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| Dataset + DataLoader（本文方案） | Dataset 只管单样本（`__len__` + `__getitem__`），DataLoader 统一负责 shuffle/batch/worker/collate；自己只写 Dataset | 绝大多数可随机访问数据的监督/自监督训练管道（CV、文本、表格），需多进程预取与 shuffle |
| IterableDataset + DataLoader（替代方案） | 流式产出样本而非按索引取；无 shuffle、无 `len` 语义，跨 worker 自行分片 | 流式/不可随机访问数据：网络流、实时生成、无限序列、超大文件顺序读取 |
| 手写批次循环（替代方案） | 自己维护索引切分、手动 shuffle、手动 stack，绕过 DataLoader | 教学演示、一次性脚本、数据量极小且无并行/预取需求 |

---
## 参考
- [torch.utils.data 官方文档（Dataset / DataLoader / Sampler / collate_fn）](https://pytorch.org/docs/stable/data.html)
- [PyTorch 官方教程：Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

---
## 具体案例
- [[Dataset 与 DataLoader 实战示例]](Dataset与DataLoader_sample.py)
