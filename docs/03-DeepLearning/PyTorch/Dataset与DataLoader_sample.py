# -*- coding: utf-8 -*-
"""Dataset 与 DataLoader —— 演示代码

覆盖要点：
1. 自定义 map-style Dataset: __len__ / __getitem__ 契约；
2. 在 __getitem__ 里做 transform（数据增强只在训练集用）;
3. DataLoader: batch_size / shuffle / drop_last / num_workers 的行为;
4. 默认 collate 与"变长序列必须自写 collate_fn(padding)"的坑;
5. 用 DataLoader 完成一次极小的端到端训练循环。

运行（仅 CPU，Windows 下也安全——所有入口都在 __main__ 保护内）：
    python Dataset与DataLoader_sample.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------
# 1) 自定义 Dataset: 合成一个"y = 2*x0 - x1 + 噪声"的回归问题
# ----------------------------------------------------------------------
class ToyDataset(Dataset):
    """每个样本: (特征 x ∈ R^4, 标签 y)。transform 是(可选的)可调用对象。"""

    def __init__(self, n=2000, transform=None, seed=0):
        self.transform = transform
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 4, generator=g)
        self.y = 2.0 * self.x[:, 0] - self.x[:, 1] + 0.3 * torch.randn(n, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 注意: 热路径里别做重活(读盘/解码), 必要时缓存
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:      # 数据增强/预处理钩子
            x = self.transform(x)
        return x, y


# 训练集增强: 加一点噪声扰动(增强只给训练集!)
def train_augment(x: torch.Tensor) -> torch.Tensor:
    return x + 0.02 * torch.randn_like(x)


def no_augment(x: torch.Tensor) -> torch.Tensor:
    return x


# ----------------------------------------------------------------------
# 2) 基本用法演示
# ----------------------------------------------------------------------
def demo_basic() -> None:
    print("=" * 66)
    print("实验1: Dataset 契约与 DataLoader 组批次")
    ds_train = ToyDataset(n=1000, transform=train_augment)
    ds_val = ToyDataset(n=200, transform=no_augment)   # 验证集不做增强
    print(f"    len(train)={len(ds_train)}  len(val)={len(ds_val)}")
    x0, y0 = ds_train[0]
    print(f"    ds[0] -> x.shape={tuple(x0.shape)} y={y0.item():.3f}")

    loader = DataLoader(ds_train, batch_size=32, shuffle=True,
                        drop_last=False, num_workers=0)
    batch_x, batch_y = next(iter(loader))
    print(f"    batch: x={tuple(batch_x.shape)} y={tuple(batch_y.shape)} "
          f"-> 默认collate把32个样本stack成一个batch张量")

    # shuffle 语义: 每 epoch 顺序不同(重排的是索引)
    first_of_ep = []
    for _ in range(3):
        it = iter(DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=0))
        first_of_ep.append(next(it)[1][0].item())
    print(f"    3个epoch各自第一个样本的y: {[f'{v:.3f}' for v in first_of_ep]} (顺序不同)")

    n_full = len(DataLoader(ds_train, batch_size=300, shuffle=True, num_workers=0))
    n_drop = len(DataLoader(ds_train, batch_size=300, shuffle=True,
                            drop_last=True, num_workers=0))
    print(f"    1000条/batch300: 共{n_full}批; drop_last=True -> {n_drop}批"
          f"(丢掉末尾不满300的尾巴)")


# ----------------------------------------------------------------------
# 3) 变长序列 + 自定义 collate_fn
# ----------------------------------------------------------------------
class SentenceBatchDataset(Dataset):
    """每条样本是长度不等的序列(模拟文本/音频帧)。"""

    def __init__(self, n=8):
        g = torch.Generator().manual_seed(7)
        self.lens = torch.randint(3, 10, (n,), generator=g)

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, idx):
        L = int(self.lens[idx])
        return {"seq": torch.randn(L, 8), "len": L}


def pad_collate(batch):
    """默认 collate 无法 stack 长度不同的张量 -> 自定义: 按最长pad成 (B, maxL, 8)。"""
    max_len = max(b["len"] for b in batch)
    feats = torch.stack([
        torch.nn.functional.pad(b["seq"], (0, 0, 0, max_len - b["len"]))
        for b in batch
    ])
    return {"seq": feats, "len": torch.tensor([b["len"] for b in batch])}


def demo_collate() -> None:
    print("=" * 66)
    print("实验2: 变长序列 -> 默认collate报错, 自定义collate_fn解决")
    ds = SentenceBatchDataset(n=8)
    try:
        next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    except RuntimeError as e:
        print(f"    ❌ 默认collate: {str(e).splitlines()[0][:70]}...")
    loader = DataLoader(ds, batch_size=4, collate_fn=pad_collate, num_workers=0)
    b = next(iter(loader))
    print(f"    ✅ 自定义pad_collate: seq={tuple(b['seq'].shape)} "
          f"各样本len={b['len'].tolist()}")


# ----------------------------------------------------------------------
# 4) 端到端小训练循环 (回归 MLP)
# ----------------------------------------------------------------------
def demo_training() -> None:
    print("=" * 66)
    print("实验3: 端到端训练循环 (1个epoch, 验证集评估)")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(ToyDataset(n=2000, transform=train_augment),
                              batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(ToyDataset(n=400, transform=no_augment),
                            batch_size=128, shuffle=False, num_workers=0)

    model.train()
    total = 0.0
    for x, y in train_loader:                       # 一个 epoch = 遍历一遍 Dataset
        opt.zero_grad()
        loss = loss_fn(model(x).squeeze(-1), y)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"    训练loss均值 = {total / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        errs = []
        for x, y in val_loader:                     # 验证: 无增强, 无 shuffle
            errs.append(((model(x).squeeze(-1) - y) ** 2).mean().item())
    print(f"    验证MSE = {sum(errs) / len(errs):.4f}  (y=2x0-x1, 理论可到~0.09噪声下限)")


# ----------------------------------------------------------------------
# 5) num_workers 说明 (Windows 下多进程需 main 保护)
# ----------------------------------------------------------------------
def demo_workers() -> None:
    print("=" * 66)
    print("实验4: num_workers (本机演示用0; >0需 main 保护)")
    loader = DataLoader(ToyDataset(n=200), batch_size=32, shuffle=True,
                        num_workers=0, persistent_workers=False)
    print(f"    num_workers=0: 主进程同步取数, 适合小数据/调试")
    print("    num_workers>0: 子进程预取; Windows 必须把代码放进 main 保护,")
    print("                  否则子进程会重新 import 整个脚本导致无限递归。")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}\n")
    demo_basic()
    demo_collate()
    demo_training()
    demo_workers()
    print("\n要点回顾:")
    print("  - 只写 Dataset(__len__/__getitem__), 批量逻辑交给 DataLoader;")
    print("  - shuffle/drop_last/collate_fn/num_workers 都在 DataLoader 上配;")
    print("  - 变长样本必须自定义 collate_fn(padding); 增强只给训练集。")
