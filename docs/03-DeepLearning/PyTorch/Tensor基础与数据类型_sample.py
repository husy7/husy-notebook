# -*- coding: utf-8 -*-
"""Tensor 基础与数据类型 —— 演示代码

覆盖要点：
1. 创建方式与 dtype/shape/device 三要素；
2. view / reshape / transpose / permute 的"视图 vs 拷贝"语义与 contiguity；
3. 就地操作(_ 后缀) 与 requires_grad 叶子的坑；
4. 与 NumPy 互转的内存共享陷阱；
5. 广播、cat/stack、item/tolist、索引取数。

运行（仅 CPU）：
    python Tensor基础与数据类型_sample.py
"""
import numpy as np
import torch


def demo_create() -> None:
    print("=" * 64)
    print("实验1: 创建与三要素 (shape / dtype / device)")
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])          # 从 Python 列表
    b = torch.randn(2, 3)                             # 标准正态
    c = torch.zeros(2, 3, dtype=torch.float64)
    idx = torch.tensor([0, 2], dtype=torch.long)      # 索引/标签默认 int64
    print(f"    a: shape={tuple(a.shape)} dtype={a.dtype} numel={a.numel()}")
    print(f"    b: shape={tuple(b.shape)} dtype={b.dtype} 默认float32")
    print(f"    c: dtype={c.dtype} (float64=双精度, 一般训练不用)")
    print(f"    idx dtype={idx.dtype} -> .to(torch.int32) 可转, 但默认长整型最稳")
    print(f"    device={a.device}")
    # 创建新 Tensor 的常用方式
    print("    arange:", torch.arange(6).tolist(), "| eye(2):",
          torch.eye(2).tolist(), "| full:", torch.full((2,), 7).tolist())
    print("    坑: torch.Tensor(3) 是[未初始化的3个元素], 不是元素值为3 ->",
          torch.Tensor(3).numel(), "个元素")


def demo_view_vs_reshape() -> None:
    print("=" * 64)
    print("实验2: view / reshape / transpose / permute 与连续性")
    x = torch.arange(12).reshape(3, 4)                # reshape 连续输入不拷贝
    print(f"    x = arange(12).reshape(3,4), is_contiguous={x.is_contiguous()}")
    print(f"    x.view(4,3): {x.view(4, 3).tolist()}  (共享存储的视图)")

    y = x.transpose(0, 1)                             # (4,3) 视图, 底层没动
    print(f"    y = x.transpose(0,1) -> shape={tuple(y.shape)}, "
          f"is_contiguous={y.is_contiguous()}")
    try:
        y.view(12)
    except RuntimeError as e:
        print(f"    ❌ y.view(12) 报错: {str(e)[:60]}...")
    z = y.reshape(12)                                  # reshape 自动 contiguous 兜底
    print(f"    ✅ y.reshape(12) 成功(内部拷贝) -> {z.tolist()}")

    # permute 任意换轴
    t = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    p = t.permute(2, 0, 1)                            # (4,2,3)
    print(f"    permute(2,0,1): {tuple(t.shape)} -> {tuple(p.shape)}, "
          f"连续={p.is_contiguous()}")
    # 共享存储验证: 改视图的底层数据, 源也变
    xv = x.view(12)
    xv[0] = -1
    print(f"    view 与源共享存储: x[0,0] 被视图改成 -1 -> {x[0, 0].item()}")
    print("    (要独立副本请 .clone())")


def demo_inplace() -> None:
    print("=" * 64)
    print("实验3: 就地操作(_) 与 autograd")
    w = torch.randn(3, requires_grad=True)            # 叶子张量
    try:
        w.add_(1.0)
    except RuntimeError as e:
        print(f"    ❌ 对 requires_grad 叶子 w.add_(1.0) -> {str(e)[:70]}")
    # 正确做法: 在 no_grad 里改, 或对非叶子操作
    with torch.no_grad():
        w.add_(1.0)
    print(f"    ✅ with torch.no_grad(): w.add_(1.0) 成功, w.sum={w.sum().item():.2f}")
    plain = torch.zeros(2, 2)
    plain.zero_()
    plain.mul_(3.0)
    print(f"    ✅ 非叶子就地: zero_() 然后 mul_(3.0) -> {plain.tolist()}")
    print("    就地优点: 省内存(不产生新张量); 代价: 破坏视图共享/autograd 图")


def demo_numpy() -> None:
    print("=" * 64)
    print("实验4: 与 NumPy 互转 (注意共享内存)")
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t_a = torch.from_numpy(np_a)                      # 共享内存!
    np_a[0, 0] = 99.0
    print(f"    from_numpy 后改 numpy, tensor 也跟着变: t_a[0,0]={t_a[0, 0].item()}")
    t_b = torch.randn(2, 2)
    np_b = t_b.numpy()
    t_b.mul_(2.0)                                     # 就地改 tensor
    print(f"    .numpy() 后改 tensor, numpy 数组也变: np_b[0,0] 已乘2")
    # 需要独立副本
    t_c = torch.from_numpy(np.array([1.0, 2.0])).clone()
    t_c.mul_(10.0)
    print(f"    .clone() 得独立副本 -> {t_c.tolist()} (源 numpy 不受影响)")
    # 精度提醒: int64->float32 大数会丢精度
    big = torch.tensor(2 ** 24 + 1)                   # 16777217 > float32 可精确表示范围
    print(f"    精度坑: int(2**24+1) 转 float32 = {big.float().item():.0f} (丢了1)")


def demo_ops() -> None:
    print("=" * 64)
    print("实验5: 广播 / cat / stack / 取值")
    a = torch.ones(3, 1)
    b = torch.arange(4).float()
    print(f"    广播: (3,1) + (4,) -> {tuple((a + b).shape)}  一维对齐为 (1,4)")
    c1 = torch.randn(2, 3)
    c2 = torch.randn(4, 3)
    print(f"    cat 沿 dim=0: {tuple(c1.shape)} + {tuple(c2.shape)} -> "
          f"{tuple(torch.cat([c1, c2], dim=0).shape)}")
    s1 = torch.randn(2, 3)
    s2 = torch.randn(2, 3)
    print(f"    stack 新建维: {tuple(torch.stack([s1, s2], dim=0).shape)} (batch轴)")
    scalar = torch.tensor(3.14)
    print(f"    标量 item() = {scalar.item():.4f}; 向量 tolist() = "
          f"{torch.tensor([1.0, 2.0]).tolist()}")
    m = torch.arange(9).reshape(3, 3)
    print(f"    高级索引 m[1]={m[1].tolist()} m[:, 0]={m[:, 0].tolist()} "
          f"mask取数={(m > 4).sum().item()}个>4")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__} / NumPy {np.__version__}\n")
    demo_create()
    demo_view_vs_reshape()
    demo_inplace()
    demo_numpy()
    demo_ops()
