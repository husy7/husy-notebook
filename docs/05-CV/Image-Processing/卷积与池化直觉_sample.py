# -*- coding: utf-8 -*-
"""
卷积与池化直觉演示（可离线运行，仅依赖 torch）

覆盖知识点：
  1. 输出尺寸公式：手算 vs F.conv2d 实测（stride/padding/dilation 三个变量全验证）
  2. 感受野递推：从输出层往回算，对比"两层3x3"与"单层5x5"的感受野
  3. 1x1 卷积 = 通道线性混合（不改变 H/W）
  4. MaxPool vs stride=2 卷积两种降采样
  5. GlobalAvgPool 替代 Flatten+Linear（参数量对比）
  6. 空洞卷积：不降分辨率放大感受野

运行：python 卷积与池化直觉_sample.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out(h_in, k, p=0, s=1, d=1):
    """输出尺寸公式：k_eff = d*(k-1)+1; H_out = floor((H+2p-k_eff)/s)+1"""
    k_eff = d * (k - 1) + 1
    return (h_in + 2 * p - k_eff) // s + 1


def receptive_field(kernels, strides):
    """从最后一层往回递推感受野。
    kernels[i] 第 i 层有效核大小；strides[i] 第 i 层 stride。
    规则：r_0=1；r_i = r_{i-1} + (k_i - 1) * Π_{j<i} s_j
    """
    r = 1
    stride_prod = 1
    for k, s in zip(kernels, strides):
        r = r + (k - 1) * stride_prod
        stride_prod *= s
    return r


def demo_output_size():
    """手算公式 vs 真实卷积输出，全部一致才算对。"""
    print("=" * 60)
    print("1) 输出尺寸公式验证 (输入 H=64, 单通道)")
    x = torch.randn(1, 1, 64, 64)
    cases = [
        dict(k=3, p=1, s=1, d=1),   # same 卷积
        dict(k=3, p=0, s=2, d=1),   # stride 2 降采样
        dict(k=5, p=2, s=1, d=2),   # 空洞 3x3 d=2 → 有效 5x5
        dict(k=7, p=0, s=2, d=1),
    ]
    for c in cases:
        w = torch.randn(1, 1, c["k"], c["k"])
        y = F.conv2d(x, w, padding=c["p"], stride=c["s"], dilation=c["d"])
        calc = conv_out(64, c["k"], c["p"], c["s"], c["d"])
        real = y.shape[2]
        flag = "OK" if calc == real else "MISMATCH!"
        print(f"  k={c['k']} p={c['p']} s={c['s']} d={c['d']}: "
              f"公式={calc} 实测={real}  [{flag}]")


def demo_receptive_field():
    """两层 3x3 堆叠 == 感受野 5x5；三层 == 7x7（用公式证明）。"""
    print("=" * 60)
    print("2) 感受野递推：小核堆叠放大感受野")
    print("  两层 3x3 s=1 :", receptive_field([3, 3], [1, 1]), "x",
          receptive_field([3, 3], [1, 1]))
    print("  三层 3x3 s=1 :", receptive_field([3, 3, 3], [1, 1, 1]), "x",
          receptive_field([3, 3, 3], [1, 1, 1]))
    print("  单层 7x7 s=1 :", receptive_field([7], [1]), "x",
          receptive_field([7], [1]))
    print("  两段含池化: 3x3(s1) + MaxPool(s2) + 3x3(s1):",
          receptive_field([3, 2, 3], [1, 2, 1]), "x",
          receptive_field([3, 2, 3], [1, 2, 1]), "  (池化按 k=2,s=2 计)")
    # 参数对比：两层 3x3 = 2*9=18 参数/通道对，一层 7x7 = 49
    print("  参数对比: 两层3x3=18 权重 vs 单层7x7=49 权重")


def demo_conv1x1():
    """1x1 卷积只做通道混合，空间尺寸不变。"""
    print("=" * 60)
    print("3) 1x1 卷积 = 通道线性混合")
    x = torch.randn(2, 64, 8, 8)
    conv = nn.Conv2d(64, 32, kernel_size=1)   # C_in=64 -> C_out=32
    y = conv(x)
    print(f"  输入 {tuple(x.shape)} -> 输出 {tuple(y.shape)}（H/W 不变，通道 64->32）")


def demo_downsample():
    """MaxPool(2) 与 stride=2 卷积都能把分辨率减半。"""
    print("=" * 60)
    print("4) 两种降采样对比（输入 1x8x8）")
    x = torch.randn(1, 8, 8, 8)
    mp = nn.MaxPool2d(kernel_size=2, stride=2)          # 无参数、取最大值
    sc = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)  # 可学习降采样
    y_mp = mp(x)
    y_sc = sc(x)
    print(f"  MaxPool(2,2)      -> {tuple(y_mp.shape)}  参数: "
          f"{sum(p.numel() for p in mp.parameters())}")
    print(f"  Conv3x3(s=2,p=1)  -> {tuple(y_sc.shape)}  参数: "
          f"{sum(p.numel() for p in sc.parameters())}")


def demo_gap():
    """GlobalAvgPool：把 HxW 压成 1x1，替代 Flatten+Linear 的参数量爆炸。"""
    print("=" * 60)
    print("5) GlobalAvgPool vs Flatten+Linear（最后一层特征图 512x7x7）")
    x = torch.randn(4, 512, 7, 7)
    num_classes = 1000
    # 方式 A：Flatten -> Linear(512*7*7, 1000)
    fc = nn.Linear(512 * 7 * 7, num_classes)
    params_a = sum(p.numel() for p in fc.parameters())
    # 方式 B：AdaptiveAvgPool(1) -> Linear(512, 1000)
    gap = nn.AdaptiveAvgPool2d(1)
    fc2 = nn.Linear(512, num_classes)
    params_b = sum(p.numel() for p in fc2.parameters())

    y_a = fc(x.flatten(1))
    y_b = fc2(gap(x).flatten(1))
    print(f"  Flatten+Linear:   {tuple(y_a.shape)}  参数 {params_a:,}")
    print(f"  GAP+Linear:       {tuple(y_b.shape)}  参数 {params_b:,}"
          f"（约 1/{params_a // params_b}）")


def demo_dilated():
    """空洞卷积：k=3,d=2 → 感受野 5x5，但 H/W 可用 padding 保持。"""
    print("=" * 60)
    print("6) 空洞卷积放大感受野但不降分辨率")
    x = torch.randn(1, 16, 32, 32)
    conv_d = nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2)
    y = conv_d(x)
    print(f"  k=3, d=2, p=2: 输入 {tuple(x.shape)} -> 输出 {tuple(y.shape)}")
    print(f"  有效感受野 = {2 * (3 - 1) + 1} x {2 * (3 - 1) + 1} = 5x5, "
          f"但每层权重仍只有 3x3x16x16 = {3 * 3 * 16 * 16} 个")


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_output_size()
    demo_receptive_field()
    demo_conv1x1()
    demo_downsample()
    demo_gap()
    demo_dilated()
    print("\n全部演示完成 ✓")
