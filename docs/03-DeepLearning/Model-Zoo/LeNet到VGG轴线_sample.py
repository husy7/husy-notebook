# -*- coding: utf-8 -*-
"""LeNet 到 VGG 演进轴线 —— 演示代码

覆盖要点：
1. LeNet-5 结构（5x5 卷积 + 池化 + 全连接）与逐层参数量;
2. VGG 风格"卷积块"：3x3(padding=1) 堆叠 + 2x2 池化降采样;
3. 关键洞察量化: 3 个 3x3 的参数量 vs 1 个 7x7(同感受野) —— 省~45%;
4. 为什么 FC 头是 VGG 的参数大头(直接打印各块占比);
5. 两个模型实际前向跑通, 验证输出形状。

运行（仅 CPU, 不训练, 直接看结构/参数/形状）:
    python LeNet到VGG轴线_sample.py
"""
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# LeNet-5 (经典版: 输入 1x32x32; MNIST 28x28 可加 2 padding)
# ----------------------------------------------------------------------
class LeNet5(nn.Module):
    """经典 LeNet-5: 输入 1x32x32 (MNIST 28x28 需 pad 到 32)。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),        # 32->28
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), # 28->14
            nn.Conv2d(6, 16, kernel_size=5),       # 14->10
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), # 10->5
        )
        # 5x5x16 = 400 -> 全连接 120/84/10 (经典尺寸)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


# ----------------------------------------------------------------------
# VGG 风格卷积块: 3x3 堆叠 + 池化
# ----------------------------------------------------------------------
def conv_block(in_ch, out_ch, n_convs):
    """一个 VGG 块: n_convs 个 3x3(padding=1)+ReLU, 随后 2x2 池化降采样。"""
    layers = []
    for i in range(n_convs):
        layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGGish(nn.Module):
    """VGG 迷你复刻(通道减半版, 便于 CPU 运行): [64,64,M,128,128,M] + FC。"""

    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_ch, 64, 2),
            conv_block(64, 128, 2),
            conv_block(128, 256, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),     # 32x32 输入池化 3 次 -> 4x4
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


# 真实 VGG-16 的配置(只用来算参数量, 不实例化省内存):
# [64,64,M, 128,128,M, 256,256,256,M, 512,512,512,M, 512,512,512,M]
VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
             512, 512, 512, "M", 512, 512, 512, "M"]


def count_params(model: nn.Module, named: bool = False):
    total = sum(p.numel() for p in model.parameters())
    if named:
        per = [(n, p.numel()) for n, p in model.named_parameters()]
        return total, per
    return total


def vgg16_conv_params(cfg):
    """不实例化, 直接按公式算 VGG-16 卷积部分参数量(含 bias)。"""
    total, in_ch = 0, 3
    for item in cfg:
        if item == "M":
            continue
        total += item * item * 3 * 3 * in_ch + item   # weight + bias
        in_ch = item
    return total


def demo_param_math() -> None:
    """3x3 堆叠 vs 大卷积核: 感受野相同, 参数省多少。"""
    print("=" * 66)
    print("实验1: 参数数学 —— 同感受野下 3x3 堆叠 vs 单一大核 (通道 C)")
    print(f"    {'C':>3} {'3x3堆叠(3个)':>14} {'单个7x7':>12} {'省参%':>8}")
    for C in (64, 128, 256, 512):
        stack = 3 * (3 * 3 * C * C)          # 27 C^2
        big = 7 * 7 * C * C                  # 49 C^2
        save = (1 - stack / big) * 100
        print(f"    {C:>3} {stack:>14,d} {big:>12,d} {save:>7.1f}%")


def demo_lenet() -> None:
    print("=" * 66)
    print("实验2: LeNet-5 —— 逐层参数量")
    net = LeNet5()
    total, per = count_params(net, named=True)
    for n, c in per:
        print(f"    {n:<22} {c:>10,d}")
    print(f"    合计: {total:,d}  (经典 LeNet-5 约 6 万参数)")
    x = torch.randn(2, 1, 32, 32)
    y = net(x)
    print(f"    前向: (2,1,32,32) -> {tuple(y.shape)}")


def demo_vgg() -> None:
    print("=" * 66)
    print("实验3: VGG 风格 —— 卷积块堆叠与形状流")
    net = VGGish()
    x = torch.randn(2, 3, 32, 32)
    # 打印每块输出形状, 直观看到"池化降采样、通道翻倍"
    h = x
    idx = 0
    for name, blk in net.features.named_children():
        h = blk(h)
        print(f"    block{idx}: out {tuple(h.shape)}  (池化后 H,W 减半)")
        idx += 1
    y = net(x)
    print(f"    展平后 -> classifier -> {tuple(y.shape)}")
    total, per = count_params(net, named=True)
    conv_params = sum(c for n, c in per if "features" in n)
    fc_params = sum(c for n, c in per if "classifier" in n)
    print(f"    VGGish 参数量: 卷积 {conv_params:,d} / 全连接 {fc_params:,d}"
          f" (小 FC 头已避免 VGG 的 1.2 亿大头)")

    # 真实 VGG-16 参数量(解析计算, 不占内存)
    conv = vgg16_conv_params(VGG16_CFG)
    fc = (7 * 7 * 512) * 4096 + 4096 + 4096 * 4096 + 4096 + 4096 * 1000 + 1000
    print(f"    真实 VGG-16: 卷积 ≈{conv / 1e6:.1f}M, 全连接 ≈{fc / 1e6:.1f}M, "
          f"合计 ≈{(conv + fc) / 1e6:.1f}M  -> FC 是绝对大头")
    print("    (教训: 大 FC 头后来被 GAP/1x1 收尾取代, 见 ResNet/NiN)")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}\n")
    demo_param_math()
    demo_lenet()
    demo_vgg()
    print("\n轴线总结:")
    print("  LeNet(5x5, ~6万) -> AlexNet(深+ReLU) -> VGG(3x3堆叠, ~1.38亿):")
    print("  小核深堆 = 同感受野更省参数 + 更多非线性; 演进主线是'卷积块'化。")
