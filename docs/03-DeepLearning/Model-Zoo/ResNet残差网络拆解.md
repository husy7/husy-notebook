---
title: "ResNet：残差网络结构拆解"
tags: [深度学习, CNN, ResNet, 残差连接]
date: 2026-08-29
---

# ResNet：残差网络结构拆解

## 定义

ResNet（Residual Network，残差网络）是何恺明（Kaiming He）等人 2015 年提出的深度卷积神经网络架构，核心构建单元是**残差块（Residual Block）**，通过**残差连接 / 捷径连接（skip / shortcut connection）**让每一层去学习"残差" $F(x)$，而不是直接拟合完整的目标映射 $H(x)$，网络整体输出为 $H(x) = F(x) + x$。

它要解决的问题是深网络的**退化（degradation）**现象：实验发现普通深网络并非"越深越好"，当层数堆到一定深度后训练误差不降反升——这并非过拟合，而是网络难以优化（梯度沿深层长链传播衰减，恒等子网络难以逼近），导致深层网络反而差于浅层网络。

核心特征有三点：(1) 显式保留恒等路径——若残差 $F(x)$ 学成 0，网络退化为恒等映射，结果"至少不更差"，从结构上保证了"深不劣于浅"；(2) 梯度可沿 shortcut 直通回浅层，缓解梯度消失与退化问题；(3) 优化目标从"拟合完整映射"降为"拟合残差"，函数空间更平滑、更易优化。

适用范畴：以图像分类为代表的计算机视觉主干网络（分类、检测、分割、人体姿态等几乎都可换用 ResNet 作 backbone）；其残差思想还延伸到了 ResNeXt、DenseNet，以及 Transformer 的 Pre-LN 连接等后续架构中，属于深度学习最基础、复用率最高的范式之一。对序列（如 LSTM 类）或图数据并非天然直接适用，需针对性改造。

## 原理

**为什么这样设计**：深层网络之所以退化，根源在于梯度在跨越多层反向传播时经过长链连乘而衰减/弥散，浅层几乎学不到有效梯度，深层恒等映射又很难被直接拟合。ResNet 引入恒等 shortcut 后，反向传播时梯度多了一条"直通"路径，可以几乎无损地传回浅层，从而让加深网络真正变得可训练。

**残差块机制**：标准 BasicBlock 内做两次 `Conv 3x3 → BN → ReLU`（第二个卷积后只过 BN 不过 ReLU），得到残差 $F(x)$，与输入 $x$ 逐元素相加后再过 ReLU，即 $H(x) = F(x) + x$。残差块的连接结构如下：

```
      x ────────────────┐
      │                │
      Conv 3x3         │
      BN + ReLU        │
      Conv 3x3         │
      BN               │
      (---F(x)---)     │
      ├─────＋─────────┤  →  F(x) + x
           ReLU
```

恒等捷径（identity shortcut）要求输入输出**通道数与空间尺寸完全一致**才能直接相加；当 stride 或通道数变化导致不一致时，需要额外加一个 **1×1 卷积（stride 对齐）升/降维**做投影适配，官方实现中该投影 shortcut 也带 BatchNorm。

**Bottleneck 块（深层变体）**：对 ResNet-50 及以上的深网络，把两个 3×3 卷积换成 **1×1 降维 → 3×3 卷积 → 1×1 升维** 三段式，中间瓶颈通道更窄，大幅减少计算量与参数量，从而支撑 101/152 层这样的更深网络。若残差为 0，网络退化为恒等映射 → 保证"深不劣于浅"。

**ResNet 系列变体参数对比**：

| 网络 | 层数 | 残差块类型 | 参数量 | 关键点 |
|------|------|-----------|--------|--------|
| ResNet-18/34 | 18/34 | BasicBlock（2×3×3） | ~11M/21M | 轻量 |
| ResNet-50 | 50 | Bottleneck（1×1→3×3→1×1） | ~25M | 深度应用首选 |
| ResNet-101/152 | 101/152 | Bottleneck | ~44M/60M | 更强 |

**为什么有效（三点）**：① 梯度直通——反向传播梯度沿 shortcut 传回浅层，避免长链连乘衰减；② 隐式恒等偏置——网络默认先学到恒等映射，再多学一点残差即可，优化更平滑；③ 实践验证——ResNet-152 相比普通 152 层网络大幅提升，ImageNet top-5 错误率降至 3.57%。

## 应用

**典型使用场景**：作为视觉任务的特征提取主干（backbone）做迁移学习/微调——图像分类（ImageNet/CIFAR 等）、目标检测（Faster R-CNN、YOLO 的 backbone）、语义分割（DeepLab、UNet 编码器）、人体关键点等；数据或算力有限时选 ResNet-18/34，追求精度且资源充足时选 ResNet-50/101，大规模训练直接用 Bottleneck 结构以节省显存与算力。

**快速上手步骤**：① 直接用 torchvision 加载预训练权重：`torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`；② 冻结除最后几层外的参数做特征提取，或整体微调；③ 替换分类头 `model.fc = nn.Linear(2048, num_classes)` 适配自己的类别数；④ 归一化输入（ImageNet mean/std）后训练；⑤ 手写网络时按"Conv→BN→ReLU→(残差相加)→ReLU"搭建残差块并严格对齐维度。

**注意事项 / 常见坑**：
- ❌ 不加 `(stride, padding)` 参数的 Conv 会导致特征图尺寸对齐失败 → 相加时形状不匹配。✅ 严格对齐 stride/padding/bias 设置。
- ❌ 忘加 `shortcut` 升维逻辑，输入输出通道不一致就相加报错。✅ 用 1×1 卷积（带 stride）对齐。
- ❌ 把残差块里 **先 BN 后相加再 ReLU** 与 "PreAct"（预激活）顺序混用。✅ 官方 ResNet 顺序为 Conv→BN→ReLU→(残差相加)→ReLU，改变顺序即变成 PreAct-ResNet，行为不同。
- 边界：ResNet 是 CNN 范式代表，对序列/图数据不直接适用；大规模时应改用 Bottleneck 结构以节约显存与算力。

```python
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet 基础残差块：学习残差 F(x)，输出 H(x) = F(x) + x"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # 两条 3x3 卷积主路径；stride 控制下采样，padding=1 保持尺寸，BN 前不加 bias
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        # 通道数或尺寸变化时，用 1x1 卷积投影适配 shortcut
        self.shortcut = None
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c))

    def forward(self, x):
        # 官方顺序：Conv → BN → ReLU → (残差相加) → ReLU，勿与 PreAct 混用
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + (self.shortcut(x) if self.shortcut else x)  # 残差相加
        return F.relu(out)

# 案例详解：以 BasicBlock 堆叠一个 ResNet-18 风格的简化主干——
# block1 = BasicBlock(64, 64)               # 恒等 shortcut，直接相加
# block2 = BasicBlock(64, 128, stride=2)    # 空间减半+通道翻倍 → 触发 1x1 shortcut 投影
# 前向时梯度经 shortcut 直通，深层梯度可无损回传，这正是 ResNet 能训练深网的原因。
```

---
## 关联
- 前置：[[CNN 卷积神经网络]]、[[批量归一化 BatchNorm]]、[[反向传播]]
- 类似：[[VGG]]（区别是 VGG 无 shortcut，靠堆叠更深的同构 3×3 卷积层直接拟合完整映射，易受退化问题限制；ResNet 学残差并显式保留恒等路径）
- 进阶：[[ResNeXt]]（分组卷积）、[[DenseNet]]（密集连接）、残差思想在 Transformer（Pre-LN 层连接）中的沿用

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案 ResNet | 残差连接学习 $F(x)$，恒等映射显式保留，梯度可直通 | 深度视觉主干、ImageNet 预训练迁移、通用分类/检测/分割 |
| 替代方案 VGG | 无 shortcut，靠堆更深的同构 3×3 卷积直接拟合目标 | 浅层轻量任务、特征可视化与风格迁移（感知损失常用 VGG） |
| 替代方案 DenseNet | 层间密集连接，特征全复用 | 参数效率优先、小数据集/需要特征复用的场景 |

---
## 参考
- [Deep Residual Learning for Image Recognition (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385)
- [PyTorch torchvision ResNet models](https://pytorch.org/vision/stable/models/resnet.html)

---
## 具体案例
- [[ResNet残差网络结构拆解 实战示例]](ResNet残差网络结构拆解_sample.py)
