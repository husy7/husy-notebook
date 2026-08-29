---
title: "ResNet：残差网络结构拆解"
tags: [深度学习, CNN, ResNet, 残差连接]
date: 2026-08-29
---

# ResNet：残差网络结构拆解

## 一、核心思想

**问题**：网络越深性能是否越好？实验发现普通深网络会**退化（degradation）**——训练误差不降反升，并非过拟合，而是难训练（梯度传播困难）。

**解决**：ResNet 提出**残差连接（skip / shortcut connection）**——让每一层学习"残差" $F(x)$，而不是直接拟合目标 $H(x)$，网络整体输出为 $H(x) = F(x) + x$。

- 若残差为 0，网络退化为恒等映射，至少不更差 → 保证了"深不劣于浅"。
- 梯度可沿 shortcut 直通，缓解了梯度消失/退化问题。

## 二、残差块结构

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

- 恒等捷径：当输入输出通道数一致、尺寸一致时直接相加；否则加一个 **1×1 卷积升维**进行匹配。

```python
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        # 通道数或尺寸变化时，用 1x1 卷积适配
        self.shortcut = None
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + (self.shortcut(x) if self.shortcut else x)  # 残差相加
        return F.relu(out)
```

## 三、ResNet 系列参数对比

| 网络 | 层数 | 残差块类型 | 参数量 | 关键点 |
|------|------|-----------|--------|--------|
| ResNet-18/34 | 18/34 | BasicBlock（2×3×3） | ~11M/21M | 轻量 |
| ResNet-50 | 50 | Bottleneck（1×1→3×3→1×1） | ~25M | 深度应用首选 |
| ResNet-101/152 | 101/152 | Bottleneck | ~44M/60M | 更强 |

**Bottleneck 块**用 1×1 降维→3×3 卷积→1×1 升维，大幅减少计算量，支撑更深网络。

## 四、为什么有效

- **梯度直通**：反向传播时梯度沿 shortcut 传回浅层，避免长链连乘衰减。
- **隐式恒等偏置**：网络默认学习恒等，多学一点残差即可，优化更平滑。
- 实践验证：ResNet-152 比普通 152 层网络大幅提升，ImageNet top-5 错误降至 3.57%。

## 五、边界与坑

- ❌ 不加 `(stride, padding)` 参数的 Conv 会导致特征图尺寸对齐失败 → 相加不匹配。✅ 严格对齐 stride/padding/bias 设置。
- ❌ 忘加 `shortcut` 升维逻辑，输入输出通道不一致就相加报错。✅ 用 1×1 卷积对齐。
- ❌ 残差块里 **先 BN 后相加再 ReLU** 与 "PreAct" 顺序混用。✅ 官方 ResNet 顺序为 Conv→BN→ReLU→(残差++)→ReLU。
- 边界：ResNet 是 CNN 范式代表，对序列/图数据不直接适用；大规模时应改用 Bottleneck 结构节显存算力。

## 六、关联

- 前置知识：CNN、批量归一化、反向传播。
- 类似概念：VGG（无 shortcut，靠堆更深的同构卷积层）vs ResNet（加残差）。
- 进阶：ResNeXt（分组卷积）、DenseNet（密集连接）、残差思想在 Transformer（Pre-LN 层连接）中的沿用。

## 七、参考

- Deep Residual Learning for Image Recognition — https://arxiv.org/abs/1512.03385
- PyTorch torchvision models — https://pytorch.org/vision/stable/models/resnet.html
