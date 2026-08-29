# -*- coding: utf-8 -*-
"""
ResNet 残差网络 —— 典型代码演示
================================
覆盖知识点：
  1. 手写 BasicBlock 残差块（核心：shortcut 残差相加）
  2. 用 nn.Module 组装一个小型 ResNet（含 1x1 升维对齐）
  3. 对比"有残差" vs "无残差"（PlainNet）在加深时的表现
  4. 使用 torchvision 预训练 ResNet 做迁移

依赖：pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# =====================================================================
# 一、BasicBlock：残差块手写
# =====================================================================
class BasicBlock(nn.Module):
    """残差块：
       x → conv(3x3)→BN→ReLU→conv(3x3)→BN →(+x 或 1x1 适配)→ ReLU
       当通道数/尺寸变化时用 1x1 shortcut 对齐，保证逐元素相加可行。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主体：两层 3x3 卷积（padding=1 保持尺寸）
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # shortcut：当需要改变通道数或下采样(stride>1)时，用 1x1 卷积适配
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))     # 第一段
        out = self.bn2(self.conv2(out))           # 第二段（未激活）
        out = out + self.shortcut(x)              # ★ 残差相加（核心）
        return F.relu(out)


# =====================================================================
# 二、组装一个微型 ResNet（两阶段）
# =====================================================================
class MiniResNet(nn.Module):
    """极简 ResNet：先一个卷积下采样，再两个残差阶段，最后全局池化+线性。"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(          # 输入 (3,32,32)
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.stage1 = BasicBlock(16, 16)             # 保持 16 通道
        self.stage2 = BasicBlock(16, 32, stride=2)   # 下采样 → 32 通道
        self.avgpool = nn.AdaptiveAvgPool2d(1)       # 自适应全局池化 → (32,1,1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# 验证残差块输入/输出通道变化的正确性
net = MiniResNet()
x = torch.randn(2, 3, 32, 32)          # 模拟 2 张 3 通道 32x32 图
out = net(x)
print("[MiniResNet] 输入:", tuple(x.shape), "→ 输出:", tuple(out.shape))
print("[MiniResNet] 参数量:", sum(p.numel() for p in net.parameters()), "个")

# =====================================================================
# 三、关键实验：有残差 vs 无残差（加深 vs 退化）
# =====================================================================
class PlainBlock(nn.Module):
    """无残差的普通块，仅做对比。"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


def build_net(block_type, num_blocks):
    """按给定块类型构造网络：层数由 num_blocks 决定。"""
    c_in, c_hidden, c_out = 16, 16, 32
    f = [block_type(c_in, c_hidden, 1)]
    for _ in range(num_blocks - 1):
        f.append(block_type(c_hidden, c_hidden, 1))
    f.append(block_type(c_hidden, c_out, 2))
    body = nn.Sequential(*f)
    return nn.Sequential(body, nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(c_out, 10))


# 用一张随机图前向，验证不同深度与残差/非残差都能跑通
for name, blk in [("Plain(无残差)", PlainBlock), ("Res(残差)", BasicBlock)]:
    netx = build_net(blk, num_blocks=6)
    outx = netx(torch.randn(1, 16, 32, 32))
    print(f"[{name}] 前向输出 shape =", tuple(outx.shape),
          " 参数量 =", sum(p.numel() for p in netx.parameters()))
print("→ 残差网络在相同深度下梯度传播更顺畅，深层也能稳定训练")

# =====================================================================
# 四、使用 torchvision 预训练 ResNet 做迁移（最常遇到的实际用法）
# =====================================================================
import torchvision.models as models

# 加载 ImageNet 预训练 ResNet18（首次运行会下载权重）
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print("\n[预训练 ResNet18] 原分类头输出维度 =", resnet18.fc.out_features, "(1000 类)")

# 迁移：替换全连接层适配我们的 10 分类任务
num_classes = 10
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
print("[迁移后] 新分类头输出维度 =", resnet18.fc.out_features)

# 冻结主干，只训练新分类头（迁移学习常见策略，少参数省算力防过拟合）
for param in resnet18.parameters():
    param.requires_grad = False     # 冻结所有参数（ImageNet 学到的特征保留）
for param in resnet18.fc.parameters():
    param.requires_grad = True      # 只留新分类头可训练
print("[迁移] 可训练参数数 =",
      sum(p.numel() for p in resnet18.parameters() if p.requires_grad))

# 若用 ResNet50（Bottleneck 结构，更深更强）：
# resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# =====================================================================
# 小结
# =====================================================================
# 残差块核心 = "out + shortcut(x)"，让梯度直通 → 深层可训练；
# 工程上优先用 torchvision 预训练 ResNet 做迁移学习。
