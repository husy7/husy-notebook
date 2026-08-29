# -*- coding: utf-8 -*-
"""
CNN 经典架构：LeNet / AlexNet / VGG / ResNet —— 典型代码演示
============================================================
覆盖知识点：
  1. 用 PyTorch 手写一个极简 LeNet（理解卷积块堆叠）
  2. VGG 的"3x3 堆叠"设计模式
  3. ResNet 残差连接（回顾并用 torchvision 直接调用）
  4. 对比各经典模型在相同输入下的输出与参数量
  5. 用 torchvision 现成模型做迁移学习

依赖：pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# =====================================================================
# 一、手写 LeNet-5（经典奠基架构）理解卷积块堆叠
# =====================================================================
class LeNet5(nn.Module):
    """
    LeNet 结构：
      输入(1,32,32) → Conv6×5x5 → Pool2x2 → Conv16×5x5 → Pool
      → flatten → FC120 → FC84 → FC10
    适用于灰度图（单通道）小图，MNIST 手写数字。
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取部分：两组 卷积+池化
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),   # (32->32)
            nn.Tanh(),
            nn.AvgPool2d(2),                             # 32->16
            nn.Conv2d(6, 16, kernel_size=5),             # 16->12
            nn.Tanh(),
            nn.AvgPool2d(2),                             # 12->6
        )
        # 分类头：全连接
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)          # 展平为 (N, 16*6*6)
        return self.classifier(x)

# 前向验证 LeNet
lenet = LeNet5()
x_mnist = torch.randn(4, 1, 32, 32)      # 4张 32x32 灰度图
out_lenet = lenet(x_mnist)
print("[LeNet5] 输入(1,32,32) → 输出", tuple(out_lenet.shape),
      f"  参数量={sum(p.numel() for p in lenet.parameters())}")

# =====================================================================
# 二、VGG 风格：只用 3x3 卷积反复堆叠 + 每段后 maxpool
# =====================================================================
def make_vgg_block(in_c, out_c, n_conv=2):
    """一个卷积块：连续 n_conv 个 3x3 卷积 + 一个 ReLU。"""
    layers = []
    for i in range(n_conv):
        layers += [nn.Conv2d(in_c if i == 0 else out_c, out_c,
                             kernel_size=3, padding=1),
                   nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class MiniVGG(nn.Module):
    """极简 VGG：两个 3x3 卷积块 + 分类头。"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = make_vgg_block(3, 16, n_conv=2)   # 3 通道 RGB
        self.block2 = make_vgg_block(16, 32, n_conv=2)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)    # 假设输入 32x32

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = torch.flatten(x, 1)
        return self.fc(x)

vgg = MiniVGG()
x_rgb = torch.randn(2, 3, 32, 32)
print("\n[MiniVGG] 输入(3,32,32) → 输出", tuple(vgg(x_rgb).shape),
      f"  参数量={sum(p.numel() for p in vgg.parameters())}")

# =====================================================================
# 三、ResNet：残差连接（与 torchvision 现成模型对照）
# =====================================================================
class BasicBlockDemo(nn.Module):
    """简版残差块用于演示，与 torchvision 的 BasicBlock 结构一致。"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Identity()            # 恒等捷径（通道/尺寸一致时）
        if stride != 1 or in_c != out_c:         # 不一致时用 1x1 卷积对齐
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))    # ★ 残差相加

# 直接用 torchvision 预训练模型（工程最常用，DataParallel 已封装好）
import torchvision.models as models

def load_pretrained(name):
    """加载 torchvision 预训练模型，并返回模型与参数量。"""
    weights_map = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    }
    model_fn = getattr(models, name)
    m = model_fn(weights=weights_map[name])        # 下载 ImageNet 权重（需联网）
    n_params = sum(p.numel() for p in m.parameters())
    return m, n_params

# 若无法联网，用随机初始化的模型做演示（仅演示 shape，不下载）
def demo_shape(model, ch=3, h=224, w=224):
    with torch.no_grad():
        out = model(torch.randn(1, ch, h, w))      # (1, 1000) 类别 logits
    return tuple(out.shape)

# 演示 ResNet（等 torchvision 模型可直接加载，这里用结构一致的随机模型）
from torchvision.models import resnet18
res_net = resnet18(weights=None)                   # 不下载权重
print("\n[ResNet18] 前向 shape =", demo_shape(res_net),
      f"  参数量={sum(p.numel() for p in res_net.parameters()):,}")

# =====================================================================
# 四、迁移学习：用预训练 ResNet 适配 10 分类新任务
# =====================================================================
model_transfer = resnet18(weights=None)
# 替换最后的全连接分类头 1000 → 10
in_features = model_transfer.fc.in_features
model_transfer.fc = nn.Linear(in_features, 10)
print("\n[迁移] 替换分类头: fc.in =", in_features, "→ fc.out = 10")
print("[迁移] 新模型参数量 =", sum(p.numel() for p in model_transfer.parameters()))

# 迁移学习冻结主干策略：
for param in model_transfer.parameters():
    param.requires_grad = False          # 冻结特征提取部分
for param in model_transfer.fc.parameters():
    param.requires_grad = True           # 只训练新分类头
trainable = sum(p.numel() for p in model_transfer.parameters() if p.requires_grad)
print("[迁移] 冻结后仅可训练参数 =", trainable, "（省算力、防过拟合）")

# =====================================================================
# 小结
# =====================================================================
# 经典 CNN 演进：LeNet(奠基) → AlexNet(规扩大) → VGG(3x3堆叠简洁)
#              → ResNet(残差打通深层)
# 工程上迁移学习：加载预训练 → 换分类头 → 冻结主干 → fine-tune。
