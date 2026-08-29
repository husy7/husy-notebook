---
title: "CNN 经典架构：LeNet / AlexNet / VGG / ResNet"
tags: [计算机视觉, CNN, 卷积网络, LeNet, AlexNet, VGG]
date: 2026-08-29
---

# CNN 经典架构：LeNet / AlexNet / VGG / ResNet

## 一、核心思想

卷积神经网络（CNN）利用**权重共享（卷积核滑动）**与**局部感受野**，在图像这种高维、平移相关的数据上远优于全连接网络：

- **卷积层**提取从低级（边缘、颜色）到高级（形状、物体部件）的层级特征。
- **池化/下采样**降低分辨率、减轻计算、增加平移不变性。
- **经典架构 = 卷积块堆叠 → 全局池化 → 全连接分类头**。

这条演进主线（LeNet → AlexNet → VGG → ResNet）代表了 CNN 加深、加宽、并在后期靠**残差连接**稳住深网训练的路径。

## 二、里程碑架构对比

| 网络 | 年代 | 关键突破 | 深度 | 特点 |
|------|------|----------|------|------|
| **LeNet-5** | 1998 | 卷积+池化奠基 | 5 层 | 手写数字（MNIST）奠定 CNN 范式 |
| **AlexNet** | 2012 | ImageNet 夺冠、ReLU/GPU/Dropout | 8 层 | 深度学习引爆点 |
| **VGG** | 2014 | 统一小卷积核 3×3 反复堆叠 | 16/19 层 | 简洁、好用、附参数量大 |
| **ResNet** | 2015 | 残差连接打通深层训练 | 18~152 层 | ImageNet top-5 降至 3.57% |

## 三、演进脉络

### 3.1 LeNet-5
卷积-池化-卷积-池化... → 全连接。证明 CNN 可行，但无力大规模。

### 3.2 AlexNet：为什么 2012 年引爆
- **ReLU** 激活取代 Sigmoid → 缓解梯度消失、加速。
- **GPU（两块 GTX580）并行**训练，让大规模训练成为可能。
- **Dropout + 数据增广**抗过拟合。
- 用 **LRN**（后已少用）做局部归一化。

### 3.3 VGG：极简主义
- 只用 **3×3 卷积**（两个 3×3 等价一个 5×5 感受野但参数更省）。
- 规律重复"Conv→Conv→MaxPool"块。简单好实现、可迁移性极强（可作特征提取器）。
- 缺点:参数多（~138M）、计算量大。

### 3.4 ResNet：解"越深越好"的陷阱
- 见 03 板块 [ResNet残差网络拆解](../../03-DeepLearning/Model-Zoo/ResNet残差网络拆解.md) —— 残差连接让梯度直通、支持上百层。

## 四、搭建/使用现成模型

```python
import torchvision.models as models
import torch

# 用预训练 VGG16 做特征提取（冻结权重）
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
features = vgg(torch.randn(1, 3, 224, 224))
print(features.shape)   # (1, 1000) 类别 logits

# ResNet 更常用作 backbone
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
```

> 💡 实际项目通常加载 torchvision 预训练权重做 fine-tune / 替换分类头迁移学习，而非从零训练。

## 五、CNN vs 后续迭代

| 方向 | 代表 | 要点 |
|------|------|------|
| **多分支/分组** | ResNeXt、Inception | 用分组/并行卷积分支提升能力与效率 |
| **轻量部署** | MobileNet、EfficientNet | 深度可分离卷积、通道缩放 |
| **注意力融入** | SE-Net、ViT | 引入通道/空间/全局注意力 |

**ViT（Vision Transformer）** 用 Transformer 把图片拆成 patch 当 token，在超大数据上达到/超越 CNN，是现代多模态与基础视觉模型（如 CLIP）趋势，详见 [Transformer结构拆解](../../03-DeepLearning/Model-Zoo/Transformer结构拆解.md)。

## 六、边界与坑

- ❌ 直接用批次图片形状与网络默认输入不匹配（VGG16 要求 224×224）。✅ 先 resize/normalize（见图像处理笔记）。
- ❌ 迁移学习未冻结或误冻 `classifier` 层 → 训练失效或权重被随机破坏。✅ 明确 `requires_grad` 策略。
- ❌ VGG 门面层（~138M 参）直接全量训练 → 显存爆、慢。✅ 优先用 ResNet/EfficientNet 或冻结。
- ❌ 小数据集全量训练深 CNN → 严重过拟合。✅ 用预训练 + 冻结/低学习率 + 强增广。
- 边界：经典 CNN 在处理**全局长程依赖**与**超大感受野**上不如 Transformer；需超大数据的视觉任务，CNN 有上限，ViT 更有潜力。

## 七、关联

- 前置知识：卷积/池化、反向传播、训练循环。
- 同板块：[图像处理与数据增广](..\Image-Processing\图像处理与数据增广.md)。
- 跨界：ResNet/ViT 结构与 03 板块共用；目标检测常以 CNN/ViT 为 backbone（见 [单阶段与两阶段目标检测](..\Object-Detection\目标检测与YOLO.md)）。

## 八、参考

- LeNet — Gradient-Based Learning Applied to Document Recognition
- AlexNet — https://arxiv.org/abs/1404.5997
- VGG — https://arxiv.org/abs/1409.1556
- torchvision 模型库 — https://pytorch.org/vision/stable/models.html
