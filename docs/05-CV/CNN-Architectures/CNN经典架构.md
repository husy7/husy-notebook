---
title: "CNN 经典架构：LeNet / AlexNet / VGG / ResNet"
tags: [计算机视觉, CNN, 卷积网络, LeNet, AlexNet, VGG]
date: 2026-08-29
---

# CNN 经典架构：LeNet / AlexNet / VGG / ResNet

## 定义

CNN（卷积神经网络）经典架构是指以 **卷积层 + 池化层堆叠 → 全局池化/展平 → 全连接分类头** 为主干范式的一组里程碑网络：LeNet-5、AlexNet、VGG、ResNet。它解决的核心问题是：在图像这类高维、**平移相关（translation-correlated）**的数据上，全连接网络参数爆炸、无法共享特征、且对空间结构无归纳偏置的问题。

- **权重共享**：卷积核（filter/kernel）在整幅图上滑动，同一组参数被所有位置复用，参数量与输入尺寸无关，只取决于卷积核大小与通道数。
- **局部感受野**：每个神经元只连接输入的一个局部区域，先看局部（边缘、颜色、纹理）再看整体（形状、部件、物体），天然匹配图像的空间结构。
- **层级特征**：浅层卷积提取低级特征（边缘、颜色、角点），深层卷积提取高级语义特征（形状、物体部件）。
- **池化/下采样**：降低特征图分辨率，压缩计算量，同时带来一定平移/形变不变性。
- **适用范围**：图像分类、目标检测、语义分割的特征提取 backbone，以及作为各种现代视觉模型的特征底座；经典 CNN 仍是中大规模视觉任务的主力、预训练迁移学习的事实标准。

LeNet → AlexNet → VGG → ResNet 这条演进主线，代表了 CNN 从"证明可行"到"加深、加宽、并用残差连接稳住深层训练"的完整路径。

## 原理

经典 CNN 的机制建立在"卷积=可学习的局部模板匹配"之上。设输入特征图尺寸为 H×W×C，一个 k×k 卷积核滑过所有位置做内积，输出特征图每个位置的值代表该局部区域与模板的相似度；多个卷积核并列堆叠成多个输出通道，每层可学习的参数约为 k²×C_in×C_out（不含 bias）。

四个里程碑的量化对比一览：

| 网络 | 年代 | 关键突破 | 深度 | 特点 |
|------|------|----------|------|------|
| **LeNet-5** | 1998 | 卷积+池化奠基 | 5 层 | 手写数字（MNIST）奠定 CNN 范式 |
| **AlexNet** | 2012 | ImageNet 夺冠、ReLU/GPU/Dropout | 8 层 | 深度学习引爆点 |
| **VGG** | 2014 | 统一小卷积核 3×3 反复堆叠 | 16/19 层 | 简洁、好用、附参数量大 |
| **ResNet** | 2015 | 残差连接打通深层训练 | 18~152 层 | ImageNet top-5 降至 3.57% |

- **卷积核堆叠的等效感受野**：VGG 证明可以用小卷积核反复堆叠替代大卷积核——**两个 3×3 卷积堆叠等价于一个 5×5 卷积的感受野**（2×(3−1)+3=5），但参数量更省：两个 3×3 为 2×9=18C²，一个 5×5 为 25C²；同时中间多一层非线性，表达能力更强。
- **池化层**：常用 2×2 MaxPool，stride=2，把分辨率减半。代价是丢失部分空间细节，收益是计算量降为 1/4、轻微平移不变性，并为下一层扩大感受野。
- **激活函数演进**：早期用 Sigmoid，深了易梯度消失、训练慢；AlexNet 改用 **ReLU**（max(0, x)），正区间梯度恒为 1，缓解梯度消失并大幅加速收敛。
- **残差连接（ResNet 的核心）**：输出 = F(x) + x，让梯度可以沿恒等映射直通浅层，从而支撑 18~152 层乃至上千层的训练；若某层学不到有用变换，F(x)→0 即可，网络退化为浅层，不会变差——这打破了"越深越难训"的困境。
- **正则化手段**：Dropout 随机丢弃神经元、数据增广（翻转/裁剪/颜色扰动）扩充样本分布，共同抑制深网过拟合；AlexNet 时代的 LRN 局部归一化后来已基本弃用。
- **训练硬件**：AlexNet 用两块 GTX 580 GPU 并行训练，使大规模数据上的深度训练首次变得现实，是 2012 年引爆深度学习的关键工程前提之一。

## 应用

典型用法是**加载预训练权重做迁移学习（fine-tune）**，而不是从零训练：把现成 backbone 当作通用特征提取器，替换最后的分类头适配自己的类别数，再按数据量决定冻结策略。

- **快速上手步骤**：(1) 用 torchvision `models.vgg16(weights=DEFAULT)` / `resnet50(...)` 加载预训练模型；(2) 输入先 resize 到网络要求的尺寸（VGG16/ResNet50 要求 224×224）并按 ImageNet 均值/方差 normalize；(3) 冻结 backbone（`requires_grad=False`）或整体低学习率微调；(4) 替换 `classifier`/`fc` 分类头输出为自己类别数；(5) 训练后用验证集评估并监控过拟合。
- **常见坑 1（输入尺寸不匹配）**：直接把任意尺寸批次图片喂给 VGG16 会报错——它默认输入是 224×224。✅ 先 resize + normalize（见 [[图像处理与数据增广]]）。
- **常见坑 2（冻结策略错误）**：迁移学习时误冻 `classifier` 层，或该冻结的 backbone 没冻结，导致训练失效或预训练权重被随机初始化破坏。✅ 明确每层的 `requires_grad` 策略与优化器参数分组。
- **常见坑 3（显存爆炸）**：VGG 全连接"门面层"参数量约 **138M**，直接全量训练极慢且吃显存。✅ 优先用 ResNet/EfficientNet，或冻结 backbone 只训分类头。
- **常见坑 4（小数据集全量训练）**：小数据集从零训练深 CNN 会严重过拟合。✅ 用预训练权重 + 冻结/低学习率 + 强数据增广。
- **边界**：经典 CNN 在处理**全局长程依赖**与**超大感受野**上不如 Transformer；需要超大规模数据的视觉任务中 CNN 有上限，ViT 更有潜力（见 [[Transformer结构拆解]]）。

```python
import torch
import torchvision.models as models

# —— 用预训练 VGG16 做特征提取 / 分类（先只验证前向形状）——
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg.eval()                                  # 切到 eval 模式，关 Dropout/BN 统计
x = torch.randn(1, 3, 224, 224)             # 注意：VGG16 默认输入必须是 224×224
logits = vgg(x)                             # 输入 → features 卷积栈 → classifier 全连接头
print(logits.shape)                         # (1, 1000) = ImageNet 类别 logits

# —— 迁移学习：ResNet50 作 backbone，替换分类头适配自己的任务 ——
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for p in resnet.parameters():               # 先冻结全部权重
    p.requires_grad = False
num_classes = 10                            # 自己的类别数
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)  # 替换分类头
# 此时只训练 fc 头（或解冻最后几层做低学习率微调），即可在自己数据集上 fine-tune

# 💡 实际项目通常加载 torchvision 预训练权重做 fine-tune / 替换分类头迁移学习，而非从零训练。
```

---
## 关联
- 前置：[[图像处理与数据增广]]（卷积/池化、归一化的基础）；另需掌握卷积/池化、反向传播、训练循环等基础知识。
- 类似：[[Transformer结构拆解]]（区别是_以自注意力替代卷积核滑动、全局建模长程依赖，需超大数据量才能超越 CNN_）
- 进阶：[[ResNet残差网络拆解]]（残差连接使梯度直通、支持上百层深网）；视觉任务下游常以 CNN/ViT 为 backbone，见 [[目标检测与YOLO]]（单阶段与两阶段目标检测）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 经典 CNN 卷积堆叠（LeNet→AlexNet→VGG→ResNet，本文方案） | 权重共享卷积核提取层级特征 + 池化降采样，ResNet 以残差连接稳住深层训练 | 中大规模数据、平移相关图像任务；分类/检测/分割的通用 backbone，预训练迁移学习 |
| ViT（Vision Transformer） | 把图片拆成 patch 当 token，用全局自注意力建模任意距离依赖 | 超大规模数据、全局长程依赖任务；现代多模态与基础视觉模型（如 CLIP）底座 |
| 轻量 CNN（MobileNet / EfficientNet） | 深度可分离卷积、通道缩放（compound scaling）压低算力 | 移动端/边缘部署、算力受限场景 |
| 多分支/注意力增强 CNN（ResNeXt / Inception / SE-Net） | 分组或并行卷积分支提升能力，通道注意力加权 | 在经典 CNN 框架内追求更高精度/效率的折中方案 |

---
## 参考
- LeNet（Gradient-Based Learning Applied to Document Recognition）— [Lecun 原文 PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- AlexNet — https://arxiv.org/abs/1404.5997
- VGG — https://arxiv.org/abs/1409.1556
- torchvision 模型库 — https://pytorch.org/vision/stable/models.html

---
## 具体案例
- [[CNN经典架构实战示例]](CNN经典架构_sample.py)
