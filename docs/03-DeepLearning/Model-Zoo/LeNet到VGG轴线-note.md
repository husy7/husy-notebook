---
title: "LeNet 到 VGG 的卷积网络演进轴线"
tags: [深度学习, CNN, LeNet, VGG, Model-Zoo]
date: 2026-08-29
---

# LeNet 到 VGG 的卷积网络演进轴线

> 一句话：从 LeNet-5 到 VGG，CNN 的设计主线是 **"用更多更小的卷积核堆更深"**：LeNet 用 5×5 直接提特征，VGG 证明 **连续堆叠 3×3 等价于一个大核的感受野，但参数量更少、非线性更强**，从此"小卷积核+深堆叠+BN"成为 CNN 主干的标准词汇表。

## 定义

本知识点梳理的是卷积神经网络（CNN）从 LeNet-5（1998）到 AlexNet（2012）再到 VGG（2014）的**架构演进轴线**，从中提炼出一条贯穿性的设计主线：**"用更多、更小的卷积核把网络堆得更深"**，并沉淀出此后 CNN 主干通用的"卷积块（Conv Block）"设计单元与"深度随数据规模增长"的选型原则。

它解决的问题是：当任务规模从小图小数据（MNIST 手写数字）放大到大图大数据（ImageNet 百万级分类）时，网络结构应当如何演进——即深度与宽度如何跟着数据规模走，以及在不断加深网络的同时如何控制参数量、计算量与过拟合风险。

核心特征可概括为三阶段演进：

| 模型 | 年份 | 结构要义 | 规模 |
| --- | --- | --- | --- |
| LeNet-5 | 1998 | 卷积(5×5)→池化→卷积→池化→全连接；奠定"卷积提取+全连接分类"骨架 | ~6 万参数，MNIST 手写数字 |
| AlexNet | 2012 | 更深(8 层) + ReLU + Dropout + 重叠池化 + GPU 并行；ImageNet 冠军引爆深度学习 | ~6000 万参数 |
| VGG | 2014 | **只堆 3×3**：2~3 个 3×3 卷积构成一个"卷积块"，块间 2×2 池化降采样，块数/通道递增 64→128→256→512 | VGG-16 ~1.38 亿，其中全连接占 ~1.2 亿 |

适用范畴：图像分类骨干网络的设计与理解、CNN 感受野与参数量分析，以及阅读 ResNet 等现代架构的必备前置知识。判据一句话：MNIST 级任务用 LeNet 级结构就够，ImageNet 级任务才需要 VGG 级的深度 + 宽度。

## 原理

**为什么"3×3 堆叠"取代"大卷积核"**，是这条轴线的核心机制，理由环环相扣：

1. **感受野等价（数学依据）**：stride=1 时，两个 3×3 卷积的感受野 ≈ 一个 5×5 卷积，三个 3×3 ≈ 一个 7×7 卷积——小核堆叠可以无损"拼出"大核的感受野，这是整个替换成立的前提。
2. **参数量更省（关键推导）**：设通道数为 C（C in / C out 均为 C），则
   - 三个 3×3：`3 × (3×3×C×C) = 27C²`
   - 一个 7×7：`7×7×C×C = 49C²`
   - 结论：**节省约 45% 参数**，且感受野完全相同。
3. **非线性更强**：同样的感受野被拆成 3 次独立的 ReLU 决策，等效于对同一区域做 3 次非线性变换，表达能力更好。
4. **隐式正则化**：参数变少本身就对模型施加了隐式正则，有利于泛化。
5. **工程友好**：3×3 是主流硬件与推理库（cuDNN 等）上优化最成熟的卷积尺寸。

**VGG 卷积块设计（后来 CNN 的通用单元）**：

```
ConvBlock(k) = [ Conv3x3(k) -> ReLU ] × 2~3      # 同分辨率内堆叠
            后接 MaxPool2x2(stride=2)              # 空间减半、通道加倍
```

- 每个块内保持 H×W 不变（`padding=1`），只把通道数翻倍 → 计算量可控、信息不丢；
- 用池化做"降采样"而不是卷积 stride=2 直接降（当时惯例；ResNet 之后两者混用）。

**VGG-16 结构速览（数字 = 每层后通道数）**：

```
[64, 64, 池化] → [128, 128, 池化] → [256,256,256, 池化]
→ [512,512,512, 池化] → [512,512,512, 池化] → FC(4096) → FC(4096) → FC(1000, softmax)
```

输入 224×224×3；参数量级：卷积部分 ~1470 万，全连接 ~1.24 亿。

**这条轴线留下的重要教训**：VGG 的三个全连接层吃掉了约 1.2 亿参数（占总量 ~87%），参数量被尾部"浪费"——这推动后续架构用**全局平均池化（GAP）/ 1×1 卷积**收尾（NiN → ResNet 尾部），从根上消除巨型 FC 头。

## 应用

**典型使用场景**：

- 作为 CNN 骨干的选型依据与"词汇表"：理解任何现代主干（ResNet / DenseNet / EfficientNet）时，先认出其内部的 VGG 风格卷积块。
- 用 VGG 式"小核深堆 + 通道翻倍"的配方，在自己的数据集上搭建或复刻骨干网络。
- 作为参数量 / 感受野 / FLOPs 估算的教学级案例（3×3 vs 7×7 的对比可手推验证）。

**快速上手步骤**：

1. 先判断数据规模与输入尺寸：MNIST 级小图用 LeNet 级浅网即可；ImageNet 级大图才需要 VGG 级深度 + 宽度（结构复杂度跟着数据规模走）。
2. 实际复刻时先用**小变体**（VGG-ish / 通道减半版）验证想法，再放大到目标规模。
3. 逐层核对特征图尺寸：3×3 卷积必须 `padding=1`，分辨率由池化负责减半。
4. 用参数量统计脚本验证手算推导（见下方代码示例）。
5. 小任务用 GAP 或小 FC 头收尾，不要照搬 138M 参数的全连接头。

**常见坑 ❌✅**：

- ❌ 把 LeNet 当通用架构直接用到 224×224 大图上 → 深不下去也表达不了复杂特征。
- ✅ 记住演进关系：结构复杂度跟着**数据规模**走（MNIST 级用 LeNet 就够，ImageNet 级才需要 VGG 级深度+宽度）。
- ❌ 手算参数量忘算 bias / 忘算全连接 `7×7×512×4096` 的巨大体量。
- ✅ 用 `sum(p.numel() for p in model.parameters())` 验证自己的推导。
- ❌ 在 3×3 卷积里忘了 `padding=1` → 每层 H×W 都在缩水，深层直接算到负尺寸。
- ✅ VGG 风格块保持分辨率 → 池化才负责降采样。
- ❌ 盲目照搬 138M 参数的全连接头做小任务 → 过拟合 + 显存爆炸。
- ✅ 小任务用 GAP 或小 FC 头；这恰是历史给出的教训。

```python
# 例1：验证"三个 3×3 ≈ 一个 7×7 感受野、省 ~45% 参数"的推导
import torch.nn as nn

def conv_params(C, k):
    """单个卷积层参数量（C_in=C_out=C），含 bias —— 手算常忘 bias，这里补上"""
    return (k * k * C * C) + C

C = 256
stack3 = 3 * conv_params(C, 3)   # 三个 3×3
single7 = conv_params(C, 7)      # 一个 7×7（感受野与三个 3×3 等价）
print(f"三个3x3={stack3:,}   一个7x7={single7:,}   节省={1 - stack3/single7:.1%}")
# 输出约 45% —— 感受野相同，参数省近一半，还多出两次 ReLU 非线性

# 例2：VGG 风格小变体（通道减半版 + GAP 收尾），避免 138M 巨型 FC 头
class VGGish(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 64, 2), nn.MaxPool2d(2),    # -> 112×112
            self._conv_block(64, 128, 2), nn.MaxPool2d(2),  # -> 56×56
            self._conv_block(128, 256, 3), nn.MaxPool2d(2), # -> 28×28
            self._conv_block(256, 256, 3), nn.MaxPool2d(2), # -> 14×14
        )
        # GAP 替代 VGG 的 FC(4096)×3：小任务不照搬全连接头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 全局平均池化，压成 256 维
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _conv_block(in_c, out_c, n):
        """同分辨率内堆叠 n 个 3×3；padding=1 保证 H×W 不变——忘了就每层缩水"""
        layers = []
        for i in range(n):
            layers.append(nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.features(x))

model = VGGish()
print(f"小变体参数量: {sum(p.numel() for p in model.parameters()):,}")
# 结果远小于 VGG-16 的 1.38 亿 —— 主因正是没有 7×7×512×4096 级别的全连接层；
# 案例要点：结构 = 卷积块(小核+深堆) + 池化降采样 + GAP 收尾，即本轴线沉淀的通用配方
```

---
## 关联
- 前置：[[LeNet-5]]（奠定"卷积提取 + 全连接分类"骨架）；[[感受野]]
- 类似：[[AlexNet]]（区别是____它靠 11×11/5×5 大卷积核 + ReLU/Dropout/重叠池化/GPU 并行取胜，卷积核仍偏大，未收敛到 3×3）；[[NiN / 1×1 卷积]]（区别是____它用 1×1 卷积融合通道并用 GAP 取代全连接收尾，直击 VGG 尾部参数浪费问题）
- 进阶：[[ResNet]]（在 VGG 卷积块上加恒等残差 → 更深不退化）；[[参数量与 FLOPs 估算]]（本文 3×3 vs 7×7 推导是其典型用例）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（3×3 深堆叠 / VGG 风格块） | 连续堆叠 2~3 个 3×3 拼出大感受野：参数省 ~45%、非线性更强；块内 padding=1 保持分辨率、块间池化降采样、通道翻倍 | 中大规模图像分类骨干 / 通用特征提取主干（ImageNet 级），及作为 ResNet 等现代架构的底座 |
| 替代方案：大卷积核直提（LeNet-5 / AlexNet 风格） | 单个 5×5 / 7×7 / 11×11 卷积一步覆盖大感受野，网络浅、实现直接 | 小图小数据（MNIST 级）、极浅网络、快速原型 |
| 替代方案：GAP + 1×1 卷积收尾（NiN 思路） | 1×1 卷积做通道融合，GAP 替代全连接，大幅削减尾部参数 | 小任务防过拟合、显存敏感场景；现代主干收尾的标准做法 |

---
## 参考
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG, arXiv:1409.1556)](https://arxiv.org/abs/1409.1556)
- [Gradient-Based Learning Applied to Document Recognition (LeNet-5, LeCun 1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [torchvision.models.vgg16 官方文档](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html)

---
## 具体案例
- [[LeNet 到 VGG 轴线复刻案例]](LeNet到VGG轴线_sample.py)
