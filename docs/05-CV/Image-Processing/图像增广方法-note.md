---
title: "图像增广方法（Data Augmentation）"
tags: [计算机视觉, 数据增强, 深度学习]
date: 2026-08-30
---

# 图像增广方法（Data Augmentation）

## 定义
一句话总结：增广 = 在训练时对样本施加**保持语义的随机变换**，用"廉价的新样本"换取泛化能力，本质是一种**隐式正则化**，与权重衰减、Dropout 并列。

图像增广（Data Augmentation）是深度学习中在**训练阶段**对每个输入样本随机施加一组不改变其语义（标签不变）的变换（几何、光度、擦除、混合等），从而从有限数据中"免费"扩展出分布多样的训练样本的方法。它要解决的核心问题是：深度模型容量大而训练集数量有限时，模型会直接**记答案**——过拟合训练分布，表现为验证集掉点、对平移/光照等扰动脆弱。

其核心特征可概括为四点：

- **注入先验 / 领域不变性**：分类、检测任务中物体平移、翻转、轻微遮挡、亮度变化**不应改变标签**，增广把这些先验知识编码进训练分布，让特征对这些变化不敏感（鲁棒性）。
- **随机性充当正则**：同一张图每次前向看到不同版本，相当于在样本层面加噪声，降低模型对特定像素的依赖。
- **缓解类别不平衡 / 小样本**：对少数类做更多增广（过采样式增广）可部分缓解长尾问题。
- **适用范畴**：图像分类、目标检测、分割、自监督预训练等视觉任务的训练流水线；对无归纳偏置的 ViT 几乎是必需品（见「关联/进阶」）。

两个边界必须清醒：① 增广**不能凭空创造信息量**，它只是把"假设的真实分布"放大，过度增广（把"猫"裁得只剩耳朵）会**毁掉语义**反而掉点；② **Normalize 不是增广**——它是让输入分布落在模型期望范围内（与预训练统计一致）的**确定性预处理**，训练/验证/测试都要做且参数必须一致。

## 原理
为什么增广有效？本质是**用数据分布的视角解释正则化**：模型只在训练分布上优化，若训练分布太"窄"，大容量网络就记忆训练样本而非学习可迁移特征。增广把与标签无关的变换族采样进训练分布，等价于在样本层面对目标函数做平滑，迫使网络学到的特征对这批变换不敏感，从而缩小训练分布与真实分布的差距。

按变换性质，核心方法可分为五类（机制各异）：

| 类别 | 代表方法 | 作用 / 直觉 |
|---|---|---|
| 几何增广 | RandomResizedCrop、水平翻转、旋转、平移、缩放、Cutout 区域遮挡 | 让模型对位置、尺度、朝向不敏感；RandomResizedCrop 同时模拟"尺度 + 裁剪 + 比例"变化，是 ImageNet 训练标配 |
| 色彩/光度增广 | ColorJitter(亮度/对比度/饱和/色调)、灰度化、PCA 抖动 | 对光照、白平衡变化鲁棒；**注意对医学图/红外图慎用**（色彩可能承载语义） |
| 噪声/擦除 | GaussianNoise、RandomErasing、Cutout | 让模型不依赖局部纹理，鼓励利用全局上下文 |
| 样本混合 | Mixup（像素线性插值 + 标签线性插值）、CutMix（裁剪粘贴 + 标签按面积比混合） | 在样本**之间**制造"插值样本"，软化决策边界、抑制对单样本的过拟合；CutMix 还能让模型学会用局部特征 |
| 组合流水线 | torchvision `transforms.Compose`、Albumentations、imgaug | 多种随机变换按概率/强度叠加 |

**关键机制与公式**：

- **单样本随机变换**：每个 epoch 每张图以不同随机参数过一遍 pipeline（RandomResizedCrop 先随机取 `scale∈(0.08,1.0)`、`ratio∈(0.75,1.333)` 再裁剪缩放），让模型在每个 epoch 看到同一语义的不同外观；`scale` 下限 0.08 ≈ 目标占原图 ≥8%，过低裁剪会丢失语义。
- **样本间插值（Mixup）**：像素与标签同比例线性混合：`x̃ = λ·x_a + (1−λ)·x_b`，`ỹ = λ·y_a + (1−λ)·y_b`（软标签，loss 用 CE 计算），λ 是**整体混合系数**；它制造位于两类之间的样本，把决策边界从"贴边过拟合"拉成平滑过渡。
- **区域拼贴（CutMix）**：在原图内随机裁 patch 交叉粘贴到另一张图，标签按**像素面积比例 λ** 混合（与 Mixup 的 λ 含义不同，别混用）；被裁剪区域逼着模型用剩余局部特征，兼顾上下文与局部。
- **训练/验证流程必须区分**（验证集用随机增广会**污染指标**，测不出真实水平）：

```
训练集:   随机读取 → 随机增广(RandomResizedCrop/Flip/ColorJitter/…)
          → ToTensor → Normalize → 可选 batch 级 Mixup/CutMix → 前向
验证/测试: 读取 → 确定性 Resize+CenterCrop → ToTensor → Normalize → 前向
```

唯一合法例外是 **TTA（Test-Time Augmentation）**：推理时对同一图做多种确定性增广（翻转/多尺度），把多次预测**平均**，用计算换精度，属于推理技巧而非训练增广。

**与模型结构的关系**：CNN 自带平移等变性，对平移增广的"需求"低于无归纳偏置的 ViT——ViT 论文明确：**没有强增广（RandAugment + Mixup + CutMix）和蒸馏，ViT 在 ImageNet-1k 从头训练打不过 ResNet**，这是"增广补归纳偏置"的经典例证。增广还与 Dropout、权重衰减、标签平滑等正则手段协同（Mixup/CutMix 常与 label smoothing 同出现在现代配方里）；检测里 **Mosaic**（YOLO 系四图拼贴）是增广家族一员，直接服务于小目标与上下文多样性。

## 应用
**典型使用场景**：图像分类/检测/分割的迁移学习或从零训练、自监督对比学习、小样本与类别不平衡任务（对少数类过采样式增广）、ViT 等弱归纳偏置架构的从头训练。快速上手步骤如下：

1. 按任务语义约束选择增广族：分类用 RandomResizedCrop+水平翻转+ColorJitter 起步；数字/文字类**禁用旋转与翻转**（旋转 90° 让"6"变"9"）；检测小目标把 RandomResizedCrop 的 `scale` 下限调大，避免小目标被裁没。
2. 用 `transforms.Compose`（或 Albumentations）按概率/强度组合，`ToTensor` 后再 `Normalize`；**换用 pretrain 权重必须用其训练时的 mean/std**（ImageNet 用 `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`），自己从零训练可任意归一。
3. Mixup/CutMix 放在 `DataLoader` 之后、模型 forward 之前（需拿到整个 batch 做线性组合/拼贴）。
4. **不固定增广的随机种子**（增广随机性应在 sample 级，全局固定会让增广被"背下来"）。
5. 验证/测试只走确定性 `Resize+CenterCrop`；指标异常先自查验证集是否误用了随机增广。

**常见坑与正确做法**：

| ❌ 常见错误 | ✅ 正确做法 |
|---|---|
| 验证集也套训练用随机增广 → 指标虚高或虚低、不可复现 | 验证/测试只做确定性 `Resize+CenterCrop`（或直接 Resize 到网络输入） |
| Mixup 只混合图像、标签还是 one-hot | 标签同步混合：`y = α·y_a + (1−α)·y_b`（软标签）；计算 loss 时对混合后标签用 CE |
| CutMix 中 α 取错 / 裁出的 patch 与原图重叠 | patch 在原图内随机裁、交叉粘贴；标签按**像素面积比例** `λ` 混合（与 Mixup 的 λ 含义不同，Mixup 是整体比例） |
| 增广强度过大破坏语义（小目标被裁掉、旋转 90° 让"6"变"9"） | 按任务语义约束增广族；数字/文字类避免旋转翻转；检测小目标用大 scale 下限 |
| RandomResizedCrop 直接用到检测/分割 | 检测的分辨率变化要**同步改标注框坐标**，torchvision 的 `RandomResizedCrop` 对检测不自带 bbox 变换，需用检测专用库（albumentations 或自己写 bbox 变换） |
| 忘了 Normalize，或 Normalize 统计与预训练不一致 | 换用 pretrain 权重必须用其训练时的 mean/std；自己从零训练可任意归一 |
| 每个 epoch 用相同 seed 导致增广可被"背下来" | 不固定增广的随机种子（增广随机性应在 sample 级，不在全局固定） |
| 大批量 + 强增广同时上却掉点就删增广 | 先检查是不是增广语义被破坏 / 学习率是否需要配合增广强度调整；强增广常需更长训练或更大 epoch |
| 把 BatchNorm 统计和增广耦合误解 | BatchNorm 与增广无直接冲突，但 Mixup 混合后的输入分布偏移较大时，配合 label smoothing 或 BN 统计小心调 |

**排查口诀**：验证集准确率远低于训练集 → 优先怀疑过拟合（先查验证集是否误用随机增广，再加强增广/正则/数据）；两者都低 → 欠拟合或数据/标签问题（加大模型容量、调学习率、检查 pipeline 与 Normalize 统计）。另外，"训练时故意不增广反而更好"的场景真实存在：数据规模足够大、或任务对原始外观敏感（OCR、医学病灶纹理等）且增广假设与真实分布偏差过大时，少增广或不增广更稳。

```python
# ================= 图像增广：训练 / 验证双流水线（torchvision，2020~2025 兼容）=================
import torch
from torchvision import transforms

# —— 训练 pipeline：随机增广，每个 epoch 每个样本看到不同版本 ——
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),  # 随机"尺度+裁剪+宽高比"
    transforms.RandomHorizontalFlip(p=0.5),            # 水平翻转（数字/文字/朝向敏感任务要删）
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),   # 光度扰动（医学图/红外图慎用）
    transforms.ToTensor(),                             # HWC(0~255) -> CHW(0~1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 用预训练权重时 mean/std 必须与其一致
                         std=[0.229, 0.224, 0.225]),
])

# —— 验证 pipeline：必须确定性（同一图每次结果一致，指标才可复现）——
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# —— 案例详解：
#  ① RandomResizedCrop(scale=(0.08, 1.0)) 的 0.08 下限 ≈ 目标至少占原图 8%，
#     再低会把"猫"裁成"耳朵"，破坏语义反而掉点；检测小目标请把下限调大。
#  ② 想临时评估某个增广的强度，可用 transforms.RandomApply / RandomChoice 叠加试跑。
#  ③ 验证集若误用 train_tf，指标会虚高/虚低且不可复现——验证/测试只用 val_tf。

# —— Mixup / CutMix 需拿到整个 batch：放在 DataLoader 之后、模型 forward 之前 ——
def mixup(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample()  # Beta 分布采样整体混合系数 λ
    idx = torch.randperm(x.size(0))                       # 本 batch 内随机配对的下标
    x_mix = lam * x + (1 - lam) * x[idx]                  # 像素线性插值
    y_mix = lam * y + (1 - lam) * y[idx]                  # 标签同步混合成软标签 y=λ·y_a+(1−λ)·y_b
    return x_mix, y_mix                                   # loss 对软标签用 CE 计算
# 注：CutMix 则是原图内随机裁 patch 交叉粘贴，标签按像素面积比例 λ 混合
# （CutMix 的 λ = 保留区域面积占比，与 Mixup 的整体混合系数含义不同，别混用）。
```

---
## 关联
- 前置：[[卷积与池化直觉]]（CNN 自带平移等变性/归纳偏置，先理解"架构已自带什么"，才知道增广该补什么）
- 类似：[[交叉验证与数据泄漏]]（区别是：增广是在训练分布内合法制造新样本；而把随机增广误用到验证/测试集属于指标层面的数据泄漏，两者都关乎"数据如何划分与变换"）
- 进阶：[[ViT视觉Transformer]]（区别是：该笔记主角是 ViT 架构，讲清无卷积归纳偏置的模型必须靠 RandAugment+Mixup+CutMix 强增广与蒸馏才能打平 ResNet，是"增广=注入先验"的极端例证；检测方向的 Mosaic 增广见 [[YOLO单阶段系列]]）

---
## 对比选型

| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（训练期随机增广，Compose + RandomResizedCrop/Flip/ColorJitter…） | 对每个样本施加保持语义的随机变换，扩展训练分布、充当隐式正则 | 训练/微调阶段防过拟合，提升对平移、光照、遮挡的鲁棒性 |
| 替代方案：确定性预处理（Resize+CenterCrop+Normalize） | 不引入随机性，只做尺寸统一与输入标准化 | 验证/测试/推理的干净评测（随机增广会污染指标，禁止用于评测） |
| 替代方案：TTA（Test-Time Augmentation） | 推理期对同图多次确定性变换并平均预测 | 不重训只推理时的精度提升，用计算换精度 |
| 替代方案：样本混合增广（Mixup/CutMix） | 在样本之间插值/拼贴并同步混合标签，软化决策边界 | 强分类任务中配合 label smoothing 的现代训练配方 |

---
## 参考
- [PyTorch TorchVision — Transforms 官方文档（RandomResizedCrop / ColorJitter / Compose 等 API 与参数说明）](https://pytorch.org/vision/stable/transforms.html)
- [Albumentations 官方文档（检测/分割场景的 bbox 感知增广）](https://albumentations.ai/docs/)

---
## 具体案例
- [[图像增广方法 实战示例]](图像增广方法_sample.py)
