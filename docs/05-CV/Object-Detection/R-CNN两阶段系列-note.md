---
title: "R-CNN 两阶段系列"
tags: [目标检测, 两阶段检测, 深度学习]
date: 2026-08-30
---

# R-CNN 两阶段系列

## 定义

两阶段目标检测（Two-stage Object Detection）是"先粗定位候选，再精分类回归"的检测范式：第一阶段从图像中找出"哪里有物体候选"（Region Proposal），第二阶段对每个候选区域（RoI）做分类 + 边界框回归。本笔记的 R-CNN 两阶段系列指这一范式的三代奠基工作：**R-CNN（2014）→ Fast R-CNN（2015）→ Faster R-CNN（2016）**，它们分别回答了三个递进问题——"CNN 能不能直接用于检测""能不能避免对每个框各算一次 CNN""能不能把找框这一步也训练进网络"。

- **解决什么问题**：检测 = 定位 + 分类。朴素滑窗穷举要在全图所有位置 × 尺度上跑分类器，代价过高；R-CNN 系列给出"传统候选框算法/可学习 RPN + 深度特征 + 判别分类器"的组合路线，把检测从"穷举"变成"先粗筛再细看"。
- **核心特征**：三代演进的主线是**消灭重复计算与外部依赖**——从"每框过一遍 CNN"（R-CNN），到"全图一次 CNN + RoI 共享特征"（Fast R-CNN），再到"把找框本身变成网络的一部分 RPN"（Faster R-CNN），逐步"共享化 / 可学习化"，最终端到端可训练。
- **一句话总结**：两阶段检测 = "先粗定位，再精分类"——第一阶段找"哪里有物体候选（Region Proposal）"，第二阶段对每个候选做 RoI 级分类 + 框回归；R-CNN 三代演进的主线是消灭重复计算：从"每框过一遍 CNN"，到"全图一次 CNN + RoI 共享特征"，再到"把找框本身也训练进网络（RPN）"。
- **适用范畴**：精度敏感场景（学术榜单、小目标/密集目标、医疗/遥感影像）至今仍以两阶段为强基线；它也是 Mask R-CNN、Cascade R-CNN、FPN 检测头等一系列衍生工作的地基；torchvision（`fasterrcnn_resnet50_fpn`）与 Detectron2 均有开箱实现。
- **与单阶段的分野**：两阶段先筛选再细看，精度天花板更高，但速度慢、结构复杂（RPN + RoI + 两套 head）；单阶段（YOLO/SSD）一次前向直接出框 + 类别，速度快，但早期小目标/密集目标弱（取舍详见关联 [[YOLO单阶段系列]]）。

## 原理

两阶段的核心机制是"候选框 → 共享特征 → RoI 级精修"的流水线，三代各自消除一类冗余：

### 1. R-CNN（2014）—— 首次把深度特征与候选框结合
```
输入图 → Selective Search ~2000 候选框 → 每框 warp 到 227×227 → CNN(AlexNet) 提特征
      → SVM 分类 + 线性回归精修框
```
- 候选框用**传统算法 Selective Search**（CPU、不可学习）产出 ~2000 个；每个框独立 warp 到固定 227×227 再过一次 CNN。
- 优点：首次刷新 PASCAL VOC mAP；缺点 ❌：2000 框 × 每框一次 CNN = 推理约 47s/图；训练分三阶段（CNN 预训练 → SVM → 回归器）极繁琐；warp 破坏宽高比。

### 2. Fast R-CNN（2015）—— 共享特征 + RoI Pooling
```
输入图 → CNN 整图一次 → 共享特征图
      → 每个候选框映射到特征图 → RoI Pooling 池化到固定 7×7 → FC → 分类头 + 框回归头
```
- 关键发明 **RoI Pooling**：把任意大小候选框对应的特征区域**切分成固定网格（如 7×7），每格做 MaxPool**，得到固定尺寸特征，从而让不同大小的候选共享同一套 FC 头。
- **多任务联合训练**：分类 loss（softmax）+ 框回归 loss（smooth-L1）合并为一个 loss 端到端反传；FC 层用 SVD 分解加速。推理约 0.3s/图，比 R-CNN 快约 200 倍。
- 瓶颈：候选框仍是外部 Selective Search（CPU、不可学习），只是"算一次"而非"算两千次"。

### 3. Faster R-CNN（2016）—— RPN：把"找框"也变成网络
```
输入图 → CNN 骨干(共享) → 特征图
      ├─ 主干：RoI Pooling/Align → 检测头（分类+回归）
      └─ RPN：在特征图上滑 3×3 → 每个锚点(anchor)铺 k 个预设框(anchor)
             → 二分类(有/无目标) + 框回归 → NMS → 输出 ~2k/300 个 proposal
```
- **RPN（Region Proposal Network）** 与检测头**共享同一份特征图**，把候选框生成变成可学习的网络层；训练从原论文的 4 步交替训练演进为现代实现的端到端联合训练（torchvision / Detectron2 均为后者）。
- **Anchor（锚框）机制**：在特征图每个位置预置不同**尺度 × 宽高比**的参考框（典型 3 scale × 3 ratio = 9 个）。RPN 学的是两件事：这些预设框里"哪些含物体"（二分类）+ "相对预设框的偏移量"（回归）。**anchor 把"预测任意框"降维成"预测相对预设框的小偏移"**，极大简化了回归任务；RPN 输出经 NMS 抑制冗余后送给检测头。
- 后续变体：FPN 在骨干上叠多尺度特征金字塔、各层分别出 RoI（`fasterrcnn_resnet50_fpn` 即此结构）；Mask R-CNN 在检测头旁加分割分支，并把 **RoI Pooling 换成 RoI Align**。

### RoI Pooling vs RoI Align（一图流）
- RoI Pooling：候选框坐标除以特征图 stride 后**取整**，网格切分再**取整** → **两次量化误差**，小目标/分割任务上偏差显著。
- RoI Align（Mask R-CNN）：**不做取整**，网格内按浮点坐标**双线性采样** → 亚像素精度，消除像素错位。

### 演进本质
三代各自消灭的"重复计算/外部依赖"可归纳为：R-CNN 消灭不了 2000 次 CNN 前向 → Fast 用整图一次前向 + RoI Pooling 共享特征消灭之，但保留外部 Selective Search → Faster 用 RPN + anchor 把候选生成也网络化，全流程可训练。

## 应用

两阶段检测的典型使用场景是**精度优先、算力可接受**的任务：学术评测与榜单、小目标/密集目标检测、医疗影像与遥感影像（对漏检容忍度低）、以及作为实例分割（Mask R-CNN）等任务的基座。落地时**直接用 Faster R-CNN 及其 torchvision/Detectron2 实现**即可，不必复刻 R-CNN/Fast 的历史训练流程。

快速上手步骤（以 torchvision 为例）：① 加载预训练 `fasterrcnn_resnet50_fpn` 权重；② 输入 [N,3,H,W] 的 float 张量、像素归一化到 [0,1]；③ 前向得到 `{boxes, labels, scores}` 字典（decode + NMS 已内置）；④ 按 score 阈值筛选最终框，或按需微调（数据增广、anchor 调整、FPN 多尺度）提升小目标召回。

注意事项与常见坑（❌ → ✅）：

| ❌ 常见错误 | ✅ 正确做法 |
|---|---|
| 把 R-CNN 当成"对每个框重算一遍 CNN"还在用 | 理解演进即可，落地直接用 Faster R-CNN / 其 torchvision 实现 |
| RoI 坐标直接拿原图像素坐标去特征图上取 | 先除以特征图 stride（映射），RoI 归一化坐标可选；RoI Align 用浮点更稳 |
| 以为 RPN 和检测头必须两阶段交替训练 | 现代实现（torchvision/Detectron2）已端到端联合训练；交替训练只是原论文历史路径 |
| 忽略 anchor 超参（尺度/比例/数量）对召回影响 | 小目标/极端宽高比场景要按数据调 anchor 或直接用 FPN 多尺度 |
| 训练时忘记 NMS 后处理 / 测试时阈值不一致 | 训练时 RPN 用 NMS 控制 proposal 数（如 2000→200/300）；推理时按 score 阈值 + 类别内 NMS 出最终框 |
| 不知道检测输出要后处理（box decode + NMS） | torchvision 推理已内置 decode+NMS，返回 `boxes/scores/labels` 字典；自己实现时别漏 |
| 把检测 mAP 当分类 acc 一样算 | 检测用 IoU 匹配 + mAP（见 [[NMS与mAP评测]]） |
| RoI Pooling 量化误差在分割任务里无所谓 | Mask R-CNN 必须 RoI Align；纯检测小目标也建议 Align |

选型提示：精度敏感（学术榜单、小目标、医疗/遥感）→ 两阶段仍是强基线；实时场景 → 单阶段；另外数据增广（随机翻转/多尺度训练）对检测同样关键。

```python
# 案例：用 torchvision 的 Faster R-CNN（ResNet50 + FPN）做整图推理
# 要点：torchvision 推理已内置「RPN 提候选 → RoI 分类回归 → box decode → NMS」全流程，
#       返回 dict，无需手写后处理；输入约定为 [N,3,H,W] 的 float 张量，像素归一化到 [0,1]。
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image

# 1) 加载 COCO 预训练权重（91 类）
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")   # 内部为 ResNet50 骨干 + FPN + RPN + 检测头
model.eval()

# 2) 预处理：PIL → Tensor → 归一化到 [0,1]（无需手动减均值）
img = Image.open("demo.jpg").convert("RGB")
x = transforms.ToTensor()(img).unsqueeze(0)          # [1,3,H,W]

# 3) 前向：整图只过一次骨干，RPN 与检测头共享这份特征图
with torch.no_grad():
    out = model(x)[0]
# out: {"boxes":[N,4] xyxy 像素坐标, "labels":[N] 类别 id, "scores":[N] 置信度}
#      —— decode + NMS 已由模型内置完成，这正是"两阶段"封装好的产物

# 4) 按置信度阈值筛选最终框（阈值 0.5~0.9 之间权衡漏检 vs 误检）
keep = out["scores"] > 0.5
boxes = out["boxes"][keep].cpu().numpy()             # xyxy 格式
labels = out["labels"][keep].cpu().numpy()
scores = out["scores"][keep].cpu().numpy()

# 案例详解：
# - 这一步"单次前向"正是两阶段共享特征的意义：R-CNN 时代每框过一遍 CNN 的
#   重复计算被消除，RPN 和检测头复用同一特征图，速度差约 200 倍。
# - 若自己实现训练/推理，勿漏三步：RoI 坐标先按 stride 映射回特征图、
#   训练时 RPN 用 NMS 控制 proposal 数量并采样正负样本、推理按 score 阈值 + 类别内 NMS。
# - 小目标/极端宽高比召回不足时，优先检查 anchor 尺度/比例设置，或换 FPN 多尺度特征。
```

---
## 关联
- 前置：[[卷积与池化直觉]]（骨干 CNN 特征图的感受野/池化直觉，理解特征图共享与 RoI 映射的基础）；[[图像增广方法]]（随机翻转/多尺度训练等增广对检测同样关键）
- 类似：[[YOLO单阶段系列]]（区别是____ 单阶段没有显式 Region Proposal 阶段，一次前向直接回归出框 + 类别，速度更快、结构更简，但早期版本小目标/密集目标精度弱于两阶段）
- 进阶：[[NMS与mAP评测]]（两阶段输出必经的后处理与评测协议）；Mask R-CNN / FPN（在 Faster R-CNN 上加分割分支、RoI Align、多尺度金字塔）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：Faster R-CNN（两阶段） | 先 RPN+anchor 提候选、再 RoI 级分类+回归，全流程共享特征并端到端可训练 | 精度敏感：小目标/密集目标、医疗/遥感、学术榜单；实例分割等衍生任务的基座 |
| 替代方案：单阶段检测（YOLO/SSD） | 无候选阶段，一次前向在网格/anchor 上直接回归框与类别 | 实时推理、边缘端部署、速度优先且目标尺度相对常规的场景 |
| 历史方案：R-CNN / Fast R-CNN | 每框独立过 CNN（R-CNN）；整图一次 + RoI Pooling 共享特征但候选仍靠 Selective Search（Fast） | 学习演进脉络/复现论文；工程落地已无必要，已被 Faster R-CNN 取代 |

---
## 参考
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (arXiv:1506.01497)](https://arxiv.org/abs/1506.01497)
- [Fast R-CNN (arXiv:1504.08083)](https://arxiv.org/abs/1504.08083)
- [Rich feature hierarchies for accurate object detection and semantic segmentation (R-CNN, arXiv:1311.2524)](https://arxiv.org/abs/1311.2524)
- [torchvision 官方文档：torchvision.models.detection.fasterrcnn_resnet50_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)

---
## 具体案例
- [[R-CNN 两阶段系列 实战示例]](R-CNN两阶段系列_sample.py)
