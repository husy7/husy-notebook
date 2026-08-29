---
title: "目标检测：两阶段与单阶段（R-CNN / YOLO）"
tags: [计算机视觉, 目标检测, R-CNN, YOLO, IoU, mAP]
date: 2026-08-29
---

# 目标检测：两阶段与单阶段（R-CNN / YOLO）

## 一、核心思想

目标检测不只分类，还要**定位**：输出图像中每个目标的**类别** + **边界框（bounding box，(x,y,w,h)）**。即回答"图里有什么、分别在哪一块"。

两大流派：
- **两阶段（two-stage）**：先生成候选区域（proposal），再对每个候选分类+回归框。精度高、较慢。代表：R-CNN 系、Faster R-CNN。
- **单阶段（one-stage）**：直接在特征图上划分密集锚框/网格，一步同时预测类别与框。速度快、精度逐步追平。代表：YOLO、SSD。

## 二、关键评估指标

### 2.1 IoU（Intersection over Union）— 框重合度

$$\text{IoU} = \frac{\text{预测框} \cap \text{真值框}}{\text{预测框} \cup \text{真值框}}$$

- IoU 越接近 1 越好；常用阈值 IoU@0.5（IoU≥0.5 判为"命中"）。

### 2.2 mAP（mean Average Precision）

对每个类别计算 precision-recall 曲线下面积（AP），再对所有类别取平均得 **mAP**。

- mAP@0.5：用 IoU 阈值 0.5 判命中时全类别的平均 AP。
- mAP@0.5:0.95：在 0.5~0.95 多档 IoU 上求平均（COCO 标准，更严格）。

## 三、两阶段：R-CNN 家族

| 版本 | 候选区域方式 | 分类 | 特点 |
|------|------------|------|------|
| **R-CNN** | Selective Search 生成 ~2k 候选 | CNN 逐候选分类 | 慢（每候选各过 CNN） |
| **Fast R-CNN** | Selective Search + RoI Pooling | 整图一次卷积，RoI 池化共享特征 | 大幅加速 |
| **Faster R-CNN** | **RPN（Region Proposal Network）** 端到端生成候选 | 共享特征 | 全可微、主流两阶段 |

关键演进：把"候选区域"也变成可学习网络（RPN），与分类共享特征，实现端到端。

## 四、单阶段：YOLO 系（You Only Look Once）

### 4.1 核心思想：一次前向直接输出

把图像分为 S×S 网格，每格预测若干个**锚框（anchor box）**的坐标、置信度与类别概率。整图**只看一次**（one look），速度远快于两阶段。

### 4.2 YOLO 版本脉络

| 版本 | 变化 |
|------|------|
| **YOLOv1** | 开创 idea，但小物体/密集目标弱 |
| **YOLOv3** | 多尺度特征金字塔，精度大幅提升（流行） |
| **YOLOv5/v8** | 工程化框架，易用、生态好（ultralytics） |
| **YOLOX / RT-DETR** | 无锚框、Transformer 检测头等改进 |

### 4.3 Ultralytics YOLOv8 快速使用

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")            # 预训练 nano 版本
results = model("bus.jpg", conf=0.5)  # 直接推理

# 结果含边界框与类别
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()   # (N,4) (x1,y1,x2,y2)
    classes = r.boxes.cls.cpu().numpy()  # 类别索引
    confs = r.boxes.conf.cpu().numpy()   # 置信度
    print(boxes, classes, confs)
```

## 五、两阶段 vs 单阶段

| 维度 | 两阶段（Faster R-CNN） | 单阶段（YOLO/SSD） |
|------|----------------------|--------------------|
| 流程 | 候选 + 分类两段 | 一步端到端 |
| 速度 | 较慢 | 快（实时） |
| 精度 | 通常更高（尤其小/密集目标） | 已追平，胜在实时 |
| 适用 | 精度优先、离线 | 实时视频、边缘部署 |

> 近年有 Transformer 检测（DETR）与稀疏查询范式，进一步统一流程但需大数据。

## 六、边界与坑

- ❌ 检测时未做**非极大值抑制（NMS）** → 同一目标输出多个重叠框。✅ 后处理 NMS 去重。
- ❌ 用 `mAP@0.5` vs `mAP@0.5:0.95` 口径不一致直接比较 → 误导。✅ 明确标注评测协议（COCO/YOLO）。
- ❌ 小目标检测直接用大体 VGG/深 backbone → 高层特征丢失小物体。✅ 用特征金字塔多尺度（FPN/PAN）。
- ❌ 训练数据框**标注不齐/类别不平衡** → 漏检与误检。✅ 数据校正 + 类别加权。
- ❌ 实时部署继续跑原尺寸高分辨率 → 帧率不足。✅ 可在工程推理时降采样/量化/tensorRT。
- 边界：检测依赖**两个头**（分类+回归），加一个类别就多一组输出维度，anchors 的参数耦合需一起重训。

## 七、关联

- 前置知识：CNN 特征提取、卷积、训练循环。
- 同板块：[CNN 经典架构](..\CNN-Architectures\CNN经典架构.md)（backbone）、[图像处理与数据增广](..\Image-Processing\图像处理与数据增广.md)。
- 进阶：实例分割（Mask R-CNN）、多尺度特征金字塔、DETR（Transformer 检测）。

## 八、参考

- Faster R-CNN — https://arxiv.org/abs/1506.01497
- YOLO 论文 — https://arxiv.org/abs/1506.02640
- Ultralytics YOLO 官方文档 — https://docs.ultralytics.com/
- COCO 评估标准（mAP）— https://cocodataset.org/#detection-eval
