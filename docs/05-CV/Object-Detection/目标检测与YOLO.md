---
title: "目标检测：两阶段与单阶段（R-CNN / YOLO）"
tags: [计算机视觉, 目标检测, R-CNN, YOLO, IoU, mAP]
date: 2026-08-29
---

# 目标检测：两阶段与单阶段（R-CNN / YOLO）

## 定义

目标检测（Object Detection）不只做分类，还要求**定位**：对图像中每个目标同时输出**类别（class）**与**边界框（bounding box，(x,y,w,h)）**，即回答"图里有什么、分别在哪一块"。它与纯图像分类的本质区别在于：一张图里可能有多个目标，且每个目标都要有位置输出。

检测方法按流程分为两大流派：**两阶段（two-stage）** 先生成候选区域（proposal），再对每个候选做分类+框回归，精度高但较慢，代表是 R-CNN 系、Faster R-CNN；**单阶段（one-stage）** 直接在特征图上划分密集锚框/网格，一步同时预测类别与框，速度快、精度已逐步追平，代表是 YOLO、SSD。

评价一个检测器是否"命中"依赖框重合度指标 **IoU**，综合精度则用 **mAP**（对全部类别的平均精度取均值），它们构成检测任务与分类任务不同的评估体系。

适用范畴：自动驾驶感知、安防监控、工业质检、视频分析、医学影像等一切需要"哪里有什么目标"的视觉任务；选型时需在精度与实时性之间权衡，这正是两阶段/单阶段分流的由来。

## 原理

**任务建模**：检测网络的输出头是"双头"结构——分类头 + 回归头（box regression），因此每加一个类别，输出维度就多一组，且锚框（anchor）参数与类别耦合，需一起重训。

**IoU（Intersection over Union）**——框重合度：

$$\text{IoU} = \frac{\text{预测框} \cap \text{真值框}}{\text{预测框} \cup \text{真值框}}$$

IoU 越接近 1 越好；常用判定阈值 IoU@0.5（IoU≥0.5 判为"命中"）。

**mAP（mean Average Precision）**：对每个类别计算 precision-recall 曲线下面积（AP），再对所有类别取平均得 mAP。两个常用口径：`mAP@0.5` 用 IoU 阈值 0.5 判命中；`mAP@0.5:0.95` 在 0.5~0.95 多档 IoU 阈值上求平均，是 COCO 标准、更严格。两者口径不同，直接比较会误导。

**两阶段（R-CNN 家族）关键演进**：

| 版本 | 候选区域方式 | 分类 | 特点 |
|------|------------|------|------|
| **R-CNN** | Selective Search 生成 ~2k 候选 | CNN 逐候选分类 | 慢（每候选各过 CNN） |
| **Fast R-CNN** | Selective Search + RoI Pooling | 整图一次卷积，RoI 池化共享特征 | 大幅加速 |
| **Faster R-CNN** | **RPN（Region Proposal Network）** 端到端生成候选 | 共享特征 | 全可微、主流两阶段 |

核心机制是把"候选区域生成"也变成可学习的网络（RPN），与分类共享卷积特征，实现端到端训练。

**单阶段（YOLO 系：You Only Look Once）**：把图像划分为 S×S 网格，每格预测若干个**锚框（anchor box）**的坐标、置信度与类别概率，整图只"看一次"（one look）即一次前向输出全部结果，速度远快于两阶段。版本脉络：

| 版本 | 变化 |
|------|------|
| **YOLOv1** | 开创 idea，但小物体/密集目标弱 |
| **YOLOv3** | 多尺度特征金字塔，精度大幅提升（流行） |
| **YOLOv5/v8** | 工程化框架，易用、生态好（ultralytics） |
| **YOLOX / RT-DETR** | 无锚框、Transformer 检测头等改进 |

近年有 Transformer 检测（DETR）与稀疏查询范式，进一步统一流程，但需要大数据。

**两阶段 vs 单阶段（维度对比）**：

| 维度 | 两阶段（Faster R-CNN） | 单阶段（YOLO/SSD） |
|------|----------------------|--------------------|
| 流程 | 候选 + 分类两段 | 一步端到端 |
| 速度 | 较慢 | 快（实时） |
| 精度 | 通常更高（尤其小/密集目标） | 已追平，胜在实时 |
| 适用 | 精度优先、离线 | 实时视频、边缘部署 |

## 应用

**典型使用场景**：实时视频流（摄像头/直播）、边缘与嵌入式部署（帧率敏感）、快速落地业务（选工程化 YOLO 系）；离线高精度任务（小目标、密集目标、对召回要求苛刻）则倾向 Faster R-CNN 两阶段。

**快速上手步骤（Ultralytics YOLOv8）**：① `pip install ultralytics`；② 加载预训练权重 `YOLO("yolov8n.pt")`（n/s/m/l/x 由小到大）；③ 一行推理 `model("bus.jpg", conf=0.5)`；④ 从 `results` 中取出 `boxes`（xyxy 坐标）、`cls`（类别索引）、`conf`（置信度）即可可视化或后续处理；训练自有数据用 `model.train(data="dataset.yaml", epochs=...)`，随后 `model.val()` 观察 mAP。

**注意事项 / 常见坑**：

- ❌ 检测后处理未做**非极大值抑制（NMS）** → 同一目标输出多个重叠框。✅ 用 NMS 去重（ultralytics 推理内部已自动执行）。
- ❌ `mAP@0.5` 与 `mAP@0.5:0.95` 口径不一致就相互比较 → 误导结论。✅ 明确标注评测协议（COCO/YOLO）。
- ❌ 小目标检测直接套用大体量 VGG/深 backbone → 高层特征丢失小物体。✅ 用特征金字塔做多尺度（FPN/PAN）。
- ❌ 训练数据框**标注不齐/类别不平衡** → 漏检与误检。✅ 数据校正 + 类别加权。
- ❌ 实时部署仍跑原尺寸高分辨率 → 帧率不足。✅ 推理阶段降采样/量化/TensorRT 加速。
- 边界：检测依赖**分类+回归两个头**，新增类别会连带 anchors 参数耦合，需整体重训，不能只改分类层。

```python
# Ultralytics YOLOv8 快速使用示例（单阶段检测：一次前向输出所有框）
from ultralytics import YOLO

# 1) 加载预训练模型：n = nano（最小最快），另有 s / m / l / x 档位
model = YOLO("yolov8n.pt")

# 2) 直接推理：conf=0.5 只保留置信度 >=0.5 的框
#    （推理内部已自动完成 NMS 后处理，无需手写非极大值抑制）
results = model("bus.jpg", conf=0.5)

# 3) 解析结果：results 是列表，每个元素对应一张输入图
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()   # (N,4) 边界框，坐标格式 (x1,y1,x2,y2)
    classes = r.boxes.cls.cpu().numpy()  # 每框类别索引
    confs = r.boxes.conf.cpu().numpy()   # 每框置信度
    print(boxes, classes, confs)

# 案例详解：
# - xyxy 为左上/右下角坐标；需要中心点+宽高 (x,y,w,h) 时改用 r.boxes.xywh。
# - classes 是索引，用 model.names[int(idx)] 映射成类别名（如 bus / person）。
# - 框数量受 conf 阈值控制：调低 conf 召回更多但引入误检，调高则反之。
# - 训练自有数据：model.train(data="dataset.yaml", epochs=100)，
#   训练完用 model.val() 查看 mAP@0.5 / mAP@0.5:0.95（注意协议口径）。
```

---
## 关联

- 前置：[[CNN 经典架构]]（检测的 backbone 基础：卷积特征提取）、[[图像处理与数据增广]]（数据预处理/增广，改善标注噪声与不平衡）
- 类似：[[SSD]]（区别是 SSD 在多个尺寸的特征图上直接铺密集默认框做多尺度预测，而本文 YOLO 用 S×S 网格 + 锚框一次前向输出；同属单阶段，锚点分布策略不同，YOLO 工程化生态更好）
- 进阶：[[Mask R-CNN]]（实例分割：检测 + 像素级掩码）、[[DETR]]（Transformer 稀疏查询，无锚框、免 NMS，需大数据）、[[FPN 特征金字塔]]（多尺度检测的关键结构）

---
## 对比选型

| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（单阶段 YOLO） | 一次前向：S×S 网格 + 锚框同时预测类别与框，端到端、实时 | 实时视频流、边缘/嵌入式部署、快速工程落地 |
| 替代方案（两阶段 Faster R-CNN） | RPN 生成候选区域 + 分类/框回归两段式，共享特征、精度优先 | 离线高精度任务、小目标与密集目标场景 |
| 替代方案（DETR / RT-DETR） | 稀疏查询 + Transformer 解码做端到端集合预测，无锚框免 NMS | 数据充足、追求流程统一与免后处理的新项目 |

---
## 参考

- Faster R-CNN — https://arxiv.org/abs/1506.01497
- YOLO 论文 — https://arxiv.org/abs/1506.02640
- Ultralytics YOLO 官方文档 — https://docs.ultralytics.com/
- COCO 评估标准（mAP）— https://cocodataset.org/#detection-eval

---
## 具体案例

- [[目标检测与YOLO 实战示例]](目标检测与YOLO_sample.py)
