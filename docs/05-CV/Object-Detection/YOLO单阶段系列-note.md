---
title: "YOLO 单阶段系列（YOLOv1 → Ultralytics YOLOv8）"
tags: [目标检测, YOLO, 深度学习]
date: 2026-08-30
---

# YOLO 单阶段系列（YOLOv1 → Ultralytics YOLOv8）

## 定义

YOLO（You Only Look Once）单阶段系列是一类把目标检测当作**单次前向回归**的算法，时间跨度从 2015 年的 YOLOv1 一直演进到 2023 年 Ultralytics 的 YOLOv8。一句话总结：单阶段检测把"找框 + 分类"合并成一次前向回归——把图像划成网格，每个网格单元直接预测"中心落在本格的目标框 + 类别"，速度远快于两阶段，代价是早期版本对密集/小目标/任意长宽比召回弱。

它解决的问题是：给定一张图像，**同时**输出每个目标的位置（边界框）与类别，而不像两阶段那样先粗筛候选框再逐框精判。核心特征：输入图直接回归出一个 `S×S×(B×5 + C)` 张量（S 为网格数、每格 B 个框、每框 5 个数 = x,y,w,h + 置信度、C 个类别概率），全程无 proposal、无 RoI 池化，推理是一条直线。

适用范畴：实时/视频流检测、边缘与嵌入式部署、需要快速训练与工程化落地的项目——Ultralytics 提供 `yolo` 一条命令走通训练/导出/推理。系列演进主线：**网格直出 → anchor 回归 → 多尺度预测 → anchor-free + 解耦头**，每一代都在解决前一代"定位粗 / 小目标弱 / 超参多"的痛点。

## 原理

**为什么单阶段能快这么多**：Faster R-CNN 要"RPN 出 proposal → RoI 再精判"两步；YOLO 的哲学是把检测当成一个**纯回归问题**——网络一次前向直接输出所有框与类别，无 proposal、无 RoI，因此推理延迟远低于两阶段，这是"单阶段快"的本质来源。

**各代关键改进（机制演进表）**：

| 版本 | 关键思想 | 一句话贡献 |
|---|---|---|
| YOLOv1 (2015) | S×S 网格，每格回归 B 框 + 类别 | 端到端实时检测；但**每格只能出一个目标**、小目标/密集场景崩、框定位粗 |
| YOLOv2 (2016) | **Anchor 先验**（k-means 从数据聚出）、BatchNorm、高分辨率 | 借鉴 Faster R-CNN anchor 思想，召回与定位大提升 |
| YOLOv3 (2018) | **多尺度预测（类 FPN 三条支路）+ 逻辑回归代替 softmax 分类** | 不同尺度支路负责大/中/小目标，小目标显著改善；主干换 Darknet-53 |
| YOLOv4/5 (2020) | CSPDarknet、Mish、**Mosaic 增广**、CIoU loss、自对抗训练 | 工程化集大成：精度/速度/部署全方位实用化 |
| YOLOX/YOLOv7 (2021-22) | **Anchor-free + 解耦头（分类/回归分开）+ SimOTA 标签分配** | 去掉 anchor 超参；解耦头消除分类/回归竞争 |
| YOLOv8 (Ultralytics, 2023) | Anchor-free、C2f 结构、TaskAlignedAssigner 分配、**一体化 CLI/API** | 默认落地首选：`yolo` 一条命令训练/导出/推理 |

**v8 的 Anchor-free 直觉**：每个网格位置直接预测"目标中心距本格偏移 (x,y) + 宽高 (w,h)"，不再依赖 preset anchor；正负样本由 TaskAlignedAssigner 按"分类分 × IoU 分"动态分配——少了一堆超参，对小目标更友好。

**v1"每格一个目标"限制的两条解决路径**（后续版本正是沿这两条走）：① 多尺度/多位置——v3 用三条尺度支路（类 FPN）分开负责大中小目标，v8 每网格位置独立预测；② 每位置多候选 + 动态分配——v2 引入 anchor 先验（每个位置多个预设框），v8 演化为 TaskAlignedAssigner 按"分类分×IoU 分"动态指派正负样本。

**YOLO 标签与输出格式（必须会换算）**：
- 训练标签（Ultralytics）：每行一个目标，**归一化 xywh**（中心点 + 宽高，均除以图像宽高）：`class_id  x_center  y_center  w  h`，txt 文件名 = 图像名。数据集结构：
  ```
  dataset/
    images/train/  images/val/
    labels/train/  labels/val/      # 每张图对应同名 .txt
  ```
- 模型原始输出 → 后处理：解码（中心点/宽高 → 角点 xyxy）→ 按类别置信度阈值过滤 → **NMS** 去重。原始输出张量不是最终结果，漏掉任何一步都会直接得到错框/重复框。

## 应用

**典型使用场景**：实时视频流/工业质检/自动驾驶感知等延迟敏感任务、边缘设备部署、快速原型验证与数据集迭代——这些场景下单阶段"一次前向直出结果"的延迟优势最大。推理输入是 **RGB，BGR→RGB 转换由库内部处理**；`imgsz` 训练与推理尽量保持一致，自写推理时还要自己做 letterbox 并在输出端还原坐标。

**快速上手（CLI 一条命令流）**：

```bash
pip install ultralytics            # 自带 torch 依赖
yolo predict model=yolov8n.pt source=bus.jpg     # CLI：推理（自动下载权重，需联网）
yolo train data=coco8.yaml model=yolov8n.pt epochs=50 imgsz=640
```

**坑 ❌ 与 做法 ✅**：

| ❌ 常见错误 | ✅ 正确做法 |
|---|---|
| 拿 COCO 预训练权重直接测自己的域（工业/卫星图）不微调 | 换域必须微调；至少跑通后看混淆与 PR 曲线再定 |
| 标签写成像素 xyxy 或未归一化 | Ultralytics 要求**归一化 xywh**（除以图宽高），类别从 0 开始连续编号 |
| 训练/推理 imgsz 不一致或忘了 letterbox | 用同一 imgsz；库内部 letterbox 补边，导出/自写推理时也要 letterbox + 还原坐标 |
| conf 阈值设太低 → 一堆假阳性；太高 → 漏检 | 按 PR 曲线/F1 曲线挑（Ultralytics 训练日志给各阈值曲线），一般 conf≈0.25、NMS iou≈0.45 起步 |
| 密集/极小目标场景直接用默认 yolov8n | 小目标：提高 imgsz、加 mosaic、或用 P2 层版本/切图推理 |
| 忘后处理：把解码前原始张量当结果 | 原始输出须先解码再类别过滤再 NMS（见下方 sample 代码自实现解码器） |
| 类别不平衡/标签漏标导致 mAP 虚低 | 先检查标签质量与分布；mAP 计算口径见关联的 `NMS与mAP评测` |
| 与两阶段对比只看精度不看速度 | 单阶段赢在延迟；要精度天花板（小目标/密集）两阶段仍可参考 `R-CNN两阶段系列` |

**自测与换算练习**（检验是否真正理解，要点均见上文原理与应用）：
1. YOLOv1 的"每格一个目标"限制是怎么被后续版本解决的（两条路径）？
2. Anchor-free 与 Anchor-based 各自多/少了什么超参与计算？
3. 一个 640×640、stride=32 的 YOLO 输出特征图是几×几？若 3 个尺度呢？
4. 训练标签 (0.5, 0.5, 0.1, 0.2) 表示什么？若图像是 640×480，对应像素宽高多少？

```python
from ultralytics import YOLO

# 1) 加载模型：传 .pt 权重文件 = 加载预训练权重（首次自动下载，需联网）
#    传 .yaml 配置文件   = 从零开始按结构定义网络（YOLO("yolov8n.yaml")）
model = YOLO("yolov8n.pt")

# 2) 训练：data 指向 ultralytics 格式数据集（images/ + labels/ 同名 .txt）
#    标签每行 = class_id x_center y_center w h（xywh 均归一化到 0~1）
model.train(data="coco8.yaml", epochs=50, imgsz=640)

# 3) 推理：conf / iou 是后处理阈值（类别置信度过滤 / NMS 去重）
res = model.predict("bus.jpg", conf=0.25, iou=0.45, save=True)

# 4) 结果解读：res[0].boxes 提供 xyxy / conf / cls（torch tensor），res[0].names 为类别名
boxes = res[0].boxes          # .xyxy 角点坐标、.conf 置信度、.cls 类别 id
names = res[0].names          # {0: 'person', 1: 'bus', ...}

# ⚠️ 若自实现推理：网络原始输出 ≠ 最终框，管线必须为
#    解码（中心点/宽高 → xyxy 角点）→ 按类别置信度阈值过滤 → NMS 去重
#    可视化/存盘时再按需把 xyxy 转回 xywh
#    （对照 YOLO单阶段系列_sample.py 的自实现解码器）
```

---
## 关联
- 前置：[[目标检测与YOLO]]（检测任务定义、单/两阶段两派路线总览入口）
- 类似：[[R-CNN两阶段系列-note|R-CNN 两阶段系列]]（区别是：两阶段先由 RPN 生成 proposal、再对每个候选框做 RoI 精判，"找框"与"判框"分两步走，精度天花板更高、小目标/密集场景更强，但推理多一道流程更慢；本文单阶段把两步合并成一次前向回归，快但早期版本对密集/小目标召回弱）
- 进阶：[[NMS与mAP评测-note|NMS 与 mAP 评测]]（YOLO 训练日志的 mAP50 / mAP50-95 与推理后处理 NMS 正是这套口径，自实现解码器后必须补上）
- 增广：[[图像增广方法-note|图像增广方法]]（Mosaic/CutMix 等强增广是 YOLO 精度关键）
- 骨干：[[卷积与池化直觉-note|卷积与池化直觉]]（Darknet-53/C2f 仍是卷积堆叠）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：YOLO 单阶段（Anchor-free，如 v8） | 每个网格位置直接回归"目标中心偏移 (x,y) + 宽高 (w,h)"，TaskAlignedAssigner 按 分类分×IoU 分 动态分配正负样本，一次前向直出框+类 | 通用落地首选：实时/视频流、边缘部署、快速迭代（`yolo` 一条命令训练/导出/推理） |
| 本文早期替代：Anchor-based（v2/v3） | k-means 从训练数据聚类出 anchor 先验，回归相对 anchor 的偏移；v3 用三条尺度支路（类 FPN）分别负责大/中/小目标 | 复现历史论文、对照演进实验、理解"预设先验"型检测器的工作方式 |
| 替代方案：两阶段 Faster R-CNN（R-CNN 家族） | RPN 先生成 proposal → RoI 池化后逐框精判（分类+回归），两步走 | 精度天花板场景（小目标/密集）、算力充裕且延迟不敏感；对照刷分参考 |

---
## 参考
- [YOLOv1 论文：You Only Look Once: Unified, Real-Time Object Detection（arXiv:1506.02640）](https://arxiv.org/abs/1506.02640)
- [Ultralytics YOLOv8 官方文档（Docs）](https://docs.ultralytics.com/)
- [ultralytics/ultralytics（GitHub 仓库）](https://github.com/ultralytics/ultralytics)

---
## 具体案例
- [[YOLO 单阶段系列 实战示例]](YOLO单阶段系列_sample.py)（自实现解码器/后处理示例，对应"坑：忘后处理——把解码前原始张量当结果"）
