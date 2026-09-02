---
title: "NMS 与 mAP 评测（检测后处理与指标口径）"
tags: [Object-Detection, NMS, mAP]
date: 2026-08-30
---

# NMS 与 mAP 评测（检测后处理与指标口径）

## 定义

- **NMS（Non-Maximum Suppression，非极大值抑制）**是目标检测推理阶段的**后处理去重**算法：模型（尤其滑窗 / anchor 类检测器）对同一目标往往输出多个高度重叠的预测框，NMS 按置信度贪心保留每个目标的最优框、删除被判定为"重复"的其余框，解决"一个目标被输出一堆重叠框"的问题。
- **mAP（mean Average Precision，平均精度均值）**是目标检测的**离线评测指标**：把检测结果按 IoU 逐框判定为命中/误检/漏检，逐类别计算 PR 曲线下面积 AP，再对各类别取平均，解决"检测质量怎么客观量化"的问题——漏检、误检、定位偏差都被折算成一个可跨模型比较的分数。
- 两者的共同地基是 **IoU（Intersection over Union，交并比）**：`IoU(A,B) = |A∩B| / |A∪B|`，范围 [0,1]。IoU 决定哪些框"算命中"（评测匹配阈值），也决定两个预测框"算重复"（NMS 抑制阈值）——是同一把尺子的两种用法。
- 适用范畴：任何输出"带置信度的边框集合"的检测器（R-CNN 两阶段系、YOLO 单阶段系、anchor-free 稠密预测头）做推理收口与竞赛级评测（VOC / COCO / 工业质检）都依赖这套后处理与指标口径。
- 核心特征：几何量（IoU）驱动阈值化离散决策；NMS 抑制阈值与评测匹配阈值是**两个独立旋钮**，不可混为一谈，报告指标必须写清 IoU 口径。

## 原理

IoU 是纯几何量：两框交集面积 / 并集面积，与类别无关、对称，IoU=1 完全重合，IoU=0 不相交。检测中两处关键用途：**匹配**（预测框 vs GT 是否算命中）与**抑制**（两个预测框是否算重复）。

**NMS 贪心流程**（同一类别内）：
```
输入: 同一类别的一堆 (box, score)，阈值 iou_th（常用 0.45~0.5）
1. 按 score 从高到低排序
2. 取当前最高分框加入保留集
3. 删掉与该框 IoU > iou_th 的所有框（视为重复）
4. 重复 2~3 直到空
```
- 必须**类别内做**（跨类不抑制）；不同实现先按类别分组再各自 NMS。
- 阈值语义：iou_th 越小删得越狠（0.5 表示重叠过半就删）。
- 局限与变体：贪心 NMS 会把"真正挨着的两个目标"误删一个 → **Soft-NMS**（重叠框的分数按 IoU 连续衰减而不是硬删）、**DIoU/CIoU-NMS**（重叠度量加中心距离/宽高比惩罚）；稠密场景（行人/细胞）另配中心点式 NMS。

**Precision / Recall / PR 曲线 / AP 机制**：把检测结果按 score 降序逐框判定——**TP**：与某 GT 的 IoU ≥ 阈值且该 GT 未被占用 → 命中；**FP**：没匹配上任何 GT；**FN**：没被任何预测框命中的 GT。随 score 阈值从高到低扫描得到一组 (precision, recall) 点 → 画 PR 曲线。**AP = PR 曲线下面积**（对该类别）；**mAP = 各类别 AP 平均**。
- 早期（VOC）用 11 点插值 / 101 点插值近似；
- COCO 用**逐点积分（all-point interpolation）**：`AP = Σ (r_{n+1}-r_n) * p_interp(r_{n+1})`，其中 `p_interp(r) = max_{r'≥r} p(r')`——先把曲线单调化（取后缀最大精度），容忍抖动。

**匹配规则（坑点密集区）**：
1. 预测按 score 排序，**每个 GT 只能被匹配一次**（先到先得；后续命中同一 GT 的预测框即使 IoU 够高也算 FP，"双检测"不算双 TP）。
2. 用 **IoU 阈值**判定匹配（VOC 0.5；COCO 有多档）。
3. 类别分开算：预测类别必须等于 GT 类别才参与该类别匹配。

**mAP 的两种主流口径（务必分清）**：

| 口径 | 含义 | 使用场景 |
|---|---|---|
| **mAP@0.5** (AP50) | 只在 IoU=0.5 匹配下算 mAP | VOC、工业界常用；宽松、"定位差不多就行" |
| **mAP@0.5:0.95** (AP) | IoU 从 0.5 到 0.95、步长 0.05（共 10 档）各算 AP 再平均 | **COCO 官方主指标**，对定位精度极其敏感——框偏一点 AP 就掉，逼着检测器精修 |
| mAP@0.75 (AP75) | 只在 IoU=0.75 匹配下算 | 高精度定位的专项指标 |

同一模型 mAP@0.5 可能 0.9，而 mAP@.5:.95 只有 0.6——**两者差得大 = 定位不够准**。报告指标必须写清口径，Ultralytics 训练日志里 `mAP50` 与 `mAP50-95` 就是这两个。COCO 还按目标尺寸拆：`AP_small/medium/large`（面积 <32² / 32²~96² / >96²）与 `AR`（固定每图框数下的召回）——小目标 AP 通常远低，是评测报告的常规体检项。

## 应用

**典型使用场景**：检测器训练完成后（1）推理侧对原始输出做 NMS 去重再送下游；（2）评测侧用 mAP 系列指标量化模型质量、对比模型选型；（3）调优定位精度时对照 mAP@0.5 与 mAP@0.5:0.95 的差距判断问题在"召回"还是"框不准"。

**快速上手步骤**：
1. 预测与 GT 统一到同一坐标系、同一表示：都转成**像素角点 xyxy**（归一化坐标先乘回原图）；进评测函数前 clip 到图像范围、丢弃 w/h ≤ 0 的越界框。
2. 按类别分组，每组各自跑 NMS（iou_th 常用 0.45~0.5），得到送入评测的框集。
3. 评测直接用官方实现：`pycocotools`（COCO 格式 json，注意 annotations 用 `[x,y,w,h]` 左上角+宽高，**非中心 xywh**）+ Ultralytics 内置 `val` 报告 mAP50/mAP50-95。
4. 报告一律标注口径：`mAP@0.5` 或 `mAP@0.5:0.95`；同口径才可比，不要拿 11 点插值 AP 与 COCO 逐点积分混着比。

**常见坑与正确做法**：

| ❌ 常见错误 | ✅ 正确做法 |
|---|---|
| NMS 全局跨类别做 | 每类各自 NMS（跨类框重叠不该互相删） |
| 忘了 GT 只匹配一次 → 双检测算 TP | 匹配后打标记；同一 GT 的第二个匹配必是 FP |
| 把"分类 accuracy"思路搬来：只看框多准不看召回 | 看 AP（PR 曲线下面积），它同时惩罚漏检与误检 |
| 报告 mAP 不写 IoU 口径（0.5 / 0.5:0.95 差出一大截） | 一律标注口径：`mAP@0.5` 或 `mAP@0.5:0.95` |
| 预测框越界/负宽高没 clip | 评测前 clamp 到图像范围、丢弃 w/h≤0 的框 |
| 坐标系统一错误（归一化 vs 像素、xywh vs xyxy） | 进评测函数前全部转成同一种（如像素 xyxy） |
| NMS 阈值与 IoU 匹配阈值混为一谈 | NMS 阈值管"去重"，匹配阈值管"算不算命中"，两者独立 |
| 用 11 点插值 AP 和 COCO 逐点积分混着比 | 同口径才可比；直接抄官方库(pycocotools/Ultralytics)避免自造轮子偏差 |

**更多工程注意**：空预测（无框）→ Precision=0；空 GT 类别的 AP 处理各库不同（pycocotools 会跳过无 GT 类）；评测数据划分要与训练独立；硬负例/重复检测会直接压 AP；别在训练与评测用不同阈值后处理而不自知（会影响 PR 曲线口径，虽然后处理通常不影响 mAP 计算本身的 GT 匹配，但影响送入评测的框集）。

**快速自检（原文保留）**：
1. 一个 GT 被两个高分预测框同时以 IoU=0.8 命中，按匹配规则结果如何（TP/FP）？若两个框分别命中两个不同 GT 呢？
2. 为什么 mAP@.5:.95 比 mAP@.5 更能鞭策检测器把框画准？
3. NMS 的 iou_th 从 0.5 调到 0.9，框数变多还是变少？对密集场景选哪个方向？
4. PR 曲线的单调化（p_interp）解决什么问题？为什么 11 点插值不精确？

```python
# -*- coding: utf-8 -*-
"""NMS 与 mAP 评测最小示例（案例详解见代码内注释）。
只演示核心机制：IoU → 类别内贪心 NMS → 匹配口径说明；
完整 mAP 计算直接交给官方库（pycocotools / Ultralytics val），不要自造轮子。"""
import numpy as np


def iou(box_a, box_b):
    """两个像素角点 xyxy 框的交并比，范围 [0,1]。
    纯几何量、对称：IoU=1 完全重合，IoU=0 不相交；
    检测中同一把尺子两用：评测匹配阈值 与 NMS 抑制阈值。"""
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)     # 无交集时交集面积为 0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-12)


def nms(boxes, scores, iou_th=0.5):
    """同一类别内的贪心 NMS（类外必须先分组，跨类不抑制）。
    1) score 从高到低排序  2) 最高分框入保留集
    3) 删掉与其 IoU > iou_th 的框  4) 重复到空。
    iou_th 越小删得越狠（0.5 = 重叠过半即删）；
    此处为硬删，Soft-NMS 改成分数按 IoU 衰减而非直接删除。"""
    order = np.argsort(-np.asarray(scores, dtype=np.float64))   # 从高到低
    keep = []
    while order.size > 0:
        i = int(order[0])              # 当前最高分框
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        ov = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ov <= iou_th]     # 只留下与它重叠不超阈值的框
    return keep


# ---- 案例详解：两个高分框几乎盖住同一目标 A，另有一个框盖住目标 B ----
boxes = [[10, 10, 60, 60],     # 目标 A 的主框 score=0.95
         [15, 12, 58, 58],     # 与 A IoU≈0.79(>0.5) 的重复框 score=0.80
         [90, 90, 150, 150]]   # 目标 B 的框 score=0.90
scores = [0.95, 0.80, 0.90]
keep = nms(boxes, scores, iou_th=0.5)
print("保留框下标:", keep)                       # [0, 2]：框 1 被当作框 0 的重复删除
print("框0 vs 框1 IoU =", round(iou(boxes[0], boxes[1]), 3))   # 0.791

# ---- mAP 评测路径（不自行实现积分，交给官方实现）----
# 1. 预测与 GT 统一到同一坐标系：像素角点 xyxy（归一化坐标先乘回原图）；
# 2. pycocotools：COCO json 的 annotations 用 [x, y, w, h]（左上角+宽高），
#    不是中心点 xywh！匹配时预测按 score 降序逐框判定、每个 GT 只匹配一次；
# 3. 口径二选一并写清：mAP@0.5（VOC 宽松）或 mAP@0.5:0.95（COCO 主指标，
#    对定位精度极敏感，框偏一点 AP 就掉）；
# 4. Ultralytics：model.val() 直接输出 mAP50 / mAP50-95 / AP_s..l / AR。
```

---
## 关联
- 前置：[[Object-Detection/YOLO单阶段系列]]、[[Object-Detection/R-CNN两阶段系列]]——锚框/滑窗类检测器输出"同一目标多个重叠框"，正是 NMS 的输入来源，先了解其原始输出形态再读本文。
- 类似：[[Object-Detection/IoU 损失（GIoU/DIoU/CIoU）]]（区别是____两者都以 IoU 度量框重合，但损失函数用于训练、要求平滑可导并参与反向传播；NMS 抑制阈值与评测匹配阈值用于推理/评测的离散决策，不参与梯度）；[[Classification/Accuracy 与 Top-1]]（区别是____分类只需单标签判对错，检测需要按 IoU 匹配框、同时惩罚漏检 FN 与误检 FP，因此看 PR 曲线下面积 AP 而非 accuracy）。
- 进阶：[[Image-Processing/图像增广方法]]（评测前须确定性预处理，避免训练/评测增广差异造成的不公平比较）；[[Object-Detection/Soft-NMS 与 DIoU-NMS 变体]]；[[Object-Detection/标签分配 TaskAlignedAssigner]]（IoU 同时用于训练期标签分配）。

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：类别内贪心 NMS + mAP 评测 | 按 score 贪心保留、IoU 超阈值即删；AP=PR 曲线下面积、mAP 多档平均（COCO 逐点积分） | 通用检测器推理后处理 + 需写清口径的竞赛级评测（COCO/VOC/工业质检） |
| 替代方案：Soft-NMS | 重叠框分数按 IoU 连续衰减而非硬删，保留邻近框但降权 | 密集/互相遮挡场景（行人、拥挤人群、细胞），避免挨着的两个真目标被误删 |
| 替代方案：DIoU/CIoU-NMS | 重叠判定引入中心距离/宽高比惩罚，几何更鲁棒 | 同类目标紧邻、细长框（车辆、棒状物），进一步抑制误删 |
| 替代方案：中心点式 NMS | 无锚/稠密输出时按中心点距离去重 | anchor-free 或端到端稀疏检测器（行人/细胞稠密场景的另一种实现口径） |

---
## 参考
- [COCO Detection Evaluation（mAP 官方指标口径定义）](https://cocodataset.org/#detection-eval)
- [pycocotools（COCO 评测官方实现，annotations 用 [x,y,w,h]）](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
- [Ultralytics YOLO Validation（mAP50 / mAP50-95 / AP_s..l / AR 报告）](https://docs.ultralytics.com/modes/val/)

---
## 具体案例
- [[NMS 与 mAP 评测 实战示例]](NMS与mAP评测_sample.py)
