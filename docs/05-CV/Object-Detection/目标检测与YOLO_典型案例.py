# -*- coding: utf-8 -*-
"""
目标检测：两阶段与单阶段（R-CNN / YOLO）—— 典型代码演示
========================================================
覆盖知识点：
  1. 关键指标实现：IoU（边界框重合度）手写
  2. 手写一个"暴力检测框架"理解 y_single_shot 思想（滑动窗口 + 分类）
  3. NMS（非极大值抑制）去重算法手写
  4. 用 Ultralytics YOLOv8 做真实推理（若安装了 ultralytics）
  5. Faster R-CNN 在 torchvision 中的使用

依赖：pip install torch torchvision ultralytics(可选) numpy
"""

import numpy as np
import torch

# =====================================================================
# 一、IoU：边界框重合度（命中的度量）
# =====================================================================
def iou(box1, box2):
    """计算两个边界框的交并比。框格式: (x1, y1, x2, y2) —— 左上与右下角。
         IoU = 两框交集面积 / 两框并集面积
    """
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    # 交集矩形的左上与右下角
    ix1, iy1 = max(x1, x1b), max(y1, y1b)
    ix2, iy2 = min(x2, x2b), min(y2, y2b)
    # 交集宽高（若为负说明不相交）
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    # 各自面积
    area1, area2 = (x2 - x1) * (y2 - y1), (x2b - x1b) * (y2b - y1b)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

box_a = [10, 10, 60, 60]        # 50x50 框
box_b = [20, 20, 70, 70]        # 50x50 与 box_a 重叠 40x40
box_c = [200, 200, 250, 250]    # 相距很远，无交集
print("[IoU] 框A与框B IoU =", round(iou(box_a, box_b), 3), "(重叠大,应>0.4)")
print("[IoU] 框A与框C IoU =", iou(box_a, box_c), "(不相交=0)")
print("[IoU] 完全重合 IoU =", round(iou(box_a, box_a), 3), "(应为1.0)")

# =====================================================================
# 二、NMS：非极大值抑制 —— 去重关键后处理
# =====================================================================
def nms(boxes, scores, iou_threshold=0.5):
    """去掉同一目标重复的重叠框，保留得分最高的框。
    Args:
        boxes:  (N,4) 框坐标
        scores: (N,)  每个框的置信度
    """
    # 按置信度从高到低排序
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]                     # 保留当前最高分框
        keep.append(i)
        if order.size == 1:
            break
        # 与其余框计算 IoU，剔除重叠过大的
        rest = order[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= iou_threshold]   # 只保留与当前框 IoU≤阈值 的
    return keep

# 演示：同一目标由模型输出了 5 个重叠框，分数依次递减
boxes = np.array([
    [20, 20, 80, 80],   # 高分框（真实目标）
    [22, 22, 82, 82],   # 与①高度重叠
    [21, 21, 81, 81],   # 与①重叠
    [40, 40, 120, 120], # 部分重叠
    [200, 200, 280, 280],  # 另一个独立目标
])
scores = np.array([0.95, 0.88, 0.80, 0.30, 0.90])
kept = nms(boxes, scores, iou_threshold=0.5)
print("\n[NMS] 保留的框索引(应选中两个不同目标):", kept,
      "  对应分数:", scores[kept])

# =====================================================================
# 三、手写"暴力滑动窗口检测"理解单阶段思想（示意，非高效）
# =====================================================================
from sklearn.linear_model import LogisticRegression
# 制造垃圾箱:将"图像"简化为一维特征向量，滑动窗口采样作为"候选框"
# 这里重在演示"先生成候选 → 再分类"的 pipeline，真实会用卷积特征图。

def simple_window_detector(feature_grid, window_size=8, stride=4):
    """在特征网格上滑动取窗口作为候选，返回所有候选框。"""
    candidates = []
    for x in range(0, feature_grid.shape[0] - window_size + 1, stride):
        for y in range(0, feature_grid.shape[1] - window_size + 1, stride):
            candidates.append([x, y, x + window_size, y + window_size])
    return np.array(candidates)

fake_feat = np.random.RandomState(0).randn(32, 32)   # 模拟特征图(通道数为1)
cands = simple_window_detector(fake_feat)
print(f"\n[滑动窗口] 特征图32x32, 窗口8, 步长4 → 生成候选框 {len(cands)} 个")
print("          (两阶段方法即先生成候选, 再对每个候选分类/回归)")

# =====================================================================
# 四、用 Ultralytics YOLOv8 做真实推理（单阶段实时检测的典型实现）
# =====================================================================
try:
    from ultralytics import YOLO
    # 载入预训练 nano 模型（首次运行自动下载），并用示例图片推理
    model = YOLO("yolov8n.pt")
    # 若本地无示例图，可用网图；这会触发联网下载
    results = model("/workspace/bus.jpg") if False else None
    print("\n[Ultralytics YOLOv8] 模型已就绪 (需真实图片与联网可运行)")
    print("[YOLOv8] 用法: results = model('img.jpg'); 每个 result 含 boxes/class/conf")
except Exception as e:
    print(f"\n[Ultralytics] 未安装或无法下载（{type(e).__name__}），跳过真实推理。")

# =====================================================================
# 五、用 torchvision Faster R-CNN 做真实检测
# =====================================================================
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    model_rcnn = fasterrcnn_resnet50_fpn(weights=None)  # 不下载，仅演示结构
    model_rcnn.eval()
    with torch.no_grad():
        out = model_rcnn([torch.randn(3, 300, 300)])
    # 输出包含: boxes(N,4) / labels / scores
    print("\n[Faster R-CNN(torchvision)] 对一张 300x300 随机图输出键:",
          list(out[0].keys()))
    print("       预测框 shape:", tuple(out[0]["boxes"].shape),
          " 得分前3:", out[0]["scores"][:3].round(decimals=3) if False else "需真实图")
except ImportError:
    print("\n[torchvision.detection] 未安装，跳过。")

# =====================================================================
# 小结与评测口径
# =====================================================================
print("""
[评测口径]
 mAP@0.5   : IoU 阈值 0.5 下的全类平均精度（YOLO 常用）
 mAP@0.5:0.95 : 多档 IoU 平均（COCO 标准，更严格）
两阶段(Faster R-CNN)：精度优先   |   单阶段(YOLO)：速度优先
后处理离不开 NMS 去重，指标比较必须先对齐评测协议。
""")
