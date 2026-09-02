# -*- coding: utf-8 -*-
"""
R-CNN 两阶段系列演示（离线可运行，torch + torchvision）

覆盖知识点：
  Part A  Anchor 生成：特征图每个位置 × 多尺度 × 多宽高比（Faster R-CNN RPN 核心）
  Part B  RoI 对齐：把任意大小候选框映射到特征图并池化成固定尺寸
          （RoIAlign vs RoIPool 输出对比；含手写简化 RoIPool 便于理解）
  Part C  torchvision.ops.nms 去重 demo（与 NMS/mAP 笔记呼应）
  Part D  官方 Faster R-CNN（fasterrcnn_resnet50_fpn）随机权重前向，
          weights=None 不下载；检测输出字典 boxes/scores/labels 结构展示。

环境：torch>=1.9, torchvision>=0.10（未装 torchvision 时 Part B/D 自动跳过）。
"""
import torch
import torch.nn.functional as F

# ---------- Part A：Anchor 生成 ----------
def generate_anchors(feat_h, feat_w, stride, scales=(32, 64, 128),
                     ratios=(0.5, 1.0, 2.0), base=16):
    """RPN anchor：以原图坐标计，每个特征图位置生成 len(scales)*len(ratios) 个锚框。

    每个 scale 给基础边长 s=base*scale；每个 ratio 再换算 w/h 满足 w*h=s²。
    返回 (feat_h*feat_w*k, 4) 的 xyxy 框。
    """
    anchors = []
    # 先算每个 (scale, ratio) 组合的宽高（锚框模板，与位置无关）
    for s in scales:
        for r in ratios:
            area = (base * s) ** 2
            w = round((area * r) ** 0.5)
            h = round((area / r) ** 0.5)
            for cy in range(feat_h):
                for cx in range(feat_w):
                    # 特征图 (cx,cy) -> 原图中心 (cx+0.5)*stride
                    xc = (cx + 0.5) * stride
                    yc = (cy + 0.5) * stride
                    anchors.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return torch.tensor(anchors)


def part_a_anchors():
    print("=" * 62)
    print("Part A: RPN Anchor 生成（尺度×宽高比×位置）")
    # 模拟 38x50 特征图（stride=16，对应约 608x800 输入）
    # scale 语义: 边长 = base(16) * scale，即 8/16/32 → 128/256/512 px 见方
    anchors = generate_anchors(feat_h=38, feat_w=50, stride=16, scales=(8, 16, 32))
    k = len(anchors)
    print(f"  特征图 38x50 × {k // (38 * 50)} 个模板/位置 = {k} 个 anchor")
    print(f"  每个位置模板 = 3 scale(8/16/32 → 128/256/512px) × "
          f"3 ratio(0.5/1/2)")
    # 打印位置 (cy=10, cx=10) 的 9 个模板框做直观感受
    # 注意：generate_anchors 是"模板外循环、空间内循环"——同一 cell 的 9 个
    # 模板在数组里相隔 feat_h*feat_w 个位置，需按步长取样
    cell = 10 * 50 + 10                     # 展平后的 cell 序号
    stride_between_tpl = 38 * 50            # 每个模板覆盖全部 cell 数
    tpl_idx = [cell + t * stride_between_tpl for t in range(9)]
    tpl = anchors[tpl_idx]
    cx0, cy0 = (10 + 0.5) * 16, (10 + 0.5) * 16   # 该 cell 中心原图坐标
    for a in tpl:
        w = a[2] - a[0]
        h = a[3] - a[1]
        print(f"    anchor 中心=({a[0] + w / 2:.0f},{a[1] + h / 2:.0f}) "
              f"(cell中心=({cx0:.0f},{cy0:.0f}))  w={w:.0f} h={h:.0f}  "
              f"ratio={w / h:.2f}")
    # RPN 回归的是相对 anchor 的偏移 (dx,dy,dw,dh)，示意解码：
    print("  RPN 回归目标示意: 框 = anchor + (dx,dy,dw,dh) 的小偏移修正")


# ---------- Part B：RoI 池化 ----------
def part_b_roi():
    print("=" * 62)
    print("Part B: RoI Pooling vs RoI Align（torchvision.ops）")
    try:
        from torchvision.ops import RoIPool, RoIAlign
    except ImportError:
        print("  [跳过] 未安装 torchvision")
        return
    # 假想：原图 800x800，stride=16 → 特征图 50x50，通道 256
    C, H, W = 256, 50, 50
    feat = torch.randn(1, C, H, W)
    # 原图坐标的候选框（第 1 维 batch index）
    boxes = torch.tensor([[0, 100, 100, 300, 300],   # 200x200 大框
                          [0, 400, 400, 500, 500]])  # 100x100 小框
    pool = RoIPool(output_size=7, spatial_scale=1.0 / 16)
    align = RoIAlign(output_size=7, spatial_scale=1.0 / 16, sampling_ratio=2)
    y_pool = pool(feat, boxes)
    y_align = align(feat, boxes)
    print(f"  特征图 {tuple(feat.shape)}，2 个候选框(原图坐标, 除以16映射)")
    print(f"  RoIPool 输出:  {tuple(y_pool.shape)}  (每框 7x7xC，固定尺寸→共享FC头)")
    print(f"  RoIAlign 输出: {tuple(y_align.shape)}  (浮点双线性采样，无取整误差)")
    print("  含义: 任意大小候选框都能得到固定 7x7 特征 → 检测头参数可共享")
    # 手写直觉版 RoIPool（简化：最近邻取整 + adaptive max pool）
    def simple_roipool(feat, box, out=7, scale=1 / 16):
        b, x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = int(x1 * scale), int(y1 * scale)
        x2, y2 = int(x2 * scale), int(y2 * scale)
        region = feat[:, :, y1:y2, x1:x2]
        return F.adaptive_max_pool2d(region, (out, out))
    manual = simple_roipool(feat[0], boxes[0].tolist())
    print(f"  手写简化 RoIPool(最近邻取整)输出: {tuple(manual.shape)} —— 概念等价")


# ---------- Part C：NMS 简演示 ----------
def part_c_nms():
    print("=" * 62)
    print("Part C: torchvision.ops.nms（同类内按 score 贪心去重）")
    try:
        from torchvision.ops import nms
    except ImportError:
        print("  [跳过] 未安装 torchvision")
        return
    torch.manual_seed(0)
    boxes = torch.tensor([[10, 10, 60, 60], [12, 12, 62, 62],   # 高度重叠(同一目标)
                          [100, 100, 160, 160], [150, 150, 210, 210]])  # 部分重叠(不同目标)
    scores = torch.tensor([0.95, 0.60, 0.90, 0.80])
    keep = nms(boxes, scores, iou_threshold=0.5)
    print(f"  输入 4 框, scores={scores.tolist()}")
    print(f"  NMS(0.5) 保留索引: {keep.tolist()}  -> 高置信框留下，重叠低分框被抑制")


# ---------- Part D：官方 Faster R-CNN ----------
def part_d_fasterrcnn():
    print("=" * 62)
    print("Part D: torchvision Faster R-CNN（随机权重，离线）")
    try:
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
    except ImportError:
        print("  [跳过] 未安装 torchvision")
        return
    # 兼容 2020~2025 接口：>=0.13 用 weights=None；更老版本用 pretrained=False
    try:
        model = fasterrcnn_resnet50_fpn(weights=None)
    except TypeError:
        model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 300, 300)          # 随机"图像"（内部会缩放到 800 短边）
    with torch.no_grad():
        pred = model(x)
    d = pred[0]
    print(f"  输出类型: {type(d).__name__}, 键: {sorted(d.keys())}")
    print(f"  boxes : {tuple(d['boxes'].shape)}   scores: {tuple(d['scores'].shape)}"
          f"   labels: {tuple(d['labels'].shape)}")
    n = min(3, len(d["scores"]))
    if n:
        print("  前 3 个预测（随机权重 → 数值无意义，仅验证流程/结构）:")
        for i in range(n):
            print(f"    score={d['scores'][i]:.3f} label={int(d['labels'][i])} "
                  f"box={[round(v, 1) for v in d['boxes'][i].tolist()]}")
    print("  注: 想用 COCO 预训练改 weights=COCO_V1 或 pretrained=True，需联网下载")


if __name__ == "__main__":
    torch.manual_seed(0)
    part_a_anchors()
    part_b_roi()
    part_c_nms()
    part_d_fasterrcnn()
    print("\n全部演示完成 ✓（未下载任何权重）")
