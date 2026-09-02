# -*- coding: utf-8 -*-
"""
YOLO 单阶段系列核心机制演示（离线可运行，仅依赖 torch/numpy）

检测网络输出的本质是"一大张数字"，必须解码 + 过滤 + NMS 才是框。本脚本：
  Part A  标签格式：像素 xyxy <-> 归一化 xywh（Ultralytics txt 格式）互转
  Part B  网格化输出：模拟 YOLO 特征图张量 (B, H, W, 4+1+C) 的语义切片
  Part C  解码器：v3 风格 anchor 解码 与 v8 风格 anchor-free 解码各来一遍
  Part D  后处理：置信度阈值 + 手写 NMS → 最终框
  Part E  附赠：Ultralytics YOLOv8 官方 API 示例（若已 pip install ultralytics
          且能联网下载权重；默认 try/except 跳过，不影响离线部分）

运行：python YOLO单阶段系列_sample.py
"""
import torch


def xyxy_to_xywh_norm(box, img_w, img_h):
    """像素角点框 -> 归一化中心宽高（类别行前 4 个数的训练标签格式）。"""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def xywh_norm_to_xyxy(t, img_w, img_h):
    """归一化中心宽高 -> 像素角点框（预测输出可视化常用）。"""
    cx, cy, w, h = t
    return (cx - w / 2) * img_w, (cy - h / 2) * img_h, \
           (cx + w / 2) * img_w, (cy + h / 2) * img_h


def part_a_label_format():
    print("=" * 62)
    print("Part A: 标签格式换算（像素 xyxy <-> 归一化 xywh）")
    img_w, img_h = 640, 480
    box = (100, 120, 300, 360)                     # 像素角点
    t = xyxy_to_xywh_norm(box, img_w, img_h)
    print(f"  图像 {img_w}x{img_h}, 目标像素框 {box}")
    print(f"  -> 归一化 xywh: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}, {t[3]:.4f})")
    print(f"  -> 训练 txt 行: 0 {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {t[3]:.6f}")
    back = xywh_norm_to_xyxy(t, img_w, img_h)
    print(f"  -> 还原 xyxy  : {tuple(round(v, 1) for v in back)}（应≈原框）")


def make_grid(side, stride):
    """side x side 网格每个 cell 中心在输入图的坐标 (side, side, 2)。

    纯广播构造，不依赖 meshgrid 的新旧 indexing 语义，兼容任意 torch 版本。
    第 (y, x) 个 cell 中心 = ((x+0.5)*stride, (y+0.5)*stride)
    """
    coords = (torch.arange(side).float() + 0.5) * stride   # (side,)
    xs = coords.view(1, side).expand(side, side)           # 每行相同
    ys = coords.view(side, 1).expand(side, side)           # 每列相同
    return torch.stack([xs, ys], dim=-1)                   # (side, side, 2)


def decode_anchor_based(raw, anchors, stride, conf_th=0.5, num_cls=3):
    """YOLOv3 风格解码。

    raw: (B, H, W, num_anchors*(5+num_cls))，每个 anchor 对应 5+num_cls 个数：
         tx,ty,tw,th(网络原始偏移) + objectness + 类别分。
    anchors: 该尺度预设 anchor 的 (w,h)，单位是"网格"，乘 stride 变像素。
    公式: cx = (grid_x + sigmoid(tx))*stride, cy 同理；
          w = anchor_w * exp(tw) * stride, h 同理。
    """
    B, H, W, _ = raw.shape
    na = len(anchors)
    grid = make_grid(H, stride)                     # (H, W, 2) 中心
    raw = raw.view(B, H, W, na, 5 + num_cls)
    obj = raw[..., 4].sigmoid()
    cls = raw[..., 5:].softmax(-1)
    conf, cls_id = (obj.unsqueeze(-1) * cls).max(dim=-1)  # 目标度×类别度
    tx, ty, tw, th = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
    # 每 (H,W) 位置广播 anchor 中心
    gx = grid[:, :, 0].view(1, H, W, 1)
    gy = grid[:, :, 1].view(1, H, W, 1)
    cx = (gx + tx.sigmoid()) * stride
    cy = (gy + ty.sigmoid()) * stride
    aw = torch.tensor([a[0] for a in anchors]).view(1, 1, 1, na)
    ah = torch.tensor([a[1] for a in anchors]).view(1, 1, 1, na)
    w = aw * tw.exp() * stride
    h = ah * th.exp() * stride
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    boxes = boxes[conf > conf_th]
    scores = conf[conf > conf_th]
    ids = cls_id[conf > conf_th]
    return boxes, scores, ids


def decode_anchor_free(raw, stride, conf_th=0.5, num_cls=3):
    """YOLOv8 风格 anchor-free 解码。

    raw: (B, H, W, 4+num_cls)：x,y 是相对本 cell 的 sigmoid 偏移（中心点），
         w,h 直接回归像素宽高（训练时用 CIoU loss 监督）。
    简化：x,y 偏移直接乘 stride；w,h 视为已学好的像素宽高（演示用）。
    """
    B, H, W, _ = raw.shape
    grid = make_grid(H, stride)                     # (H, W, 2)
    x, y, w, h = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
    conf, cls_id = raw[..., 4:].softmax(-1).max(dim=-1)
    gx = grid[:, :, 0].view(1, H, W)
    gy = grid[:, :, 1].view(1, H, W)
    cx = (gx + x.sigmoid()) * stride
    cy = (gy + y.sigmoid()) * stride
    w, h = w.abs() * stride, h.abs() * stride
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    keep = conf > conf_th
    return boxes[keep], conf[keep], cls_id[keep]


def nms(boxes, scores, iou_th=0.45):
    """手写贪心 NMS：按分数从高到低，抑制与该框 IoU>阈值 的其余框。"""
    if not len(boxes):
        return torch.empty(0, dtype=torch.long)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0].item()
        keep.append(i)
        xx1 = torch.clamp(x1[order[1:]], min=x1[i])
        yy1 = torch.clamp(y1[order[1:]], min=y1[i])
        xx2 = torch.clamp(x2[order[1:]], max=x2[i])
        yy2 = torch.clamp(y2[order[1:]], max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return torch.tensor(keep)


def part_bcd_decode_pipeline():
    print("=" * 62)
    print("Part B/C/D: 网格输出 -> 解码 -> 过滤 -> NMS（模拟 13x13 网格，stride=32）")
    torch.manual_seed(1)
    img_size = 416
    side, stride = 13, 32                       # 416/32
    num_cls = 3
    # --- v3 风格: 每格 3 anchors × (5+3) ---
    anchors = [(0.5, 0.6), (1.2, 1.3), (2.5, 2.8)]     # 网格单位
    raw_v3 = torch.randn(1, side, side, len(anchors) * (5 + num_cls))
    b, s, c = decode_anchor_based(raw_v3, anchors, stride, conf_th=0.5, num_cls=num_cls)
    if len(b):
        print(f"  [v3 anchor] 解码后命中 {len(b)} 个候选, 例: "
              f"score={s[0]:.3f} cls={c[0]} box={[round(v, 1) for v in b[0].tolist()]}")
    else:
        print("  [v3 anchor] 该随机输出无候选通过 conf 阈值（阈值调低或换 seed 即有）")
    # 类别独立 NMS（同一类才互相抑制）
    final = []
    for ci in range(num_cls):
        m = c == ci
        if m.sum():
            final.append(nms(b[m], s[m], iou_th=0.45))
    print(f"  类别内 NMS 后合计保留 {sum(len(f) for f in final)} 个框")

    # --- v8 风格 anchor-free: 每格 (4+3) ---
    raw_v8 = torch.randn(1, side, side, 4 + num_cls)
    b2, s2, c2 = decode_anchor_free(raw_v8, stride, conf_th=0.5, num_cls=num_cls)
    if len(b2):
        print(f"  [v8 anchor-free] 解码后命中 {len(b2)} 个候选"
              f"（无 anchor 超参）, 最高分 {s2.max():.3f}")
    else:
        print("  [v8 anchor-free] 该随机输出无候选通过 conf 阈值")
    # 同一位置同一目标只保留最高分（简化：conf 过滤已够演示）


def part_e_ultralytics():
    print("=" * 62)
    print("Part E: Ultralytics YOLOv8 官方用法（需 ultralytics + 联网下载权重）")
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("  [跳过] 未安装 ultralytics。离线部分不受影响。")
        print("  用法备忘:")
        print("    pip install ultralytics")
        print("    from ultralytics import YOLO")
        print("    model = YOLO('yolov8n.pt')")
        print("    model.train(data='coco8.yaml', epochs=50, imgsz=640)")
        print("    res = model.predict('bus.jpg', conf=0.25, iou=0.45)")
        print("    res[0].boxes.xyxy / .conf / .cls  # 框/置信度/类别")
        return
    from ultralytics import YOLO
    # 联网可下载 ~6MB 权重；网络不可用时以下会抛错，捕获后友好提示
    try:
        model = YOLO("yolov8n.pt")
        print(f"  已加载 {model.task} 模型: {model.model_name if hasattr(model, 'model_name') else 'yolov8n'}")
    except Exception as e:
        print(f"  [跳过] 加载 yolov8n.pt 失败(可能无网络): {type(e).__name__}")
        print("  联网后重跑本函数即可体验: model.predict / model.train / model.export")


if __name__ == "__main__":
    part_a_label_format()
    part_bcd_decode_pipeline()
    part_e_ultralytics()
    print("\n全部演示完成 ✓（核心解码部分离线可跑）")
