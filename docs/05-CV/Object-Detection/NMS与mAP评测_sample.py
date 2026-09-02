# -*- coding: utf-8 -*-
"""
NMS 与 mAP 评测演示（离线可运行，仅依赖 torch/numpy）

自洽实现完整评测链路（不依赖 COCO API，方便看透原理）：
  Part A  IoU 计算（含 torchvision.ops.box_iou 对照，若可用）
  Part B  手写 NMS + Soft-NMS 对比（重复框去重，Soft-NMS 分数衰减）
  Part C  玩具数据集上走完整评测：预测按 score 排序 -> 与 GT 匹配(每GT只匹配一次)
          -> 逐点插值 PR -> AP
  Part D  mAP@0.5  vs  mAP@0.5:0.95 的差异演示（用"框画得不准"的模型对比）
  Part E  附 torchvision / pycocotools 官方用法速览（不强制安装）

运行：python NMS与mAP评测_sample.py
"""
import torch


# ---------- Part A: IoU ----------
def box_iou(a, b):
    """向量化 IoU：a:(N,4) b:(M,4) -> (N,M)，xyxy 角点。"""
    ax1, ay1, ax2, ay2 = a[:, None, 0], a[:, None, 1], a[:, None, 2], a[:, None, 3]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    iw = torch.clamp(torch.min(ax2, bx2) - torch.max(ax1, bx1), min=0)
    ih = torch.clamp(torch.min(ay2, by2) - torch.max(ay1, by1), min=0)
    inter = iw * ih
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-9)


def part_a():
    print("=" * 62)
    print("Part A: IoU 手写 vs torchvision.ops.box_iou")
    a = torch.tensor([[0.0, 0, 10, 10], [20, 20, 30, 30]])
    b = torch.tensor([[5.0, 5, 15, 15], [20, 20, 28, 28]])
    iou = box_iou(a, b)
    print(f"  手写 IoU 矩阵:\n{iou.numpy().round(3)}")
    try:
        from torchvision.ops import box_iou as tv_iou
        assert torch.allclose(iou, tv_iou(a, b), atol=1e-6)
        print("  与 torchvision.ops.box_iou 结果一致 ✓")
    except Exception:
        print("  (torchvision 不可用，跳过对照)")


# ---------- Part B: NMS & Soft-NMS ----------
def nms(boxes, scores, iou_th=0.5):
    """标准贪心 NMS：高置信保留，重叠 > 阈值 的硬删除。"""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][iou <= iou_th]
    return torch.tensor(keep)


def soft_nms(boxes, scores, sigma=0.5, score_th=0.01):
    """Soft-NMS：不硬删重叠框，而是按 IoU 高斯衰减其分数、每轮重取最大分，
    低于 score_th 才出局（稠密/粘连场景比硬 NMS 保留更多真框）。"""
    out = scores.clone().float()
    keep = []
    while out.max() >= score_th:
        i = int(out.argmax())
        keep.append(i)
        iou = box_iou(boxes[i:i + 1], boxes)[0]     # 与所有框的 IoU
        out = out * torch.exp(-(iou ** 2) / sigma)  # 重叠越大分数掉越多
        out[i] = -1.0                               # 已选帧出局
    return torch.tensor(keep), out


def part_b():
    print("=" * 62)
    print("Part B: NMS vs Soft-NMS")
    boxes = torch.tensor([
        [10., 10, 60, 60], [12, 12, 62, 62], [11, 11, 61, 61],  # 同一目标三重框
        [100, 100, 160, 160],                                    # 独立目标
    ])
    scores = torch.tensor([0.95, 0.90, 0.85, 0.80])
    k = nms(boxes, scores, iou_th=0.5)
    ks, out_s = soft_nms(boxes, scores)
    decayed = [round(float(out_s[i]), 3) for i in ks.tolist()]
    print(f"  原始 scores : {[round(float(s), 2) for s in scores.tolist()]}")
    print(f"  标准 NMS    : 保留索引 {k.tolist()}  (重叠 3 框只留最高分)")
    print(f"  Soft-NMS    : 保留索引 {ks.tolist()}")
    print(f"    处理顺序上的分数: {decayed}  (低分框衰减到 <0.01 才出局，"
          f"且每轮按最新分数重排)")
    print("  -> 粘连/稠密场景 Soft-NMS 通常召回更好（代价：多一步衰减计算）")


# ---------- Part C: 匹配 + PR + AP ----------
def match_and_count(preds, gts, iou_th):
    """逐框匹配（每 GT 只匹配一次）。preds: [(score, box)]; gts: [(label, box)]"""
    preds = sorted(preds, key=lambda x: -x[0])     # score 降序
    tp, fp = [], []
    used = [False] * len(gts)
    for score, pbox in preds:
        best_i, best_iou = -1, iou_th
        for gi, (_, gbox) in enumerate(gts):
            if used[gi]:
                continue
            v = box_iou(pbox[None], gbox[None])[0, 0].item()
            if v > best_iou:
                best_iou, best_i = v, gi
        if best_i >= 0:
            used[best_i] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    return tp, fp, used


def ap_from_tp_fp(tp, fp, n_gt):
    """逐点插值 AP = Σ(r_{n+1}-r_n)·p_interp(r_{n+1})，p_interp 向右取 max。"""
    tp = torch.tensor(tp, dtype=torch.float)
    fp = torch.tensor(fp, dtype=torch.float)
    ctp, cfp = tp.cumsum(0), fp.cumsum(0)
    rec = ctp / max(n_gt, 1)
    prec = ctp / (ctp + cfp + 1e-9)
    # 单调化：每个 recall 点取"该点右侧最大 precision"
    p_interp = torch.flip(torch.cummax(torch.flip(prec, [0]), 0)[0], [0])
    # 积分：在 recall 变化处加矩形
    rec_ext = torch.cat([torch.zeros(1), rec, torch.ones(1)])
    p_ext = torch.cat([torch.zeros(1), p_interp, torch.zeros(1)])
    ap = torch.sum((rec_ext[1:] - rec_ext[:-1]) * p_ext[1:]).item()
    return ap


def part_c():
    print("=" * 62)
    print("Part C: 玩具样例完整评测链路（单类 3 GT，阈值 IoU=0.5）")
    gts = [(0, torch.tensor([10., 10, 60, 60])),        # GT A
           (0, torch.tensor([80., 80, 120, 120])),      # GT B
           (0, torch.tensor([200., 200, 240, 240]))]    # GT C
    # 模型输出 5 个预测（含 1 个"抢了已匹配 GT"的重复 + 1 误检 + 1 漏检）
    preds = [
        (0.95, torch.tensor([11., 11, 61, 61])),        # 命中 A
        (0.85, torch.tensor([81., 81, 121, 121])),      # 命中 B
        (0.70, torch.tensor([86., 86, 124, 124])),      # 与 B IoU≈0.61>0.5，
                                                        #   但 B 已被占用 → FP!
        (0.60, torch.tensor([300., 300, 330, 330])),    # 误检 → FP
        # GT C 无人命中 → FN
    ]
    tp, fp, used = match_and_count(preds, gts, iou_th=0.5)
    n_gt = len(gts)
    print(f"  TP 序列: {tp}\n  FP 序列: {fp}")
    print(f"  FN = {sum(1 for u in used if not u)} (GT C 未被命中)")
    ap = ap_from_tp_fp(tp, fp, n_gt)
    print(f"  AP@0.5 = {ap:.4f}")
    # 与 pycocotools 同口径思路对照：AP 衡量"排序质量"，漏检/误检都会压 AP


# ---------- Part D: mAP@0.5 vs mAP@0.5:0.95 ----------
def coco_style_ap(preds, gts):
    """COCO 口径：10 个 IoU 阈值 (0.5,0.55,...,0.95) 的 AP 均值。"""
    aps = []
    for t in torch.arange(0.5, 0.95 + 1e-6, 0.05):
        tp, fp, _ = match_and_count(preds, gts, iou_th=float(t))
        aps.append(ap_from_tp_fp(tp, fp, len(gts)))
    return aps, sum(aps) / len(aps)


def make_model_preds(offset):
    """模拟"框画得偏 offset 像素"的检测器：6 个 GT 各带 1 个预测。"""
    gts = [(i, torch.tensor([float(x), 100 + i * 50, x + 40, 140 + i * 50]))
           for i, x in enumerate(range(10, 70, 10))]
    preds = [(0.9, torch.tensor([float(x) + offset, 100 + i * 50 + offset,
                                 x + 40 + offset, 140 + i * 50 + offset]))
             for i, x in enumerate(range(10, 70, 10))]
    return gts, preds


def part_d():
    print("=" * 62)
    print("Part D: mAP@0.5 vs mAP@0.5:0.95（同一模型，框偏 2px vs 偏 5px）")
    for off in (2, 5):
        gts, preds = make_model_preds(off)
        aps, ap_coco = coco_style_ap(preds, gts)
        ap50 = aps[0]
        print(f"  框偏 {off}px : mAP@0.5={ap50:.4f} | "
              f"mAP@0.5:0.95={ap_coco:.4f} "
              f"(10 档均值: {[round(v, 2) for v in aps]})")
    print("  -> 框偏 5px 时 AP50 仍=1.0(宽松阈值不在乎)，"
          "但 mAP@.5:.95 明显掉(更高档阈值不再命中)："
          "该指标对定位精度极其敏感")


def part_e():
    print("=" * 62)
    print("Part E: 官方工具用法速览（需安装对应包）")
    print("  torchvision.ops.nms / box_iou  -> 后处理")
    print("  pycocotools: COCOeval(anns, preds, 'bbox'); "
          "eval.evaluate(); eval.accumulate(); eval.summarize()")
    print("    # summarize() 输出 AP@[.5:.95], AP@.5, AP@.75, AP_s/m/l, AR...")
    print("    # 注意 COCO 标注框格式是 [x, y, w, h]（左上角+宽高），非 xywh 中心")
    print("  Ultralytics: model.val() 训练后直接给 mAP50 / mAP50-95")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    print("\n全部演示完成 ✓")
