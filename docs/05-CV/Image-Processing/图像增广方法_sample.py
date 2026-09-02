# -*- coding: utf-8 -*-
"""
图像增广方法演示（可离线运行）

本脚本不下载任何权重、不依赖网络：
  1. 用随机 tensor 模拟图像，演示 torchvision 几何/色彩增广的输出形状；
  2. 用 PIL 随机图直观展示 RandomResizedCrop / 翻转 / ColorJitter 效果并保存对比图；
  3. 手写实现 batch 级 Mixup 与 CutMix（含标签软混合），验证输出形状与标签数值。

依赖：torch>=1.8；torchvision>=0.9 与 Pillow 用于几何/色彩演示（缺失时自动跳过，
不影响 Mixup/CutMix 等 torch 核心演示）。
"""
import copy
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image

# torchvision 可选：没有它时跳过几何/色彩/Normalize 演示，其余照常运行
try:
    from torchvision import transforms
    HAS_TV = True
except ImportError:
    transforms = None
    HAS_TV = False

OUT_DIR = "aug_output"
os.makedirs(OUT_DIR, exist_ok=True)


def make_dummy_image(seed=0):
    """造一张看得见“形状/颜色差异”的演示图：彩色圆环 + 横竖条纹。"""
    random.seed(seed)
    img = Image.new("RGB", (320, 320), (255, 255, 255))
    px = img.load()
    import math
    for y in range(320):
        for x in range(320):
            d = math.hypot(x - 160, y - 160)
            if 90 <= d <= 130:
                px[x, y] = (30, 90, 200)          # 蓝色圆环
            elif d < 45:
                px[x, y] = (220, 60, 60)          # 红色内圆
            if (x // 16) % 2 == 0 and 240 <= y < 320:
                px[x, y] = (40, 160, 80)          # 底部绿条纹
    return img


def demo_geometric_color():
    """几何 + 色彩增广：随机化 + 冻结 seed 复现同一变换（便于肉眼对比）。"""
    if not HAS_TV:
        print("[跳过] 未安装 torchvision，无法演示几何/色彩增广（装好后重跑即可）")
        return
    img = make_dummy_image()
    print("原始图像:", img.size, img.mode)

    # 训练 pipeline（随机）
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.15),
    ])
    # 验证 pipeline（确定性）
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    # 冻结随机种子 → 每张图与它打印的 seed 一一对应
    for i in range(4):
        torch.manual_seed(100 + i)
        out = train_tf(img)
        out.save(os.path.join(OUT_DIR, f"aug_train_{i}.png"))
        print(f"训练增广 #{i} (seed={100 + i}): 输出尺寸 {out.size}")

    val_out = val_tf(img)
    val_out.save(os.path.join(OUT_DIR, "aug_val_center.png"))
    print("验证确定性 pipeline: 输出尺寸", val_out.size, "(Resize 256 + CenterCrop 224)")


def demo_normalize():
    """Normalize 是确定性预处理而非增广：ToTensor -> (x/255) -> (x-mean)/std。"""
    if not HAS_TV:
        print("\n[跳过] 未安装 torchvision，无法演示 ToTensor+Normalize pipeline")
        return
    t = torch.tensor([[[128.0]]])  # 单像素灰度 128
    print("\nNormalize 演示（单像素）: 128/255 =", round(t.item() / 255, 4))
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    # 造 1x1 灰度 PIL 图做全流程
    img = Image.new("L", (1, 1), 128)
    x = tf(img)
    print("  Normalize 后 =", round(x.item(), 4),
          "(≈ (128/255 - 0.485) / 0.229)  注意必须与预训练统计一致")


def mixup_batch(x, y, alpha=0.2):
    """Mixup：同一 batch 内按随机置换对样本做线性插值，标签也线性混合。"""
    lam = torch.distributions.Beta(alpha, alpha).sample()
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]   # y 为 one-hot 软标签
    return x_mix, y_mix, lam.item()


def cutmix_batch(x, y, alpha=1.0):
    """CutMix：把样本 b 的随机矩形区域粘到样本 a 上，标签按面积比 λ 混合。"""
    lam = torch.distributions.Beta(alpha, alpha).sample()
    idx = torch.randperm(x.size(0))
    H, W = x.size(2), x.size(3)
    # 依据论文：裁剪框尺寸由 λ 反推，保证粘贴面积 ≈ 1-λ
    cut_ratio = (1 - lam).sqrt()
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = int(torch.randint(W, (1,)))
    cy = int(torch.randint(H, (1,)))
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, x1 + cut_w)
    y2 = min(H, y1 + cut_h)
    x[:, :, y1:y2, x1:x2] = x[idx][:, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)   # 保留 a 的面积占比（float）
    y_mix = lam * y + (1 - lam) * y[idx]
    return x, y_mix, lam


def demo_mixup_cutmix():
    """batch 级 Mixup / CutMix：8 张 3x32x32 随机图 + 8 个 one-hot 标签。"""
    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32)
    y = F.one_hot(torch.arange(8) % 4, num_classes=4).float()  # (8,4)

    xm, ym, lam_m = mixup_batch(copy.deepcopy(x), copy.deepcopy(y))
    print("\nMixup  : x", tuple(xm.shape), "| 标签和=1?",
          bool(torch.allclose(ym.sum(dim=1), torch.ones(8))), "| λ =", round(lam_m, 3))
    # λ 相同 → batch 内所有样本混合比例相同（Mixup 特点）

    xc, yc, lam_c = cutmix_batch(x.clone(), y.clone())
    print("CutMix : x", tuple(xc.shape), "| 标签和=1?",
          bool(torch.allclose(yc.sum(dim=1), torch.ones(8))), "| 平均λ =",
          round(lam_c, 3))
    # 对比：CutMix 每个样本粘贴位置/面积都不同；Mixup 整体 λ 相同。
    # 用混合软标签 → loss 直接用 nn.CrossEntropyLoss 会报错，应手动 CE：
    # loss = -(ym * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def demo_pil_save_grid():
    """把变换后的图拼成一行便于肉眼对比（可选，无 matplotlib 也能跑）。"""
    # 利用上面已保存的文件；此处只打印提示
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".png"))
    print("\n已保存增广结果图到", os.path.abspath(OUT_DIR), "->", files)


if __name__ == "__main__":
    demo_geometric_color()
    demo_normalize()
    demo_mixup_cutmix()
    demo_pil_save_grid()
    print("\n全部演示完成 ✓（未下载任何权重/数据，可离线运行）")
