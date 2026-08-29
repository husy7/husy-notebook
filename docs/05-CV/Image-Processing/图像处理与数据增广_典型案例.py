# -*- coding: utf-8 -*-
"""
图像处理基础：张量表示、归一化与增广 —— 典型代码演示
====================================================
覆盖知识点：
  1. 图像 → 张量（CHW / 归一化到[0,1]）
  2. 用 torchvision transforms 做标准化与数据增广
  3. 卷积与池化算子的直观演示
  4. 常见坑：OpenCV(BGR/HWC) vs PyTorch(RGB/CHW)
  5. 可辨别的可视化辅助（用像素统计而非依赖 matplotlib）

依赖：pip install torch torchvision pillow numpy opencv-python(可选)
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

torch.manual_seed(0)

# =====================================================================
# 一、制造一张"虚拟图像"来演示张量表示（避免依赖真实图片文件）
# =====================================================================
# torchvision.io 或 PIL 读图 → 形状与值域
from PIL import Image
# 生成一张纯色/渐变图（假图片，3 通道 64x64）
np_img = np.zeros((64, 64, 3), dtype=np.uint8)
np_img[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]   # R 渐变
np_img[:, :, 1] = 128                                                  # G 固定
np_img[:, :, 2] = 32                                                   # B 固定
pil_img = Image.fromarray(np_img, "RGB")
print("[读图] PIL Image 模式:", pil_img.mode, " 尺寸:", pil_img.size)

# 转张量：ToTensor 会把值缩放到 [0,1]，并转成 CHW
tensor_img = T.ToTensor()(pil_img)     # -> (3, 64, 64), float，范围 [0,1]
print("[ToTensor] shape =", tuple(tensor_img.shape),
      " 值域 = [%.2f, %.2f]" % (tensor_img.min().item(), tensor_img.max().item()),
      " 布局 = (通道C, 高H, 宽W)")

# 手动验证：PIL(RGB, HWC) → PyTorch(CHW)
manual = np_img.transpose(2, 0, 1).astype(np.float32) / 255.0   # HWC→CHW + 归一化
print("[手动转CHW] 与 ToTensor 误差 =", np.abs(manual - tensor_img.numpy()).max())

# =====================================================================
# 二、标准化：用均值方差把每个通道归一到接近"标准正态"
# =====================================================================
# ImageNet 统计量（配合 torchvision 预训练模型必须用）
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

normalize = T.Normalize(mean=MEAN, std=STD)
norm_tensor = normalize(tensor_img)
print("\n[标准化] 每通道 (原均值→规范化):")
for c in range(3):
    ch = norm_tensor[c]
    print(f"   通道{c}:  min={ch.min():.3f} max={ch.max():.3f} "
          f"mean≈{(ch.mean() + 0):.4f}")

# =====================================================================
# 三、数据增广：随机变换让模型更鲁棒（只用于训练集）
# =====================================================================
train_aug = T.Compose([
    T.RandomResizedCrop(48, scale=(0.8, 1.0)),   # 随机裁剪缩放到 48x48
    T.RandomHorizontalFlip(),                    # 随机水平翻转
    T.ColorJitter(brightness=0.2, contrast=0.2), # 颜色扰动
    T.ToTensor(),                                # -> CHW float [0,1]
    normalize,                                   # 标准化
])
# 注意：ToTensor 之后才能 Normalize；且增广要作用在训练集而非验证集

aug1 = train_aug(pil_img)
aug2 = train_aug(pil_img)
print("\n[增广] 同一张原图两次随机变换:")
print("       增广1 shape", tuple(aug1.shape), " mean", round(aug1.mean().item(), 3))
print("       增广2 shape", tuple(aug2.shape), " mean", round(aug2.mean().item(), 3))
print("       (两次不同 → 增广引入了数据多样性)")

val_transform = T.Compose([          # 验证/评估：只做 resize + 归一化，不做随机
    T.Resize(48),
    T.ToTensor(),
    normalize,
])

# =====================================================================
# 四、卷积与池化：CNN 的基础算子
# =====================================================================
def conv_pool_demo(t_in):
    """对一个 3 通道小图像做卷积+池化，观察 shape 变化。"""
    conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    pool = nn.MaxPool2d(2)                    # 2x2 最大池化，尺寸减半
    x = t_in.unsqueeze(0)                     # (C,H,W) -> (1,C,H,W) 加 batch
    out_conv = conv(x)
    out_pool = pool(out_conv)
    print(f"输入 {tuple(x.shape)} → 卷积(3x3)-> {tuple(out_conv.shape)} "
          f"→ 池化(2x2)-> {tuple(out_pool.shape)}")
    return out_pool

conv_pool_demo(norm_tensor)

# 解释：卷积提取局部特征（16 个不同卷积核）；池化降维、增平移不变性。

# =====================================================================
# 五、常见坑：OpenCV() 通道序 / Resize 后接 model
# =====================================================================
# 坑1：OpenCV 读图是 BGR 且 HWC
# import cv2
# img_bgr = cv2.imread("img.jpg")           # (H, W, 3), BGR
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # 必须转 RGB
# img_chw = img_rgb.transpose(2, 0, 1)      # 转 CHW
# tensor_color_correct = torch.from_numpy(img_chw).float() / 255.0

# 坑2：ToTensor 与 Normalize 顺序
# 正确：先 ToTensor([0,1]) 再 Normalize(均值/std)；颠倒了会算错。

# =====================================================================
# 六、把它们拼成完整管线（模拟真实训练的前处理）
# =====================================================================
def data_pipeline(images, train=True):
    """把一张 PIL 图像按 train/val 模式做完整预处理，返回模型可吃的张量。"""
    aug = train_aug if train else val_transform
    return torch.stack([aug(img) for img in images])   # (N,C,H,W)

demo_batch = data_pipeline([pil_img, pil_img], train=False)
print("\n[完整管线] 单个 48x48 输入 → batch shape:", tuple(demo_batch.shape))

print("\n[小结] 读取→ToTensor[0,1]→Normalize(用ImageNet均/std)→(训练时)增广→送入 CNN")
