---
title: "计算机视觉（05）"
description: "图像预处理、CNN 经典架构与目标检测。"
tags: [计算机视觉, CV, CNN, 目标检测]
---

# 计算机视觉（05）

> **板块定位**：图像表示与预处理、卷积神经网络架构、以及以目标检测为代表的核心视觉任务。

## 覆盖主题

| 子目录 | 覆盖内容 |
| :--- | :--- |
| `Image-Processing` | 图像的张量表示、归一化、增广、卷积/池化基础 |
| `CNN-Architectures` | LeNet/AlexNet/VGG、ResNet、ViT |
| `Object-Detection` | 两阶段与单阶段检测、R-CNN 系、YOLO 系、评估（IoU/mAP） |

## 前置知识

- 机器学习基础（02）、深度学习与 PyTorch（03）：张量、卷积运算、训练循环。
- ResNet 拆解详见 03 板块。

## 写作约定

- 每个知识点配 PyTorch/`torchvision` 可运行片段。
- CNN 架构类附**结构对比表**与**参数量**。
