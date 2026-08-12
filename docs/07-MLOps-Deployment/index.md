---
title: "MLOps 与部署（07）"
description: "ONNX 转换、Triton 推理服务器、FastAPI 服务与 Docker-K8s 部署。"
tags: [MLOps, 部署]
---

# MLOps 与部署（07）

> **板块定位**：模型从训练到生产的最后一公里——转换、服务化、容器化与编排。

## 覆盖主题

| 子目录 | 覆盖内容 | 状态 |
| :--- | :--- | :--- |
| `ONNX-Conversion` | PyTorch → ONNX 导出、算子兼容、优化 | 🔴 待填充 |
| `Triton-Server` | Triton 推理服务、动态 batching、多模型管理 | 🔴 待填充 |
| `FastAPI-Serving` | FastAPI 接口、pydantic 校验、并发与压测 | 🔴 待填充 |
| `Docker-K8s` | Dockerfile 优化、镜像瘦身、K8s 部署与扩缩容 | 🔴 待填充 |

## 规划笔记

- [ ] PyTorch 模型导出 ONNX 全流程
- [ ] Triton 动态 Batching 原理
- [ ] FastAPI 高性能推理服务模板
- [ ] Docker 镜像瘦身实战

## 写作约定

- 涉及命令/配置均给出**可直接复制的完整示例**与**验证步骤**。
- 性能类结论附**实测数据**（延迟 / 吞吐 / 显存）。
