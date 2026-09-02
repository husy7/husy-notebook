---
title: "ViT 视觉 Transformer（Vision Transformer）"
tags: [计算机视觉, Transformer, ViT]
date: 2026-08-30
---

# ViT 视觉 Transformer（Vision Transformer）

## 定义

ViT（Vision Transformer，视觉 Transformer）是 Dosovitskiy 等人 2020 年提出（ICLR 2021，*An Image is Worth 16x16 Words*）的架构：**把图像切成固定大小的 patch（图像块）序列，当作一串 token 喂给标准 Transformer Encoder**（自注意力 + MLP + LayerNorm + 残差连接）做全局建模，分类任务取额外拼上的可学习 `[class] token` 的输出。一句话总结：把图像切成 **patch 序列**喂给标准 Transformer，丢掉 CNN 的平移等变/局部性先验，换来**任意长程依赖**与**数据/算力规模下的可扩展性（scaling）**，代价是**需要远超 CNN 的数据量或强增广/蒸馏才能从头训好**。

- **解决什么问题**：CNN 的归纳偏置是双刃剑——参数少、数据效率高、天然平移等变，但感受野靠堆层慢慢增长 → 长程依赖（图像里相距很远的两个区域互相影响）建模弱且低效；局部先验在数据足够大时反而成了"天花板"（NLP 已证明 scaling 的威力，CV 想复制）。
- **核心特征**：**最小化归纳偏置**——只保留"图像 = 一组 patch 序列"这一条假设，其余（哪些 patch 该互相看、看多重）全交给注意力去学；第一个 self-attention 层就能完成全图交互（每个 token 直接看到全图），长程交互一步到位，效果随模型/数据一起涨。
- **代价/局限**：没有 CNN 的平移等变与数据效率；从头训 ImageNet-1k（1.28M）会**低于**同量级 ResNet，需要 ImageNet-21k / JFT-300M 级预训练或 DeiT 式**蒸馏 + 强增广**；自注意力随 token 数 **O(N²)**，更高分辨率/更小 patch 计算量会爆炸。
- **适用范畴**：图像分类起家，如今广泛作为检测/分割/视频等任务的通用骨干（骨干替换不影响评测指标）；数据 <1M 量级、追求数据效率的场景仍优先 CNN 或 Hybrid（CNN stem + ViT）缓解小数据弱势。

## 原理

**1) 为什么这样设计**：把 CV 问题"翻译"成 NLP 式序列建模——图像 = 一组 patch token，自注意力是唯一的信息交换机制，先验减到最少；这样当数据/算力足够大时，性能随模型与数据规模一起涨（scaling），长程交互也不再有"逐层传播"的瓶颈。自注意力的设计动机：让每个位置**一步**看到序列里所有其他位置，权重完全由内容相似度（Q·K）决定，而不是由卷积核的空间位置决定。

**2) 核心流程（以 ViT-B/16、输入 224×224×3 为例）**

1. **Patchify + Embedding**：切成 16×16 的 patch，共 N = (224/16)² = 196 个；每个 patch 展平后经线性投影 → D=768 维 token。数学等价：`Conv2d(C→D, kernel=16, stride=16)`（即 kernel=stride=patch_size），随后把输出重排成序列。
2. **拼接 `[class] token`**：一个可学习向量（或取所有 patch 的 GAP，二选一），分类时取它的输出；序列长度变为 N+1 = 197——"序列长度 197"的由来：196 个 patch token + 1 个 class token。
3. **加位置编码**：可学习 1D，或 2D sin-cos / 2D 可学习，形状 (N+1, D)，与 token 逐位相加。注意力本身是无序集合运算，位置信息必须显式注入。
4. **L=12 个 Transformer Encoder Block**：每个 Block = `LayerNorm → Multi-Head Self-Attention → 残差 → LayerNorm → MLP(两线性层+GELU) → 残差`（Pre-LN，LN 在子层**前**，与 ViT 一致）。
5. **分类头**：取 `[class] token` 输出 → LayerNorm → `Linear(D, num_classes)`。

**3) 张量形状流动**（B=批次）：`(B,3,224,224)` → patch embed `(B,196,768)` → 拼 class token `(B,197,768)` → +pos `(B,197,768)` → ×12 Block 形状不变 `(B,197,768)` → 取 `x[:,0]` `(B,768)` → head `(B,C)`。

**4) 自注意力机制**（单头直觉）：`Q,K,V = x·Wq, x·Wk, x·Wv`；注意力权重 `A = softmax(QKᵀ/√d)`，形状 (B,197,197)——就是"每个 patch 该看哪些 patch、看多重"的**全图交互矩阵**；输出 `= A·V`。多头 = 多组 QKV 并行，各学一种交互模式再拼接。√d 缩放防止点积过大把 softmax 推到饱和区。

**5) 复杂度推导**：单层 self-attention 对 token 数 O(N²)。224²/16² = 196 token 很轻松；输入从 224 提到 448 → token 数变 784（×4），自注意力 FLOPs 约 ×4² = **×16**，而 CNN 局部运算随分辨率只约 ×4（近线性）——所以更大分辨率/更小 patch 会让 ViT 计算量爆炸。

**6) 与 CNN 的逐维机制差异**

| 维度 | CNN | ViT |
|---|---|---|
| 归纳偏置 | 局部性 + 权值共享 + 平移等变 | 只有"patch 化"；交互全靠学 |
| 数据需求 | 百万级可训好（数据效率高） | 从头训 ImageNet-1k(1.28M) 会**低于**同量级 ResNet；需 ImageNet-21k/JFT-300M 预训练，或 DeiT 式**蒸馏 + 强增广** |
| 尺度/平移 | 卷积滑动共享 → 平移等变；尺度靠数据 | 无平移等变；全靠增广与数据覆盖 |
| 全局建模 | 靠深堆大感受野，浅层看局部 | 第一层 self-attention 就能全局交互 |
| 复杂度 | 随分辨率线性（局部运算） | 随 token 数 **O(N²)** |
| 位置 | 隐含在卷积坐标 | 显式位置编码，须处理"分辨率变了编码怎么办" |
| 收敛 | 小数据快 | 小数据慢/差；大算力大数据强 |

## 应用

**典型使用场景**：① 大数据（ImageNet-21k / JFT 级预训练）下的图像分类与通用特征骨干；② 下游检测/分割把 ViT 当 backbone（DETR 类端到端直接用 Transformer，两阶段检测骨干仍常用 CNN）；③ 中小数据微调（加载预训练权重 + 强增广）。快速上手直接使用 torchvision 内置实现（≥0.12 提供 `torchvision.models.vit_b_16` / `vit_b_32` / `vit_l_16`，权重 `ViT_B_16_Weights.IMAGENET1K_V1`，配置在 `torchvision.models.vision_transformer.ViT_B_16_Weights`）。

**快速上手步骤**：加载预训练模型 + 配套 `weights.transforms()` 预处理（Normalize 必须与训练一致）→ 分类任务替换 `model.heads.head = nn.Linear(768, num_classes)`（torchvision 的 vit 分类头在 `model.heads` 里）→ 解冻最后几层 Block + head 一起小学习率微调（或全量小 lr），**只训 head 通常不够** → 训练配强增广（RandAugment/Mixup/CutMix，DeiT 证明是 ViT 小数据训练的**必需品**，补缺失的归纳偏置）。

**高分辨率推理要点**：ViT 的 `pos_embed` 按 224 训练位置学习，推理更大图必须**插值 pos_embed**——把 (N+1,D) 中后 N 个 patch 位置 reshape 成 2D 网格做 bicubic 插值到 (N'+1,D)，**class token 那一维单独保留不插值**；patch 必须与训练一致（仍是 16×16），插值后最好短程微调。更大分辨率 = 更多 token = O(N²) 更贵 → 后续 Swin 用**窗口局部注意力 + 层次化下采样**解决；ViT 系另发展出 Hybrid（CNN stem + ViT）缓解小数据弱势。

**易错点/坑 与 正确做法**：

| ❌ 常见错误 | ✅ 正确做法 |
|---|---|
| 拿小数据集从零训 ViT 并期望超过 ResNet | 数据 <1M 量级优先 CNN 或 Hybrid；必须 ViT 就用 ImageNet-21k 预训练微调，或 DeiT 蒸馏方案 |
| 换了更大的输入分辨率却**不插值**位置编码 | 先插值 pos_embed（保持 patch size），并用与训练一致的 Normalize/预处理；插值后最好短程微调 |
| 位置编码插值时把 class token 那 1 维一起 reshape | 先拆出 `pos[:, 0]`（class），对剩余 (1,N,D) 按 (H,W) reshape 后 `F.interpolate` 再拼回 |
| 混淆"分类取 CLS token 还是 GAP" | torchvision ViT 默认取 class token（forward 用 `x[:, 0]`）；自己实现时二选一并在微调时保持一致 |
| patch 尺寸改了却复用原 pos_embed 形状 | patch size / 分辨率任一变化都会改变 token 数 N，pos_embed 形状必须同步 |
| 直接用 nn.TransformerEncoder 忘了它的默认结构差异 | 手工搭 Block 更直观：LN 在子层**前**（Pre-LN，与 ViT 一致）；`nn.MultiheadAttention` 的 batch_first 注意 |
| 微调时把整个 backbone 冻结只训 head | 常用做法：先解冻最后几层 block + head 微调（或全量小 lr）；只训 head 通常不够 |
| 误以为 ViT 不需要 CNN 式强增广 | 恰恰相反：DeiT 证明强增广(RandAugment/Mixup/CutMix)是 ViT 小数据训练的**必需品** |

```python
# ViT-B/16 案例详解：加载预训练 → 前向推理 → 换头微调 → 高分辨率 pos_embed 插值
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

# 1) 加载 ImageNet-1k 预训练权重（torchvision >= 0.12）
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
model.eval()

# 2) 必须用与训练一致的预处理管线（归一化均值/方差不一致会明显掉点）
preprocess = weights.transforms()
# x = preprocess(PIL_image)          # -> (C,224,224)，再 unsqueeze(0) 成 (1,3,224,224)

# 3) 前向：torchvision 内部取 class token 输出（等价 x[:, 0]）再过分类头
with torch.no_grad():
    logits = model(torch.randn(1, 3, 224, 224))      # (1,1000)
pred = logits.argmax(dim=-1)                         # 类别索引

# 4) 微调换任务：torchvision 的 ViT 分类头在 model.heads 里，ViT-B/16 的 D=768
num_classes = 10
model.heads.head = nn.Linear(768, num_classes)       # 替换分类头
# 常见做法：解冻最后几层 Block + head 用小学习率微调，而不是整体冻结只训 head

# 5) 更高分辨率（如 448x448）推理：位置编码必须插值，class token 那一维单独保留
def interpolate_pos_embed(pos_embed, new_h, new_w):
    # pos_embed 形状 (1, N+1, D)：第 0 个是 class token 位置，其余 N = H*W 个 patch 位置
    cls_token = pos_embed[:, :1]                          # (1,1,D) 保留，不参与插值
    grid = pos_embed[:, 1:]                               # (1,N,D)
    h = w = int(grid.shape[1] ** 0.5)                     # 训练网格：224/16 -> 14x14
    grid = grid.reshape(1, h, w, -1).permute(0, 3, 1, 2)  # (1,D,14,14)
    grid = F.interpolate(grid, size=(new_h, new_w), mode="bicubic")  # (1,D,28,28)
    grid = grid.flatten(2).transpose(1, 2)                # (1,784,D)
    return torch.cat([cls_token, grid], dim=1)            # (1,785,D)

# 448x448、patch=16 -> 28x28=784 个 patch token + 1 个 class token
model.pos_embed = nn.Parameter(interpolate_pos_embed(model.pos_embed, 28, 28))
# 注意：token 数 784 时自注意力 O(N^2) 明显变贵；插值后最好在目标任务上短程微调
```

---
## 关联

- 前置：[[Image-Processing/卷积与池化直觉-note]]（patch embedding 数学等价于 kernel=stride=patch_size 的卷积，需要卷积直觉打底）
- 类似：[[CNN-Architectures/CNN经典架构]]（区别是 CNN 以局部性 + 权值共享 + 平移等变为归纳偏置、数据高效但长程依赖弱；ViT 只保留"patch 化"假设、第一层即可全图交互、靠规模 scaling，小数据反而劣势）
- 进阶：[[Image-Processing/图像增广方法-note]]（DeiT 证明强增广补的是 ViT 缺失的归纳偏置，是小数据训练的必需品）
- 进阶：[[Object-Detection/R-CNN两阶段系列-note]]（ViT 当骨干用于检测/分割：DETR 类直接用 transformer，两阶段检测骨干仍常用 CNN）
- 进阶：[[Object-Detection/NMS与mAP评测-note]]（骨干换成 ViT 属骨干替换，评测指标与其无关）

---
## 对比选型

| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（ViT） | 图像 = patch 序列 + 标准 Transformer Encoder，最小化归纳偏置，全局自注意力随数据/算力规模 scaling | 大数据（ImageNet-21k / JFT 级预训练）下的图像分类与通用骨干；需要一步到位的长程交互 |
| CNN（ResNet 等） | 局部感受野 + 权值共享 + 平移等变，卷积堆叠出层级特征 | 中小数据集（<1M）、资源受限、追求数据效率与平移等变的任务 |
| Hybrid（CNN stem + ViT） | 卷积 stem 先提取局部特征/降采样，再交给 ViT 做全局建模 | 数据不够大、又想用 Transformer 全局建模时的折中方案 |
| Swin Transformer | 窗口局部自注意力 + 层次化下采样（移位窗口实现跨窗通信），复杂度近似随分辨率线性 | 高分辨率输入与密集预测（检测/分割），缓解 ViT 的 O(N²) 开销 |

---
## 参考

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（ViT 原论文，ICLR 2021）](https://arxiv.org/abs/2010.11929)
- [Training data-efficient image transformers & distillation through attention（DeiT）](https://arxiv.org/abs/2012.12877)
- [torchvision.models.vit_b_16 官方文档](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)

---
## 具体案例

- [[ViT视觉Transformer 实战示例]](ViT视觉Transformer_sample.py)
