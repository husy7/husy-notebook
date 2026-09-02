# -*- coding: utf-8 -*-
"""
ViT 视觉 Transformer 演示（离线可运行）

Part A：手写 mini-ViT（~120 行），用随机权重完整走一遍 ViT 的数据流：
       patch embed(等价 Conv) -> class token -> 位置编码 -> L 个
       Pre-LN Transformer Block(MHSA+MLP+残差) -> CLS -> 分类头。
       完全本地、无下载，shape 全部打印，便于对照"为什么 token 数是 197"。

Part B：torchvision 官方 vit_b_16 接口速览（weights=None = 随机权重，
       不需要联网；想用 ImageNet 预训练可改为 weights=IMAGENET1K_V1，
       首次运行需下载 ~330MB，本脚本默认不下载）。

运行：python ViT视觉Transformer_sample.py   （仅依赖 torch）
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Patch Embedding：把 (B,3,H,W) -> (B, N, D)，N=(H/16)*(W/16)。

    实现用 kernel=stride=patch_size 的 Conv2d（等价于先切 patch 再线性投影），
    输出重排成序列时 spatial 顺序即 patch 的左上→右下自然顺序。
    """

    def __init__(self, in_ch=3, D=192, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, D, kernel_size=patch, stride=patch)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)                 # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Block(nn.Module):
    """Pre-LN Transformer Block（ViT 采用 LN 在子层前）。

    顺序: LN -> MHSA -> 残差 -> LN -> MLP -> 残差
    """

    def __init__(self, D, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, int(D * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(D * mlp_ratio), D),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MiniViT(nn.Module):
    """自洽 mini-ViT：patch_embed + [cls] + pos_embed + blocks + head"""

    def __init__(self, img_size=224, patch=16, D=192, depth=6, num_heads=8,
                 num_classes=10, in_ch=3):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, D, patch)
        n = (img_size // patch) ** 2          # 196 (224/16=14 → 14*14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, D))
        # 初始化为小值（论文做法），scale 按维度
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.Sequential(*[Block(D, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, num_classes)

    def forward(self, x, return_feats=False):
        B = x.shape[0]
        x = self.patch_embed(x)                       # (B,196,D)
        cls = self.cls_token.expand(B, -1, -1)        # (B,1,D)
        x = torch.cat([cls, x], dim=1)                # (B,197,D)
        x = x + self.pos_embed                        # 加位置编码
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        x = self.norm(x)
        cls_out = x[:, 0]                             # 取 [cls] token
        if return_feats:
            return self.head(cls_out), feats
        return self.head(cls_out)


def part_a_mini_vit():
    """手写 mini-ViT 走一遍全流程，观察各阶段 shape。"""
    print("=" * 62)
    print("Part A: 手写 Mini-ViT 数据流（随机权重，离线）")
    torch.manual_seed(0)
    model = MiniViT(img_size=224, patch=16, D=192, depth=3)
    x = torch.randn(2, 3, 224, 224)                  # (B,C,H,W)
    pe = model.patch_embed
    with torch.no_grad():
        t = pe(x)
        print(f"  输入图像        : {tuple(x.shape)}")
        print(f"  PatchEmbed 后   : {tuple(t.shape)}  "
              f"(196 = (224/16)^2 patch token)")
        # 模拟 class token 拼接 + pos
        B = x.shape[0]
        t2 = torch.cat([model.cls_token.expand(B, -1, -1), t], dim=1)
        print(f"  拼上 [cls] 后   : {tuple(t2.shape)}  (序列长 197)")
        print(f"  位置编码形状    : {tuple(model.pos_embed.shape)}  (逐 token 相加)")
        logits, feats = model(x, return_feats=True)
        print(f"  Block 输出不变  : {tuple(feats[-1].shape)}")
        print(f"  取 x[:,0] 后    : {tuple(feats[-1][:, 0].shape)}")
        print(f"  分类 logits     : {tuple(logits.shape)}")
    print(f"  模型总参数量     : {sum(p.numel() for p in model.parameters()):,}")
    # 训练一步（验证反向传播完整）
    loss = F.cross_entropy(model(x), torch.randint(0, 10, (2,)))
    loss.backward()
    print(f"  一次 backward OK, loss={loss.item():.4f}")


def part_b_torchvision_vit():
    """torchvision 官方 vit_b_16：随机权重版（weights=None，无下载）。"""
    print("=" * 62)
    print("Part B: torchvision vit_b_16 接口速览（weights=None → 随机权重）")
    try:
        import torchvision.models as tvm
    except ImportError as e:
        print("  未安装 torchvision，跳过 Part B。", e)
        return
    if not hasattr(tvm, "vit_b_16"):
        print("  torchvision 版本过旧(无 vit_b_16，需 >=0.12)，跳过 Part B。")
        return
    # 版本兼容写法：>=0.13 用 weights=None；更老版本用 pretrained=False
    try:
        model = tvm.vit_b_16(weights=None)
    except TypeError:
        model = tvm.vit_b_16(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"  vit_b_16(weights=None) 输出: {tuple(out.shape)}  (ImageNet 1000 类)")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("  注: 需 ImageNet 预训练时改用:\n"
          "      weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1\n"
          "      首次会联网下载权重(~330MB)，脚本默认不下载。")
    # 微调换任务示例（不执行，仅展示 API）
    # num_classes = 5
    # model.heads.head = nn.Linear(768, num_classes)


if __name__ == "__main__":
    part_a_mini_vit()
    part_b_torchvision_vit()
    print("\n全部演示完成 ✓（无任何网络下载）")
