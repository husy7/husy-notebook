---
title: "DBSCAN：基于密度、任意形状、自带噪声"
tags: ["Clustering", "DBSCAN", "密度聚类", "eps", "min_samples", "HDBSCAN"]
date: 2026-08-29
---

# DBSCAN：基于密度、任意形状、自带噪声

## 定义

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法。它不预设簇数量 K、不假设簇为球形，而是把"稠密相连"的点圈成一个簇：**核心点**（半径 eps 内至少 min_samples 个邻居）向外扩张密度可达（density-reachable）的点，经核心点链式合并成密度相连（density-connected）的连通区域；既不是核心点、又不在任何核心点邻域内的点"密度够不着"任何簇，被标成**噪声（label = −1）**。

它解决的核心问题是：K-Means 等中心型算法无法处理非凸、细长、弧形、嵌套等任意形状的簇，也无法处理簇内带噪声的数据。DBSCAN 改用**局部密度**定义簇——"簇 = 密度足够高的连通区域，簇与簇之间被低密度区隔开"——从而不给定 K 也能还原任意形状的簇，并顺带完成异常检测。

核心特征：① 自动定簇数（不需要给定 K）；② 支持任意形状（球形假设被彻底抛弃）；③ 自带噪声点输出（免费的离群检测）；④ 确定性算法（给定参数与访问顺序一致，结果稳定）；⑤ 只依赖 eps 与 min_samples 两个参数。

适用范畴：簇形状不规则、数据含噪声点、簇数量未知的场景，例如地理空间点聚集、客户分群、图像像素区域划分、点云分割等；不适合簇间密度差异大或高维密度失效的情形（应换 HDBSCAN/OPTICS，或先降维）。

## 原理

**为什么用"密度"而不是"距离中心"**：K-Means 用"到中心的距离"隐含球形簇假设；DBSCAN 改用**局部密度**定义簇——簇是密度足够高的连通区域，簇与簇之间被低密度区隔开。于是细长、弧形、嵌套的簇都能被还原，不再受球形假设束缚。

**一个 eps + min_samples 共同定义"稠密"**：两点是否"密度可达"由 eps（邻域半径）与 min_samples（邻域内最少点数，即密度阈值）共同决定。`min_samples` 还间接充当**最小簇规模**：低于它的点不构成核心，聚不成簇。邻域查询可用 KDTree/BallTree 加速，sklearn 的 `algorithm='auto'` 会自动选择。

**三类点**：核心点（邻域点数 ≥ min_samples）、边界点（自身邻域不足，但落在某核心点邻域内）、噪声点。簇与簇之间用"核心点可达性"定义而非任意点相连，因此比单链聚类更抗链式粘连；簇的边界（核心/边界判定）是噪声判定上唯一模糊处。

**核心流程**：
1. 对每个点计算 eps 邻域内的邻居数（KDTree/BallTree 加速）；
2. 标记核心点（≥ min_samples）；任取一个未访问的核心点，把与它**密度相连**（经核心点链式可达）的所有点全部并入同一簇；
3. 重复直到没有未访问核心点；其余点标为噪声（−1）。

**参数选择机制**：`min_samples` 经验取 `≥ 2×dim`（dim 为特征维数）；`eps` 看 **k-距离图**：把每个点第 k 近邻（k = min_samples）的距离升序排列，取曲线"拐点/膝盖"处作为 eps。

## 应用

典型使用场景：不规则形状的空间数据聚类（地理坐标、雷达点云）；簇数量未知且含大量噪声点的探索性分析；把噪声输出（−1）当离群检测用（与孤立森林、GMM 低密度判异常同思路）；作为"先按区域聚拢、再细处理"的上游步骤。

快速上手步骤：
1. **先标准化**（如 StandardScaler），否则距离由量纲大的特征主导，eps 邻域含义失真；
2. 设 `min_samples ≥ 2×dim`（保守值取 2×dim）；
3. 画 **k-距离图**找拐点定 eps，再按业务微调；
4. 调用 `DBSCAN(eps=..., min_samples=...).fit_predict(X)`，解读各簇标签与 −1 噪声。

常见坑：
- ❌ **簇间密度差异大**（一簇很密、一簇很疏）→ 一个全局 eps 顾此失彼：小了稀疏簇全成噪声，大了密集簇被并掉。✅ 用 **HDBSCAN**（把 eps 变成自适应：按层次密度合并）、**OPTICS**，或先分域再各跑 DBSCAN。
- ❌ 随便拍 eps → 太小全是噪声、太大全并成一坨。✅ 先画 k-距离图找拐点，再按业务调。
- ❌ `min_samples` 设 1~2 → 边界/噪声判定退化，噪声会大量混进簇。
- ❌ 忘记缩放 → 距离由量纲大的特征主导，"eps 邻域"含义失真。✅ 先标准化再选 eps。
- ❌ 高维数据直接跑 → 高维空间所有距离都趋同，密度概念失效；先降维或换模型。
- ❌ 以为参数边界稳定：DBSCAN 是确定性的，但 eps 参数边界附近的点归属可能对数据小扰动敏感。

```python
# -*- coding: utf-8 -*-
"""
DBSCAN 实战示例：任意形状簇 + 噪声点 + eps 的 k-距离图选参法
案例：两个交错的半月（非凸/任意形状，K-Means 会失败），噪声 5%
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# 1. 造数据：300 个点、两个交错的半月形簇（任意形状，球形假设失效）
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# 2. 关键坑：先标准化！否则 eps 邻域被量纲大的特征主导，邻域含义失真
X = StandardScaler().fit_transform(X)

# 3. 选参
#    - min_samples 经验值 >= 2 * dim（dim=2 → 取 4~5 即可）
#    - eps 用 k-距离图选：k = min_samples，取曲线拐点/膝盖处
min_samples = 5
nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
dist, _ = nn.kneighbors(X)
k_dist = np.sort(dist[:, -1])            # 每个点的第 k 近邻距离，升序排列

plt.plot(k_dist)
plt.xlabel("points sorted by k-distance")
plt.ylabel(f"{min_samples}-th neighbor distance")
plt.title("k-distance plot: pick eps at the knee")
plt.show()
# 从图读拐点：本例约 eps = 0.18；拐点不明显说明密度不均 → 换 HDBSCAN/OPTICS

# 4. 建模：核心点（eps 内 >= min_samples）向外扩张密度可达的点，
#    够不着的点标 -1（噪声）——免费异常检测
model = DBSCAN(eps=0.18, min_samples=min_samples)
labels = model.fit_predict(X)

# 5. 结果解读：-1 是噪声，其余是各簇标签；簇数自动得出，无需给定 K
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())
print(f"自动发现的簇数: {n_clusters}, 噪声点数: {n_noise}")

# 可视化：两个半月被正确分开，少量噪声点显示为深色
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20)
plt.title("DBSCAN result")
plt.show()

# 易错点回顾：
# - eps 调小 → 全是噪声；调大 → 全并成一坨 → 先 k-距离图，再按业务微调
# - 各簇密度差异大时全局 eps 顾此失彼 → 换 HDBSCAN / OPTICS
```

---
## 关联
- 前置：[[K-Means]]（先理解"到中心的距离 + 球形假设"的局限，才能体会密度聚类动机）
- 类似：[[K-Means]]（区别是球形簇 vs 任意形状、需给定 K vs 自动定簇、无噪声概念 vs 天然输出 −1 噪声标签）
- 类似：[[谱聚类]]（区别是谱聚类也能抓非凸簇，但需先建相似度图、要指定 K、对参数更敏感）
- 类似：[[孤立森林]] / GMM（区别是它们专做异常检测，而 DBSCAN 的 −1 噪声只是聚类副产品，但"低密度 = 异常"的思路一致）
- 进阶：[[HDBSCAN]] / [[OPTICS]]（区别是解决"密度不均时 eps 选不动"：HDBSCAN 按层次密度合并把 eps 自适应化，OPTICS 沿密度轴扫描）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| DBSCAN（本文方案） | 局部密度：核心点经密度可达向外扩张成簇，够不着标 −1 噪声 | 簇任意形状、含噪声点、K 未知、各簇密度较均匀 |
| K-Means | 到中心距离最小化，隐含球形簇假设 | 簇近似球形、需给定 K、大规模数据追求速度 |
| HDBSCAN / OPTICS | 自适应 eps：按层次密度合并 / 沿密度轴连续扫描 | 簇间密度差异大、全局 eps 选不动、想少调参 |

---
## 参考
- [scikit-learn 用户指南：Clustering（DBSCAN 章节）](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [sklearn.cluster.DBSCAN API 文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

---
## 具体案例
- [[DBSCAN 实战示例]](DBSCAN_sample.py)
