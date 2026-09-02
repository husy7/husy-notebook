---
title: "K-Means：划分聚类与 K 的选择"
tags: ["Clustering", "K-Means", "肘部法", "轮廓系数", "EM"]
date: 2026-08-29
---

# K-Means：划分聚类与 K 的选择

## 定义

K-Means（K 均值）是一种**划分式（partitional）聚类算法**：给定样本集与预先指定的簇数 K，它把每个样本分配到 K 个簇中的一个，使得**簇内平方和（SSE / inertia）**——即每个样本到其所在簇中心的平方欧氏距离之和——最小。它是无监督学习里最基础、最常用的聚类方法，核心是"中心点（均值）代表簇"这一思想。

它解决的核心问题是：**在没有标签的情况下，如何把数据自动分成 K 个同质小组**，常用于客户分群、图像压缩（向量量化）、异常检测、以及作为其他算法的预处理/降维步骤。

核心特征有三个：① **硬分配**——每个点只属于一个簇（归属非 0 即 1），与 GMM 的软分配相对；② **簇中心 = 簇内所有点的均值**，且距离度量用欧氏距离（平方距离）；③ **K 必须事先给定**，且目标函数对 K 单调下降，因此"选多少个簇"需要靠肘部法、轮廓系数或业务语义外部决定。

适用范畴：适合簇形状近似**球形（各向同性）、大小相近、密度均匀**的数据，数据量大时依然高效（近似 O(N·K·I)，I 为迭代次数）；对细长簇、环形簇、密度不均、含离群点的数据会系统性失效，应改用 DBSCAN、GMM、谱聚类等。

## 原理

目标函数为联合优化簇归属与中心：`min Σ_i ||x_i − μ_k||²`，其中 μ_k 是簇 k 的中心。直接同时求最优归属与最优中心是 **NP 难**的；但**固定中心求归属**（每点归到最近中心）与**固定归属求中心**（中心取簇内均值）各自都有闭式解。因此算法采用**交替优化（坐标下降）**：每次迭代都保证目标函数不增，从而实用地收敛——尽管只保证收敛到**局部最优**。

它本质上是 **EM 算法的硬分配特例**：E 步（分配）把每个点归到距离最近的中心，等价于以 0/1 后验做硬分配；M 步把中心更新为簇内均值。这一目标函数的期望解释等价于"每个簇服从各向同性高斯分布"时的极大似然硬分配版本——这正是为什么它隐含"簇是球形、等方差"的假设。若换成别的距离度量，"均值"就不成立（如 L1 距离对应中位数），算法也随之变成 K-medoids 一类。

**为什么 K 必须事先给定**：SSE 随 K 增大单调不增（K=N 时每个点自成簇，惯性=0），所以"最小化 SSE"本身无法定出 K。选 K 的常见做法：肘部法看 SSE–K 曲线拐点（主观）、轮廓系数 silhouette（∈[−1,1]，越高表示簇越紧凑且分离越好，需对每个候选 K 各跑一遍聚类）、Gap statistic、以及业务可解释性兜底。

**标准流程**：
1. 用 K-Means++ 选 K 个初始中心（按距离加权概率采样，理论上有 O(log K) 的近似保证）；
2. E 步（硬分配）：每个点归到最近中心；
3. M 步：把每个中心更新为簇内所有点的均值；
4. 重复 2–3 直到中心不再移动（或惯性变化 < tol）；
5. 评价/选 K：肘部法、轮廓系数或业务语义。

由于目标非凸，**初始化质量决定结果**：初始化差会掉进坏局部最优，因此用 K-Means++ 起步，并用 `n_init` 多次随机重启、取惯性最低的那次做兜底。

## 应用

**典型使用场景**：客户/用户分群（RFM 等画像聚类）；图像与信号压缩（把聚类中心当码本，即向量量化，K-Means ≈ 码本学习）；文档/特征向量的粗聚类与检索加速；异常检测的基线（离群点离所有中心都远）；以及高维数据的可视化与下游建模前的探索性分段。

**快速上手步骤**：① 清洗数据并**先做特征缩放**（StandardScaler/MinMaxScaler），再谈距离；② 可视化（如 PCA/UMAP 降维）确认簇大致是球形、等方差、密度均匀的形态，否则换算法；③ 对候选 K 逐个跑聚类，用肘部法 + 轮廓系数综合选 K，再结合业务可解释性定稿；④ 显式设置 `n_init`（如 10，新版 sklearn 默认 auto）并比较各次惯性，避免坏局部最优；⑤ 用簇中心与簇内样本解读每个簇的业务含义。

**注意事项 / 常见坑**：
- ❌ 不缩放直接聚类 → 量纲大的特征主导欧氏距离，聚类结果被"单位"绑架。✅ 先 StandardScaler/MinMaxScaler。
- ❌ 数据不是"球形、等方差、密度相近"的簇（细长簇、环形、密度不均）→ K-Means 必然切错。✅ 先可视化，换 DBSCAN/GMM/谱聚类。
- ❌ 只用默认 `n_init=1` → 一次随机初始化可能撞上差局部最优。✅ 显式 `n_init=10`（新版默认 auto）并比较惯性。
- ❌ 用"惯性最低"选 K → 惯性永远随 K 降。✅ 用肘部/轮廓系数/Gap statistic；轮廓系数对小 K 有偏好，要结合数据量看。
- ❌ 数据有离群点 → 均值会被拉走，整个簇中心偏移。✅ 清洗或换 K-medoids / DBSCAN（天然给噪声标签）。
- ❌ 高维直接算欧氏距离 → 距离趋于均匀（维数灾难），先降维（PCA/UMAP 仅作可视化参考）。

```python
# K-Means 完整上手示例：缩放 → 选 K（肘部法 + 轮廓系数）→ 聚类 → 解读
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. 造数据：3 个球形簇（真 K=3），含轻微噪声
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.9, random_state=42)

# 2. 特征缩放：量纲统一，避免某列主导欧氏距离（重要，勿省）
X = StandardScaler().fit_transform(X)

# 3. 选 K：对候选 K 各跑一遍（固定 n_init），比较 惯性(肘部) 与 轮廓系数
K_range = range(2, 7)
inertias, sil_scores = [], []
for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))  # ∈[-1,1]，越大越好

# 肘部：惯性随 K 下降，找"拐点"（主观）；轮廓系数：取峰值对应 K（客观参考）
print("K=", list(K_range), "inertia=", np.round(inertias, 2))
print("K=", list(K_range), "silhouette=", np.round(sil_scores, 3))
best_k = K_range[int(np.argmax(sil_scores))]          # 本例应选出 best_k=3

# 4. 用选定的 K 正式聚类：k-means++ 初始化 + n_init=10 多次重启取最优
km = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42)
labels = km.fit_predict(X)

# 5. 结果解读：centers 是簇的业务代表"原型"，labels 给每个样本打上簇号
print("最终惯性(SSE):", round(km.inertia_, 3))
print("簇中心:\n", np.round(km.cluster_centers_, 3))
print("每簇样本数:", np.bincount(labels))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            marker="*", s=200, c="red", label="centers")
plt.legend()
plt.show()
```

**案例详解**：以上用 `make_blobs` 生成 3 个球形簇模拟真实分群任务。关键步骤的顺序有讲究——先缩放再算距离，否则聚类结果被量纲绑架；选 K 时不看"惯性最低"（它永远随 K 单调降，见原理），而是结合肘部拐点与轮廓系数峰值，本例 silhouette 在 K=3 取最大，恰好还原真 K；最终聚类显式给足 `n_init=10` 并固定 `random_state`，保证结果可复现且避开单次随机初始化的坏局部最优。

---
## 关联
- 前置：[[EM 算法]]（K-Means 是 EM 在 spherical、等方差、硬分配下的极限，先理解"交替优化/E步-M步"思想再看它）
- 类似：[[K-medoids]]（区别是中心取簇内实际样本点而非均值（PAM/L1 中位数），对离群点更鲁棒，但计算量更高、不适合大样本）
- 类似：[[DBSCAN]]（区别是它按密度连通定义簇、自动定簇数、可识别噪声，不假设球形——是 K-Means 失效场景的直接替代）
- 进阶：[[GMM]]（区别是软分配 + 每个簇可有椭圆协方差，能描述非球形簇，但参数更多、需防过拟合；K-Means 是其特例）
- 向量量化视角：K-Means ≈ 码本学习，聚类中心可直接当压缩码本

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（K-Means） | 硬分配 + 均值中心交替迭代，最小化 SSE；K 需先定 | 大规模、球形等方差、密度均匀的簇，要快、要好解释 |
| 替代方案（DBSCAN） | 密度可达定义簇，自动定簇数，标记噪声点 | 任意形状 / 密度不均 / 含大量离群点，K 未知 |
| 替代方案（GMM） | 软分配 EM，每簇一个高斯（可椭圆协方差） | 簇形状/方差不同、需要概率归属（软聚类） |
| 替代方案（K-medoids） | 用簇内实际样本点作代表中心 | 离群点多或使用非欧氏距离的场合 |

---
## 参考
- [scikit-learn: KMeans 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn: K-Means 用户指南（含 K-Means++ / Mini-Batch 细节）](https://scikit-learn.org/stable/modules/clustering.html#k-means)

---
## 具体案例
- [[K-Means 实战示例]](K-Means_sample.py)
