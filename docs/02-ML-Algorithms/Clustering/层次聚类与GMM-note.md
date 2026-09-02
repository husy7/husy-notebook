---
title: "层次聚类（凝聚式）与 GMM（软聚类）"
tags: ["Clustering", "层次聚类", "GMM", "EM", "树状图", "软聚类"]
date: 2026-08-29
---

# 层次聚类（凝聚式）与 GMM（软聚类）

## 定义

本文知识点覆盖两套风格迥异、互为补充的聚类方案：**层次聚类（凝聚式，AGNES）** 与 **GMM（高斯混合模型 / 软聚类）**。

- **解决什么问题**：层次聚类解决"事先不知道聚几类、且希望看到类别之间的层级包含关系"的问题；GMM 解决"样本可能以不同概率同时属于多个簇、需要软标签置信度与密度模型"的问题。
- **层次聚类的核心特征**：从"每个样本各自一簇"出发，自底向上重复合并"距离最近的两簇"（贪心），合并历史天然构成一棵**树状图（dendrogram）**；想聚几类就在树状图上横切一刀得到对应 K 个簇——**无需预设 K**，还能观察子簇的嵌套关系。代表算法是凝聚式 AGNES（与之相对的是分裂式 DIANA）。
- **GMM 的核心特征**：假设数据由 K 个高斯分布按权重 π_k 混合生成，用 **EM** 迭代估计参数，输出每个样本属于各簇的**软概率（责任度 γ_ik）**；同时它本身就是一个**密度模型** p(x) = Σ_k π_k·N(x | μ_k, Σ_k)，可顺带做密度估计与异常检测。
- **适用范畴**：层次聚类适合样本量适中（约 n ≤ 1 万）、需要层级结构或 K 不确定的场景（如类目树、系统发育树）；GMM 适合簇呈椭球/不同大小方向、需要软分配或密度建模的场景（如图像分割、异常检测）。
- **共同边界**：层次聚类需要 O(n²) 距离矩阵内存、算法 O(n²~n³)，大数据跑不动；GMM 对多峰/重尾/非高斯数据表达不足。K-Means 可视为 GMM 在"球形协方差 + 等方差 + 硬分配"下的退化特例，因此本知识点是从硬聚类走向软聚类、从几何走向概率建模的桥梁。

## 原理

**层次聚类（凝聚式 AGNES）为什么这么设计**

- **自底向上合并**：每一步都是贪心——合并当前"距离最近的两簇"，合并顺序本身编码了样本间的层级包含关系，因此能输出整棵树的树状图；用户选择切割高度 = 选择 K，还能查看任意层的子簇构成。
- **关键设计点 linkage（簇间距离如何定义，直接决定结果形状）**：
  - `single` 单链（两簇最近点距离）：能抓细长/非凸簇，但**容易链式粘连**——一列噪声点就能把两簇串成一簇；
  - `complete` 全链（两簇最远点距离）：簇紧实、抗粘连，但对离群点敏感、结果偏向球形；
  - `average` 平均距离：上述两者的折中；
  - `ward`：使合并后总簇内 SSE（误差平方和）增量最小，本质与 K-Means 是同一目标 → 结果偏球形、最常用。
- **代价**：需先建 O(n²) 距离矩阵（内存），算法 O(n²~n³)，**大数据直接跑不动**。

**GMM + EM 为什么这么设计**

- **动机**：真实数据常是多个"子总体"的混合，每个子总体近似高斯。GMM 用协方差矩阵参数化簇的**中心、大小与方向**（`covariance_type` 提供 full / diag / tied / spherical 四档表达），比 K-Means"半径相同的圆"表达力强得多。
- **为什么用 EM 而不是直接极大似然**：每个样本归属于哪个高斯是**隐变量**，直接对 log-likelihood 求导没有闭式解。EM 交替迭代，且保证每次迭代似然不降、收敛到（局部）最优：
  - **E 步**：固定当前参数，按贝叶斯公式计算责任度（软分配）：
    γ_ik = P(簇k | x_i) = π_k · N(x_i | μ_k, Σ_k) / Σ_j π_j · N(x_i | μ_j, Σ_j)
  - **M 步**：以责任度 γ_ik 为权重，加权重估混合权重 π_k = Σ_i γ_ik / n、均值 μ_k 与协方差 Σ_k（加权均值 / 加权协方差）。
- **软输出与密度**：`predict_proba` 输出后验概率（可当置信度）；`score_samples` 输出 log p(x)，低密度处打分低 → 可直接做**密度估计 / 异常检测**。
- **退化关系**：若把各簇协方差固定为等方差球、并把软分配退化为硬分配（取 argmax），GMM 的迭代就退化为 K-Means。
- **收敛注意**：EM 与 K-Means 同病——对初值敏感、会撞局部最优，因此实践中用 KMeans 初始化 + `n_init` 多次重启。

## 应用

- **典型使用场景**：层次聚类——客户分群汇报层级（让业务人员直接看树状图选切分高度）、商品类目树构建、系统发育树；GMM——图像/信号软分割（像素或帧属于各分量的概率）、语音混合分离、欺诈/工业质检**异常检测**（`score_samples` 低密度打分）、以及任何需要"椭球簇 + 置信度"的聚类。
- **层次聚类快速上手**：1) 选 linkage（默认优先 `ward`，除非明确要找细长簇）；2) 计算并绘制树状图；3) 按业务可解释的高度横切选 K（可用肘部法/轮廓系数辅助）；4) 用子簇嵌套关系解释层级含义。
- **GMM 快速上手**：1) 用 **BIC/AIC** 扫描定 K 与分量数；2) 选协方差类型（默认 `full`，样本少时降级）；3) `init_params='kmeans'` 初始化 + `n_init` 多次重启；4) 收敛后用 `predict_proba` 取软标签（argmax 即硬标签）、用 `score_samples` 做密度估计/异常检测。
- **常见坑与对策**：
  - ❌ 层次聚类对大样本直接跑 → O(n²) 距离矩阵直接内存爆掉。✅ n > 1 万先采样 / mini-batch（如 Birch）或换 K-Means。
  - ❌ 用 `single` linkage 遇链式数据 → 两簇被噪声桥接成一簇。✅ 用 `ward`/`complete`，除非明确要找细长簇。
  - ❌ GMM 用 `covariance_type='full'` 但样本少 → 每个簇要估 O(d²) 个协方差参数，极易**奇异/不收敛**。✅ `reg_covar` 加抖动、或降级 `diag`/`tied`/`spherical`。
  - ❌ EM 不重启 → 撞局部最优（与 K-Means 同病）。✅ `n_init` + `init_params='kmeans'`。
  - ❌ 把 GMM 当纯聚类 → 忘了它是密度模型：`score_samples` 可直接做异常检测。
  - ❌ 数据多峰/重尾/非高斯时硬套 GMM → 表达不足，考虑 Dirichlet 过程混合（自动定 K）或换聚类。

```python
# -*- coding: utf-8 -*-
# 案例：小规模"客户行为"数据，分别用层次聚类与 GMM 聚类，
# 演示软分配（责任度）、硬标签与基于密度的异常检测雏形。
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# 1. 构造 300 个样本、2 个簇心的样例数据（真实场景换成你的特征矩阵 X 即可）
X, _ = make_blobs(n_samples=300, centers=2, cluster_std=1.2, random_state=42)
X = StandardScaler().fit_transform(X)   # 先归一化，避免某维度主导距离/方差

# ---------- 方案 A：层次聚类（凝聚式） ----------
# 注意：数据 >1 万时慎用，需先建 O(n²) 距离矩阵，内存/耗时都爆炸
hier = AgglomerativeClustering(n_clusters=2, linkage="ward")  # ward 偏球形、最常用
# linkage 取舍：single=最近点(细长簇但易链式粘连) / complete=最远点 / average=折中
labels_hier = hier.fit_predict(X)       # 等价于"树状图在高度 K=2 处横切一刀"

# ---------- 方案 B：GMM（软聚类）----------
gmm = GaussianMixture(
    n_components=2,          # 簇数 K：实践中用 BIC/AIC 扫描选择
    covariance_type="full",  # full=每簇独立协方差(椭球/方向)；样本少易奇异，可降 diag/tied/spherical
    reg_covar=1e-6,          # 给协方差对角加抖动，防奇异矩阵/不收敛
    n_init=10,               # 多次重启，规避 EM 撞局部最优（与 K-Means 同病）
    init_params="kmeans",    # KMeans 初始化，收敛更快更稳
    random_state=42,
)
gmm.fit(X)

# 核心输出 1：责任度矩阵（软分配）——每行是 P(样本属于各簇)，行和为 1，可当置信度
resp = gmm.predict_proba(X)             # shape (n_samples, K)
labels_gmm = gmm.predict(X)             # 硬标签 = 责任度取 argmax

# 核心输出 2：密度模型打分 p(x)=Σ π_k·N(x|μ_k,Σ_k)，
# score_samples 返回 log p(x)，分数最低的一批样本即异常检测候选（低密度区）
log_density = gmm.score_samples(X)
anomaly_idx = np.argsort(log_density)[:5]   # 取对数密度最低的 5 个样本
print("层次聚类标签:", np.unique(labels_hier, return_counts=True))
print("GMM 责任度样例:", resp[:3].round(3))
print("GMM 硬标签样例:", labels_gmm[:10])
print("疑似异常样本索引:", anomaly_idx)
```

案例详解：先造 300 个样本、2 个簇心（带随机噪声）的数据模拟"客户行为特征"，`StandardScaler` 归一化后——方案 A 用 `AgglomerativeClustering(linkage='ward', n_clusters=2)` 一次得到硬标签，等价于"树状图横切一刀"；方案 B 用 `GaussianMixture` 拟合，`predict_proba` 得到每行和为 1 的责任度矩阵（软标签/置信度），`predict` 取 argmax 得到硬标签，`score_samples` 得到每个样本的对数密度，挑出分数最低的若干样本即为异常检测雏形。参数注释已覆盖全部易错点：大样本勿直接跑层次聚类、`ward` 抗链式粘连、`reg_covar` 防奇异、`n_init` + kmeans 初始化防局部最优。

---
## 关联
- 前置：[[EM 算法]]（GMM 的求解框架，同样用于缺失值填充、隐马尔可夫模型）；层次聚类的基础是[[距离度量与相似度]]
- 类似：[[K-Means 聚类]]（区别是 K-Means 是 GMM"球形+等方差+硬分配"的退化特例，输出硬标签、无置信度与密度估计能力；而 GMM 输出软概率责任度且兼作密度模型）
- 进阶：[[Dirichlet 过程混合模型]]（贝叶斯 GMM，数据自适应决定 K，免去 BIC/AIC 扫描）；[[DBSCAN 密度聚类]]（任意形状簇 + 自动标记噪声，无需指定 K）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：层次聚类（凝聚式 AGNES） | 自底向上贪心合并最近两簇，输出树状图，按高度切分选 K | 中小数据（n ≲ 1 万）、K 不确定、需要层级/嵌套结构（类目树、系统发育树） |
| 本文方案：GMM（软聚类，EM 求解） | 混合高斯建模 + 责任度软分配，兼作密度模型 | 簇为椭球/大小方向各异、需要软标签置信度、密度估计或异常检测 |
| 替代方案：K-Means | 等方差球形簇 + 硬分配，最小化簇内 SSE | 大数据量、簇近似球形等大小、只求快速硬聚类 |
| 替代方案：DBSCAN | 基于密度可达划分簇并标记噪声点 | 任意形状簇、含噪声/离群点、无需预设 K |
| 替代方案：Dirichlet 过程混合 | 贝叶斯 GMM，分量数由数据自适应推断 | 分布多峰/重尾、K 完全未知且希望自动决定 |

---
## 参考
- [scikit-learn 官方文档：聚类算法总览（含层次聚类）](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn 官方文档：GaussianMixture 混合模型](https://scikit-learn.org/stable/modules/mixture.html)
- [SciPy 官方文档：scipy.cluster.hierarchy（linkage 与 dendrogram）](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)

---
## 具体案例
- [[层次聚类与GMM 实战示例]](层次聚类与GMM_sample.py)
