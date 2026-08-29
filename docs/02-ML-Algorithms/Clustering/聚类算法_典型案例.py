# -*- coding: utf-8 -*-
"""
聚类算法 —— 典型代码演示
=========================
覆盖知识点：
  1. 数据生成：人为构造"球形 / 非球形"数据，观察算法差异
  2. K-Means：原理复现 + sklearn 使用，K 的选择（肘部法 / 轮廓系数）
  3. DBSCAN：基于密度，能处理任意形状簇并识别噪声
  4. 评估：轮廓系数（无监督内部指标）

依赖：pip install scikit-learn numpy matplotlib
"""

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 固定随机种子保证结果可复现
rng = np.random.RandomState(0)

# =====================================================================
# 一、制造两类数据
# =====================================================================
# 1) 球形簇（K-Means 擅长）
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4,
                              cluster_std=0.6, random_state=0)
# 2) 月牙形（环形/非凸，K-Means 会失败，DBSCAN 擅长）
X_moons, y_moons = make_moons(n_samples=300, noise=0.06, random_state=0)

# =====================================================================
# 二、K-Means 实战
# =====================================================================
km = KMeans(n_clusters=4, n_init=10, random_state=0)   # n_init 多次初始化取最优
labels_km = km.fit_predict(X_blobs)                    # 训练并直接得到簇标签
print("[K-Means] 质心:\n", np.round(km.cluster_centers_, 2))
print("[K-Means] 惯性(簇内平方和,越小越紧)=", round(km.inertia_, 1))

# 用真实标签验证聚类质量（adjusted_rand_index 越接近 1 越好；仅演示用）
# 注意：无监督场景通常没有真实标签，这里用 sklearn 内部构造的数据演示
print("[K-Means] ARI(与真实标签一致性) =",
      round(adjusted_rand_score(y_blobs, labels_km), 3))


def manual_kmeans(X, k, max_iter=100, seed=0):
    """手写一个极简 K-Means，帮助理解算法迭代过程。"""
    rng = np.random.RandomState(seed)
    # 1) 随机选 k 个样本作为初始质心
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        # 2) 将每个样本分给最近的质心（欧氏距离）
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        assign = dist.argmin(axis=1)                    # 每个样本的簇编号
        # 3) 用簇内均值更新质心
        new_centers = np.array(
            [X[assign == j].mean(axis=0) for j in range(k)])
        if np.allclose(new_centers, centers):           # 质心不再变化则停止
            break
        centers = new_centers
    return assign, centers

labels_manual, centers_manual = manual_kmeans(X_blobs, 4)
print("[手动K-Means] 与 sklearn 划分一致性 ARI =",
      round(adjusted_rand_score(labels_km, labels_manual), 3))


def pick_k_by_elbow(X, max_k=8):
    """肘部法：画出惯性随 K 的下降曲线，找明显拐点。"""
    inertias = []
    for k in range(1, max_k + 1):
        km_temp = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km_temp.inertia_)
    # 返回惯性列表（真实作图时拐点即"肘部"，通常选 3~5 之间骤降后变缓处）
    return inertias

print("[肘部法] 各 K 的惯性值（找拐点）:",
      [round(v, 1) for v in pick_k_by_elbow(X_blobs)])

# 泊线系数评估 K 的选择（越大越好）
print("[轮廓系数] K=4 时 =",
      round(silhouette_score(X_blobs, labels_km), 3))

# =====================================================================
# 三、在"月牙形"数据上：K-Means 失败 → DBSCAN 成功
# =====================================================================
# K-Means 假设球形簇，把两个月牙强行切成若干"球"，结果错乱
km_moon = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X_moons)
print("\n[月牙数据] K-Means 与真实标签 ARI =",
      round(adjusted_rand_score(y_moons, km_moon), 3), "(往往很低，说明切错)")

# 先标准化（DBSCAN 依赖距离，量纲敏感）
X_moons_s = StandardScaler().fit_transform(X_moons)
db = DBSCAN(eps=0.3, min_samples=5)          # 半径 0.3，最少 5 个邻居为簇
labels_db = db.fit_predict(X_moons_s)
# 标签 -1 表示噪声点
print("[月牙数据] DBSCAN: 发现簇数 =", len(set(labels_db)) - (1 if -1 in labels_db else 0),
      " 噪声点 =", int((labels_db == -1).sum()))
print("[月牙数据] DBSCAN 与真实标签 ARI =",
      round(adjusted_rand_score(y_moons, labels_db), 3), "(接近1 → 切分正确)")

# =====================================================================
# 小结与选择建议
# =====================================================================
# K-Means：快、需指定 K、仅球形簇，必做标准化。
# DBSCAN：不需 K、任意形状、自带噪声识别，但 eps/min_samples 需调参，
#         且各簇密度差异大时不稳定。
# 无标签场景评估用内部指标（轮廓系数），有标签才用 ARI/AMI。
