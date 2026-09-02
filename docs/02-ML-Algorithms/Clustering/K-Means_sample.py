# -*- coding: utf-8 -*-
"""K-Means 案例
覆盖要点：
1. make_blobs 球形簇上跑通 K-Means：分配->更新 迭代的本质；
2. 选 K：肘部法(SSE 曲线) vs 轮廓系数(候选 K 逐个算)；
3. K-Means++ vs 纯随机初始化：多次重启取最优惯性的必要性；
4. 特征缩放的影响：量纲不同时欧氏距离被大尺度特征主导；
5. 球形假设的边界：对细长/环形簇 K-Means 会切错（预告 DBSCAN）。

运行：python K-Means_sample.py （依赖 numpy, scikit-learn；绘图可选）
"""
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def demo_basic():
    """标准球形簇：跑通算法 + 展示中心收敛"""
    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.2,
                           random_state=42)
    km = KMeans(n_clusters=4, n_init=10, random_state=0)
    km.fit(X)
    print("=" * 72)
    print("球形簇 K-Means（K=4 已知，与真标签对比）")
    print("=" * 72)
    # 用大多数投票式的简化对比：直接看各簇尺寸与惯性
    print(f"惯性(SSE) = {km.inertia_:.2f}   迭代轮数 = {km.n_iter_}")
    print(f"簇中心(前 3 维):\n{km.cluster_centers_[:3]}")
    # 预测量化质量：因为标签编号不对齐，用 v-measure/ARI 更公平
    from sklearn.metrics import adjusted_rand_score
    print(f"ARI(与真标签) = {adjusted_rand_score(y_true, km.labels_):.3f}"
          f"  （1 = 完全一致）")


def demo_choose_k():
    """选 K：肘部法 + 轮廓系数"""
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.4,
                      random_state=42)
    print("\n" + "=" * 72)
    print("选 K：肘部法（SSE）与轮廓系数（越高越好）")
    print("=" * 72)
    print(f"{'K':>3} {'SSE 惯性':>14} {'轮廓系数':>10}")
    inertias, silhs = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, km.labels_)
        silhs.append(sil)
        print(f"{k:>3} {km.inertia_:>14.1f} {sil:>10.4f}")
    # 肘部：惯性下降斜率从 k=4 后明显变缓；轮廓：k=4 最高
    best_k = 2 + int(np.argmax(silhs))
    print(f"\n-> 轮廓系数建议 K = {best_k}（真实为 4，符合预期）")
    print("   肘部法看上面 SSE 列的'拐点'（主观，需结合业务解释）")


def demo_init_and_scale():
    """初始化的影响 + 特征缩放的影响"""
    # (a) K-Means++ vs 随机初始化（坏初始可能困在局部最优）
    X, _ = make_blobs(n_samples=400, centers=5, cluster_std=2.5,
                      random_state=7)
    print("\n" + "=" * 72)
    print("初始化策略对比（同一数据重复 20 次的惯性分布）")
    print("=" * 72)
    inertias_rand = []
    for seed in range(20):
        km = KMeans(n_clusters=5, init="random", n_init=1, random_state=seed)
        km.fit(X)
        inertias_rand.append(km.inertia_)
    km_pp = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=0)
    km_pp.fit(X)
    print(f"随机初始化 x1   : 惯性 min={min(inertias_rand):.0f}  "
          f"max={max(inertias_rand):.0f}  (波动=坏局部最优)")
    print(f"KMeans++ x10重启 : 惯性 = {km_pp.inertia_:.0f}（接近 min -> 更稳）")

    # (b) 缩放的影响：构造量纲差 1000 倍的两特征
    print("\n" + "=" * 72)
    print("特征缩放的影响（x1 量纲 ~0-1，x2 量纲 ~0-1000）")
    print("=" * 72)
    Xa = np.random.default_rng(3).uniform(0, 1, (300, 1))
    Xb = np.random.default_rng(3).uniform(0, 1000, (300, 1))
    # 造两簇：一个只靠 x2 区分，一个 x1/x2 都要
    Xraw = np.hstack([Xa, Xb + (np.arange(300) < 150) * 300])
    for name, data in [("原始特征", Xraw),
                       ("标准化后", StandardScaler().fit_transform(Xraw))]:
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data)
        sil = silhouette_score(data, km.labels_)
        print(f"{name:<8} 轮廓系数 = {sil:.4f}"
              f"  （{'不缩放会被 x2 主导，切分失真' if name == '原始特征' else '缩放后距离更合理'})")


def demo_shape_limit():
    """球形假设边界：K-Means 对两半月(非凸)束手无策 -> 预告 DBSCAN"""
    X, y_true = make_moons(n_samples=300, noise=0.06, random_state=0)
    print("\n" + "=" * 72)
    print("非凸簇（two moons）：K-Means 的球形假设失败")
    print("=" * 72)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    from sklearn.metrics import adjusted_rand_score
    print(f"K-Means 在 moons 上的 ARI = "
          f"{adjusted_rand_score(y_true, km.labels_):.3f}  （接近 0 = 基本切错）")
    print("-> 两个半月是连通的弧形，K-Means 只能竖切一刀；")
    print("   DBSCAN/谱聚类按密度连通性聚类才能恢复弧形结构")


if __name__ == "__main__":
    demo_basic()
    demo_choose_k()
    demo_init_and_scale()
    demo_shape_limit()

    # 可选：画肘部曲线与聚类散点
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.4,
                          random_state=42)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        inertias = []
        ks = list(range(2, 11))
        for k in ks:
            inertias.append(KMeans(n_clusters=k, n_init=10,
                                   random_state=0).fit(X).inertia_)
        axes[0].plot(ks, inertias, "o-")
        axes[0].set_title("肘部法：SSE vs K（拐点在 4 附近）")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("SSE 惯性")
        km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
        axes[1].scatter(X[:, 0], X[:, 1], c=km.labels_, s=12, cmap="viridis")
        axes[1].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                        c="red", marker="x", s=120, label="centers")
        axes[1].set_title("K-Means 聚类结果")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig("kmeans_elbow.png", dpi=110)
        print("\n[绘图] 已保存 kmeans_elbow.png")
    except Exception as exc:
        print(f"\n[绘图] 跳过（需要 matplotlib）：{type(exc).__name__}")
