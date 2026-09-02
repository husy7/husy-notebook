# -*- coding: utf-8 -*-
"""层次聚类与 GMM 案例
覆盖要点：
1. 凝聚式层次聚类：不同 linkage（single/complete/ward）对链式/离群数据的差异；
2. 树状图 dendrogram：横切高度 = 选 K（依赖 scipy，可选）；
3. GMM 软分配：predict_proba 输出责任度（软聚类证据）；
4. covariance_type 的影响（spherical/diag/tied/full）：表达力 vs 参数量；
5. GMM 是密度模型：score_samples 低分 = 异常检测；
6. BIC/AIC 辅助选分量数 K。

运行：python 层次聚类与GMM_sample.py （依赖 numpy, scikit-learn；树状图需要 scipy）
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def demo_linkage():
    """不同 linkage 对'细长链 + 球形簇'数据的差异"""
    rng = np.random.default_rng(0)
    chain = np.linspace([0, 0], [6, 6], 200) + rng.normal(0, 0.15, (200, 2))
    blob = rng.normal([6, 0], 0.5, (200, 2))
    X = np.vstack([chain, blob])
    y_true = np.r_[np.zeros(200, int), np.ones(200, int)]

    print("=" * 72)
    print("凝聚式层次聚类：linkage 对'斜链 + 球簇'的影响（ARI，1=完美）")
    print("=" * 72)
    for link in ["ward", "average", "complete", "single"]:
        agg = AgglomerativeClustering(n_clusters=2, linkage=link).fit(X)
        print(f"linkage = {link:<9} ARI = {adjusted_rand_score(y_true, agg.labels_):.3f}")
    print("-> single 会被链式结构串扰；ward/complete 倾向紧实簇，此数据更合适")


def demo_dendrogram(X, y_true, title="dendrogram"):
    """画树状图（可选，需 scipy）：横切高度决定簇数"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage

        # 小样本才画得动
        idx = np.random.default_rng(0).choice(len(X), 60, replace=False)
        Z = linkage(X[idx], method="ward")
        fig, ax = plt.subplots(figsize=(9, 4))
        dendrogram(Z, ax=ax, no_labels=True)
        ax.set_title(f"{title}：横切高度越高簇越少")
        plt.tight_layout()
        plt.savefig("dendrogram.png", dpi=110)
        print("\n[绘图] 已保存 dendrogram.png（横切某高度即得到对应簇数）")
    except Exception as exc:
        print(f"\n[绘图/树状图] 跳过（需要 scipy + matplotlib）：{type(exc).__name__}")


def demo_gmm_soft():
    """GMM 软分配：重叠的两簇，输出责任度而非硬标签"""
    X, _ = make_blobs(n_samples=500, centers=2, cluster_std=2.0,
                      random_state=3)  # cluster_std 大 -> 两簇严重重叠
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          n_init=5, random_state=0).fit(X)
    hard = gmm.predict(X)
    prob = gmm.predict_proba(X)  # 责任度矩阵 gamma

    print("\n" + "=" * 72)
    print("GMM 软分配（两簇重叠）：predict 给硬标签，predict_proba 给责任度")
    print("=" * 72)
    uncertain = np.where(np.abs(prob[:, 0] - 0.5) < 0.1)[0]
    print(f"样本数 = {len(X)}，其中责任度接近 0.5（高度不确定）的样本 = {len(uncertain)}")
    print(f"示例（责任度）：样本 {uncertain[0]} -> 簇0: {prob[uncertain[0],0]:.3f}, "
          f"簇1: {prob[uncertain[0],1]:.3f}  <- 软聚类给出置信度，K-Means 给不了")
    # K-Means 对照：它也"硬切"了重叠区
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    print("K-Means 只能给硬标签；GMM 多出的软概率可用于下游(如阈值决策)")


def demo_covariance():
    """covariance_type 对非各向同性数据的影响 + BIC 选 K"""
    # 构造"拉长的椭圆"双簇：spherical 假设必然吃亏
    rng = np.random.default_rng(5)
    X1 = rng.multivariate_normal([0, 0], [[6.0, 0.0], [0.0, 0.3]], 300)
    X2 = rng.multivariate_normal([4, 4], [[0.4, 0.0], [0.0, 4.0]], 300)
    X = np.vstack([X1, X2])
    y_true = np.r_[np.zeros(300, int), np.ones(300, int)]

    print("\n" + "=" * 72)
    print("协方差类型对比（椭圆双簇；ARI 越高越好）")
    print("=" * 72)
    for cov in ["spherical", "diag", "tied", "full"]:
        gmm = GaussianMixture(n_components=2, covariance_type=cov,
                              n_init=5, random_state=0).fit(X)
        print(f"covariance_type = {cov:<10} ARI = "
              f"{adjusted_rand_score(y_true, gmm.predict(X)):.3f}")

    print("\nBIC/AIC 选分量数 K（越低越好）:")
    print(f"{'K':>3} {'BIC':>12} {'AIC':>12}")
    for k in range(1, 6):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=5, random_state=0).fit(X)
        print(f"{k:>3} {gmm.bic(X):>12.1f} {gmm.aic(X):>12.1f}")


def demo_anomaly():
    """GMM 当密度模型：score_samples 低分处判异常"""
    rng = np.random.default_rng(9)
    normal = rng.normal(0, 1.0, (800, 2))
    outliers = rng.uniform(-6, 6, size=(20, 2))  # 均匀撒的远点
    X = np.vstack([normal, outliers])
    gmm = GaussianMixture(n_components=2, n_init=5, random_state=0).fit(X)
    scores = gmm.score_samples(X)  # 对数似然密度
    thr = np.quantile(scores, 0.02)  # 取最低 2% 当异常
    flagged = np.where(scores < thr)[0]
    hits = np.sum(flagged >= 800)  # 命中真实异常(后 20 个)的个数
    print("\n" + "=" * 72)
    print("GMM 密度模型做异常检测（分数最低的 2% 标为异常）")
    print("=" * 72)
    print(f"标出 {len(flagged)} 个异常候选，其中 {hits} 个是真离群点（共 20 个）")


if __name__ == "__main__":
    demo_linkage()

    # 顺便给标准的球形数据画一张树状图（可选步骤）
    Xb, yb = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=0)
    demo_dendrogram(StandardScaler().fit_transform(Xb), yb)

    demo_gmm_soft()
    demo_covariance()
    demo_anomaly()
