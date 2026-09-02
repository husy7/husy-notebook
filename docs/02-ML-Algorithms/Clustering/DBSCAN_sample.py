# -*- coding: utf-8 -*-
"""DBSCAN 案例
覆盖要点：
1. 非凸簇（two moons）：K-Means 失败、DBSCAN 成功（任意形状）；
2. 带噪声数据：DBSCAN 自动把孤立点标成 -1（免费异常检测）；
3. eps / min_samples 两个参数的作用 + 参数过大过小的后果；
4. 用 k-距离图辅助选 eps（拐点法）；
5. 密度差异大时单个全局 eps 的局限（提示 HDBSCAN）。

运行：python DBSCAN_sample.py （依赖 numpy, scikit-learn；绘图可选）
"""
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def demo_nonconvex():
    """非凸簇：K-Means vs DBSCAN"""
    X, y_true = make_moons(n_samples=400, noise=0.06, random_state=0)
    X = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    db = DBSCAN(eps=0.25, min_samples=8).fit(X)  # 参数需与数据尺度匹配

    print("=" * 72)
    print("two moons（非凸簇）: K-Means vs DBSCAN")
    print("=" * 72)
    print(f"K-Means  ARI = {adjusted_rand_score(y_true, km.labels_):.3f}"
          f"  <- 球形假设，切不出弧形")
    print(f"DBSCAN   ARI = {adjusted_rand_score(y_true, db.labels_):.3f}"
          f"  <- 密度连通，恢复弧形结构")
    print(f"DBSCAN 找到簇数 = {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)}"
          f"，噪声点数 = {(db.labels_ == -1).sum()}")


def demo_noise():
    """在 blobs 周围撒噪声点：DBSCAN 把噪声标成 -1"""
    rng = np.random.default_rng(42)
    X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.6,
                           random_state=0)
    noise = rng.uniform(-8, 8, size=(60, 2))
    Xn = np.vstack([X, noise])
    db = DBSCAN(eps=0.5, min_samples=10).fit(Xn)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = int((db.labels_ == -1).sum())
    print("\n" + "=" * 72)
    print("带孤立噪声点的数据：DBSCAN 自动识别噪声")
    print("=" * 72)
    print(f"识别出簇数 = {n_clusters}，噪声点 = {n_noise}（真实噪声 60）")
    print(f"-> 噪声点与真实噪声重合度 ≈ "
          f"{np.isin(np.where(db.labels_ == -1)[0], np.arange(400, 460)).mean():.0%}")


def demo_eps_sweep():
    """扫 eps：观察簇数与噪声变化，理解参数语义"""
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
    X = StandardScaler().fit_transform(X)
    print("\n" + "=" * 72)
    print("扫 eps（min_samples=8 固定）")
    print("=" * 72)
    print(f"{'eps':>6} {'簇数':>5} {'噪声数':>6}   解读")
    for eps in [0.05, 0.15, 0.3, 0.6, 1.5]:
        db = DBSCAN(eps=eps, min_samples=8).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        note = {0.05: "eps 太小：几乎全成噪声",
                0.15: "恢复 2 簇但噪声偏多",
                0.3: "理想区间：2 簇、几乎无噪声",
                0.6: "eps 偏大：开始合并/扩张",
                1.5: "eps 太大：全并成一簇"}.get(eps, "")
        print(f"{eps:>6.2f} {n_clusters:>5} {n_noise:>6}   {note}")


def demo_k_distance():
    """k-距离图：辅助选 eps（升序排列的第 k 近邻距离，拐点即 eps）"""
    X, _ = make_moons(n_samples=400, noise=0.06, random_state=0)
    X = StandardScaler().fit_transform(X)
    k = 8  # 与 min_samples 对应
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nn.kneighbors(X)
    k_dist = np.sort(dists[:, -1])  # 每个点的第 k 近邻距离，升序
    print("\n" + "=" * 72)
    print(f"k-距离图辅助选 eps（k = min_samples = {k}）")
    print("=" * 72)
    # 打印分位数：拐点大致在距离"突然陡升"的位置
    for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f"  分位 {q:>5.0%} -> 第 {k} 近邻距离 ≈ {np.quantile(k_dist, q):.4f}")
    print("-> 拐点出现在曲线陡升处（这里约 0.15~0.3），选其左侧作 eps 较稳")


def demo_density_gap():
    """密度差异大的两簇：一个全局 eps 顾此失彼 -> 提示 HDBSCAN"""
    rng = np.random.default_rng(1)
    dense = rng.normal(0, 0.2, (600, 2))            # 很密的簇
    sparse = rng.normal([6, 6], 1.6, (200, 2))      # 很疏的簇
    X = np.vstack([dense, sparse])
    print("\n" + "=" * 72)
    print("密度差异大的数据（左密右疏）：全局 eps 的困境")
    print("=" * 72)
    for eps in [0.3, 1.0, 3.0]:
        db = DBSCAN(eps=eps, min_samples=8).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"eps = {eps:.1f}: 簇数 = {n_clusters}, 噪声 = {n_noise}"
              + ("  <- 保住稀疏簇但密集簇被撕碎" if eps < 0.6
                 else "  <- 保住密集簇但稀疏簇丢了" if eps < 2 else "  <- 全糊成一簇"))
    print("-> 单一 eps 无法同时适配两种密度：此时应上 HDBSCAN / OPTICS")


if __name__ == "__main__":
    demo_nonconvex()
    demo_noise()
    demo_eps_sweep()
    demo_k_distance()
    demo_density_gap()

    # 可选：画两幅对比图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y_true = make_moons(n_samples=400, noise=0.06, random_state=0)
        Xs = StandardScaler().fit_transform(X)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xs)
        db = DBSCAN(eps=0.25, min_samples=8).fit(Xs)
        axes[0].scatter(Xs[:, 0], Xs[:, 1], c=y_true, s=8, cmap="viridis")
        axes[0].set_title("真实标签")
        axes[1].scatter(Xs[:, 0], Xs[:, 1], c=km.labels_, s=8, cmap="viridis")
        axes[1].set_title("K-Means（切错）")
        axes[2].scatter(Xs[:, 0], Xs[:, 1], c=db.labels_, s=8, cmap="viridis")
        axes[2].set_title("DBSCAN（恢复弧形；紫色=-1 噪声）")
        plt.tight_layout()
        plt.savefig("dbscan_moons.png", dpi=110)
        print("\n[绘图] 已保存 dbscan_moons.png")
    except Exception as exc:
        print(f"\n[绘图] 跳过（需要 matplotlib）：{type(exc).__name__}")
