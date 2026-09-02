# -*- coding: utf-8 -*-
"""Ridge (L2) 与 Lasso (L1) 对比案例
覆盖要点：
1. 高维 + 特征相关 + 含噪声的合成回归数据上，OLS 过拟合而正则化更稳；
2. 对比系数形态：Ridge 全部收缩但不为零（稠密）vs Lasso 多数精确为 0（稀疏）；
3. RidgeCV / LassoCV / ElasticNetCV 自动选 alpha；
4. 特征选择视角：Lasso 找回真正相关的少数特征；
5. alpha 过大/过小对测试误差的影响（验证集选择）。

运行：python 正则化Ridge与Lasso_sample.py （依赖 numpy, scikit-learn；绘图为可选项）
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # ---------------------------------------------------------------
    # 1. 合成数据：300 个样本，80 个特征，只有 6 个真正有用，且特征彼此相关
    #    -> 高维稀疏真实模型，最适合展示 L1 稀疏与 L2 收缩的差别
    # ---------------------------------------------------------------
    X, y = make_regression(
        n_samples=300,
        n_features=80,
        n_informative=6,      # 真正相关的特征只有 6 个
        noise=25.0,           # 较大噪声
        random_state=42,
    )
    # 人为制造特征相关：把部分无关特征设成"有用特征 + 噪声"的线性组合
    # 这会让 OLS 的系数在相关组之间互相拉扯（不稳定）
    rng = np.random.default_rng(0)
    X[:, 20:50] += X[:, :6] @ rng.normal(0, 0.6, size=(6, 30)) + rng.normal(0, 0.1, (len(X), 30))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7
    )

    # 统一先标准化（惩罚项对量纲敏感！）——必须只 fit 在训练集上，防止泄漏
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    # ---------------------------------------------------------------
    # 2. 基线对比：OLS vs Ridge vs Lasso vs ElasticNet
    #    OLS 无惩罚：高维 + 相关特征下测试误差通常最大
    # ---------------------------------------------------------------
    models = {
        "OLS(无惩罚)": LinearRegression(),
        "Ridge(alpha=10)": Ridge(alpha=10.0),
        "Lasso(alpha=1)": Lasso(alpha=1.0, max_iter=100_000),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=100_000),
    }
    print("=" * 78)
    print("模型对比（在保留的测试集上评估）")
    print("=" * 78)
    results = {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        yp = model.predict(Xte)
        mse = mean_squared_error(y_test, yp)
        r2 = r2_score(y_test, yp)
        results[name] = model
        print(f"{name:<16}  test MSE = {mse:9.2f}   R2 = {r2:.4f}")

    # ---------------------------------------------------------------
    # 3. 系数形态对比：L2 全部收缩 vs L1 精确稀疏
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("系数形态（80 维特征上，|w| > 1e-6 视为非零）")
    print("=" * 78)
    true_support = set(range(6))  # 真实相关的 6 个特征下标
    for name, model in results.items():
        coef = np.asarray(model.coef_).ravel()
        nonzero = np.count_nonzero(np.abs(coef) > 1e-6)
        # Lasso/ElasticNet 找回了几个真实特征？
        if hasattr(model, "sparse_coef_"):  # Lasso 系
            found = len(true_support & set(np.where(np.abs(coef) > 1e-6)[0]))
        else:
            found = len(true_support & set(np.argsort(-np.abs(coef))[:6]))
        print(f"{name:<16}  非零系数 = {nonzero:3d}/80   (真实 6 个里命中 {found} 个)")

    # 单独强调 L1 与 L2 的本质差别：看前 8 个系数的具体数值
    print("\n前 8 个特征的系数数值：")
    print(f"{'特征':<6}{'Ridge':>14}{'Lasso':>14}{'ElasticNet':>14}")
    for j in range(8):
        def cv(name):
            return np.asarray(results[name].coef_).ravel()[j]
        print(f"x{j:<5}{cv('Ridge(alpha=10)'):>14.3f}{cv('Lasso(alpha=1)'):>14.3f}{cv('ElasticNet'):>14.3f}")

    # ---------------------------------------------------------------
    # 4. 用 CV 自动选 alpha（不用手动瞎试网格）
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("交叉验证自动选 alpha")
    print("=" * 78)
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(Xtr, y_train)
    lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=100_000,
                       random_state=0).fit(Xtr, y_train)
    enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, max_iter=100_000,
                           random_state=0).fit(Xtr, y_train)
    print(f"RidgeCV 选出的 alpha      = {ridge_cv.alpha_:.4f}  -> test MSE = "
          f"{mean_squared_error(y_test, ridge_cv.predict(Xte)):.2f}")
    print(f"LassoCV 选出的 alpha      = {lasso_cv.alpha_:.4f}  -> test MSE = "
          f"{mean_squared_error(y_test, lasso_cv.predict(Xte)):.2f}")
    print(f"ElasticNetCV 选 l1_ratio  = {enet_cv.l1_ratio_:.2f}, alpha = {enet_cv.alpha_:.4f}"
          f"  -> test MSE = {mean_squared_error(y_test, enet_cv.predict(Xte)):.2f}")

    # ---------------------------------------------------------------
    # 5. alpha 过小(≈OLS) / 过大(欠拟合) 的演示：Ridge
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Ridge 的 alpha 选择影响（验证集上观察 U 型曲线）")
    print("=" * 78)
    for a in [1e-6, 1e-2, 1.0, 1e2, 1e6]:
        m = Ridge(alpha=a).fit(Xtr, y_train)
        print(f"alpha = {a:>9.0e}   train MSE = {mean_squared_error(y_train, m.predict(Xtr)):9.2f}"
              f"   test MSE = {mean_squared_error(y_test, m.predict(Xte)):9.2f}")

    # ---------------------------------------------------------------
    # 6.（可选）画出 OLS/Ridge/Lasso 随 alpha 的系数路径
    # ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        alphas = np.logspace(-4, 1, 100)

        # Ridge 系数路径：所有系数平滑收缩，都不归零
        coefs_r = []
        for a in alphas:
            coefs_r.append(Ridge(alpha=a).fit(Xtr, y_train).coef_)
        axes[0].plot(alphas, np.array(coefs_r))
        axes[0].set_xscale("log")
        axes[0].set_title("Ridge 系数路径：收缩但不归零")
        axes[0].set_xlabel("alpha")

        # Lasso 系数路径：系数逐个精确跳到 0
        coefs_l = []
        for a in alphas:
            coefs_l.append(Lasso(alpha=a, max_iter=200_000).fit(Xtr, y_train).coef_)
        axes[1].plot(alphas, np.array(coefs_l))
        axes[1].set_xscale("log")
        axes[1].set_title("Lasso 系数路径：逐个精确变 0（稀疏）")
        axes[1].set_xlabel("alpha")

        plt.tight_layout()
        plt.savefig("ridge_lasso_coef_path.png", dpi=110)
        print("\n[绘图] 已保存 ridge_lasso_coef_path.png")
    except Exception as exc:  # 无 matplotlib 也能跑，只是少一张图
        print(f"\n[绘图] 跳过（需要 matplotlib）：{type(exc).__name__}")


if __name__ == "__main__":
    main()
