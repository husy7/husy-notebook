# -*- coding: utf-8 -*-
"""回归指标案例
覆盖要点：
1. MSE / RMSE / MAE / 中位绝对误差 的数值差异与单位解释；
2. 异常值敏感性：加入几个极端离群点后 MSE 暴涨而 MAE 稳（RMSE≫MAE 即诊断）；
3. R² 语义：=1 / =0 / <0（比猜均值还差）三种情形实测；
4. 鲁棒回归对比：OLS(L2) vs Huber(L1/L2 折中) 在污染数据上的表现；
5. MAPE 的除零风险提示（y 接近 0 时爆炸）。

运行：python 回归指标_sample.py （依赖 numpy, scikit-learn；绘图可选）
"""
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             median_absolute_error, r2_score)
from sklearn.dummy import DummyRegressor


def metrics_block(y_true, y_pred, title):
    """打印一组回归指标"""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))))
    print(f"\n--- {title} ---")
    print(f"  MSE   = {mean_squared_error(y_true, y_pred):10.4f}  (平方误差均值, 离群值被放大)")
    print(f"  RMSE  = {rmse:10.4f}  (单位与 y 一致, 可业务直读)")
    print(f"  MAE   = {mae:10.4f}  (绝对误差均值, 对离群鲁棒)")
    print(f"  MedAE = {medae:10.4f}  (中位绝对误差, 最鲁棒)")
    print(f"  R²    = {r2:10.4f}  (1-残差平方和/总平方和)")
    print(f"  MAPE  = {mape:10.4%}  (相对误差, y->0 会爆炸, 慎用)")
    return rmse, mae


def demo_outlier_sensitivity():
    """同一模型 + 少量离群点：MSE 暴涨、MAE 变化小"""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    y = 1.5 * x + rng.normal(0, 1.0, len(x))
    X = x.reshape(-1, 1)
    y_clean = y.copy()
    # 往测试集里塞 5 个离群点（把 5 个 y 改成 +40 偏移）
    y_dirty = y.copy()
    y_dirty[::40] += 40.0

    print("=" * 76)
    print("异常值敏感性：同样的线性拟合，只污染测试集的 5/200 个点")
    print("=" * 76)
    m = LinearRegression().fit(X, y_clean)   # 干净数据上训练
    rmse_c, mae_c = metrics_block(y_clean, m.predict(X), "干净测试集")
    rmse_d, mae_d = metrics_block(y_dirty, m.predict(X), "含 5 个离群点的测试集")
    print(f"\n-> RMSE: {rmse_c:.3f} -> {rmse_d:.3f} (暴涨 {rmse_d/rmse_c:.1f}x) | "
          f"MAE: {mae_c:.3f} -> {mae_d:.3f} (温和)")
    print("-> RMSE 远大于 MAE 本身就是'存在大离群误差'的诊断信号")


def demo_r2_meaning():
    """R² 的三种情形：完美 / 等于猜均值 / 比猜均值还差"""
    rng = np.random.default_rng(1)
    n = 300
    x = np.linspace(0, 10, n)
    y = 2.0 * x + rng.normal(0, 2.0, n)
    X = x.reshape(-1, 1)

    print("\n" + "=" * 76)
    print("R² 语义实测：好模型 / 猜均值 / 故意更差的模型")
    print("=" * 76)
    good = LinearRegression().fit(X, y).predict(X)
    mean_model = DummyRegressor(strategy="mean").fit(X, y).predict(X)
    bad = good + 5.0 * np.ones(n) + 3.0 * x  # 系统性偏差 -> 比猜均值更差

    for name, pred in [("线性拟合(好)", good), ("常数猜均值", mean_model),
                       ("人为变差模型", bad)]:
        r2 = r2_score(y, pred)
        print(f"{name:<12} R² = {r2:>8.4f}"
              + {1.0: "", 0.0: "  <- R²=0 等价于猜均值",
                 0.8: "  <- 解释了 80% 的方差"}.get(round(r2, 1), "")
              + ("  <- R²<0：比'猜均值'还差, 通常是离群/模型方向错了" if r2 < 0 else ""))


def demo_robust_regression():
    """训练数据被离群点污染时：OLS(L2) vs Huber"""
    rng = np.random.default_rng(2)
    n = 300
    x = np.linspace(0, 10, n)
    y = 1.0 * x + rng.normal(0, 0.8, n)
    # 注入 8% 的粗大误差（不同机制生成的大偏差）
    idx = rng.choice(n, int(n * 0.08), replace=False)
    y[idx] += rng.normal(0, 20, len(idx))
    X = x.reshape(-1, 1)

    print("\n" + "=" * 76)
    print("训练集含 8% 粗大离群点：OLS vs Huber（评估在干净测试集上）")
    print("=" * 76)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    models = {
        "OLS(L2)": LinearRegression(),
        "Huber(鲁棒)": HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000),
    }
    for name, m in models.items():
        m.fit(Xtr, ytr)
        metrics_block(yte, m.predict(Xte), f"{name} 在干净测试集上")
    print("\n-> 平方损失被离群点拉偏斜率; Huber 对大残差降权, 拟合更接近真实规律")
    print("   若离群是'数据错误'应优先清洗, 若是'长尾真实分布'才适合鲁棒损失")


def demo_mape_trap():
    """MAPE 的除零/爆炸风险：y 里混入接近 0 的值"""
    y_true = np.array([100.0, 50.0, 0.5, 0.02, 200.0])   # 有一个接近 0
    y_pred = np.array([105.0, 52.0, 0.6, 0.10, 210.0])
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12)))
    print("\n" + "=" * 76)
    print("MAPE 陷阱：y 接近 0 时相对误差爆炸")
    print("=" * 76)
    print(f"真实值 {y_true.tolist()}")
    print(f"预测值 {y_pred.tolist()}  ->  单个样本相对误差 "
          f"{np.abs((y_true[3]-y_pred[3])/y_true[3]):.0%}")
    print(f"整体 MAPE = {mape:.1%}  （被 0.02 那个样本主导, 失真）")
    print("-> 换成 SMAPE / WAPE（按总量加权）或对数空间评估更稳")


if __name__ == "__main__":
    demo_outlier_sensitivity()
    demo_r2_meaning()
    demo_robust_regression()
    demo_mape_trap()

    # 可选：残差图（诊断用）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 200)
        y = 1.5 * x + rng.normal(0, 1.0, len(x))
        y[::40] += 40.0
        m = HuberRegressor().fit(x.reshape(-1, 1), y)
        resid = y - m.predict(x.reshape(-1, 1))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(x, y, s=10)
        axes[0].plot(x, m.predict(x.reshape(-1, 1)), "r-", label="Huber 拟合")
        axes[0].set_title("数据与拟合（可见离群点）")
        axes[0].legend()
        axes[1].scatter(m.predict(x.reshape(-1, 1)), resid, s=10)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_title("残差图（应无结构；漏斗形=异方差）")
        plt.tight_layout()
        plt.savefig("regression_metrics.png", dpi=110)
        print("\n[绘图] 已保存 regression_metrics.png")
    except Exception as exc:
        print(f"\n[绘图] 跳过（需要 matplotlib）：{type(exc).__name__}")
