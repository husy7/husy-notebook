# -*- coding: utf-8 -*-
"""GBDT / XGBoost 案例
覆盖要点：
1. 手写"伪残差提升"：用 sklearn 的浅决策树逐轮拟合残差，看清 GBDT 机制；
2. 残差拟合 vs 直接拟合：浅树串行累加逼近非线性函数；
3. learning_rate 与 n_estimators 的配合 + 早停；
4. sklearn GradientBoosting 与 HistGradientBoosting 对比；
5. （可选）若装了 xgboost / lightgbm，加一轮三方对比。

运行：python GBDT与XGBoost_sample.py （核心依赖 numpy, scikit-learn）
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

rng = np.random.default_rng(0)


def target(x):
    """带噪声的非线性真值函数"""
    return np.sin(x) + 0.3 * x + 0.05 * rng.normal(size=len(x))


# ----------------------------------------------------------------------
# 1. 手写梯度提升：回归 MSE 的负梯度 = 残差 y - f(x)
#    每轮：拟合残差 -> 乘以 learning_rate 累加进模型
# ----------------------------------------------------------------------
def manual_gbdt(Xtr, ytr, Xte, yte, n_estimators=120, lr=0.1, max_depth=2):
    Xtr, Xte = Xtr.reshape(-1, 1), Xte.reshape(-1, 1)
    # 第 0 步：常数初始化 = 训练均值（MSE 下的最优常数）
    f = np.full(len(ytr), ytr.mean())
    trees = []
    print("\n" + "=" * 70)
    print("手写 GBDT（MSE 损失 -> 拟合残差），每 30 轮打印一次训练残差均方")
    print("=" * 70)
    for m in range(n_estimators):
        residual = ytr - f                    # 负梯度 = 残差
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        tree.fit(Xtr, residual)               # 用浅树拟合残差
        f = f + lr * tree.predict(Xtr)        # 收缩更新
        trees.append(tree)
        if (m + 1) % 30 == 0:
            print(f"round {m + 1:>4}: 训练残差 RMSE = "
                  f"{np.sqrt(np.mean((ytr - f) ** 2)):.4f}")
    # 预测：把训练时对每棵树的累加搬到测试集上
    f_pred = np.full(len(yte), ytr.mean())
    for tree in trees:
        f_pred = f_pred + lr * tree.predict(Xte)
    return f_pred


def demo_manual():
    X = np.linspace(-4, 4, 400)
    y = target(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    pred = manual_gbdt(Xtr, ytr, Xte, yte)
    print(f"手写 GBDT  测试 RMSE = {np.sqrt(mean_squared_error(yte, pred)):.4f}")
    # 对比：单棵深树 vs 单棵浅树
    deep = DecisionTreeRegressor(max_depth=None, random_state=0).fit(
        Xtr.reshape(-1, 1), ytr)
    shallow = DecisionTreeRegressor(max_depth=2, random_state=0).fit(
        Xtr.reshape(-1, 1), ytr)
    print(f"单棵深树(不剪枝)  测试 RMSE = "
          f"{np.sqrt(mean_squared_error(yte, deep.predict(Xte.reshape(-1,1)))):.4f}"
          f"  （过拟合，测试反而差）")
    print(f"单棵浅树(depth=2)  测试 RMSE = "
          f"{np.sqrt(mean_squared_error(yte, shallow.predict(Xte.reshape(-1,1)))):.4f}"
          f"  （欠拟合：弱学习器要串行累加才够强）")


# ----------------------------------------------------------------------
# 2. sklearn 封装版：learning_rate 与 n_estimators 配合、早停
# ----------------------------------------------------------------------
def demo_sklearn_gbdt():
    X = np.linspace(-4, 4, 500).reshape(-1, 1)
    y = target(np.linspace(-4, 4, 500))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    print("\n" + "=" * 70)
    print("learning_rate vs n_estimators（lr 小 -> 需要更多树，且更稳）")
    print("=" * 70)
    for lr in [1.0, 0.1, 0.02]:
        gb = GradientBoostingRegressor(
            n_estimators=400, learning_rate=lr, max_depth=2,
            random_state=0).fit(Xtr, ytr)
        tr = mean_squared_error(ytr, gb.predict(Xtr))
        te = mean_squared_error(yte, gb.predict(Xte))
        print(f"lr = {lr:<5}  n_estimators = {gb.n_estimators_:>4}"
              f"  train MSE = {tr:.4f}   test MSE = {te:.4f}")

    # 早停：大 n_estimators + 验证集自动截断
    gb_es = GradientBoostingRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=2,
        validation_fraction=0.2, n_iter_no_change=20, random_state=0)
    gb_es.fit(Xtr, ytr)
    print(f"\n早停生效：实际用了 {gb_es.n_estimators_:>4} 棵树（2000 上限内自动截断）")
    print(f"          test MSE = {mean_squared_error(yte, gb_es.predict(Xte)):.4f}")


# ----------------------------------------------------------------------
# 3. RF vs GBDT vs HistGBDT：噪声数据上的表现
# ----------------------------------------------------------------------
def demo_compare():
    X = np.linspace(-4, 4, 800).reshape(-1, 1)
    y = target(np.linspace(-4, 4, 800))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=0,
                                              n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=2, random_state=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=7, random_state=0),
    }
    print("\n" + "=" * 70)
    print("RF vs GBDT vs HistGBDT（带噪正弦回归，测试 MAE）")
    print("=" * 70)
    for name, m in models.items():
        m.fit(Xtr, ytr.ravel())
        print(f"{name:<20} test MAE = "
              f"{mean_absolute_error(yte, m.predict(Xte)):.4f}")


# ----------------------------------------------------------------------
# 4.（可选）xgboost / lightgbm 安装时做一轮快速对比
# ----------------------------------------------------------------------
def demo_external_if_available():
    X = np.linspace(-4, 4, 800).reshape(-1, 1)
    y = target(np.linspace(-4, 4, 800))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    print("\n" + "=" * 70)
    print("可选三方库对比（未安装则跳过）")
    print("=" * 70)
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                             max_depth=2, random_state=0,
                             early_stopping_rounds=30)
        m.fit(Xtr, ytr.ravel(), eval_set=[(Xte, yte.ravel())], verbose=False)
        print(f"XGBoost (已安装)      test MAE = "
              f"{mean_absolute_error(yte, m.predict(Xte)):.4f}")
    except ImportError:
        print("xgboost 未安装，跳过")
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              max_depth=2, random_state=0)
        m.fit(Xtr, ytr.ravel(), eval_set=[(Xte, yte.ravel())],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        print(f"LightGBM (已安装)     test MAE = "
              f"{mean_absolute_error(yte, m.predict(Xte)):.4f}")
    except ImportError:
        print("lightgbm 未安装，跳过")


if __name__ == "__main__":
    demo_manual()
    demo_sklearn_gbdt()
    demo_compare()
    demo_external_if_available()
