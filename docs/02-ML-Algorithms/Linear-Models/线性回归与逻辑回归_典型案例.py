# -*- coding: utf-8 -*-
"""
线性回归与逻辑回归 —— 典型代码演示
==================================
覆盖知识点：
  1. 线性回归（Linear Regression）模型、MSE 损失、梯度下降
  2. 逻辑回归（Logistic Regression）模型、交叉熵损失、概率预测
  3. 正则化：Ridge(L2) / Lasso(L1) / ElasticNet 对比
  4. 关键工程细节：特征标准化、类别处理

依赖：pip install scikit-learn numpy
"""

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Ridge, Lasso, ElasticNet,
)
from sklearn.metrics import mean_squared_error, accuracy_score

# =====================================================================
# 一、线性回归：用 sklearn 一行训练
# =====================================================================
# 载入糖尿病数据集（回归任务：预测病情量化指标）
diab_X, diab_y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    diab_X, diab_y, test_size=0.2, random_state=0,
)

# 创建并训练线性回归模型
lin = LinearRegression()
lin.fit(X_tr, y_tr)                 # 拟合学习参数：系数 coef 与截距 intercept

# 在测试集上评估：均方误差越小越好
y_pred = lin.predict(X_te)
print("[线性回归] 测试集 MSE =", round(mean_squared_error(y_te, y_pred), 3))
print("[线性回归] 特征系数", np.round(lin.coef_, 3))

# =====================================================================
# 二、手动实现梯度下降式的线性回归（理解原理）
# =====================================================================
# 只取前 1 个特征做可视化简化，演示参数更新过程
X1 = diab_X[:, 0].reshape(-1, 1)
X1 = np.c_[np.ones(len(X1)), X1]    # 在左边加一列全 1，把截距并进权重向量
theta = np.zeros(X1.shape[1])       # 初始化权重 [b, w]

def mse_gradient(theta, X, y):
    """计算 MSE 对权重的梯度。"""
    n = len(y)
    pred = X @ theta                # 线性预测：X·θ
    grad = (2 / n) * X.T @ (pred - y)   # 2/n · X^T(ŷ - y)
    return grad

lr = 0.01
for _ in range(5000):               # 梯度下降迭代
    theta -= lr * mse_gradient(theta, X1, diab_y)
print("[手动梯度下降] 学到的 [偏置, 权重] =", np.round(theta, 3))

# =====================================================================
# 三、正则化对比：Ridge(L2) vs Lasso(L1) vs ElasticNet
# =====================================================================
# 构造一个"高维稀疏真值"的数据：只有少数特征有贡献
rng = np.random.RandomState(0)
n, p = 200, 20
X_reg = rng.randn(n, p)
true_w = np.zeros(p); true_w[[0, 3, 8]] = [3.0, -2.0, 1.5]   # 只有 3 个特征有作用
y_reg = X_reg @ true_w + 0.1 * rng.randn(n)

# L2 正则化（Ridge）：系数整体缩小但都不为 0
ridge = Ridge(alpha=1.0).fit(X_reg, y_reg)
# L1 正则化（Lasso）：自动把无关特征系数压到 0，实现特征选择
lasso = Lasso(alpha=0.1).fit(X_reg, y_reg)
# ElasticNet：L1+L2 混合
en = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_reg, y_reg)

print("\n[正则化对比] 真值非零权重:", true_w.nonzero()[0])
print("[正则化对比] Ridge 非零系数数:", int(np.sum(np.abs(ridge.coef_) > 1e-6)))
print("[正则化对比] Lasso 非零系数数:", int(np.sum(np.abs(lasso.coef_) > 1e-6)),
      "（L1 会把无关特征压到 0）")
print("[正则化对比] ElasticNet 非零系数数:", int(np.sum(np.abs(en.coef_) > 1e-6)))

# =====================================================================
# 四、逻辑回归：用交叉熵做二分类 + 概率输出
# =====================================================================
iris_X, iris_y = load_iris(return_X_y=True)
# 构造二分类：只保留前两类（setosa / versicolor）
mask = iris_y < 2
Xb = iris_X[mask]
yb = iris_y[mask]

# 关键：逻辑回归对特征量纲敏感，先标准化再训练
scaler = StandardScaler()
Xb_scaled = scaler.fit_transform(Xb)

clf = LogisticRegression(max_iter=1000)   # 增加迭代次数以免未收敛告警
clf.fit(Xb_scaled, yb)

# 预测类别（硬判断）与概率（软输出）
pred = clf.predict(Xb_scaled)
proba = clf.predict_proba(Xb_scaled)      # 每行 = [P(类0), P(类1)]
print("\n[逻辑回归] 训练集准确率 =", round(accuracy_score(yb, pred), 3))
print("[逻辑回归] 第一个样本的类别概率 =", np.round(proba[0], 3))

# 手动复现 sigmoid 函数，理解概率来源
def sigmoid(z):
    """Sigmoid 激活：把线性得分 z 映射到 (0,1) 表示概率。"""
    return 1 / (1 + np.exp(-z))

# 用学到的权重手动算逻辑回归的预测概率，验证与 sklearn 一致
z = Xb_scaled @ clf.coef_.T + clf.intercept_   # 线性得分
manual_prob = sigmoid(z.ravel())
print("[逻辑回归] 手动 sigmoid 与 sklearn 概率误差 =",
      round(np.abs(manual_prob - proba[:, 1]).max(), 8))
