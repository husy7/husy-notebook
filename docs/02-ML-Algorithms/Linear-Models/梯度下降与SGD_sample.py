# -*- coding: utf-8 -*-
"""
梯度下降与 SGD —— 案例代码（线性模型细分）
==========================================
覆盖：
  1. 手写 BGD / SGD / Mini-batch 三种梯度下降解线性回归
  2. 观察学习率过大发散 / 过小收敛极慢
  3. 观察特征量纲对收敛的影响（标准化重要性）
  4. 与 sklearn SGDRegressor 对照
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无显示器环境也可保存图
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression

# 生成回归数据：50 样本，噪声适中
X, y = make_regression(n_samples=50, n_features=1, noise=15, random_state=0)

def add_bias(X):
    """给特征矩阵左侧加一列全1(当截距项)，使 b 也算进权重向量。"""
    return np.c_[np.ones(len(X)), X]

def loss_mse(w, X, y):
    pred = X @ w
    return np.mean((pred - y) ** 2)     # 注意用 mean，统一步长尺度

# =====================================================================
# 一、BGD：每次用全部样本算真实梯度
# =====================================================================
def gradient_descent(X, y, lr=0.01, epochs=2000, batch_size=None):
    """
    通用梯度下降批版本。
    batch_size=None → BGD(全量)  ；=1 → SGD ；>1 → mini-batch(演进版)
    返回学习过程中的损失序列。
    """
    Xb = add_bias(X)                    # (n,2)：[:,0]=bias, [:,1]=斜率权重
    w  = np.zeros(Xb.shape[1])
    losses = []
    n = len(Xb)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(idx)      # 每轮打散顺序
        if batch_size is None or batch_size >= n:
            batch_idx = idx
        else:
            batch_idx = idx[:batch_size]
        Xb_m, y_m = Xb[batch_idx], y[batch_idx]
        grad = (2.0 / len(batch_idx)) * Xb_m.T @ (Xb_m @ w - y_m)   # 2/n·X^T(Xw−y)
        error = (Xb_m @ w - y_m)
        losses.append(np.mean(error ** 2))
        w -= lr * grad
    return w, losses

# 全量 BGD：应收敛，解接近解析解 w≈[截距, 斜率]
w_bgd, loss_bgd = gradient_descent(X, y, lr=0.02, epochs=300)
print("[BGD] 学到 [bias, w] =", np.round(w_bgd, 3),
      " | 最终 loss = %.2f" % loss_bgd[-1])

# =====================================================================
# 二、SGD：batch_size=1（噪声大，loss 曲线抖但不影响趋势）
# =====================================================================
w_sgd, loss_sgd = gradient_descent(X, y, lr=0.02, epochs=300, batch_size=1)
print("[SGD] 学到 [bias, w] =", np.round(w_sgd, 3),
      " | 末段 loss(跳动) ≈ %.2f" % np.mean(loss_sgd[-30:]))

# =====================================================================
# 三、学习率影响对比
# =====================================================================
def run_lr_rate(lr):
    w, ls = gradient_descent(X, y, lr=lr, epochs=300)
    return ls[-1] if np.isfinite(ls[-1]) else float("inf")

# lr 太小 → loss 降得极慢；lr 适中 → 收敛；lr 过大 → 发散(NaN)
for lr in [0.001, 0.02, 10.0]:
    final = run_lr_rate(lr)
    tag = "→ 发散!" if np.isinf(final) else ("→ 收得很慢" if lr <= 0.001 else "→ 正常")
    print(f"[学习率 lr={lr:6.3f}] 末段loss={final:12.3f} {tag}")

# =====================================================================
# 四、特征量纲：量纲差异大会拖慢收敛
# =====================================================================
X_scaled = StandardScaler().fit_transform(X)      # 标准化(均值0、方差1)
w_plain,  ls_plain  = gradient_descent(X,        y, lr=0.02, epochs=300)
w_scaled, ls_scaled = gradient_descent(X_scaled, y, lr=0.02, epochs=300)
print("\n[量纲] 未标准化末段loss=%.2f, 标准化后=%.2f" %
      (ls_plain[-1], ls_scaled[-1]))

# =====================================================================
# 五、与 sklearn SGDRegressor 对照（核对数量级一致）
# =====================================================================
reg = SGDRegressor(max_iter=500, tol=1e-3, eta0=0.02,
                   learning_rate="constant", random_state=0)
reg.fit(X, y)
print("\n[sklearn SGDRegressor] intercept=%.3f coef=%.3f" %
      (reg.intercept_[0], reg.coef_[0]))
print("[手写 BGD 对照]        [bias, w] ≈", np.round(w_bgd, 3))

# =====================================================================
# 小结
# =====================================================================
# BGD稳、SGD快但抖；学率太小慢、太大发；特征量纲差异大必须标准化；
# 深度学习实践中常用 Mini-batch + 自适应优化器(Adam)。
