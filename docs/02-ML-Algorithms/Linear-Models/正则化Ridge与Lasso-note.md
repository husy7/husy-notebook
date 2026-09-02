---
title: "正则化：Ridge (L2) 与 Lasso (L1)"
tags: [Linear Models, Ridge, Lasso, 正则化, ElasticNet]
date: 2026-08-29
---

# 正则化：Ridge (L2) 与 Lasso (L1)

## 定义

正则化（Regularization）是在线性回归等模型的损失函数上追加一项对模型系数（权重）大小的惩罚，是应对**过拟合**最经典的手段之一。过拟合的典型症状是**系数过大、方差高**——模型为了硬拟合训练集里的噪声把权重推得很大，导致换一批数据预测就剧烈波动；正则化的思路是用"一点偏差"换"大幅降方差"，让系数整体变小、模型更平滑、泛化更稳。

- **Ridge（岭回归，L2 正则）**：惩罚项为系数平方和 λ·Σw_j²，把所有系数**等比收缩但不归零**，输出稠密的小系数，适合保留全部特征、应对多重共线性的场景。
- **Lasso（套索回归，L1 正则）**：惩罚项为系数绝对值之和 λ·Σ|w_j|，能把不重要的系数**精确压成 0**，得到稀疏解，天然具备**特征选择**能力，模型可解释性更强。
- 二者统一在"损失 + 惩罚"框架下，区别只在惩罚项选 L1 范数还是 L2 范数；惩罚强度 λ 在 sklearn 中称为 `alpha`。适用范畴覆盖线性回归、逻辑回归、SVM、神经网络（weight decay）等几乎所有带参数的监督模型，尤其适合特征数量多、特征相关性强或 n < p（样本数少于特征数）的数据。

## 原理

动机来源：最小二乘（OLS）解在特征相关性强或 n < p 时**系数爆炸、极不稳定**——矩阵近似奇异导致闭式解对数据扰动极其敏感。加入惩罚项后优化目标变为：

```
J(w) = MSE(w) + λ · penalty(w)
penalty_L2 = Σ w_j²      （Ridge）
penalty_L1 = Σ |w_j|     （Lasso）
```

- **为什么 L2 只收缩不稀疏**：对 w 求梯度，梯度下降的更新里会多出 `w ← w(1 − 2ηλ)` 这一项——每个系数被**乘以一个小于 1 的因子**等比例缩小，任何非零系数只会无限逼近 0 而不会恰好等于 0。几何上，约束 `Σw² ≤ t` 是**球面**，与损失函数等值线相切时，切点几乎不可能落在坐标轴上，所以系数都是非零小量。
- **为什么 L1 会稀疏**：L1 在 w=0 处不可导，其**次梯度区间包含 0**，一旦某系数落入这个区间就会被"钉死"在 0 上。几何上，约束 `Σ|w| ≤ t` 是**菱形**，顶点（坐标轴上的角点）突出，等值线很容易在角点处相切，从而令对应系数恰好为 0。贝叶斯视角：L1 惩罚对应**拉普拉斯先验**（尖峰厚尾），L2 惩罚对应**高斯先验**——拉普拉斯先验把更多概率质量压在 0 附近，因此 Ridge / Lasso 分别等价于这两种先验下的最大后验（MAP）估计。
- **λ 的作用**：λ 是惩罚强度（sklearn 里 `alpha` 即 λ）。λ→0 时退化为普通最小二乘 OLS；λ→∞ 时系数全部趋于 0，模型退化为只剩截距。λ 的选取直接决定"偏差—方差"的落点，通常用交叉验证确定。
- **求解方式差异**：Ridge 的目标函数是凸二次型，有**闭式解**（岭估计 (XᵀX + λI)⁻¹Xᵀy）；Lasso 的目标含不可导的绝对值项，通常用**坐标下降**（coordinate descent）逐坐标迭代求解。
- **为什么需要 ElasticNet**：特征强相关时，Lasso 会从一组相关特征里"随机"只挑一个，系数路径不稳定；且 Lasso 在 n < p 时最多只能选出 n 个非零系数（受样本量约束）。ElasticNet = `λ₁Σ|w| + λ₂Σw²`，用 L1 保证稀疏、用 L2 让相关特征组"同进同出"（群体效应），兼顾两者。

## 应用

**典型使用场景**：特征数量多或存在强相关/多重共线性的回归建模；需要自动筛特征、追求可解释稀疏模型（Lasso）；以及任何出现过拟合、系数过大迹象的线性模型训练——Ridge/Lasso 是其标准解法。

**快速上手步骤（sklearn 流程）**：
1. **先标准化**（StandardScaler）——惩罚项对量纲敏感，不缩放时量纲大的特征会被罚得更狠，这通常不是我们想要的。
2. 用 `RidgeCV` / `LassoCV` / `ElasticNetCV` 选 α（内置交叉验证自动扫 alpha，代价几乎可忽略）；Lasso 求解走坐标下降，Ridge 有闭式解。
3. 查看系数：Ridge 输出稠密系数（都小但非零），Lasso 输出稀疏系数（多数恰好 0.0），据此做特征筛选。
4. 验证：用交叉验证比较 OLS / Ridge / Lasso / ElasticNet 的测试误差与系数稳定性。

**注意事项 / 常见坑**：
- ❌ 不缩放特征就直接加 L2/L1 惩罚 → 量纲大的特征被罚得最狠，结果失真。✅ 一律 `Pipeline([StandardScaler(), Ridge(...)])`，且**缩放器只能 fit 训练折**（在全体数据上先 fit 再交叉验证属于交叉验证泄漏）。
- ❌ 手动在巨大网格上扫 α → `RidgeCV` / `LassoCV` / `ElasticNetCV` 内置留一法/交叉验证自动选 α，代价几乎可忽略。
- ❌ 特征强相关还用 Lasso 选特征 → 选中哪个全看运气，系数路径不稳定；✅ 换 ElasticNet（设 `l1_ratio`），或先做相关性分析去掉冗余特征。
- ❌ 把 Lasso 稀疏性当"因果筛选"，只信留下的特征 → 稀疏只是优化结果，不代表因果。
- ❌ 忘记 sklearn 中惩罚强度叫 `alpha`（≈λ），而逻辑回归/SVM 里是 `C = 1/λ`——**C 越大正则越弱，方向相反**，换库时极易写反。

```python
# 对比 OLS / Ridge / Lasso / ElasticNet：测试误差、系数稳定性与稀疏性
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

# 样本少、特征多 + 强相关特征组，模拟真实过拟合场景（系数爆炸）
X, y = make_regression(n_samples=80, n_features=50, n_informative=5,
                       noise=10.0, random_state=42)
X[:, 5:10] = X[:, 0:5] + np.random.RandomState(0).normal(0, 0.1, (80, 5))  # 制造相关组

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 关键坑 1：缩放器必须放进 Pipeline —— 交叉验证时每个折内部独立 fit，
# 若先在整个数据集 fit_transform 再用 CV，会引入交叉验证泄漏。
ridge = Pipeline([("sc", StandardScaler()),
                  ("m", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))])
lasso = Pipeline([("sc", StandardScaler()),
                  ("m", LassoCV(alphas=np.logspace(-3, 3, 50), cv=5, random_state=42))])
enet = Pipeline([("sc", StandardScaler()),
                 ("m", ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1.],
                                    cv=5, random_state=42))])

for name, model in [("OLS", LinearRegression()), ("Ridge", ridge),
                    ("Lasso", lasso), ("ElasticNet", enet)]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    print(f"{name:10s} RMSE={(-scores.mean())**0.5:.2f}")

# 拟合后观察系数：Ridge 稠密小系数；Lasso 稀疏、多数恰为 0.0，可用于特征筛选
lasso.fit(X, y)
coef = lasso.named_steps["m"].coef_
print("Lasso 保留的非零系数个数:", int(np.sum(coef != 0.0)))
print("Lasso 自动选出的 alpha:", lasso.named_steps["m"].alpha_)

# 关键坑 2：sklearn 回归里惩罚强度叫 alpha(≈λ)；逻辑回归/SVM 里叫 C = 1/λ，
# C 越大正则越弱，与 alpha 方向相反，跨库使用时不要写反。
```

---
## 关联
- 前置：[[线性回归]]（Ridge/Lasso 就是在 OLS 损失上追加惩罚项；偏差-方差权衡中的"高方差"正来自系数不稳定）
- 前置：[[交叉验证]]（α 的选取与缩放器 fit 时机都依赖正确的 CV 流程）
- 类似：[[ElasticNet]]（区别是同时包含 L1 与 L2 两项惩罚，用 l1_ratio 调配，稀疏与群体效应兼得）
- 类似：[[权重衰减 Weight Decay]]（区别是它在神经网络/优化器里对 L2 惩罚的等价实现，思想相同、载体不同）
- 进阶：[[L0 正则化与子集选择]]（直接统计非零系数个数做子集选择，NP-hard 不可行，L1 是它的凸松弛）
- 进阶：[[贝叶斯线性回归]]（Ridge = 高斯先验下的 MAP，Lasso = 拉普拉斯先验下的 MAP）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| Ridge (L2)（本文方案） | 惩罚 Σw²，系数等比收缩但不归零，稠密解，有闭式解 | 特征强相关/多重共线性，希望保留全部特征、系数稳定 |
| Lasso (L1)（本文方案） | 惩罚 Σ\|w\|，系数精确归零 → 稀疏解 + 特征选择，坐标下降求解 | 特征多、需要自动筛特征与可解释稀疏模型 |
| ElasticNet（替代方案） | λ₁Σ\|w\| + λ₂Σw²：L1 保稀疏、L2 让相关特征组同进同出 | 特征强相关但仍需稀疏，避免 Lasso 随机挑一个的不稳定 |
| 普通最小二乘 OLS（替代方案） | 无惩罚，直接最小化 MSE，等价于 λ=0 | 特征少且近似独立、n ≫ p、无过拟合担忧时的基线对照 |

---
## 参考
- [Ridge — scikit-learn 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Lasso — scikit-learn 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [1.1. Linear Models（Ridge / Lasso / ElasticNet 用户指南）](https://scikit-learn.org/stable/modules/linear_model.html)

---
## 具体案例
- [[正则化：Ridge (L2) 与 Lasso (L1) 实战示例]](正则化：Ridge (L2) 与 Lasso (L1)_sample.py)
