# -*- coding: utf-8 -*-
"""
模型评估与验证 —— 典型代码演示
===============================
覆盖知识点：
  1. 分类指标：准确率/精确率/召回率/F1/AUC，混淆矩阵
  2. K 折交叉验证：cross_val_score
  3. 过拟合 vs 欠拟合的判别：用训练/测试误差对比
  4. 回归指标：MSE/MAE/R²
  5. 数据泄漏防范：Pipeline 保证"先划分再 fit"

依赖：pip install scikit-learn numpy matplotlib
"""

import numpy as np
from sklearn.datasets import make_classification, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error,
    mean_absolute_error, r2_score,
)

# =====================================================================
# 一、构造不平衡数据，看为何不能只看准确率
# =====================================================================
from sklearn.datasets import make_classification
# n_classes=1 表示二分类，weights=[0.9,0.1] 让正例很少（10%）
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, weights=[0.9, 0.1],
                           flip_y=0.05, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=0, stratify=y)
print("训练集正例占比 =", round(y_tr.mean(), 3),
      " 测试集正例占比 =", round(y_te.mean(), 3))

# 训练逻辑回归
model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)
y_pred, y_proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]

# 关键：类别不平衡时 accuracy 会"虚高"
acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred)   # 预测为正里面有多少"对的"
rec  = recall_score(y_te, y_pred)      # 真正的正样本找回了多少
f1   = f1_score(y_te, y_pred)
auc  = roc_auc_score(y_te, y_proba)    # 与阈值无关的排序能力

print("\n=========== 分类指标 ===========")
print(f"准确率 Accuracy = {acc:.3f}  （对不平衡数据参考价值低）")
print(f"精确率 Precision = {prec:.3f}  （误报代价高时重视）")
print(f"召回率 Recall   = {rec:.3f}  （漏报代价高时重视，如癌症筛查）")
print(f"F1 = {f1:.3f}   （精召调和平均，兼顾两者）")
print(f"AUC = {auc:.3f}  （排序区分能力，类不平衡更可靠）")

print("\n混淆矩阵:")
print(confusion_matrix(y_te, y_pred))
#     预测 →     [ [TN  FP]
#                [FN  TP] ]

# =====================================================================
# 二、K 折交叉验证：比单次划分更可信
# =====================================================================
# 5 折交叉验证：把数据切成 5 份，轮流用 4 份训练、1 份验证，取平均
cv_scores = cross_val_score(LogisticRegression(max_iter=1000),
                            X, y, cv=5, scoring="roc_auc")
print("\n[5折交叉验证] 每折 AUC =", [round(s, 3) for s in cv_scores])
print("[5折交叉验证] 平均 AUC =", round(cv_scores.mean(), 3),
      "±", round(cv_scores.std(), 3))

# =====================================================================
# 三、回归指标：MSE / MAE / R²
# =====================================================================
dia_X, dia_y = load_diabetes(return_X_y=True)
X_d, X_dt, y_d, y_dt = train_test_split(dia_X, dia_y, test_size=0.2,
                                        random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_d, y_d)
y_r = reg.predict(X_dt)

print("\n=========== 回归指标 ===========")
print("MSE =", round(mean_squared_error(y_dt, y_r), 2),
      "  RMSE =", round(np.sqrt(mean_squared_error(y_dt, y_r)), 2))
print("MAE =", round(mean_absolute_error(y_dt, y_r), 2))
print("R²  =", round(r2_score(y_dt, y_r), 3), "(越接近 1 越好)")

# =====================================================================
# 四、直接观察过拟合：高次多项式线性回归
# =====================================================================
# 造一个带噪声的二次数据
x = np.linspace(-3, 3, 60)
y_true = 0.5 * x**2 - x + 1
y_noisy = y_true + np.random.RandomState(0).normal(0, 1, len(x))

# 用 PolynomialFeatures 把特征升到高阶（相当于加大模型复杂度）
from sklearn.preprocessing import PolynomialFeatures
def poly_score(degree):
    """返回给定多项式阶数在 训练集 与 测试集 上的 MSE。"""
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree)),
        ("lin", LinearRegression()),
    ])
    xf = x.reshape(-1, 1)
    ix = np.arange(len(x)); np.random.seed(0); np.random.shuffle(ix)
    tr, te = ix[:45], ix[45:]
    pipe.fit(xf[tr], y_noisy[tr])
    return mean_squared_error(y_noisy[tr], pipe.predict(xf[tr])), \
           mean_squared_error(y_noisy[te], pipe.predict(xf[te]))

print("\n=========== 欠拟合/恰拟合/过拟合 ===========")
for d in [1, 2, 15]:                       # 1 = 欠拟合, 2 = 恰当, 15 = 过拟合
    tr_e, te_e = poly_score(d)
    print(f"阶数={d:2d}: 训练MSE={tr_e:.3f}  测试MSE={te_e:.3f}"
          + ("   ← 严重过拟合!" if tr_e < te_e * 0.3 and d >= 10 else ""))

# 口诀：训练MSE很高→欠拟合;训练低但测试高→过拟合

# =====================================================================
# 五、防数据泄漏：用 Pipeline 保证"标准化只在训练折上 fit"
# =====================================================================
# 错误做法（泄漏）：先对整个 X fit StandardScaler 再做交叉验证 → 验证集统计进了训练
# 正确做法（Pipeline）：每次 training 折内单独 fit
pipe = Pipeline([
    ("scaler", StandardScaler()),      # 在 Pipeline 内，只会用训练折的均值/方差
    ("clf", LogisticRegression(max_iter=1000)),
])
safe_scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print("\n[Pipeline] 无泄漏 5折AUC =", round(safe_scores.mean(), 3))

# =====================================================================
# 小结
# =====================================================================
# 评估铁律：只用未见数据评估；类不平衡看 F1/AUC；要稳定用交叉验证；
# 训练集和验证集之间绝不能有信息泄漏。
