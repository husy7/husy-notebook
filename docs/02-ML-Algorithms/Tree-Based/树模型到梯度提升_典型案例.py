# -*- coding: utf-8 -*-
"""
树模型到梯度提升 —— 典型代码演示
================================
覆盖知识点：
  1. 决策树：划分准则（信息增益 / 基尼指数）、树的可视化
  2. 随机森林（Bagging）：并行多棵树，抗过拟合
  3. GBDT / XGBoost（Boosting）：串行拟合残差
  4. 特征重要度比较、过拟合 vs 集成效果对比

依赖：pip install scikit-learn xgboost
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# 载入乳腺癌二分类数据集（30 个特征）
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=0, stratify=y)

# =====================================================================
# 一、单棵决策树：不剪枝极易过拟合
# =====================================================================
# 不加 max_depth 限制 → 深度可到极限，直到每个叶子纯净（完美记住训练集）
dt_full = DecisionTreeClassifier(random_state=0)
dt_full.fit(X_tr, y_tr)
print("[不剪枝决策树] 训练集准确率 =",
      round(dt_full.score(X_tr, y_tr), 3),   # 往往接近 1.0（过拟合）
      " 测试集准确率 =", round(dt_full.score(X_te, y_te), 3))

# 限制深度 max_depth=3 → 降低复杂度、提高泛化
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=0)
dt_shallow.fit(X_tr, y_tr)
print("[限制深度 max_depth=3] 树深度 =", dt_shallow.get_depth(),
      " 测试集准确率 =", round(dt_shallow.score(X_te, y_te), 3))

# 可视化决策树（可保存为 png；Console 输出需要 graphviz）
print("\n[决策树] 用到的特征：",
      [data.feature_names[i] for i in dt_shallow.feature_importances_.argsort()[::-1][:3]])

# =====================================================================
# 二、随机森林：多棵树 + 列采样，降低方差、抗过拟合
# =====================================================================
rf = RandomForestClassifier(
    n_estimators=100,      # 100 棵树
    max_features="sqrt",   # 每个分裂随机用 sqrt(特征数) 个特征（随机森林的关键）
    max_depth=10,          # 限制单树深度，集成可容纳更深
    oob_score=True,        # 用袋外样本估计泛化性能
    random_state=0,
)
rf.fit(X_tr, y_tr)
train_acc = accuracy_score(y_tr, rf.predict(X_tr))
test_acc  = accuracy_score(y_te, rf.predict(X_te))
test_auc  = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
print(f"\n[随机森林] 训练准确率={train_acc:.3f} 测试准确率={test_acc:.3f} "
      f"AUC={test_auc:.3f}  OOB={rf.oob_score_:.3f}")

# =====================================================================
# 三、XGBoost：Boosting 串行拟合残差 + 正则化
# =====================================================================
xgb_model = xgb.XGBClassifier(
    n_estimators=50,       # 迭代 50 轮（每轮加一棵树）
    max_depth=3,           # 单棵树深度（浅树 → 弱学习器）
    learning_rate=0.1,     # 学习率：每一步对残差的贡献，越小越稳但要多轮
    subsample=0.8,         # 每棵树只用 80% 样本（随机性）
    colsample_bytree=0.8,  # 每棵树只用 80% 特征
    eval_metric="logloss",
    use_label_encoder=False,   # 避免旧版 API 告警
    random_state=0,
)
xgb_model.fit(X_tr, y_tr)
xgb_acc = accuracy_score(y_te, xgb_model.predict(X_te))
xgb_auc = roc_auc_score(y_te, xgb_model.predict_proba(X_te)[:, 1])
print(f"[XGBoost] 测试准确率={xgb_acc:.3f} 测试AUC={xgb_auc:.3f}")

# 交叉验证得到更可信的 XGBoost 性能（K=5 折）
cv = cross_val_score(xgb_model, X, y, cv=5, scoring="accuracy")
print("[XGBoost] 5折交叉验证平均准确率 =", round(cv.mean(), 3),
      "±", round(cv.std(), 3))

# =====================================================================
# 四、特征重要度对比：看不同模型认为的"关键特征"
# =====================================================================
def top_features(model, n=3):
    """返回模型认为最重要的前 n 个特征名。"""
    idx = np.argsort(model.feature_importances_)[::-1][:n]
    return [data.feature_names[i] for i in idx]

print("\n[重要特征] 决策树:", top_features(dt_shallow))
print("[重要特征] 随机森林:", top_features(rf))
print("[重要特征] XGBoost:", top_features(xgb_model))

# =====================================================================
# 小结
# =====================================================================
# 通常集成（随机森林/XGBoost）> 单棵决策树（尤其在泛化与稳定性上）。
# Boosting 比 Bagging 精度上限更高，但对噪声更敏感、更易过拟合，
# 需靠学习率/正则/early stopping 控制。
