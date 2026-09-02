# -*- coding: utf-8 -*-
"""随机森林案例
覆盖要点：
1. 单棵树 vs Bagging vs 随机森林 vs ExtraTrees 的精度与稳定性对比；
2. 方差来源演示：同一数据上，单棵树对随机种子极敏感，RF 稳定；
3. n_estimators 的边际收益递减（方差分解：加树只灭第二项）；
4. max_features 的作用（树间去相关）；
5. OOB 分数 ≈ 交叉验证分数（免费验证）；
6. 特征重要性（impurity）及其高估连续特征的坑（与 permutation 对比）。

运行：python 随机森林_sample.py （依赖 numpy, scikit-learn）
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier)
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score


def make_data():
    """构造：高维、非线性、有噪声、特征部分冗余的分类数据"""
    X, y = make_classification(
        n_samples=800, n_features=40, n_informative=12, n_redundant=10,
        n_clusters_per_class=1, flip_y=0.08, random_state=42,
    )
    return X, y


def compare_models(X, y):
    """单树 / Bagging(全特征) / RF / ExtraTrees 交叉验证"""
    models = {
        "单棵决策树": DecisionTreeClassifier(random_state=0),
        "Bagging(全特征)": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=0),
            n_estimators=100, random_state=0, n_jobs=-1),
        "随机森林RF": RandomForestClassifier(
            n_estimators=100, random_state=0, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100, random_state=0, n_jobs=-1),
    }
    print("=" * 72)
    print("模型对比（5 折交叉验证准确率）—— 注意集成相对单树提升")
    print("=" * 72)
    fitted = {}
    for name, m in models.items():
        scores = cross_val_score(m, X, y, cv=5, n_jobs=-1)
        print(f"{name:<20} acc = {scores.mean():.4f} ± {scores.std():.4f}")
        fitted[name] = m.fit(X, y)
    return fitted


def variance_demo(X, y):
    """方差演示：换随机种子，单树结构大变，RF 结果稳定"""
    print("\n" + "=" * 72)
    print("方差来源演示：同一数据换 5 个随机种子")
    print("=" * 72)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    tree_accs, rf_accs = [], []
    for seed in range(5):
        tree = DecisionTreeClassifier(random_state=seed).fit(Xtr, ytr)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                    n_jobs=-1).fit(Xtr, ytr)
        tree_accs.append(accuracy_score(yte, tree.predict(Xte)))
        rf_accs.append(accuracy_score(yte, rf.predict(Xte)))
    print(f"单棵树   acc 序列 = {[f'{a:.3f}' for a in tree_accs]}  "
          f"std = {np.std(tree_accs):.4f}")
    print(f"随机森林 acc 序列 = {[f'{a:.3f}' for a in rf_accs]}  "
          f"std = {np.std(rf_accs):.4f}")
    print("-> 树结构对数据/种子扰动极敏感（高方差），平均化后显著变稳")


def n_estimators_demo(X, y):
    """加树收益递减：n_estimators 从 10 加到 500"""
    print("\n" + "=" * 72)
    print("n_estimators 边际收益递减（测试集 acc）")
    print("=" * 72)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for n in [1, 10, 50, 200, 500]:
        rf = RandomForestClassifier(n_estimators=n, random_state=0,
                                    n_jobs=-1).fit(Xtr, ytr)
        acc = accuracy_score(yte, rf.predict(Xte))
        print(f"n_estimators = {n:>4}  test acc = {acc:.4f}")


def max_features_demo(X, y):
    """max_features 的作用：太小树太弱，太大树太像"""
    print("\n" + "=" * 72)
    print("max_features 扫描（每次划分随机考察的特征数）")
    print("=" * 72)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    n_feat = X.shape[1]
    for mf in [1, 4, int(np.sqrt(n_feat)), 10, 20, n_feat]:
        rf = RandomForestClassifier(n_estimators=100, max_features=mf,
                                    random_state=0, n_jobs=-1).fit(Xtr, ytr)
        acc = accuracy_score(yte, rf.predict(Xte))
        note = " <- 全特征=纯Bagging(树高度相关)" if mf == n_feat else ""
        print(f"max_features = {mf:>3}  test acc = {acc:.4f}{note}")


def oob_and_importance(X, y):
    """OOB 免费验证 + 特征重要性（impurity vs permutation）"""
    print("\n" + "=" * 72)
    print("OOB 分数 vs 交叉验证 + 特征重要性")
    print("=" * 72)
    rf = RandomForestClassifier(n_estimators=200, oob_score=True,
                                random_state=0, n_jobs=-1).fit(X, y)
    cv = cross_val_score(rf, X, y, cv=5, n_jobs=-1).mean()
    print(f"OOB score = {rf.oob_score_:.4f}    5折CV 平均 = {cv:.4f}  <- 两者应接近")

    imp = rf.feature_importances_
    top_imp = np.argsort(-imp)[:5]
    print("impurity 重要性 Top5 特征:", sorted(top_imp.tolist()))
    print("(真相关特征集中在 0..11，注意 impurity 也常把冗余/高基数特征排前面)")

    # permutation 重要性：随机打乱某列看精度掉多少，更稳健
    perm = permutation_importance(
        rf, X, y, n_repeats=5, random_state=0, n_jobs=-1)
    top_perm = np.argsort(-perm.importances_mean)[:5]
    print("permutation 重要性 Top5 特征:", sorted(top_perm.tolist()))


if __name__ == "__main__":
    X, y = make_data()
    compare_models(X, y)
    variance_demo(X, y)
    n_estimators_demo(X, y)
    max_features_demo(X, y)
    oob_and_importance(X, y)
