# -*- coding: utf-8 -*-
"""决策树与划分准则案例
覆盖要点：
1. 手算基尼/熵：理解"不纯度"与"划分后不纯度下降"；
2. sklearn (CART) 的 criterion：gini vs entropy(log_loss) 在鸢尾花上几乎无差；
3. 预剪枝：max_depth / min_samples_leaf 对过拟合的控制（训练 vs 测试误差）；
4. 后剪枝：cost_complexity_pruning_path + ccp_alpha；
5. 特征重要性与树的可读结构（rules）。

运行：python 决策树与划分准则_sample.py （依赖 numpy, scikit-learn；绘图为可选项）
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score


# ----------------------------------------------------------------------
# 0. 手算不纯度：基尼 / 熵 都只依赖"类别比例 p_k"
# ----------------------------------------------------------------------
def gini(probs):
    """Gini = 1 - sum(p_k^2)：随机抽两个样本类别不一致的概率"""
    return 1.0 - np.sum(np.square(probs))


def entropy(probs):
    """熵 = -sum(p_k * log2 p_k)：把类别编码到最优时的期望码长"""
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def impurity_gain(y, split_mask, criterion="gini"):
    """按 split_mask 把 y 分成左右两组后，父节点不纯度 - 加权子不纯度"""
    def node_gini(yy):
        _, cnt = np.unique(yy, return_counts=True)
        return gini(cnt / cnt.sum())

    def node_ent(yy):
        _, cnt = np.unique(yy, return_counts=True)
        return entropy(cnt / cnt.sum())

    fn = node_gini if criterion == "gini" else node_ent
    y = np.asarray(y)
    parent = fn(y)
    n = len(y)
    left, right = y[split_mask], y[~split_mask]
    if len(left) == 0 or len(right) == 0:
        return 0.0
    child = (len(left) / n) * fn(left) + (len(right) / n) * fn(right)
    return parent - child


def demo_manual_impurity():
    """演示：一个节点 30 样本 {A:20, B:10}，按某特征切开后纯度变化"""
    y = np.array([0] * 20 + [1] * 10)          # 父节点
    split_ok = np.array([True] * 15 + [False] * 15)   # 一组全 A，一组 5A/10B
    split_bad = np.zeros(30, dtype=bool)              # 切了等于没切
    print("=" * 70)
    print("手算不纯度（父节点: 20 个 A + 10 个 B）")
    print("=" * 70)
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    print(f"父节点纯度   Gini = {gini(p):.4f}    Entropy = {entropy(p):.4f}")
    print(f"好切分增益   Gini 下降 = {impurity_gain(y, split_ok, 'gini'):.4f}"
          f"   熵下降 = {impurity_gain(y, split_ok, 'entropy'):.4f}")
    print(f"无意义切分   Gini 下降 = {impurity_gain(y, split_bad, 'gini'):.4f}  <- 树不会选它")


def demo_criterion_iris():
    """gini / entropy / log_loss 三种 criterion 效果对比"""
    iris = load_iris()
    X, y = iris.data, iris.target
    print("\n" + "=" * 70)
    print("criterion 对比（鸢尾花，5 折交叉验证平均准确率）")
    print("=" * 70)
    for crit in ["gini", "entropy", "log_loss"]:
        clf = DecisionTreeClassifier(max_depth=4, random_state=0, criterion=crit)
        scores = cross_val_score(clf, X, y, cv=5)
        print(f"criterion = {crit:<10} acc = {scores.mean():.4f} ± {scores.std():.4f}")


def demo_pruning():
    """预剪枝：max_depth / min_samples_leaf 控制训练-测试误差差距"""
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    print("\n" + "=" * 70)
    print("预剪枝参数扫描（观察过拟合：训练很高、测试跟不上）")
    print("=" * 70)
    for depth in [1, 2, 3, 5, 10, None]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(Xtr, ytr)
        tr = accuracy_score(ytr, clf.predict(Xtr))
        te = accuracy_score(yte, clf.predict(Xte))
        print(f"max_depth = {str(depth):<5}  train acc = {tr:.4f}   test acc = {te:.4f}")

    # 后剪枝：ccp_alpha（成本复杂度剪枝）
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(Xtr, ytr)
    alphas = path.ccp_alphas
    best_a, best_acc = 0.0, -1.0
    for a in alphas:
        t = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(Xtr, ytr)
        acc = accuracy_score(yte, t.predict(Xte))
        if acc > best_acc:
            best_a, best_acc = a, acc
    print(f"\n后剪枝(ccp_alpha)：最优 alpha = {best_a:.4f} -> test acc = {best_acc:.4f}"
          f"  （alpha 越大树越浅）")


def demo_structure():
    """把树打出来看规则，顺便演示特征重要性"""
    X, y = load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    print("\n" + "=" * 70)
    print("深度 3 的决策树规则（可解释性：if-then）")
    print("=" * 70)
    print(export_text(clf, feature_names=load_iris().feature_names[:4]))
    imp = dict(zip(load_iris().feature_names, clf.feature_importances_))
    print("特征重要性（平均不纯度减少归一化）:")
    for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:.3f}")


if __name__ == "__main__":
    demo_manual_impurity()
    demo_criterion_iris()
    demo_pruning()
    demo_structure()

    # 可选：可视化树结构
    try:
        import matplotlib
        matplotlib.use("Agg")
        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        X, y = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
        fig = plt.figure(figsize=(12, 6))
        plot_tree(clf, feature_names=load_iris().feature_names,
                  class_names=load_iris().target_names, filled=True,
                  impurity=True, fontsize=8)
        plt.savefig("decision_tree_iris.png", dpi=110, bbox_inches="tight")
        print("\n[绘图] 已保存 decision_tree_iris.png")
    except Exception as exc:
        print(f"\n[绘图] 跳过（需要 matplotlib）：{type(exc).__name__}")
