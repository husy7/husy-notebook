# -*- coding: utf-8 -*-
"""交叉验证与数据泄漏案例
覆盖要点：
1. StratifiedKFold vs KFold：不平衡分类下普通 K 折某折可能缺类；
2. 特征选择泄漏演示：先在全体数据上 SelectKBest 再 CV（虚高）vs
   Pipeline 内每折选特征（诚实）——对比两者分数差；
3. 缩放泄漏演示：先 fit 全体 scaler 再 CV vs Pipeline 内缩放；
4. GroupKFold：同组样本横跨 train/valid 造成的虚假高分；
5. 提示：嵌套交叉验证 / TimeSeriesSplit / SMOTE 只能放折内。

运行：python 交叉验证与数据泄漏_sample.py （依赖 numpy, scikit-learn）
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier


def demo_stratified():
    """KFold vs StratifiedKFold：每折中正类样本数"""
    X, y = make_classification(n_samples=1000, weights=[0.9, 0.1],
                               random_state=0)
    print("=" * 74)
    print("KFold vs StratifiedKFold（10% 正类，5 折每折理论正类 ~20）")
    print("=" * 74)
    for name, kf in [("KFold", KFold(5, shuffle=True, random_state=0)),
                     ("StratifiedKFold", StratifiedKFold(5, shuffle=True,
                                                         random_state=0))]:
        per_fold = []
        for _, test_idx in kf.split(X, y):
            per_fold.append(int(y[test_idx].sum()))
        print(f"{name:<16} 每折正类数 = {per_fold}  "
              f"({'某折可能缺类/失衡' if min(per_fold) < 10 else '分布稳定'})")


def leaky_pipeline_demo():
    """特征选择泄漏：全体选特征 vs 折内选特征"""
    # 200 个特征里只有 6 个有用 -> 特征选择对分数影响巨大，泄漏效果最明显
    X, y = make_classification(
        n_samples=800, n_features=200, n_informative=6, n_redundant=4,
        flip_y=0.05, random_state=1,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    # ---- 错误做法 1：先在全体训练集上 fit SelectKBest，再做 CV ----
    selector_leaky = SelectKBest(f_classif, k=10).fit(Xtr, ytr)
    Xtr_leaky = selector_leaky.transform(Xtr)
    score_leaky = cross_val_score(
        LogisticRegression(max_iter=2000), Xtr_leaky, ytr, cv=5).mean()

    # ---- 错误做法 2：更隐蔽 —— 连测试集一起参与选择（最严重）----
    Xall = np.vstack([Xtr, Xte])
    yall = np.concatenate([ytr, yte])
    selector_bad = SelectKBest(f_classif, k=10).fit(Xall, yall)
    Xall_bad = selector_bad.transform(Xall)
    # 重新切分（此时选择已看过"未来"测试信息）
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        Xall_bad, yall, test_size=0.3, random_state=0)
    leaky_model = LogisticRegression(max_iter=2000).fit(Xb_tr, yb_tr)
    score_full_leak = roc_auc_score(yb_te, leaky_model.predict_proba(Xb_te)[:, 1])

    # ---- 正确做法：Pipeline 把选择放进每一折内部 ----
    pipe = Pipeline([
        ("select", SelectKBest(f_classif, k=10)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    score_clean = cross_val_score(pipe, Xtr, ytr, cv=5).mean()

    print("\n" + "=" * 74)
    print("特征选择泄漏（SelectKBest）——分数虚高的典型")
    print("=" * 74)
    print(f"[泄漏-折外选特征] CV AUC = {score_leaky:.4f}")
    print(f"[泄漏-含测试集选] AUC    = {score_full_leak:.4f}  <- 最严重")
    print(f"[正确-Pipeline]   CV AUC = {score_clean:.4f}")
    print(f"-> 泄漏可把分数抬 {score_leaky - score_clean:.3f}+，上线必翻车")


def scaling_leak_demo():
    """缩放泄漏：全体 fit scaler vs Pipeline 内缩放（kNN 对尺度敏感）"""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (600, 30))
    X[:, 0] *= 5  # 放大一个特征的量纲，让缩放真正起作用
    y = (X[:, 0] + rng.normal(0, 1, 600) > 0).astype(int)

    # 错误：先 fit 全体 scaler
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    leaky = cross_val_score(KNeighborsClassifier(3), X_scaled, y, cv=5).mean()

    # 正确：Pipeline，scaler 每折只 fit 训练部分
    pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(3))])
    clean = cross_val_score(pipe, X, y, cv=5).mean()

    print("\n" + "=" * 74)
    print("缩放泄漏（StandardScaler + kNN）")
    print("=" * 74)
    print(f"[泄漏-全体fit缩放] CV AUC = {leaky:.4f}")
    print(f"[正确-Pipeline缩放] CV AUC = {clean:.4f}")
    print("-> 折外缩放等于让验证折的均值/方差参与训练；数据越不平稳差得越多")


def group_leak_demo():
    """GroupKFold：同一组的样本横跨 train/valid = 变相看到答案"""
    rng = np.random.default_rng(7)
    n_users, per_user = 40, 20
    user_bias = rng.normal(0, 3, n_users)      # 每个用户一个固有偏移
    X, y, groups = [], [], []
    for u in range(n_users):
        for _ in range(per_user):
            x = rng.normal(user_bias[u], 1.0, 1)
            X.append(x)
            y.append(int(x[0] + rng.normal(0, 0.5) > 0))
            groups.append(u)
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    print("\n" + "=" * 74)
    print("组结构泄漏：GroupKFold vs 普通 KFold（同一用户样本横跨两边）")
    print("=" * 74)
    # 普通 KFold：同一用户的行会同时进 train 与 valid -> 学到的用户偏移直接可用
    def auc_with(kf):
        aucs = []
        for tr, va in kf.split(X, y, groups):
            clf = LogisticRegression().fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1]))
        return float(np.mean(aucs))

    plain = auc_with(KFold(5, shuffle=True, random_state=0))
    grouped = auc_with(GroupKFold(n_splits=5))
    print(f"普通 KFold     CV AUC = {plain:.4f}  <- 同组横跨，虚高")
    print(f"GroupKFold     CV AUC = {grouped:.4f}  <- 对'新用户'的真实泛化")
    print(f"-> 差 {plain - grouped:.3f}：上线遇到全新用户时，普通 KFold 的分数会严重高估")


def demo_timeseries_note():
    """时间序列提示（文字演示）"""
    print("\n" + "=" * 74)
    print("其它泄漏形式（代码演示从略，要点如下）")
    print("=" * 74)
    print("""1. 时间序列：必须 TimeSeriesSplit（只用过去预测未来），普通 KFold 会让未来进训练；
2. SMOTE/过采样：只能在每一折的训练部分内合成，否则合成样本跨折复制；
3. 反复用同一测试集调参：测试集被'训练'，最终要留一块从未碰过的数据；
4. 模型选择更严谨的姿势：嵌套交叉验证（内层选参，外层估泛化）。""")


if __name__ == "__main__":
    demo_stratified()
    leaky_pipeline_demo()
    scaling_leak_demo()
    group_leak_demo()
    demo_timeseries_note()
