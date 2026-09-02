# -*- coding: utf-8 -*-
"""分类指标与混淆矩阵案例
覆盖要点：
1. 构造 95:5 的类不平衡数据，演示 accuracy 骗人、F1/AUC 说真话；
2. 混淆矩阵四项 + classification_report（P/R/F1/support）解读；
3. P-R 与 R 的此消彼长：扫阈值观察 trade-off，选阈值不能默认 0.5；
4. ROC-AUC（与阈值无关的排序能力）vs PR-AUC（不平衡更敏感）对比；
5. macro vs micro 平均在多分类/不平衡下的差异。

运行：python 分类指标与混淆矩阵_sample.py （依赖 numpy, scikit-learn；绘图可选）
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_curve, roc_curve, auc,
                             average_precision_score, accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.dummy import DummyClassifier


def make_imbalanced():
    """95:5 类不平衡、10 个特征、含冗余与噪声"""
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=4, n_redundant=3,
        weights=[0.95, 0.05], flip_y=0.03, random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=0)


def demo_baseline(Xtr, Xte, ytr, yte):
    """傻瓜基线（全猜多数类）vs 逻辑回归：accuracy 视角 vs F1 视角"""
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=0).fit(Xtr, ytr)
    print("=" * 74)
    print("95:5 不平衡数据：accuracy 会骗人")
    print("=" * 74)
    for name, m in [("全猜多数类(基线)", dummy), ("逻辑回归(平衡权重)", clf)]:
        yp = m.predict(Xte)
        print(f"{name:<18} accuracy = {accuracy_score(yte, yp):.4f}"
              f"   F1(正类) = {f1_score(yte, yp, pos_label=1):.4f}")
    print("-> accuracy 上基线高达 ~0.95 看着'很好'，但 F1(正类)=0 暴露它什么都没学")
    return clf


def demo_confusion(clf, Xte, yte):
    """混淆矩阵 + classification_report"""
    yp = clf.predict(Xte)
    tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
    print("\n" + "=" * 74)
    print(f"混淆矩阵（正类=1，{len(yte)} 个测试样本）")
    print("=" * 74)
    print(f"           预测负(0)   预测正(1)")
    print(f"真实负(0)   TN={tn:<9} FP={fp}")
    print(f"真实正(1)   FN={fn:<9} TP={tp}")
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    print(f"\nPrecision = {p:.3f}  （正类预测里真对的占比）")
    print(f"Recall    = {r:.3f}  （真阳性里找回的比例）")
    print(f"F1        = {2*p*r/(p+r):.3f}  （P、R 的调和平均）")
    print("\nclassification_report 全表：")
    print(classification_report(yte, yp, digits=3))


def demo_threshold(clf, Xte, yte):
    """阈值扫描：P 与 R 此消彼长；默认 0.5 未必最优"""
    proba = clf.predict_proba(Xte)[:, 1]
    print("\n" + "=" * 74)
    print("阈值扫描（P/R 随阈值变化，业务成本决定选哪点）")
    print("=" * 74)
    print(f"{'阈值':>6} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for thr in [0.2, 0.35, 0.5, 0.65, 0.8]:
        yp = (proba >= thr).astype(int)
        print(f"{thr:>6.2f} {precision_score(yte, yp, pos_label=1):>10.3f}"
              f" {recall_score(yte, yp, pos_label=1):>8.3f}"
              f" {f1_score(yte, yp, pos_label=1):>8.3f}")
    print("-> 阈值低:召回高精度低(捞全)；阈值高:精度高召回低(宁缺毋滥)")
    print("   若业务'漏检代价 > 误报代价'，0.5 以下才是该用的点")


def demo_curves(clf, Xte, yte):
    """ROC-AUC vs PR-AUC"""
    proba = clf.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, proba)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(yte, proba)
    pr_auc = average_precision_score(yte, proba)  # = PR 曲线下面积(AP)

    print("\n" + "=" * 74)
    print("ROC-AUC 与 PR-AUC")
    print("=" * 74)
    print(f"ROC-AUC = {roc_auc:.4f}  （阈值无关的排序能力；不平衡下偏乐观）")
    print(f"PR-AUC  = {pr_auc:.4f}  （不平衡下更敏感、更可信）")
    # PR 曲线关键点：找 F1 最大的 (P, R)
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    i = int(np.argmax(f1s))
    print(f"PR 曲线上 F1 最大点: P = {prec[i]:.3f}, R = {rec[i]:.3f}"
          f"  -> 该点对应阈值常优于默认 0.5")


def demo_macro_micro():
    """多分类：macro 与 micro 平均的差别"""
    from sklearn.datasets import make_classification
    from sklearn.metrics import f1_score
    X, y = make_classification(n_samples=3000, n_features=12, n_informative=6,
                               n_classes=3, weights=[0.8, 0.15, 0.05],
                               random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    yp = clf.predict(Xte)
    print("\n" + "=" * 74)
    print("多分类（类别分布 80/15/5）：macro vs micro F1")
    print("=" * 74)
    print(f"micro F1 = {f1_score(yte, yp, average='micro'):.4f}"
          f"  （按全体样本计，被 80% 的大类主导）")
    print(f"macro F1 = {f1_score(yte, yp, average='macro'):.4f}"
          f"  （三类等权平均，稀有类同样算数）")
    print("-> 大类别占比高时 micro≈accuracy；关心稀有类就盯 macro 与各类别明细")


if __name__ == "__main__":
    Xtr, Xte, ytr, yte = make_imbalanced()
    clf = demo_baseline(Xtr, Xte, ytr, yte)
    demo_confusion(clf, Xte, yte)
    demo_threshold(clf, Xte, yte)
    demo_curves(clf, Xte, yte)
    demo_macro_micro()

    # 可选：画 ROC 与 PR 曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        proba = clf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        prec, rec, _ = precision_recall_curve(yte, proba)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(fpr, tpr, label=f"ROC-AUC = {auc(fpr, tpr):.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC 曲线")
        axes[0].legend()
        axes[1].plot(rec, prec, label=f"AP = {average_precision_score(yte, proba):.3f}")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="PR 曲线（不平衡更可信）")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig("roc_pr_curves.png", dpi=110)
        print("\n[绘图] 已保存 roc_pr_curves.png")
    except Exception as exc:
        print(f"\n[绘图] 跳过（需要 matplotlib）：{type(exc).__name__}")
