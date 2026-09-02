---
title: "分类指标与混淆矩阵：P/R/F1、AUC 与类不平衡"
tags: ["Evaluation", "混淆矩阵", "Precision", "Recall", "F1", "AUC", "类不平衡"]
date: 2026-08-29
---

# 分类指标与混淆矩阵：P/R/F1、AUC 与类不平衡

## 定义
混淆矩阵是分类评估的"总账"：它按"真实类别 × 预测类别"把样本落入 TP / FP / FN / TN 四格（sklearn 默认行=真实、列=预测），而**所有分类指标都只是这张表的不同投影**。其中 TP = 实际为正且预测为正（真阳性）、FP = 实际为负但预测为正（误报，第一类错误）、FN = 实际为正但预测为负（漏报，第二类错误）、TN = 实际为负且预测为负。

指标家族分三层：① **答对多少**——accuracy = (TP+TN)/总数，最直觉但只在类别均衡时诚实；② **正类查得准不准、全不全**——Precision P = TP/(TP+FP)（说"是"的人里多少真对）、Recall R = TP/(TP+FN)（真阳里找回了多少），以及两者的折中 F1 = 2PR/(P+R)、可调权重的 Fβ；③ **排序能力**——ROC-AUC 与 PR-AUC（AP），与阈值无关地衡量"模型给正样本打的得分是否系统性更高"。

它解决的核心问题是**类别不平衡下的评估失真**：95:5 的数据全猜多数类就有 95% 准确率，模型可能什么都没学到；P/R/F1/AUC 都绕过多数类的规模，直接回答"正类（通常是小类 = 业务目标）被找得多准、多全"。

适用范畴：二分类与多分类评估（多分类用 macro/micro 平均）、代价不对称下的阈值选择、模型排序能力对比；**不适用**回归评估——分类指标几乎全部建立在"阈值/排序"上，回归则建立在"误差大小"上，两者不可混用。

## 原理
**为什么 P 与 R 天然此消彼长**：二者共享分子 TP，分母不同——P 的分母是"预测为正的全体"(TP+FP)，R 的分母是"真实为正的全体"(TP+FN)。把判定阈值调低会多捞正例（TP 与 FP 同增，R↑）但误伤负例（P↓）；调高则反向。业务不同取舍不同：垃圾邮件宁误报（重 P），癌症筛查宁漏检为 0（重 R）。这一权衡只能在 P-R 曲线上整体观察，任何单点数字都无法脱离它解释。

**为什么 F1 用调和平均**：F1 = 2PR/(P+R) = 2/(1/P + 1/R)。调和平均比算术平均更狠地惩罚"P 与 R 一个高一个低"（例如 P=1.0、R=0.01 时算术平均≈0.5 而 F1≈0.02），因此一个数就能概括两者的平衡；`Fβ` 用 β 给 Recall 加权（β>1 更重召回），适配"漏报更贵"的场景。

**为什么看 AUC**：ROC 曲线扫过**所有阈值**描出 (FPR, TPR)，AUC 即曲线下面积，等价于"随机抽一个正样本、一个负样本，正样本得分更高的概率"。因此 AUC **与阈值无关**，衡量的是模型的"排序能力"而非某一阈值下的表现；当只需要比较"谁排得更对"时用它。

**为什么不平衡下 AUC 仍偏乐观**：ROC 的横轴是 FPR = FP/(TN+FP)，负样本基数大时 FPR 分母被撑大，FP 的小幅增长被稀释，曲线显得很好看；而 PR 曲线的 precision 分母是"预测为正的全体"，只关注正类相关错误，**对类不平衡更敏感**，所以极端不平衡下用 AUC-PR（AP）更可信。

**accuracy 骗人的机理**：多数类规模直接进分母，全猜多数类即可拿到与多数类占比相同的 accuracy；P/R/F1/AUC 的构造都不含"多数类规模"这一项，从而把评估焦点锁在正类上。

**多分类平均的口径差异**：micro 先把所有类别的 TP/FP/FN 汇总再算，被大类别主导；macro 对每类算指标后取算术平均，平等对待每一类（也因此可能掩盖某类崩坏，需配合每类的 classification_report 全表看）。**阈值/校准补充**：AUC 高只说明排序好，不代表输出概率准——要概率可用需看校准曲线与 Brier 分数。

## 应用
典型使用场景：类别不平衡分类（欺诈检测、疾病筛查、罕见事件预警）、代价不对称的业务决策、多分类模型对比与"挑阈值"上线前评估。核心流程如下：

1. 拿**预测得分而非硬标签**，先算混淆矩阵（`confusion_matrix`），确认 axis 无误（行=真实、列=预测是 sklearn 默认，正负别写反）；
2. 无偏对比先看 `classification_report`（每类一行的 P/R/F1/support），别只报一个汇总数；
3. 画 ROC 定 AUC（看全局排序能力）；画 PR 定 AP（不平衡时更可信）；
4. 按业务成本在 P-R 曲线上**挑阈值**（不是默认 0.5）——FN 比 FP 贵就调低直到 P-R 平衡点/成本最优；多分类用 macro/micro 平均（micro 被大类别主导，macro 平等对待每类）。

易错点/常见坑：① 不平衡数据只看 accuracy → 应看 F1/AUC/PR，尤其少数类的 recall；② 把默认 0.5 当永远的最优阈值 → 若 FN 比 FP 贵得多，应调低阈值直到成本最优；③ 只报 macro F1 不报各类别 F1 → macro 把稀有类与多数类等权，会掩盖具体某类崩坏，要看全表；④ 同一模型在不同类别分布的数据集之间直接比 AUC → AUC 依赖样本类别构成，跨分布不可比；⑤ 混淆矩阵"正负"写反（TN/FP 行对调）→ 先想清楚 axis 语义；⑥ 忽略校准 → AUC 高 ≠ 概率准，需要可信概率时看校准曲线/Brier 分数。另外记住：**不平衡处理（重采样/SMOTE/class_weight）只是手段，先把评估指标选对才谈得上对比**；真实工程中挑阈值应在验证集而非测试集上完成，避免过拟合测试集。

```python
# 分类评估完整流程：混淆矩阵 → P/R/F1 → ROC/PR-AUC → 按业务成本挑阈值
# 场景：二分类，正类（少数类，如"欺诈/患病"）占 5%；业务上 FN（漏报）比 FP（误报）更贵
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve)

# 构造 95:5 的不平衡二分类数据（4 个特征，少数类占 5%）
X, y = make_classification(n_samples=2000, n_features=4, n_informative=3,
                           n_redundant=0, weights=[0.95, 0.05], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=42)  # stratify 保持分布

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_tr, y_tr)
# 关键：拿"预测得分"而非硬标签做评估，阈值之后可以再调
proba = clf.predict_proba(X_te)[:, 1]

# 第 1 步：先看混淆矩阵（sklearn 默认 行=真实、列=预测；TP/FP 别写反）
cm = confusion_matrix(y_te, proba >= 0.5)
print("混淆矩阵（默认阈值 0.5）：\n", cm)

# 第 2 步：无偏对比先看 classification_report（每类一行 P/R/F1/support）
print(classification_report(y_te, proba >= 0.5,
                            target_names=["多数类", "少数类(正类)"]))

# 第 3 步：全局排序能力——AUC 与阈值无关；不平衡场景再补 PR-AP
print(f"ROC-AUC = {roc_auc_score(y_te, proba):.3f}")    # 类不平衡下可能偏乐观
print(f"PR-AP   = {average_precision_score(y_te, proba):.3f}")  # 对不平衡更敏感、更可信

# 第 4 步：按业务成本挑阈值——默认 0.5 并非最优
# 本例 FN（漏报欺诈）比 FP（误报）更贵 → 应把阈值调低到 P-R 平衡点附近
prec, rec, ths = precision_recall_curve(y_te, proba)
f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)     # 逐点算 F1
best = int(np.argmax(f1s[:len(ths)]))                    # 只在有效阈值范围内取最大
print(f"最优 F1 阈值 ≈ {ths[best]:.3f}（F1 = {f1s[best]:.3f}）")
print("提示：真实工程应在验证集上选阈值，避免在测试集上调参导致过拟合。")
```

---
## 关联
- 前置：[[分类评估基础：混淆矩阵与准确率]]
- 类似：[[ROC 与 PR 曲线详解]]（区别是 ROC-AUC 扫全部阈值衡量全局排序、类不平衡下因 FPR 分母被负类撑大而偏乐观；PR 曲线只看正类相关错误、对极端不平衡更敏感）
- 类似：[[回归评估指标：MAE/MSE/R²]]（区别是分类指标建立在"阈值/排序"上，回归指标建立在"误差大小"上，二者不可混用）
- 进阶：[[代价敏感学习与阈值选择]]（阈值 ⇄ 代价敏感学习，用业务成本在 P-R 曲线上定阈值）
- 进阶：[[排序指标：AP 与 NDCG]]（P-R 曲线 ⇄ 排序指标 AP/NDCG 家族同源）
- 进阶：[[类不平衡处理：SMOTE 与 class_weight]]（重采样/SMOTE/class_weight 只是手段，评估指标先选对才谈得上对比）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：混淆矩阵 → P/R/F1/AUC 体系 | 以混淆矩阵四格为总账，用 P/R/F1 度量正类查准/查全/平衡、用 AUC/PR 度量排序能力，全部绕过多数类规模 | 类别不平衡、FN/FP 代价不对称、需要挑阈值或比排序能力的分类评估 |
| Accuracy / 错误率 | 直接统计"答对多少"：(TP+TN)/总数，直观但有偏 | 类别均衡、错误代价对称的简单场景（仅作第一直觉，不平衡时禁用） |
| 回归式误差指标（MAE/MSE） | 度量预测值与真实值的误差大小，与阈值/排序无关 | 回归任务；分类任务不可用 |
| 校准曲线 / Brier 分数 | 度量"输出的概率是否可信"，AUC 高不保证概率准 | 需要可信概率输出（如风控额度、决策概率）的下游场景 |

---
## 参考
- [scikit-learn: The scoring protocol / model evaluation（官方评估指南）](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn: sklearn.metrics.classification_report（P/R/F1 参考文档）](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

---
## 具体案例
- [[分类指标与混淆矩阵：P/R/F1、AUC 与类不平衡 实战示例]](分类指标与混淆矩阵：P/R/F1、AUC 与类不平衡_sample.py)
