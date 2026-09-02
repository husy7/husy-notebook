---
title: "随机森林：Bootstrap + 随机特征子集 + Bagging 降方差"
tags: ["Tree-Based", "随机森林", "Bagging", "集成学习", "OOB"]
date: 2026-08-29
---

# 随机森林（Random Forest）

## 定义

随机森林（Random Forest，RF）是一种基于决策树的**集成学习（ensemble learning）**方法，属于 **Bagging** 家族的代表：它对训练样本做 **Bootstrap（有放回抽样）** 生成 B 棵样本各异的**不剪枝 CART 树**，并在每棵树的每次划分时只从**随机特征子集**中挑选最优切分，最后用多数投票（分类）或取平均（回归）聚合所有树的输出。

它解决的核心问题是**单棵决策树方差过大**：树的划分对训练样本的小扰动极其敏感，样本稍变就会长出一棵完全不同的树。RF 通过「样本扰动（Bootstrap）+ 特征扰动（随机特征子集）」让树与树之间尽量**去相关**，再用投票/平均把方差稀释掉，从而把泛化误差压到比朴素 Bagging 更低的水平，而偏差几乎不变。

核心特征：① Bootstrap 有放回抽样（每棵树约只用 63.2% 的唯一样本）；② 每棵树不剪枝、偏置低；③ 划分时随机抽 `m_try` 个特征（分类默认 `sqrt(p)`、回归默认 `p/3`）；④ 树与树独立可并行训练；⑤ 未被抽中的约 36.8% 样本（**袋外 OOB**）可免费做无偏验证；⑥ 天然输出特征重要性（平均不纯度下降或 OOB 置换重要性）。

适用范畴：样本量足够、特征较多、含噪声、存在非线性交互的**表格型**分类/回归任务，几乎免调参即可获得强基线；可给出不确定性估计（各树投票比例 ≈ 置信度）。不适用/需慎用的场景：需要**外推**的回归任务（RF 是分片常数模型，预测值被训练目标范围框住）、超小样本（树间样本高度重叠，优势不明显）、类别极端不平衡且不做任何处理的分类任务。

## 原理

- **为什么是"平均"而不是"挑最好"**：设每棵树的方差为 σ²、两两相关系数为 ρ，则 B 棵树平均后的方差近似为 `ρσ² + (1−ρ)σ²/B`。B 再大也只能消灭第二项，**第一项由 ρ 决定**——所以降低方差的关键是让树彼此不相关，而不是无限加树。
- **Bootstrap 解决什么（样本扰动）**：每棵树从 N 个样本**有放回**抽取 N 个，树与树之间的输入有差异 → 部分去相关；同时每棵树约只用 63.2% 的唯一样本，剩余约 36.8% 从未参与该树训练，成为**袋外（OOB）**样本，可免费做无偏验证，不必再切验证集。
- **随机特征子集解决什么（特征扰动）**：若所有树都用全部特征，强特征会被每棵树抢着用，树长得高度相似（ρ 高）。分类默认每次划分只从 `sqrt(p)` 个特征里挑最优、回归用 `p/3` → 弱特征也有机会当根节点 → 树形态差异大 → ρ 下降。这是 RF 相对「Bagging + 全特征树」的关键改进，也是把方差压到比朴素 Bagging 更低的原因。
- **为什么偏差没涨**：单棵树不剪枝（偏置低），平均化不引入额外结构偏差；RF 的误差主要来自单树方差 + 不可约噪声，恰好被平均化稀释。因此「B 加大只降方差、不伤偏差」，不容易剧烈过拟合。
- **核心流程**：① 对 `b = 1…B`：从 N 个样本**有放回**抽取 N 个（Bootstrap）；② 在该样本集上长一棵**不剪枝**的 CART，但每次划分先随机抽 `m_try` 个特征再找最优切分；③ 聚合：分类多数投票 / 回归取平均；④ 用 OOB 样本评估（`oob_score=True`），特征重要性 = 平均不纯度下降（或 OOB 置换重要性）。
- **误差分解视角**：泛化误差 ≈ 噪声 + `ρσ²` + `(1−ρ)σ²/B`。RF 的本质是把单树方差拆成「相关项 + 独立项」：随机化（样本 + 特征双重扰动）压低 ρ，投票/平均压低独立项。这解释了所有调参与选型结论都围绕「降相关」而非「加树」。

## 应用

**典型使用场景**：结构化表格数据的分类/回归强基线；特征重要性排序与粗筛特征选择（须配合置换重要性）；需要不确定性估计（各树投票比例 ≈ 置信度）或并行训练的任务；数据量足够、特征多、含噪声、有非线性交互时通常显著优于单棵树；几乎免调参即可用，且不容易剧烈过拟合。

**快速上手步骤**（scikit-learn）：① 划分训练/测试集；② `RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42, n_jobs=-1)`（回归用 `RandomForestRegressor`）；③ `fit` 后先看 `oob_score_` 免费验证，再在测试集上评估，两者接近说明稳定；④ 需要选特征时用 `sklearn.inspection.permutation_importance`（放进交叉验证内），不要直接信 impurity 重要性；⑤ 调参顺序：`max_features` → `max_depth` / `min_samples_leaf` → `class_weight`，`n_estimators` 放最后。

**常见坑 / 注意事项**：
- ❌ 一上来狂调 `n_estimators` → B 增大收益边际递减（见原理中的方差分解公式），调参重点应是 `max_features`、`max_depth`、`min_samples_leaf`。
- ❌ 用 RF 的 impurity 重要性做特征选择 → 连续/高基数特征被系统性高估；✅ 换 `permutation_importance`（交叉验证内做）更稳。
- ❌ 忘了 `oob_score=True` 免费验证；OOB 与 K 折结果接近时说明模型稳定。
- ❌ 回归任务想外推 → RF 是分片常数，**不能外推**（预测值被训练目标范围框住）；✅ 外推场景改用线性模型（GBDT 同样不擅长外推，需谨慎）。
- ❌ 特征高度相关或类别极端不平衡时直接用默认参数 → 特征高度相关时 `max_features=1.0` 反而更合适；不平衡时配 `class_weight='balanced'`。
- ❌ 数据量小到树之间样本高度重叠 → RF 优势不明显，此时单树 + 好剪枝或正则化模型可能更优。

```python
# 随机森林最小实战示例（含注释 + 案例详解）
# 案例：合成 20 维含噪分类数据（8 个有效特征 + 4 个冗余特征），
#       对比「单棵不剪枝树」vs「随机森林」，演示 OOB 免费验证，
#       并用置换重要性做特征选择（比 impurity 重要性更稳）。
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# ① 数据：样本量足够、特征多、含噪声 → 随机森林的主场
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=8, n_redundant=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# ② 单棵不剪枝树作基线（方差大：样本小扰动就会换一棵树）
tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

# ③ 随机森林：Bootstrap + 随机特征子集 + 不剪枝 CART 聚合
#    - oob_score=True：用约 36.8% 的袋外样本免费验证，不用再切验证集
#    - 调参重点看 max_features（分类默认 sqrt(p)）/ max_depth /
#      min_samples_leaf，而不是盲目堆 n_estimators（B 再大也消不掉 ρσ² 项）
rf = RandomForestClassifier(n_estimators=200, oob_score=True,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("单棵树 acc :", round(accuracy_score(y_test, tree.predict(X_test)), 4))
print("随机森林acc:", round(accuracy_score(y_test, rf.predict(X_test)), 4))
print("OOB 分数   :", round(rf.oob_score_, 4))  # 应接近测试集 acc → 模型稳定

# ④ 特征选择：impurity 重要性会系统性高估连续/高基数特征，
#    更稳妥的做法是置换重要性（正式使用建议放进交叉验证内再做）
perm = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42)
print("置换重要性 Top-5 特征下标:", perm.importances_mean.argsort()[::-1][:5])

# 案例详解：若测试 acc ≈ OOB 分数，说明模型对未见数据表现稳定；
# Top-5 特征下标应基本落在前 8 个有效特征内。若发现高基数噪声
# 特征靠 impurity 重要性上榜，往往是其被系统性高估 —— 以置换
# 重要性结果为准做特征筛选。
```

---
## 关联
- 前置：[[决策树（CART）]]、[[Bootstrap 抽样]]、[[Bagging（Bootstrap 聚合）]]
- 类似：[[GBDT / Boosting]]（区别是：GBDT 用 Boosting 串行拟合残差（负梯度）、以降偏差为主，RF 用 Bagging 并行平均、以降方差为主——RF 是 Bagging 代表，GBDT 是 Boosting 代表）
- 类似：[[ExtraTrees]]（区别是：ExtraTrees 连切分点也随机抽取、不搜索最优阈值，方差更低、训练更快，但偏差略高）
- 进阶：[[孤立森林（Isolation Forest）]]（复用"树易分割孤立点"的特性做异常检测）、[[SHAP]]（树集成不可解释个体时做事后解释）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 随机森林（本文方案） | Bagging：Bootstrap 有放回抽样 + 随机特征子集 + 不剪枝 CART，投票/平均降方差 | 表格数据、特征多且含噪声、非线性交互强，需要免调参强基线 / 不确定性估计 / OOB 免费验证 |
| 单棵决策树（CART + 剪枝） | 贪心递归划分，单个模型 | 小样本、需要可解释性强的规则树 |
| ExtraTrees | 连切分阈值也全随机，进一步去相关 | 大样本追求更快训练与更低方差，可容忍略高偏差 |
| GBDT / XGBoost / LightGBM | Boosting：串行拟合残差（负梯度），以降偏差为主 | 追求最高精度、中小样本、排序与竞赛类结构化任务 |
| 线性模型（Logistic/Lasso/Ridge） | 线性假设 + 正则化 | 高维稀疏、线性可分、需要系数可解释或支持外推 |

---
## 参考
- [Breiman, Random Forests（2001 原始论文）](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [scikit-learn：RandomForestClassifier 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn：RandomForestRegressor 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [scikit-learn：集成方法用户指南（Bagging / 随机森林 / OOB）](https://scikit-learn.org/stable/modules/ensemble.html)

---
## 具体案例
- [[随机森林 实战示例]](随机森林_sample.py)
