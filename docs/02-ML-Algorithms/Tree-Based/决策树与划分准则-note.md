---
title: "决策树与划分准则：信息增益 / 增益率 / 基尼"
tags: ["Tree-Based", "决策树", "信息增益", "基尼", "CART", "剪枝"]
date: 2026-08-29
---

# 决策树与划分准则：信息增益 / 增益率 / 基尼

## 定义

决策树是一类**监督学习模型**（分类、回归皆可）：输出一棵树状结构，内部节点是"某个特征 + 某个阈值/取值"的判断，分支是该判断的结果，每个叶子给出一个常数预测（分类取多数类、回归取均值）。整棵树等价于一组互斥的 if-then 规则，把输入空间递归切成**互斥的轴对齐超矩形区域**（axis-aligned），天然处理非线性与特征交互，且预测结果可直接读成规则（可解释性）。

它的构建方式是**贪心的递归划分**（recursive partitioning）：每一步在当前节点挑选"某个特征 + 某个切分点"，把样本分成两组（或多组），使划分后子节点比父节点"更纯"；"纯不纯"由**划分准则**（splitting criterion）度量，树就一直切到叶子纯 / 样本太少 / 达到最大深度为止。因为每一步只看局部最优，它属于启发式而非全局优化（全局最优划分树是 NP 难问题）。

sklearn 里实现的树是 **CART 风格**：永远二叉、用基尼（分类默认）或熵（log-loss）或 MSE（回归）做准则、不原生支持类别特征、靠剪枝参数控制复杂度。适用范围：中小规模表格型数据的分类/回归基线、对规则可解释性有要求的场景（如风控、医疗规则提取）、特征重要性初筛，以及作为随机森林 / GBDT / XGBoost 的**基学习器**——工业高精度场景几乎总在它之上叠加集成。

## 原理

**核心机制：纯度下降最大化的贪心递归。** 每次分裂都找一个"让子节点不纯度下降最多"的特征与切点，整体目标是把标签打散程度尽量压低：熵与基尼都只依赖各类别比例 $p_k$，对类别分布"越偏"（如全一类）不纯度越小。

**熵（信息熵）**：$Ent(D) = -\sum_{k=1}^{K} p_k \log_2 p_k$，含义是把类别编码到最优时的期望码长，值越小越纯；$p_k=0$ 的项计 0。

**信息增益（ID3）**：$Gain(D,a) = Ent(D) - \sum_{v=1}^{V} \frac{|D^v|}{|D|} Ent(D^v)$，即"父节点熵 − 按特征 $a$ 划分后各子节点熵的加权和"，减得越多越好。**关键缺陷：偏爱取值多的特征**——取值越多切得越细、子集越"纯"，增益虚高、易过拟合，所以 ID 类特征不能直接喂给 ID3。

**增益率（C4.5）**：$GainRatio(D,a) = \frac{Gain(D,a)}{IV(a)}$，用固有值 $IV(a) = -\sum_{v=1}^{V} \frac{|D^v|}{|D|} \log_2 \frac{|D^v|}{|D|}$ 惩罚多取值特征；但它反过来偏爱取值少的特征，C4.5 的折中做法是**先从信息增益高于平均水平的特征里，再挑增益率最大者**。

**基尼指数（CART）**：$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$，直观含义是"随机抽两个样本、其类别不一致的概率"。划分后取子节点加权基尼，**越小越好**。无需对数运算、计算快，是 sklearn 分类默认准则。CART 强制二叉：连续特征对取值排序后扫描切点做阈值二分；对多取值类别/序数特征会自动化地合并取值找最优二分，但 sklearn 仍需先做数值编码。

**回归树与 sklearn 细节**：回归用 MSE（方差）下降度量，叶子预测取均值；sklearn 的 `criterion='entropy'` 实为对数损失（log_loss），与 gini 效果通常非常接近——不要在两者上花太多调参时间。

**剪枝（控过拟合的核心）**：树的深度/叶子数 ≈ 模型容量，剪枝 ≈ 正则化。预剪枝在建树时截断（`max_depth`、`min_samples_leaf`、`min_samples_split`）；后剪枝用成本复杂度剪枝：sklearn 提供 `cost_complexity_pruning_path` 生成 α 路径，再按 `ccp_alpha` 剪掉"复杂度上升换不来足够纯度收益"的分支，α 通常用 AUC/准确率的交叉验证来选。

## 应用

**典型使用场景**：中小型表格数据的可解释基线（先跑一棵树看规则再上集成）；风控审批、医疗诊断等需要向人解释"为什么判这一类"的场景；用单棵树的 `feature_importances_` 做特征初筛；以及作为随机森林 / GBDT 的基学习器组件。

**快速上手步骤**：① 把数据转成数值：类别特征 OneHot/Ordinal 编码（sklearn CART 只吃数值），剔除 ID 列；② 划分 train/test；③ 先不调参跑一版，然后用 `cost_complexity_pruning_path` 或直接设 `max_depth`/`min_samples_leaf` 控过拟合；④ fit 后可用 `export_text`/`plot_tree` 读规则、用 `feature_importances_` 看变量排序；⑤ 用准确率 / AUC / MSE 对比 gini 与 entropy、对比剪枝前后，确认没有过拟合。

**常见坑**：
- ❌ 不给任何剪枝约束 → 树深到每个叶子一个样本，训练误差 0、测试崩盘。✅ 默认就设 `max_depth`/`min_samples_leaf`，或用 `ccp_alpha` 后剪枝（换 AUC/准确率做选择）。
- ❌ 把类别特征/ID 列直接塞进 sklearn 树 → CART 只吃数值，ID 列按数值排序会产生无意义切分。✅ OneHot/Ordinal 编码，或换支持原生类别的库（CatBoost/LightGBM）。
- ❌ 样本类别极不平衡 → 树偏袒多数类。✅ 调 `class_weight='balanced'` 或重采样。
- ❌ 把单棵树的 `feature_importances_` 当严谨依据 → 高基数/连续特征被系统性高估，且单棵树随机波动大。
- ❌ 以为决策树不用预处理就万事大吉 → 对缩放确实鲁棒，但仍怕高基数类别、类别不平衡、噪声标签。

```python
# -*- coding: utf-8 -*-
# 决策树与划分准则：sklearn CART 最小可运行示例
# 覆盖：gini vs entropy 对比、预剪枝控过拟合、ccp_alpha 后剪枝
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

# 1) 裸树（无剪枝）：训练 1.0、测试明显下降 → 过拟合信号
bare = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
print("无剪枝 train=%.3f test=%.3f" % (
    accuracy_score(ytr, bare.predict(Xtr)), accuracy_score(yte, bare.predict(Xte))))

# 2) 预剪枝：max_depth/min_samples_leaf 在建树时截断，是默认该做的事
pre = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(Xtr, ytr)
print("预剪枝 train=%.3f test=%.3f" % (
    accuracy_score(ytr, pre.predict(Xtr)), accuracy_score(yte, pre.predict(Xte))))

# 3) 后剪枝：cost_complexity_pruning_path 生成 α 路径，再交叉验证选 ccp_alpha
path = bare.cost_complexity_pruning_path(Xtr, ytr)
ccp_alphas = path.ccp_alphas
scores = [cross_val_score(DecisionTreeClassifier(random_state=0, ccp_alpha=a),
                          X, y, cv=5).mean() for a in ccp_alphas]
best_alpha = ccp_alphas[max(range(len(scores)), key=scores.__getitem__)]
post = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha).fit(Xtr, ytr)
print("后剪枝 ccp_alpha=%.4f test=%.3f" % (best_alpha, accuracy_score(yte, post.predict(Xte))))

# 4) gini 与 entropy(log_loss) 通常几乎无差 → 别在二者上过度调参
e = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0).fit(Xtr, ytr)
print("entropy test=%.3f vs gini test=%.3f" % (
    accuracy_score(yte, e.predict(Xte)), accuracy_score(yte, pre.predict(Xte))))

print(export_text(pre, feature_names=load_iris().feature_names))
# 案例详解：
# 步骤 1 展示"不剪枝 → 训练满分、测试塌方"的典型过拟合；
# 步骤 2 用 max_depth/min_samples_leaf 预剪枝，测试分明显回升，是实战默认动作；
# 步骤 3 演示后剪枝流程——先求出 α 序列，再用交叉验证挑使 CV 分数最大的 ccp_alpha；
# 步骤 4 验证"划分准则选 gini 还是 entropy 差别很小"，把调参预算留给剪枝与特征工程；
# export_text 输出可读 if-then 规则，体现树的可解释性。完整版见同目录 sample.py。
```

---
## 关联
- 前置：[[树模型到梯度提升]]（树模型总览：熵定义与三种准则速览表、单树 → Bagging → Boosting 的整体地图，先读它建立坐标系）
- 类似：[[随机森林-note]]（区别是：本文解决"单棵树如何选特征/切分、如何控过拟合"的划分准则问题；随机森林解决"单树高方差"，用 bootstrap 重采样 + 随机特征子集并行集成多棵 CART 树投票，准则本身仍是 gini/entropy）
- 进阶：[[GBDT与XGBoost-note]]（在单树基础上做 boosting：逐棵拟合负梯度降偏差，XGBoost 进一步引入正则化、二阶导与近似分裂，是本文树的工业级演进）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：CART 决策树（sklearn，gini/entropy） | 二叉递归划分，加权基尼/熵下降最大；预剪枝 + ccp_alpha 后剪枝控复杂度 | 中小表格数据可解释单树基线、特征重要性初筛、作为 RF/GBDT 基学习器 |
| ID3（信息增益） | 熵减最大选特征、可多叉；无剪枝、偏好多取值特征 | 教学与理解"划分准则"概念；不适合真实数据（易过拟合） |
| C4.5（增益率） | 增益率 + 固有值惩罚多取值，支持连续值/缺失值、多叉 | 理论对比与课程脉络；工程上已被 sklearn CART / 集成模型取代 |
| 集成替代（随机森林 / XGBoost / LightGBM） | 多棵树 bagging 降方差或 boosting 降偏差，自带正则与原生类别支持 | 追求精度的大规模表格竞赛/工业场景；可解释性要求较低 |

---
## 参考
- [Decision Trees — scikit-learn 官方文档](https://scikit-learn.org/stable/modules/tree.html)
- [cost_complexity_pruning_path — scikit-learn API](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.cost_complexity_pruning_path)

---
## 具体案例
- [[决策树与划分准则 实战示例]](决策树与划分准则_sample.py)：手算基尼/熵理解"不纯度下降"、鸢尾花上 gini vs entropy 几乎无差、预剪枝（max_depth/min_samples_leaf）控过拟合、cost_complexity_pruning_path + ccp_alpha 后剪枝、特征重要性与树规则解读。
