---
title: "交叉验证与数据泄漏：K 折、分层与 Pipeline"
tags: ["Evaluation", "交叉验证", "数据泄漏", "StratifiedKFold", "Pipeline", "过拟合"]
date: 2026-08-29
---

# 交叉验证与数据泄漏：K 折、分层与 Pipeline

## 定义
交叉验证（Cross-Validation, CV）是一种**重复评估协议**：把有限数据切分成 K 份（折），轮流取其中 1 份做验证集、其余 K-1 份做训练集，重复 K 次后把 K 次分数取平均（±标准差）作为模型泛化能力的估计。它解决的核心问题是：单次 train/test split 在小样本下方差大、分数高低取决于"运气"（这次切分恰好分到什么样本），而 K 折让**每个样本都被验证恰好一次**，分数的均值更稳定、更可信。核心特征包括：分数以 mean±std 形式汇报、适用于模型选择与超参调优、可通过分层（StratifiedKFold）应对类别不平衡。适用范畴：中小规模数据上的分类/回归评估、GridSearchCV 等搜索器的内置评估器、以及一切需要"诚实估计泛化误差"的场景。**但交叉验证只有在"每个预处理步骤都只在训练折内 fit"时才是诚实的**——一旦缩放、特征选择、过采样等用了全体数据，测试信息就"泄漏"进训练，分数虚高、上线即翻车，这是本知识点的另一半主题。

## 原理
为什么采用"重复切分求平均"而不是单次切分：小样本下一次切分的分数方差大（运气成分），K 折让每个样本都被验证一次，取平均更稳。为什么 K 通常取 5 或 10：折太少 → 每折训练集太小、偏差大；折太多 → 各折训练集彼此太像、方差降不下来且计算更贵，5/10 是经验甜点。分层机制：分类不平衡时普通 KFold 可能让某一折恰好没有少数类样本，导致该折指标爆炸；**StratifiedKFold 保证每折类别比例 ≈ 总体比例**，因此只用于分类问题。组结构机制：同一"用户/会话/病历"的多行样本若被随机切开，会**双双出现在训练集与验证集**——模型在验证时见过该组的"答案亲戚"，分数虚高；**GroupKFold 保证整组样本不出现在两侧**。正确姿势的机制：**一个 `Pipeline` 包住全部预处理 + 模型**，再交给 `cross_val_score`/`GridSearchCV`——框架在每一折内部先对该折训练部分 fit 预处理（scaler/selector/encoder），再 transform 训练与验证折，从而从机制上保证每个 fit 都在折内发生。泄漏为何是"静默杀手"：泄漏不报错，只让分数变好看；且常在预处理处发生（缩放器、特征选择、缺失填充、SMOTE、编码器），因为它们"看起来"不是模型，最容易被忽略；其中特征选择/降维（SelectKBest/PCA）在 CV 外先做属于最严重的 "feature selection leak"（用全体数据挑特征 = 偷看验证集标签排序）。还需注意目标泄漏（特征里混着标签的未来/派生信息）任何 CV 都救不了；以及"实践泄漏"——部署后在线拿不到的特征，本质上也是泄漏思想在工程侧的延伸。

## 应用
典型使用场景：小样本模型的泛化误差估计、超参搜索（GridSearchCV/RandomizedSearchCV 内置分层 K 折）、模型对比选型、以及用 `cross_validate` 一次返回多个评估指标。快速上手步骤：① 把**全部预处理（缩放/编码/特征选择/降维/SMOTE）+ 模型**打包进一个 `Pipeline`；② 分类问题用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，含组结构用 `GroupKFold`，时间序列用 `TimeSeriesSplit`；③ 调用 `cross_val_score(pipeline, X, y, cv=skf)` 得到 mean±std；④ 最终上线评估用**从未见过的保留测试集**，模型选择用嵌套交叉验证。常见坑（务必逐条对照）：
- ❌ `scaler = StandardScaler().fit(X_all)` 后再 `cross_val_score` → 验证折的均值/方差混进了缩放，分数被污染。✅ `Pipeline([("sc", StandardScaler()), ("m", Model())])` 交给 CV。
- ❌ 特征选择/降维（SelectKBest/PCA）在 CV **外**先做 → 用全体数据挑特征 = 偷看验证集标签排序，虚高最严重（"feature selection leak"）。
- ❌ 过采样 SMOTE 在切分**前**做 → 合成样本跨 train/valid 复制，验证集被训练样本渗透。✅ 只在训练折内合成。
- ❌ 分类数据随手 `KFold` 不分层 → 某折缺类、指标爆炸。✅ 用 `StratifiedKFold`。
- ❌ 同一份数据先当测试集反复调参，最后又拿它"测" → 测试集被你训练过了（数据用完不复用）。✅ 最终评估用**从未见过的保留测试集**；模型选择用嵌套交叉验证。
- ❌ 时间序列数据随机 K 折 → 未来样本泄漏进训练。✅ 用 `TimeSeriesSplit`（只允许用过去预测未来）。
- ❌ 目标泄漏：特征里混着标签的"未来/派生"信息（如用"是否已购买"预测"是否购买"）→ 任何 CV 都救不了，先做特征溯源审计。

```python
# ============================================================
# 案例：比较"泄漏版"与"干净版"交叉验证的分数差异
# 任务：200 个样本、类别不平衡的二分类（方便体现 StratifiedKFold 的意义）
# ============================================================
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (cross_val_score, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 造一个不平衡的小数据集：少数类只占 ~10%
X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                           weights=[0.9, 0.1], random_state=42)

# ---------- ❌ 泄漏版：scaler 先在"全体数据"上 fit ----------
# 泄漏点：验证折的均值/方差混进了缩放参数，等价于验证信息进入了训练流程
scaler = StandardScaler().fit(X)          # fit(X_all) —— 这里是错的
X_scaled = scaler.transform(X)
leak_scores = cross_val_score(LogisticRegression(max_iter=1000),
                              X_scaled, y, cv=5)
print(f"泄漏版分数（虚高）: {leak_scores.mean():.4f}")

# ---------- ✅ 干净版：Pipeline + 分层 K 折 ----------
# cross_val_score 在每一折内部：只对该折训练部分 fit 缩放器，
# 再用它 transform 训练与验证折 —— 每个预处理 fit 都发生在折内
pipe = make_pipeline(StandardScaler(),          # 缩放器进 Pipeline
                     LogisticRegression(max_iter=1000))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clean_scores = cross_val_score(pipe, X, y, cv=skf)   # 分层 + 折内预处理
print(f"干净版分数（可信）: {clean_scores.mean():.4f} ± {clean_scores.std():.4f}")

# 结论：泄漏版分数明显高于干净版。干净版才代表真实泛化能力，
# 若拿泄漏版分数做模型选型，上线后实际表现会"跳水"。
```

---
## 关联
- 前置：[[训练集-验证集-测试集 三分法]]——三分法中"验证集"的角色在 K 折里由各折轮流担任，K 折本质上是它的重复版。
- 类似：[[KFold]]（区别是 StratifiedKFold 按类别比例分层切分，保证每折少数类不缺席、指标不爆炸，只用于分类）
- 类似：[[GroupKFold]]（区别是 KFold/StratifiedKFold 按样本行切分，GroupKFold 保证同一组——用户/会话/病历——整组只出现在训练或验证一侧）
- 类似：[[TimeSeriesSplit]]（区别是它按时间顺序只允许"用过去预测未来"，而随机 K 折会让未来样本泄漏进训练）
- 进阶：[[嵌套交叉验证]]——外层折做最终评估、内层折做模型选择，避免把测试集当调参集反复使用（数据用完不复用）。
- 进阶：[[GridSearchCV]] / [[RandomizedSearchCV]]——内置（分层）K 折做超参搜索；[[cross_validate]] 一次返回多个指标。

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：Pipeline + StratifiedKFold 交叉验证 | 预处理与模型整体打包成 Pipeline 交给 CV：每一折内只对训练折 fit 预处理，K 次评分取均值±std，无泄漏且估计稳定 | 中小规模数据的模型选择、超参调优、泛化误差估计（分类/回归通用） |
| 替代方案：单次 train/test split | 一次随机切分，训练集训练、测试集打分，只评估一轮 | 数据量大、只需一次粗估；作为 CV 之后的"从未见过"最终保留集 |
| 替代方案：嵌套交叉验证 | 外层折做最终评估、内层折做模型选择，两层数据互不重叠 | 同一份数据既要选模型又要诚实评估，防"测试集被训练过" |
| 替代方案：TimeSeriesSplit | 按时间顺序切分，训练集只含过去、验证集在未来 | 时间序列/面板数据，防止未来信息泄漏进训练 |

---
## 参考
- [scikit-learn: Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)

---
## 具体案例
- [[交叉验证与数据泄漏 实战示例]](交叉验证与数据泄漏_sample.py)
