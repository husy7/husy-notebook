---
title: "itertools 迭代工具"
tags: [Python, itertools, 迭代器, 链式工具]
date: 2026-08-29
---

# itertools 迭代工具

## 定义

itertools 是 Python 标准库中把迭代器组合成"高效、惰性、内存友好"处理链的工具集，与 functools（高阶函数）同属"函数式编程工具"父主题，本篇专攻 itertools 部分。它的核心特征是：**所有工具都返回新的迭代器（惰性求值），绝不在内存中一次性装下全部中间结果**，因此能流式处理超大甚至无限的数据集。按用途可粗分为三大类：

| 类别 | 典型函数 | 作用 |
|------|----------|------|
| **无限迭代** | `count` `cycle` `repeat` | 无限递增值 / 无限循环 / 重复值 |
| **缩短迭代**（近终止） | `takewhile` `dropwhile` `compress` | 按条件截断/过滤流 |
| **组合/排列** | `product` `permutations` `combinations` | 笛卡尔积、排列、组合 |

其余一批"组合器"包括 `chain`、`islice`、`groupby`、`tee`、`accumulate`、`zip_longest`、`starmap` 等，用于把一个或多个迭代器拼装、切片、分组、克隆、累积，形成可无限延长的处理管道。

## 原理

itertools 之所以"高效且惰性"，根因在于它严格遵循迭代器协议：函数内部不构造完整的序列，而是返回一个实现了 `__iter__`/`__next__` 的迭代器对象，每个元素都是"按需（on demand）逐项拉取"的。上游只产出一个值，下游才消费一个值，中间产物随即释放，内存占用恒定在 O(1) 量级（tee 的内部缓冲除外）。几个关键机制：`chain` 逐个委托子迭代器直到耗尽再切换下一个，等价于把"一堆来源"摊成单条流；`islice` 用 C 层计数器跳过前 N 项、取到 stop 即停，因此能对不支持下标切片（`it[5:10]`）的迭代器/无限序列做"切片"；`accumulate` 是左折叠的"前缀版"，逐项累加并产出每一步中间和（如 `[1,2,3,4] → [1,3,6,10]`），默认是加法，可换任意二元函数；`groupby` 只把**相邻且键相同**的项聚成一组，其分组正确性依赖输入有序，要全局分组必须先排序；`tee` 用一个源头在内部做缓冲，克隆出 n 个可独立二次遍历的迭代器——克隆本身不复制数据，代价是缓冲积压。由于函数体在 C 层实现且免去中间列表分配，链条比等价的"生成器 + list"写法更快更省内存。

## 应用

典型使用场景：① 需要流式处理超大/无限数据集（日志流、传感器读数、`range(10_000_000)` 级别序列）而不想整段装入内存；② 拼接多来源数据为单一流（省去重复循环或手动 concat）；③ 对仅可单向遍历的迭代器做切片、分组、克隆等本无法直接完成的操作；④ 枚举笛卡尔积/排列/组合做暴力搜索或测试用例生成。快速上手：先 `import itertools as it`，按需选用 `it.chain/islice/groupby/tee/accumulate`，对组合数学需求选 `product/permutations/combinations`；n 较小时可用 `list(...)` 查看结果，n 较大时务必改用 `for` 流式消费或用 `islice` 取前 N。常见坑：❌ `groupby` 前不排序 → 相同元素被拆成多组，应 `sorted(iterable, key=...)` 后再 `groupby`；❌ 对 `product/permutations` 结果直接 `list(...)` → 阶乘级内存/算力爆炸；❌ 想切迭代器却写 `it[5:10]` → 必须用 `itertools.islice`；❌ `tee` 出的多个迭代器消费速度悬殊时，慢的一侧会让源头缓冲持续积压内存（本质仍是惰性缓冲，点到为止）；⚠️ 想用 `while`+`next` 手写无限流时，优先用 `count/cycle` 更稳。

```python
import itertools as it

# 1. chain：串联多个可迭代对象，摊成单条流
print(list(it.chain([1, 2], (3, 4), "ab")))
# [1, 2, 3, 4, 'a', 'b']

# 2. islice：惰性切片，不真正下标、不吃全部内存
print(list(it.islice(range(10_000_000), 5, 10)))
# [5, 6, 7, 8, 9]  —— 只流式取索引 5..9

# 3. groupby：按"连续键"分组，混序前必须先 sorted
for key, group in it.groupby("AAABBBAA"):
    print(key, list(group))
# A ['A','A','A'] → B ['B','B','B'] → A ['A','A']  （A 被拆成两组！）

# 4. tee：克隆单个迭代器为 n 个独立可二次遍历的流
a, b = it.tee([1, 2, 3], n=2)
print(list(a), list(b))          # [1, 2, 3] [1, 2, 3]

# 5. accumulate：累积和/前缀运算（可换任意二元函数）
print(list(it.accumulate([1, 2, 3, 4])))      # [1, 3, 6, 10]

# 6. 组合数学三兄弟：全部惰性，n 大时勿 list 全量
list(it.permutations("ABC", 2))   # 有序排列：[('A','B'), ('A','C'), ('B','A'), ...]
list(it.combinations("ABC", 2))   # 无序组合：[('A','B'), ('A','C'), ('B','C')]
list(it.product([1, 2], [3, 4]))  # 笛卡尔积：[(1,3), (1,4), (2,3), (2,4)]
```

案例详解：第 1 条把 list/tuple/字符串三种不同来源直接拼成一个列表，省去手写三层循环；第 2 条对千万级 `range` 只拉取 5 个元素，内存恒定 O(1)，这正是"迭代器切片"相对 `it[5:10]` 的价值；第 3 条演示 groupby 的**连续**语义——同样的字符 A 因不相邻被分成两组，故全局分组前必须 `sorted`；第 4 条说明 tee 是"一次源头、两条支流"，两条流互不影响且都可完整遍历；第 5 条可换 `it.accumulate(xs, operator.mul)` 得到前缀积；第 6 条三兄弟全返回惰性迭代器，`permutations("ABC", 2)` 在乎顺序（AB≠BA），`combinations` 不在乎（AB=BA），`product` 是两集合的笛卡尔积，规模随输入阶乘级增长，消费时务必用 `for` 流式或 `islice` 限量。

---
## 关联
- 前置：[[迭代器与生成器协议]]
- 类似：[[functools 函数式编程工具]]（区别是 functools 组合"函数/高阶函数"（reduce/lru_cache/partial），itertools 组合"迭代器流"，二者常互补出现）
- 类似：内建 `zip/map/filter`（区别是它们也是惰性单工具，但缺少 chain/groupby/tee 这类把多个迭代器拼装、克隆的组合能力）
- 进阶：[[官方 Itertools Recipes]]

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（itertools 惰性链） | 所有工具返回新迭代器、逐项拉取，内存 O(1)，无限流可组合成处理管道 | 超大/无限数据集、需要切片/分组/克隆/排列组合的链式流处理 |
| 列表推导 / 一次性 list 全量 | 先完整算出中间序列再消费，代码直观、可下标随机访问 | 数据规模小（可整体驻留内存）且需要多次随机访问时 |
| 手写生成器 + while/for 循环 | 用 `yield` 手工控制每个拉取步骤，逻辑完全自定 | 无法用 itertools 现成组合器表达的定制拉取逻辑，或教学演示 |

---
## 参考
- [itertools — Functions creating iterators for efficient looping（官方文档）](https://docs.python.org/3/library/itertools.html)
- [Itertools Recipes（官方"菜谱"，同页末尾，含大量实用组合写法）](https://docs.python.org/3/library/itertools.html#itertools-recipes)

---
## 具体案例
- [[itertools 迭代工具 实战示例]](itertools 迭代工具_sample.py)
