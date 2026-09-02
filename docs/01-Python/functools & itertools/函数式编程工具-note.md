---
title: "函数式编程工具（functools & itertools）"
tags: [Python, 函数式编程, functools, itertools, 惰性求值, 缓存]
date: 2026-08-27
---

# 函数式编程工具（functools & itertools）

## 定义

它是什么：`functools` 与 `itertools` 是 Python 标准库中两个互补的「函数式编程工具箱」。`functools` 是"给函数加装备"的工具箱，基于高阶函数思想提供 `partial`（冻结参数）、`lru_cache`（结果缓存 / 记忆化）、`wraps`（装饰器元信息保留）、`singledispatch`（按第一参数类型分发）、`reduce`（序列折叠）等工具；`itertools` 是"给迭代加流水线"的工具箱，基于惰性迭代器提供 `chain`（拼接）、`islice`（切片截断）、`groupby`（相邻分组）、`product`（笛卡尔积）、`count` / `cycle` / `repeat`（无限流）、`accumulate` / `zip_longest` / `takewhile` 等标准化积木。

解决什么实际问题：手写循环处理序列时代码冗长、易错，且常重复造轮子（分组、累积、笛卡尔积、去重缓存）；纯函数式风格又缺少"记忆化""偏应用"这类高阶手段，导致递归逻辑重复计算、参数重复传入、性能差、状态管理散落各处。

核心特征：① 声明式——用几行组合代码替代手写循环与显式临时变量；② 惰性——`itertools` 的所有函数都返回迭代器，只在被消费时逐个产出元素，天然低内存；③ 低侵入——`functools` 的缓存、参数固化以装饰器 / 工厂形式封装，对调用方透明；④ 正交解耦——"数据怎么产生"（itertools 管道）与"逻辑怎么复用"（functools 高阶工具）彼此独立、可自由组合。

适用范畴：数据流 / 管道级联处理、递归 / 重复计算的缓存加速、回调与接口参数预填充、序列去重与分组、多序列组合与配对遍历、日志按时间窗聚合、无限序列取前 N 项。一句话概括：`functools` 解决"函数复用与性能"（参数固化、缓存、类型分发），`itertools` 解决"序列处理代码冗长"（标准化的惰性迭代积木）。

## 原理

`functools` 的核心机制是**高阶函数（higher-order function）**：Python 中函数是一等公民，可以被接收、返回并当作对象操作——`partial` 通过冻结部分参数生成新函数（偏应用），其实现等价于把原函数与预置参数打包进一个新的可调用对象；`lru_cache` 在装饰器层用哈希表（dict + LRU 链表）对"参数 → 返回值"做记忆化，内部按 `cache_info()` 统计命中 / 未命中，因为结果缓存只封装在装饰器层，才能对调用方完全透明、不侵入业务代码；`wraps` 实质是把被装饰函数的 `__name__`、`__doc__`、`__module__` 等元信息复制到 wrapper 上，弥补"手动装饰器丢元信息"的缺陷；`singledispatch` 内部维护一张按第一参数类型索引的分发表，注册新实现即向表中添加条目；`reduce` 则把二元函数反复作用到序列上折叠成单值。

`itertools` 的核心机制是**惰性求值与迭代协议（lazy evaluation & iteration protocol）**：所有函数都返回迭代器对象而非具体容器，元素在被 `next()` / `for` / `list()` 消费时才逐个产出，且整体由 C 语言实现，组合多层管道时开销极低。各积木对应固定语义：`islice` 按位置切片截断、`takewhile` 取到第一个不满足条件的元素即停、`groupby` 只归并**相邻**且 key 相同的元素、`product` 等价于嵌套 for 的多重笛卡尔积、`accumulate` 是 `reduce` 的逐步版本（保留每步中间值）、`zip_longest` 按最长序列配对并以 `fillvalue` 补位、`zip` 按最短者终止。

为什么必须这样设计：函数一等公民 + 迭代器是一次性流，两者组合使"数据产生方式"与"逻辑复用方式"正交解耦，可各自独立演进再自由拼接；对大数据集与无限序列，惰性按需产出是唯一可行的内存方案（整体物化必然卡死或耗尽内存）；而记忆化要生效就必须捕获"参数 → 结果"映射，封装在装饰器层是唯一不污染业务逻辑的做法。局限随之而来：迭代器一次性、不支持索引 / 切片 / `len()`，需重复消费时用 `tee()` 或重新生成。

## 应用

典型使用场景：递归（如斐波那契、动态规划）用 `@lru_cache` 从指数级提速到线性级；回调 / 接口参数预填充用 `partial`；序列去重分组、多序列组合遍历用 `itertools` 积木；日志按时间窗聚合用 `groupby`；数据管道级联处理用"惰性管道 + 末端一次消费"模式。

快速上手步骤：① 从 `itertools` 挑"产数据"的工具（`chain`、`islice`、`groupby`、`product`…）搭出惰性管道；② 从 `functools` 挑"改函数"的工具（`partial`、`lru_cache`、`singledispatch`…）装饰或派生函数；③ 用 `list()` / `for` 在最末端一次性消费整条管道，享受低内存与声明式风格。

常见坑与注意事项：
- ❌ 对无限迭代器（`count` / `cycle` / `repeat`）直接 `list()`，程序卡死不返回。✅ 必须配合 `islice(n)`、`takewhile`、`zip`（以最短者终止）等截断手段消费，永远不要对其整体物化。
- ❌ 对未排序序列使用 `groupby`，期望"全局按 key 分组"，实际得到大量重复碎组（如 `[1,1,2,1]` 产出 1 组与 2 组各两份）。✅ `groupby` 只对相邻相同 key 分组，使用前先 `groupby(sorted(data, key=...), key=...)`，或改用字典手动归集。
- ❌ `@lru_cache` 装饰参数为列表 / 字典等可变对象的函数，报 `TypeError: unhashable type`；或缓存了有副作用 / 依赖外部状态的函数，后续调用拿到过期结果。✅ `lru_cache` 以参数为哈希键，仅适用纯函数且参数可哈希；含不可哈希参数时先转元组 / 字符串，需要失效时用 `cache_clear()`。
- ❌ 装饰器忘加 `@wraps`，导致被装饰函数 `__name__` 变成 `wrapper`、文档字符串丢失，调试与 `help()` 输出错乱。✅ 手写装饰器始终在 wrapper 外套 `@functools.wraps(f)`。
- 边界条件：`itertools` 迭代器一次性、不支持索引 / 切片 / `len()`，重复消费需 `tee()` 或重新生成；`partial` 生成的函数签名元数据会变化（用 `inspect.signature` 查看），传给强类型校验框架时需注意；`groupby` 返回的组迭代器在前进到下一组后即失效，必须在循环体内立即消费（如转 `list`）。

```python
# 可运行示例（已实测）：functools 高阶工具 + itertools 惰性积木
from functools import partial, reduce, lru_cache, wraps, singledispatch
from itertools import count, cycle, islice, chain, groupby, product, \
                       zip_longest, accumulate, takewhile

# ---------- functools：给函数加装备 ----------
# 1) partial：固定参数，派生新函数（偏应用）
pow2 = partial(pow, exp=2)
print(pow2(3))                       # 9
# 2) reduce：把序列折叠成单值
print(reduce(lambda a, b: a + b, [1, 2, 3, 4]))   # 10
# 3) lru_cache：记忆化递归，指数级 → 线性级
@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
print(fib(50))                       # 12586269025（未缓存会卡到天荒地老）
print(fib.cache_info())              # 查看命中/未命中统计
# 4) wraps：装饰器不丢元信息（__name__/__doc__）
def deco(f):
    @wraps(f)
    def wrapper(*a, **k):
        return f(*a, **k)
    return wrapper
@deco
def hello():
    """say hello"""
print(hello.__name__, hello.__doc__)  # hello say hello（不加 wraps 则输出 wrapper None）
# 5) singledispatch：按第一参数类型分发（泛型函数）
@singledispatch
def size(obj):
    return "unknown"
@size.register
def _(obj: str):
    return len(obj)
@size.register
def _(obj: list):
    return sum(1 for _ in obj)
print(size("hello"), size([1, 2, 3]), size(3.14))  # 5 3 unknown

# ---------- itertools：给迭代加流水线 ----------
# 6) 无限迭代器：必须配合 islice 等截断工具消费
print(list(islice(count(10, 5), 4)))          # [10, 15, 20, 25]
print(list(islice(cycle("AB"), 5)))           # ['A','B','A','B','A']
# 7) 拼接与配对
print(list(chain([1, 2], [3, 4])))            # [1, 2, 3, 4]
print(list(zip_longest([1, 2, 3], "AB", fillvalue="?")))
                                               # [(1,'A'),(2,'B'),(3,'?')]
# 8) 折叠式迭代：累积（保留每步中间值）
print(list(accumulate([1, 2, 3, 4])))         # [1, 3, 6, 10]
# 9) 条件截取：取到第一个不满足处即停（可安全用于无限流）
print(list(takewhile(lambda x: x < 3, [1, 2, 3, 1, 2])))  # [1, 2]
# 10) groupby：相邻分组（注意：只对连续相同元素分组）
print([(k, list(g)) for k, g in groupby("AAABBC")])
                                               # [('A',['A','A','A']),('B',['B','B']),('C',['C'])]
# 11) 笛卡尔积：替代多层嵌套循环
print(list(product("AB", repeat=2)))          # [('A','A'),('A','B'),('B','A'),('B','B')]

# 案例详解：最小可用范式——"截断 + 折叠 + 分组"三板斧
#   list(islice(count(1), 5))            → [1,2,3,4,5]（无限流取前 5）
#   @lru_cache 加在递归函数上           → 即完成记忆化，fib(50) 秒出
#   groupby(sorted(data), key=...)       → 全局分组而非相邻碎组
```

---
## 关联
- 前置：[[Python生成器与迭代器]]、[[Python函数基础]]、[[闭包与装饰器]]
- 类似：[[Python生成器与迭代器]]（区别是生成器是自己写 `yield` 产数据，itertools 是现成的标准化迭代器积木，且底层为 C 实现）；[[列表推导式]]（区别是推导式立即求值整体占内存，itertools 惰性求值逐个产出省内存）
- 进阶：`operator` 模块（`itemgetter` / `attrgetter` 替代 lambda）、`functools.cached_property`、`functools.total_ordering`、`functools.cmp_to_key`、`itertools.pairwise`（3.10+）、`itertools.batched`（3.12）、`more_itertools` 第三方扩展库

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| functools + itertools（本文方案） | 声明式组合：高阶函数改函数 + 惰性迭代器搭管道 | 数据流处理、递归缓存、序列组合/分组，追求简洁与低内存 |
| 手写循环 / 生成器 | 用 `for` + `while` + 临时变量显式管理迭代状态 | 逻辑特殊、itertools 无对应积木、需要中途 break/复杂分支控制 |
| 第三方库（more_itertools / toolz） | 扩展更多迭代器原语（chunked、unique_everseen 等） | 标准库工具不够用、可引入第三方依赖的项目 |

---
## 参考
- [Python 官方文档：functools — 高阶函数与可调用对象操作](https://docs.python.org/3/library/functools.html)
- [Python 官方文档：itertools — 为高效循环而创建的迭代器函数](https://docs.python.org/3/library/itertools.html)
- [Python 官方 HOWTO：Functional Programming Modules](https://docs.python.org/3/howto/functional.html)

---
## 具体案例
- [[函数式编程工具示例]](函数式编程工具functools_itertools_sample.py)
