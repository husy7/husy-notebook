---
title: "函数式编程工具（functools & itertools）"
tags: [Python, 函数式编程, functools, itertools, 惰性求值, 缓存]
date: 2026-08-27
---
# 函数式编程工具（functools & itertools）
> `functools` 是"给函数加装备"的工具箱（固定参数、缓存结果、装饰器元信息），`itertools` 是"给迭代加流水线"的工具箱（组合、切片、分组序列）——两者配合，用几行声明式代码替代手写循环与状态管理。
## 原理 / 动机
- 解决什么实际问题：手写循环处理序列时代码冗长、易错（重复造轮子：分组、累积、笛卡尔积、去重缓存）；纯函数式风格又缺少"记忆化"“偏应用”这类高阶手段，导致逻辑重复、性能差。
- 核心原理（简洁全面）：`functools` 基于**高阶函数**思想——接收/返回函数作为对象：`partial` 冻结参数生成新函数，`lru_cache` 用哈希表对函数调用做记忆化，`wraps` 复制被装饰函数的元信息，`singledispatch` 按第一参数类型分发到不同实现。`itertools` 基于**惰性求值与迭代协议**——所有函数都返回迭代器，只在被消费时逐个产出元素，`C` 语言实现使组合开销极低。
- 为什么必须这样设计：函数是一等公民 + 迭代器是一次性流，两者的组合让"数据怎么产生"（itertools 管道）与"逻辑怎么复用"（functools 高阶工具）正交解耦；惰性实现是处理大数据与无限序列的唯一可行方案，而记忆化只有封装在装饰器层才能对调用方透明、不侵入业务代码。
## 应用示例
- 适用场景：递归/重复计算的缓存加速、回调/接口参数预填充、序列去重分组、多序列组合遍历、日志按时间窗聚合、数据管道级联处理。
- 快速上手步骤：
  1. 从 `itertools` 挑"产数据"的工具（`chain`、`islice`、`groupby`、`product`…）搭出惰性管道；
  2. 从 `functools` 挑"改函数"的工具（`partial`、`lru_cache`、`singledispatch`…）装饰或派生函数；
  3. 用 `list()` / `for` 在最末端一次性消费整条管道，享受低内存与声明式风格。
```python
# 可运行示例（已实测）
from functools import partial, reduce, lru_cache, wraps, singledispatch
from itertools import count, cycle, islice, chain, groupby, product, \
                       zip_longest, accumulate, takewhile
# ---------- functools ----------
# 1) partial：固定参数，派生新函数（偏应用）
pow2 = partial(pow, exp=2)
print(pow2(3))                       # 9
# 2) reduce：把序列折叠成单值
print(reduce(lambda a, b: a + b, [1, 2, 3, 4]))   # 10
# 3) lru_cache：记忆化递归，指数级 → 线性级
@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
print(fib(50))                       # 12586269025
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
print(hello.__name__, hello.__doc__)  # hello say hello
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
# ---------- itertools ----------
# 6) 无限迭代器：必须配合 islice 等截断工具
print(list(islice(count(10, 5), 4)))          # [10, 15, 20, 25]
print(list(islice(cycle("AB"), 5)))           # ['A','B','A','B','A']
# 7) 拼接与配对
print(list(chain([1, 2], [3, 4])))            # [1, 2, 3, 4]
print(list(zip_longest([1, 2, 3], "AB", fillvalue="?")))
                                               # [(1,'A'),(2,'B'),(3,'?')]
# 8) 折叠式迭代：累积
print(list(accumulate([1, 2, 3, 4])))         # [1, 3, 6, 10]
# 9) 条件截取：取到第一个不满足处即停
print(list(takewhile(lambda x: x < 3, [1, 2, 3, 1, 2])))  # [1, 2]
# 10) groupby：相邻分组（注意：只对连续相同元素分组）
print([(k, list(g)) for k, g in groupby("AAABBC")])
                                               # [('A',['A','A','A']),('B',['B','B']),('C',['C'])]
# 11) 笛卡尔积：替代多层嵌套循环
print(list(product("AB", repeat=2)))          # AA AB BA BB
```
## 边界 / 常见坑
- ❌ 错误现象：对无限迭代器（`count`/`cycle`/`repeat`）直接 `list()`，程序卡死不返回。  
  ✅ 正确做法：无限迭代器必须配合 `islice(n)`、`takewhile`、`zip`（以最短者终止）等截断手段消费，永远不要对其整体物化。
- ❌ 错误现象：对未排序的序列使用 `groupby`，期望得到"全局按 key 分组"的结果，实际得到大量重复的碎组（如 `[1,1,2,1]` 产出 `1` 组和 `2` 组各两份）。  
  ✅ 正确做法：`groupby` 只对**相邻**相同 key 元素分组；使用前先按分组 key 排序：`groupby(sorted(data, key=...), key=...)`，或改为用字典手动归集。
- ❌ 错误现象：`@lru_cache` 装饰了参数为列表/字典等可变对象的函数，报 `TypeError: unhashable type`；或缓存了有副作用/依赖外部状态的函数，后续调用拿到过期结果。  
  ✅ 正确做法：`lru_cache` 以参数为哈希键，仅适用于纯函数（无副作用、结果只由参数决定）且参数可哈希；含不可哈希参数时先转为元组/字符串，需要失效时用 `cache_clear()`。
- ❌ 错误现象：装饰器忘记加 `@wraps`，导致被装饰函数的 `__name__` 变成 `wrapper`、文档字符串丢失，调试与 `help()` 输出错乱。  
  ✅ 正确做法：手写装饰器时始终在 wrapper 外套一层 `@functools.wraps(f)`。
- 边界条件：`itertools` 返回的迭代器是一次性的、不支持索引/切片/`len()`，重复消费需 `tee()` 或重新生成；`partial` 生成的函数签名元数据变化（可用 `inspect.signature` 查看），传给强类型校验框架时需注意；`groupby` 返回的组迭代器在前进到下一组后即失效，必须在循环体内立即消费（如转 `list`）。
## 关联
- 前置知识：[[Python生成器与迭代器]]、[[Python函数基础]]、[[闭包与装饰器]]
- 类似概念：[[Python生成器与迭代器]]（区别是生成器是自己写 `yield` 产数据，itertools 是现成的标准化迭代器积木）；[[列表推导式]]（区别是推导式立即求值占内存，itertools 惰性求值省内存）
- 进阶知识：`operator` 模块（`itemgetter`/`attrgetter` 替代 lambda）、`functools.cached_property`、`functools.total_ordering`、`functools.cmp_to_key`、`itertools.pairwise`（3.10+）、`more_itertools` 第三方扩展库、Python 3.12 `itertools.batched`。
## 自我检验
1. 我能否一句话说清它解决什么问题？  
   - 答：`functools` 解决"函数复用与性能"（参数固化、缓存、类型分发），`itertools` 解决"序列处理代码冗长"（标准化的惰性迭代积木）。
2. 我能否写出最小可用示例？  
   - 答：`list(islice(count(1), 5))` 产出 `[1,2,3,4,5]`；`@lru_cache` 加在递归函数上即完成记忆化。
3. 我能否说出一个常见错误或边界？  
   - 答：`groupby` 只分相邻元素，未排序直接用会得到碎组；无限迭代器直接 `list()` 会卡死。
4. 我能否说出它和我已会的某个概念的区别？  
   - 答：与生成器的区别——生成器是"自己写产数据逻辑"，itertools 是"拿现成积木拼管道"；与手写循环的区别——声明式、惰性、C 实现更快。
## 对比与选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| functools + itertools（本文方案） | 声明式组合：高阶函数改函数 + 惰性迭代器搭管道 | 数据流处理、递归缓存、序列组合/分组，追求简洁与低内存 |
| 手写循环 / 生成器 | 用 `for` + `while` + 临时变量显式管理迭代状态 | 逻辑特殊、itertools 无对应积木、需要中途 break/复杂分支控制 |
| 第三方库（more_itertools / toolz） | 扩展更多迭代器原语（chunked、unique_everseen 等） | 标准库工具不够用、可引入第三方依赖的项目 |
## 参考
- [Python 官方文档：functools — 高阶函数与可调用对象操作](https://docs.python.org/3/library/functools.html)
- [Python 官方文档：itertools — 为高效循环而创建的迭代器函数](https://docs.python.org/3/library/itertools.html)
- [Python 官方 HOWTO：Functional Programming Modules](https://docs.python.org/3/howto/functional.html)
## 具体案例
- [[函数式编程工具示例]](函数式编程工具functools_itertools_sample.py)
