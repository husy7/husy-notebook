---
title: "Python生成器与迭代器"
tags: [Python, 迭代器, 生成器, 惰性求值, 迭代协议]
date: 2026-08-27
---

# Python生成器与迭代器

## 定义

> 迭代器是"知道自己下一个给什么"的对象，生成器是"用 `yield` 一行代码就能造出迭代器"的函数——它们共同解决"不把所有数据一次性塞进内存也能逐个取值"的问题。

**它是什么**：迭代器（iterator）是实现了迭代协议的对象——`__next__()` 每次产出一个元素，耗尽时抛出 `StopIteration` 表示结束；生成器（generator）则是函数体内含 `yield` 表达式的"生成器函数"被调用后返回的对象，它自动实现了 `__iter__`/`__next__`，因此是迭代器的一种特殊、简化的实现形态。

**解决什么问题**：处理大文件、无限序列（如斐波那契数列）或数据流时，若先用列表把全部元素装入内存，会造成内存暴涨甚至 `MemoryError`；迭代器/生成器提供"按需取一个、用完即算"的机制，让内存占用只与"当前这一步"有关、与数据总量无关。

**核心特征**：惰性求值（元素按需产出而非提前全部生成）、可迭代对象与迭代器合一（`__iter__` 返回自身）、一次性消费（耗尽后无法重新遍历）；生成器还能在 `yield` 处暂停并保留全部局部状态，暂停点可多次。

**适用范畴**：逐行读取超大日志文件、构造无限序列、生成器链式数据管道、对只遍历一次的大集合做推导（生成器表达式），以及一切"数据总量未知或极大、且不需要随机访问"的取值场景。

## 原理

- **迭代协议（底层机制）**：`for` 循环正是基于这套协议工作的——可迭代对象（iterable）实现 `__iter__()` 返回一个迭代器，迭代器实现 `__next__()` 逐个产出元素、耗尽时抛出 `StopIteration`；`for x in it:` 等价于反复调用 `next()` 直到捕获 `StopIteration`。
- **生成器的执行模型**：调用生成器函数时**不执行函数体**，只返回生成器对象；每次 `next()` 从上次暂停点执行到下一个 `yield` 处，把该值交给调用方并暂停，下次调用再从暂停点恢复；函数体执行完毕（或抛出异常）时迭代自然结束。最小形态：`def g(): yield 1; yield 2`，再 `for x in g(): print(x)` 即输出 1、2。
- **状态由谁保存**：`yield` 的暂停—恢复机制让 Python 解释器替你保存执行状态——局部变量、指令指针、求值栈——因此无需手写状态机类即可实现复杂的逐值生成逻辑；对比"类实现迭代器"需要自己维护状态字段与终止条件。
- **设计动机**：把"产出数据的逻辑"与"取数据的节奏"解耦，由消费者**拉动**数据，才能实现惰性求值；这是生成器相对"先算完再整段存内存"的根本优势，也让"产出方"无需知道消费者要取多少个。
- **边界语义**：生成器内的 `return` 只用于提前终止，其携带的值存入最终抛出的 `StopIteration.value` 属性，且会被 `for` 循环静默吞掉；生成器表达式在被消费前不执行任何计算（惰性的极端体现）。

## 应用

**典型使用场景**：逐行读取超大日志文件（不整文件入内存）；构造无限序列（斐波那契、计数器）；数据管道——多个生成器首尾相接链式处理（如 `only_even(read_lines(...))`）；一次性遍历的大集合推导（生成器表达式替代列表推导式）。

**快速上手步骤**：
1. 在函数体里用 `yield 值` 代替 `return`，函数即成为生成器函数；调用它得到生成器对象。
2. 用 `for x in gen:` 或 `next(gen)` 消费值——函数在每次 `yield` 处暂停、下次恢复。
3. 若需自定义类实现迭代器，则定义 `__iter__()`（返回 `self`）和 `__next__()`（耗尽时 `raise StopIteration`）；若同一份数据要遍历多遍，用 `itertools.tee` 分裂或每次重建生成器。

**注意事项 / 常见坑**：
- ❌ 生成器/迭代器被 `for` 遍历一次后，第二次 `for` 不报错、不输出任何值，程序"看起来正常"但结果为空。
  ✅ 迭代器是一次性的：需要多次遍历前先 `list(gen)` 物化，或用 `itertools.tee(gen, n)` 分裂出多个独立迭代器，或每次重新调用生成器函数创建新对象。
- ❌ 在生成器函数里写 `return value` 并期望调用方拿到该值——实际 `gen()` 只得到生成器对象，`return` 的值只出现在最终 `StopIteration.value` 属性里，且在 `for` 循环中被静默吞掉。
  ✅ 想逐个产出就用 `yield`；`return` 在生成器中只用于提前终止（可携带值供 `StopIteration.value` 读取），不要当作普通返回值使用。
- ❌ 混淆"可迭代对象"与"迭代器"：对同一个列表多次 `for` 都正常，但对 `iter(lst)` 返回的迭代器第二次 `for` 为空，误以为语言行为不一致。
  ✅ `list` 等容器是**可迭代对象**，每次 `iter()` 都返回**新的**迭代器；而生成器对象本身既是可迭代对象又是迭代器（`__iter__` 返回自身），因此天然只能消费一次。
- 边界：生成器/迭代器不支持索引、切片与 `len()`，无法随机访问；当确实需要反复随机访问、缓存全部结果或反向遍历时，惰性序列不再适用，应改用列表等容器。此外，生成器表达式在最终被消费前不会执行任何计算，调试时若不打印中间结果，难以察觉逻辑错误。

```python
# 可运行示例：自定义迭代器 vs 生成器（含生成器表达式、数据管道、一次性与 tee 分裂）
from itertools import tee

# 1) 类实现迭代器：手动管理状态与终止条件
class Countdown:
    def __init__(self, start):
        self.n = start
    def __iter__(self):
        return self  # 迭代器返回自身
    def __next__(self):
        if self.n <= 0:
            raise StopIteration  # 协议要求的终止信号
        self.n -= 1
        return self.n + 1

# 2) 生成器函数：等价逻辑，代码更简洁
def countdown_gen(start):
    n = start
    while n > 0:
        yield n  # 每次在此暂停，保留 n 的状态
        n -= 1

# 3) 生成器表达式：惰性版推导式（注意是圆括号）
squares = (x * x for x in range(10**8))  # 瞬间创建，不占内存
print(next(squares))  # 0
print(next(squares))  # 1

# 4) 数据管道：生成器链式处理
def read_lines(nums):
    for i in nums:
        yield i
def only_even(it):
    for x in it:
        if x % 2 == 0:
            yield x
pipeline = only_even(read_lines(range(10)))
print(list(pipeline))  # [0, 2, 4, 6, 8]

# 5) 迭代器/生成器的一次性：耗尽后再次遍历不会报错但无输出
gen = countdown_gen(3)
print(list(gen))       # [3, 2, 1]
print(list(gen))       # [] —— 已耗尽，静默返回空
# 需要多次遍历时：物化或分裂
gen2, gen3 = tee(countdown_gen(3))
print(list(gen2), list(gen3))  # [3, 2, 1] [3, 2, 1]
```

案例详解：① `Countdown` 类演示手写迭代器需自行维护状态字段 `n` 与 `StopIteration` 终止条件；② `countdown_gen` 用 `yield` 实现相同逻辑，解释器自动保存 `n` 的状态，代码量大幅精简；③ 生成器表达式用圆括号瞬间创建、不占内存——即便 `range(10**8)` 也不会 `MemoryError`，验证惰性求值；④ 两个生成器函数首尾相接构成数据管道，`only_even` 从 `read_lines` 逐个拉取并过滤，元素边产边耗、全程无中间列表；⑤ 展示"一次性"语义——耗尽后 `list()` 静默返回 `[]`，需要多遍遍历时用 `itertools.tee` 分裂成独立副本（注意 `tee` 会缓存中间元素，大数据量下慎用）。

---
## 关联
- 前置：[[Python函数基础]]、[[Python类与魔术方法]]、[[for循环与可迭代协议]]
- 类似：[[列表推导式]]（区别是生成器表达式用圆括号、惰性求值、省内存，列表推导式用方括号、立即求值、占内存）；[[装饰器]]（区别是装饰器包装并增强函数，生成器函数改变函数的调用语义为"返回可暂停的迭代器"）
- 进阶：[[itertools 模块]]（`count`、`chain`、`islice` 等迭代器代数工具）、[[yield from]]（子生成器委托）、[[生成器 send throw close 协程式交互]]（`send()`/`throw()`/`close()`）、[[PEP 479]]（生成器内 `StopIteration` 自动转为 `RuntimeError`）、[[异步生成器]]（`async def` + `yield`）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 生成器（yield / 生成器表达式，本文方案） | 函数中用 `yield` 暂停—恢复，解释器自动保存状态并实现迭代协议 | 大文件流式读取、无限序列、数据管道，追求代码简洁与低内存 |
| 类实现迭代器（`__iter__` + `__next__`） | 手写类并自行管理迭代状态与 `StopIteration` 终止 | 需要复杂状态管理、多方法封装、或将迭代能力嵌入既有类的场合 |
| 列表 / 列表推导式 | 立即求值，全部元素装入内存 | 数据量可控、需要随机访问、切片、多次遍历或 `len()` 的场景 |

---
## 参考
- [Python 官方文档：迭代器类型](https://docs.python.org/3/library/stdtypes.html#typeiter)
- [Python 官方文档：yield 表达式](https://docs.python.org/3/reference/expressions.html#yieldexpr)
- [Python Wiki：Iterator](https://wiki.python.org/moin/Iterator.html)
- [Python 官方文档：itertools — 高效循环的迭代器函数](https://docs.python.org/3/library/itertools.html)

---
## 具体案例
- [[Python生成器与迭代器_sample.py]](python生成器与迭代器_sample.py)
