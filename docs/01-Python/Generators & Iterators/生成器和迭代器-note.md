---
title: "Python生成器与迭代器"
tags: [Python, 迭代器, 生成器, 惰性求值, 迭代协议]
date: 2026-08-27
---
# Python生成器与迭代器
> 迭代器是"知道自己下一个给什么"的对象，生成器是"用 `yield` 一行代码就能造出迭代器"的函数——它们共同解决"不把所有数据一次性塞进内存也能逐个取值"的问题。
## 原理 / 动机
- 解决什么实际问题：处理大文件、无限序列（如斐波那契数列）或数据流时，若先用列表把全部元素装入内存，会造成内存暴涨甚至 `MemoryError`；需要一个"按需取一个、用完即算"的机制。
- 核心原理（简洁全面）：**迭代协议**要求可迭代对象实现 `__iter__()` 返回一个迭代器，迭代器实现 `__next__()` 逐个产出元素、耗尽时抛出 `StopIteration`，`for` 循环正是基于这套协议工作的。**生成器**是函数体内含 `yield` 表达式的"生成器函数"被调用后返回的对象：调用时不执行函数体，每次 `next()` 时执行到 `yield` 处暂停并保留全部局部状态，下次从暂停点恢复。生成器自动实现了 `__iter__`/`__next__`，因此它是迭代器的一种。
- 为什么必须这样设计：把"产出数据的逻辑"与"取数据的节奏"解耦，由消费者拉动数据，才能实现惰性求值；`yield` 的暂停—恢复机制让 Python 解释器替你保存执行状态（局部变量、指令指针、求值栈），无需手写状态机类即可实现复杂的逐值生成逻辑。
## 应用示例
- 适用场景：逐行读取超大日志文件、构造无限序列、数据管道（生成器链式处理）、一次性遍历的大集合推导。
- 快速上手：
  1. 在函数体里用 `yield 值` 代替 `return`，函数即成为生成器函数；调用它得到生成器对象。
  2. 用 `for x in gen:` 或 `next(gen)` 消费值，函数在每次 `yield` 处暂停、下次恢复。
  3. 若需自定义类实现迭代器，则定义 `__iter__()`（返回 `self`）和 `__next__()`（耗尽时 `raise StopIteration`）。
```python
# 可运行示例：自定义迭代器 vs 生成器
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
## 边界 / 常见坑
- ❌ 错误现象：把生成器/迭代器 `for` 遍历一次后，第二次 `for` 循环不报错、不输出任何值，程序"看起来正常"但结果为空。  
  ✅ 正确做法：迭代器是一次性的，需要多次遍历前先 `list(gen)` 物化，或用 `itertools.tee(gen, n)` 分裂出多个独立迭代器，或每次重新调用生成器函数创建新对象。
- ❌ 错误现象：在生成器函数里写 `return value` 并期望调用方拿到该值，实际 `gen()` 只得到生成器对象，`return` 的值只出现在最终 `StopIteration.value` 属性里，且在 `for` 循环中被静默吞掉。  
  ✅ 正确做法：想逐个产出就用 `yield`；`return` 在生成器中只用于提前终止（可携带值供 `StopIteration.value` 读取），不要当作普通返回值使用。
- ❌ 错误现象：混淆"可迭代对象"与"迭代器"——对同一个列表多次 `for` 都正常，但对 `iter(lst)` 返回的迭代器第二次 `for` 为空，误以为语言行为不一致。  
  ✅ 正确做法：记住 `list` 等容器是**可迭代对象**，每次 `iter()` 都返回**新的**迭代器；而生成器对象本身既是可迭代对象又是迭代器（`__iter__` 返回自身），因此天然只能消费一次。
- 边界条件：生成器/迭代器不支持索引、切片与 `len()`，无法随机访问；当确实需要反复随机访问、缓存全部结果或反向遍历时，惰性序列不再适用，应改用列表等容器。此外，生成器表达式在最终被消费前不会执行任何计算，调试时若不打印中间结果，难以察觉逻辑错误。
## 关联
- 前置知识：[[Python函数基础]]、[[Python类与魔术方法]]、[[for循环与可迭代协议]]
- 类似概念：[[列表推导式]]（区别是生成器表达式用圆括号、惰性求值、省内存，列表推导式用方括号、立即求值、占内存）；[[装饰器]]（区别是装饰器包装并增强函数，生成器函数改变函数的调用语义为"返回可暂停的迭代器"）
- 进阶知识：`itertools` 模块（`count`、`chain`、`islice` 等迭代器代数工具）、`yield from` 子生成器委托、`send()`/`throw()`/`close()` 协程式交互、PEP 479（生成器内 `StopIteration` 自动转为 `RuntimeError`）、异步生成器（`async def` + `yield`）。
## 自我检验
1. 我能否一句话说清它解决什么问题？  
   - 答：解决"数据量太大或无限时，不把所有元素装入内存也能逐个取值"的问题。
2. 我能否写出最小可用示例？  
   - 答：`def g(): yield 1; yield 2`，然后 `for x in g(): print(x)` 输出 1、2。
3. 我能否说出一个常见错误或边界？  
   - 答：迭代器/生成器是一次性的，遍历耗尽后再次 `for` 会静默得到空结果，需物化或重新创建。
4. 我能否说出它和我已会的某个概念的区别？  
   - 答：生成器表达式与列表推导式语法几乎相同，区别仅在圆括号 vs 方括号，以及惰性求值（按需产出）vs 立即求值（全量装内存）。
## 对比与选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 生成器（yield / 生成器表达式） | 函数中用 `yield` 暂停—恢复，解释器自动保存状态并实现迭代协议 | 大文件流式读取、无限序列、数据管道，追求代码简洁与低内存 |
| 类实现迭代器（`__iter__` + `__next__`） | 手写类并自行管理迭代状态与 `StopIteration` 终止 | 需要复杂状态管理、多方法封装、或将迭代能力嵌入既有类的场合 |
| 列表 / 列表推导式 | 立即求值，全部元素装入内存 | 数据量可控、需要随机访问、切片、多次遍历或 `len()` 的场景 |
## 参考
- [Python 官方文档：迭代器类型]([https://docs.python.org/3/library/stdtypes.html#typeiter](https://docs.python.org/3/library/stdtypes.html#typeiter)
- [Python 官方文档：yield 表达式]([https://docs.python.org/3/reference/expressions.html#yieldexpr](https://docs.python.org/3/reference/expressions.html#yieldexpr)
- [Python Wiki：Iterator](https://wiki.python.org/moin/Iterator.html)
- [Python 官方文档：itertools — 高效循环的迭代器函数](https://docs.python.org/3/library/itertools.html)
## 具体案例
- [[Python生成器与迭代器_sample.py]](python生成器与迭代器_sample.py)
